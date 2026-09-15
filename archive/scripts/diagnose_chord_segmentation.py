#!/usr/bin/env python3
"""Does musx DETECT chord CHANGES (boundaries), separate from labelling them?

The density diagnostic said dense jazz kills accuracy and pointed at
harmonic-rhythm RESOLUTION as the lever. This asks the prior question directly:
when the GT chord changes, does the model's frame-level output change there?
Boundary detection is orthogonal to which label it picks.

Three change-point streams per song, compared to GT chord-change times:
  - musx-raw   : argmax of the 23 ms frame posteriors (triad head), flicker-
                 suppressed by a median filter + min-segment merge, shifted by
                 the measured -0.113 s musx latency. This is what the MODEL sees
                 before any pipeline grid quantises it.
  - pipeline   : the shipped infer_chords_v1 chord boundaries (optional; needs
                 re-inference, --with-pipeline).
A GT change is RECALLED if some predicted change lands within ±tol; a predicted
change is PRECISE if some GT change is within ±tol. over_seg = n_pred / n_gt.

Runs on GuitarSet comp (bundled audio, trusted GT, cached musx probs from the
benchmark run → fast). Read-only.

Usage:
    .venv/bin/python scripts/diagnose_chord_segmentation.py             # cached-probs songs
    .venv/bin/python scripts/diagnose_chord_segmentation.py --with-pipeline
    .venv/bin/python scripts/diagnose_chord_segmentation.py --max 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402
from scripts.build_jaah_corpus import parse_jaah, load_lab, LABS_DIR   # noqa: E402

GS = REPO / "data" / "cache" / "guitarset"
ANN, AUD = GS / "annotation", GS / "audio"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PINS = REPO / "docs" / "research_sessions" / "jaah_source_pins.json"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
PLOT = REPO / "docs" / "plots" / "chord_segmentation_boundaries.png"

MUSX_LAT = 0.113          # musx runs late (measured 2026-07-27)
MIN_SEG_S = 0.14          # merge frame-runs shorter than this (flicker)
MED_W = 5                 # median-filter width in frames (~116 ms)
TOLS = (0.15, 0.25, 0.5)


def gt_changes(stem):
    j = json.loads((ANN / f"{stem}.jams").read_text())
    ch = [a for a in j["annotations"] if a["namespace"] == "chord"][-1]["data"]
    rows = []
    for d in ch:
        r, f, _ = parse_jaah(d["value"].split("/")[0])
        rows.append((d["time"], r, f))
    rows.sort()
    return [rows[i][0] for i in range(1, len(rows))
            if (rows[i][1], rows[i][2]) != (rows[i - 1][1], rows[i - 1][2])]


def gt_changes_lab(slug):
    rows = []
    for t0, t1, lab in load_lab(LABS_DIR / f"{slug}.lab"):
        r, f, _ = parse_jaah(lab)
        rows.append((t0, r, f))
    rows.sort()
    return [rows[i][0] for i in range(1, len(rows))
            if (rows[i][1], rows[i][2]) != (rows[i - 1][1], rows[i - 1][2])]


def _median_filter(a, w):
    if w <= 1:
        return a
    h = w // 2
    pad = np.pad(a, (h, h), mode="edge")
    return np.array([np.bincount(pad[i:i + w]).argmax() for i in range(len(a))])


def musx_raw_changes(wav):
    """Frame argmax of triad head -> (root, triad) per frame -> change times,
    flicker-suppressed and latency-corrected."""
    triad = frame_posteriors(wav)[0]              # (n_frame, 73)
    am = triad.argmax(1)
    am = _median_filter(am, MED_W)
    # merge runs shorter than MIN_SEG_S into the previous run
    min_fr = max(1, int(round(MIN_SEG_S / FRAME_DT)))
    changes, seg_start, prev = [], 0, am[0]
    for i in range(1, len(am)):
        if am[i] != prev:
            if i - seg_start >= min_fr:
                changes.append(seg_start * FRAME_DT - MUSX_LAT)
            seg_start, prev = i, am[i]
    # first change point is the first real segment boundary (drop the t≈0 start)
    return [c for c in changes if c > 0.05]


def pipeline_changes(wav):
    from harmonia.eval.accuracy_score import run_prediction
    pred = run_prediction(wav)
    ch = []
    for i in range(1, len(pred)):
        a, b = pred[i - 1], pred[i]
        ka = (None, None) if a.is_nc else (a.root_pc, __fam(a.quality))
        kb = (None, None) if b.is_nc else (b.root_pc, __fam(b.quality))
        if ka != kb:
            ch.append(b.t0)
    return ch


def __fam(q):
    from harmonia.eval.accuracy_score import chord_family
    return chord_family(q)


def prf(gt, pred, tol):
    if not gt:
        return float("nan"), float("nan"), float("nan")
    rec = np.mean([any(abs(g - p) <= tol for p in pred) for g in gt])
    prec = np.mean([any(abs(p - g) <= tol for g in gt) for p in pred]) if pred else 0.0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f


def style_of(stem):
    return stem.split("_")[1].split("-")[0].rstrip("0123456789")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-pipeline", action="store_true")
    ap.add_argument("--jaah", action="store_true",
                    help="dense-regime test: JAAH pinned songs via cached probs")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args(argv)

    # build (label, wav_path, gt_change_times, style) work items
    items = []
    if args.jaah:
        pins = json.loads(PINS.read_text())
        for slug, vid in sorted(pins.items()):
            if not (PROB_CACHE / f"{vid}.npz").exists():
                continue
            if not (LABS_DIR / f"{slug}.lab").exists():
                continue
            items.append((slug, JAAH_AUD / f"{vid}.wav", gt_changes_lab(slug), "jazz"))
    else:
        for jp in sorted(ANN.glob("*_comp.jams")):
            stem = jp.stem
            if (PROB_CACHE / f"{stem}_mic.npz").exists() and (AUD / f"{stem}_mic.wav").exists():
                items.append((stem, AUD / f"{stem}_mic.wav", gt_changes(stem), style_of(stem)))
    if args.max:
        items = items[:args.max]
    src = "JAAH (dense jazz, cached probs)" if args.jaah else "GuitarSet comp"
    print(f"{len(items)} {src} excerpts\n", flush=True)

    rows = []
    for stem, wav, gt, style in items:
        raw = musx_raw_changes(wav)
        rec = {"stem": stem, "style": style, "n_gt": len(gt),
               "n_raw": len(raw), "over_raw": len(raw) / max(1, len(gt))}
        for tol in TOLS:
            p, r, f = prf(gt, raw, tol)
            rec[f"raw_r{tol}"] = r
            rec[f"raw_p{tol}"] = p
        if args.with_pipeline:
            pip = pipeline_changes(wav)
            rec["n_pip"] = len(pip)
            rec["over_pip"] = len(pip) / max(1, len(gt))
            for tol in TOLS:
                p, r, f = prf(gt, pip, tol)
                rec[f"pip_r{tol}"] = r
                rec[f"pip_p{tol}"] = p
        rows.append(rec)
        print(f"  {stem:<26} gt={len(gt):>2} raw={len(raw):>2} "
              f"rec@.25={rec['raw_r0.25']:.2f} prec@.25={rec['raw_p0.25']:.2f}",
              flush=True)

    print("\n=== MUSX-RAW boundary detection vs GT chord changes ===")
    for tol in TOLS:
        r = np.nanmean([x[f"raw_r{tol}"] for x in rows])
        p = np.nanmean([x[f"raw_p{tol}"] for x in rows])
        print(f"  tol±{tol}s   recall={r:.2f}  precision={p:.2f}")
    print(f"  over-segmentation (n_raw/n_gt): {np.mean([x['over_raw'] for x in rows]):.2f}×")

    print("\n  recall@.25 by style (does musx SEE the changes?):")
    by = defaultdict(list)
    for x in rows:
        by[x["style"]].append(x["raw_r0.25"])
    for s in sorted(by):
        print(f"    {s:<8} {np.nanmean(by[s]):.2f}  (n={len(by[s])})")

    if args.with_pipeline:
        print("\n=== PIPELINE boundary detection vs GT ===")
        for tol in TOLS:
            r = np.nanmean([x[f"pip_r{tol}"] for x in rows])
            p = np.nanmean([x[f"pip_p{tol}"] for x in rows])
            print(f"  tol±{tol}s   recall={r:.2f}  precision={p:.2f}")
        print(f"  over-seg: {np.mean([x['over_pip'] for x in rows]):.2f}×")

    _plot(rows, args.with_pipeline)
    return 0


def _plot(rows, with_pip):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = sorted({x["style"] for x in rows})
    xs = np.arange(len(styles))
    rec = [np.nanmean([x["raw_r0.25"] for x in rows if x["style"] == s]) for s in styles]
    over = [np.mean([x["over_raw"] for x in rows if x["style"] == s]) for s in styles]
    ax[0].bar(xs, rec, color="#5aa9ff")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(styles, rotation=30)
    ax[0].set_ylabel("musx-raw recall of GT changes (±0.25s)")
    ax[0].set_title("Does musx SEE chord changes? (by style)")
    ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(xs, over, color="#ffa24d")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(styles, rotation=30)
    ax[1].set_ylabel("over-segmentation (n_raw / n_gt)")
    ax[1].set_title("Over-segmentation by style")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=110)
    print(f"\nwrote {PLOT.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
