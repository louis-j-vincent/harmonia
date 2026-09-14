"""Does `c` mean anything? — measure it against verified ground truth.

Louis, 2026-08-01: the pipeline's confidence does not have to be calibrated, it
has to be MEANINGFUL. `scripts/chord_lm_ratio_rule.py` turned that into a
number: the ratio rule only beats an LM-only threshold once the pipeline's
confidence reaches **AUC >= 0.70** at separating its own right answers from its
wrong ones. This script measures where `harmonia_min` actually sits.

Corpus: **GuitarSet** (Zenodo 3371780) — the only corpus on this disk with BOTH
real audio and 100%-verified chord ground truth shipped together, so there is no
alignment step and therefore no circularity. 180 comping excerpts, ~30 s each,
6 players x 5 styles. GT = the `chord` annotation's LAST entry (the "performed",
manually-verified one), reduced to root + 7-family.

Audited elsewhere and rejected for this purpose:
  McGill-Billboard  890 expert annotations, but NO audio ships (chroma only) —
                    musx is an audio model, so it cannot be run
  JAAH              115 label files, audio directory EMPTY
  RWC               audio directory is 0 bytes (purged)
  CHOCO             audio directory empty

What is measured, per decoded chord span:
  correct_root    the GT sounding at the span's midpoint has the same root
  correct_full    same root AND same family
  c               harmonia_min's confidence for that span

and then AUC, the reliability curve, and — the number Louis asked for — where
`c` lands against the 0.70 bar.

CAVEAT stated up front: GuitarSet is solo acoustic guitar in 30 s excerpts. It
isolates the chord model from band-mix and alignment confounds, which is what we
want for auditing `c`, but it is NOT our production domain, and 30 s is too
short for section folding to fire — so this measures `_segment_confidence`, the
ACOUSTIC branch of `c`, not the folding branch.

    .venv/bin/python scripts/chord_conf_auc_guitarset.py --max 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_jaah_corpus import parse_jaah              # noqa: E402
from harmonia_min import pipeline as minpipe             # noqa: E402

ANN = Path("data/cache/guitarset/annotation")
AUD = Path("data/cache/guitarset/audio")

# harmonia_min quality tail -> the 7-family alphabet parse_jaah returns
TAIL_FAMILY = {
    "": "maj", "^7": "maj", "^9": "maj", "6": "maj", "69": "maj", "2": "maj",
    "-": "min", "-7": "min", "-9": "min", "-6": "min", "-^7": "min",
    "7": "dom", "9": "dom", "13": "dom", "7sus4": "sus",
    "o": "dim", "o7": "dim", "h7": "hdim", "+": "aug",
    "sus2": "sus", "sus4": "sus",
}


def load_gt(jams_path: Path):
    j = json.loads(jams_path.read_text())
    chords = [a for a in j["annotations"] if a["namespace"] == "chord"]
    ann = chords[-1] if chords else None
    rows = []
    for d in (ann["data"] if ann else []):
        root, fam, _ = parse_jaah(d["value"].split("/")[0])
        rows.append((float(d["time"]), float(d["time"]) + float(d["duration"]),
                     root, fam))
    rows.sort()
    return rows


def gt_at(rows, t):
    for t0, t1, root, fam in rows:
        if t0 <= t < t1:
            return root, fam
    return None, None


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def reliability(conf, correct, n_bins=8):
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    edges = np.linspace(conf.min(), conf.max(), n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf <= hi if hi == edges[-1] else conf < hi)
        if sel.sum() >= 15:
            out.append((lo, hi, int(sel.sum()), float(conf[sel].mean()),
                        float(correct[sel].mean())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--out", default="docs/research_sessions/conf_auc_guitarset.json")
    args = ap.parse_args()

    jamses = sorted(ANN.glob("*_comp.jams"))
    if args.max:
        # spread across players and styles rather than taking the first N
        step = max(1, len(jamses) // args.max)
        jamses = jamses[::step][:args.max]
    print(f"{len(jamses)} comping excerpts\n")

    rows = []
    t_start = time.time()
    for i, jp in enumerate(jamses, 1):
        stem = jp.stem
        wav = AUD / f"{stem}_mic.wav"
        if not wav.exists():
            hits = list(AUD.glob(f"{stem}*.wav"))
            if not hits:
                continue
            wav = hits[0]
        gt = load_gt(jp)
        if not gt:
            continue
        try:
            chart = minpipe.analyze(wav, title=stem, file_key=stem)
        except Exception as e:
            print(f"  [{i}/{len(jamses)}] {stem}: FAILED {type(e).__name__}: {e}")
            continue
        n = 0
        for sec in chart.get("sections", []):
            for bar in sec.get("bars", []):
                for ch in bar:
                    if ch.get("nc") or "t0" not in ch:
                        continue
                    mid = 0.5 * (ch["t0"] + ch["t1"])
                    g_root, g_fam = gt_at(gt, mid)
                    if g_root is None:
                        continue
                    fam = TAIL_FAMILY.get(ch.get("q", ""))
                    rows.append({
                        "song": stem, "c": float(ch["c"]),
                        "dur": float(ch["t1"] - ch["t0"]),
                        "root_ok": int(ch["root"]) == int(g_root),
                        "full_ok": (int(ch["root"]) == int(g_root)
                                    and fam is not None and fam == g_fam),
                        "n_obs": ch.get("n_obs"),
                    })
                    n += 1
        print(f"  [{i}/{len(jamses)}] {stem}: {n} scored spans "
              f"({time.time()-t_start:.0f}s)", flush=True)

    if not rows:
        print("no rows scored — check paths")
        return

    c = np.array([r["c"] for r in rows])
    root_ok = np.array([r["root_ok"] for r in rows], bool)
    full_ok = np.array([r["full_ok"] for r in rows], bool)
    n_fold = sum(1 for r in rows if r["n_obs"] is not None)

    print("\n" + "=" * 74)
    print(f"RESULT — {len(rows)} decoded chord spans over "
          f"{len({r['song'] for r in rows})} excerpts")
    print("=" * 74)
    print(f"  root accuracy      {root_ok.mean():.4f}")
    print(f"  root+family        {full_ok.mean():.4f}")
    print(f"  spans from folding {n_fold} ({100*n_fold/len(rows):.1f}%) "
          "— 30 s excerpts, so this is the ACOUSTIC branch of `c`")
    print(f"  c: min {c.min():.3f}  median {np.median(c):.3f}  max {c.max():.3f}")

    a_root, a_full = auc(c, root_ok), auc(c, full_ok)
    print(f"\n  AUC(c separates right root from wrong)   {a_root:.4f}")
    print(f"  AUC(c separates right root+family)       {a_full:.4f}")
    bar = 0.70
    verdict = ("CLEARS the 0.70 bar — the ratio rule is worth using"
               if a_root >= bar else
               "BELOW the 0.70 bar — the ratio rule would be WORSE than "
               "ignoring `c` entirely")
    print(f"\n  vs the 0.70 requirement from chord_lm_ratio_rule.py: {verdict}")

    print("\n  reliability (root):")
    print(f"    {'c bin':<16} {'n':>6} {'mean c':>8} {'accuracy':>10}")
    for lo, hi, n, mc, acc in reliability(c, root_ok):
        print(f"    {f'{lo:.2f}-{hi:.2f}':<16} {n:>6} {mc:>8.3f} {acc:>10.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_spans": len(rows), "auc_root": a_root, "auc_full": a_full,
         "root_acc": float(root_ok.mean()), "full_acc": float(full_ok.mean()),
         "rows": rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
