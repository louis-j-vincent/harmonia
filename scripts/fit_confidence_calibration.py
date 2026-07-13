"""Fit the display-layer confidence calibration (audit 2026-07-13, step 1b).

Runs the REAL production pipeline (infer_chords_v1, post-#25 defaults) on
held-out jazz1460 songs, collects per-output-chord (raw fused score, correct)
pairs, fits an isotonic map raw→P(correct), and saves it to
data/cache/confidence_calibration.npz, which chord_pipeline_v1 auto-loads
(_get_conf_calibrator) for the `confidence` field the app displays.

Definitions:
  raw score  = confidence_raw × root_conf   (quality conf × span root posterior;
               falls back to confidence_raw when the root model is off)
  correct    = GT at the chord's midpoint matches BOTH the root pc and the
               5-way quality family (maj/min/dom/hdim/dim) — i.e. "the chord
               the app displays is right at the granularity it displays".

Splits (all disjoint from eval_two_pass_801d.py's 70..95 eval set):
  fit  = jazz1460 4/4 songs idx 20..50   (n≈30)
  test = jazz1460 4/4 songs idx 100..112 (n≈12) — reliability/ECE reported on
         this split for raw, fused, and calibrated so the gate (ECE < 0.05,
         audit step 1c) is honest.

Usage:
    .venv/bin/python scripts/fit_confidence_calibration.py           # fit + save + report
    .venv/bin/python scripts/fit_confidence_calibration.py --dry-run # report only, no save
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from eval_two_pass_801d import MMA_TO_Q5  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models import chord_pipeline_v1 as P  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
OUT = REPO / "data" / "cache" / "confidence_calibration.npz"
NOTE_TO_PC = {n: i for i, n in enumerate(P.NOTE)}


def collect(songs, renderer, sf2) -> tuple[np.ndarray, np.ndarray]:
    """(raw fused score, correct) per output chord over a list of DB records."""
    scores, correct = [], []
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = Path(cache_dir)
        for i, rec in enumerate(songs):
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            if not spans:
                continue
            print(f"  [{i + 1}/{len(songs)}] {rec['song_id']}", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                chart = P.infer_chords_v1(tmp, cache_dir=cache)
            finally:
                tmp.unlink(missing_ok=True)
            for c in chart.chords:
                label = c["label"]
                if ":" not in label:
                    continue
                mid = 0.5 * (c["start_s"] + c["end_s"])
                gt = next(((r, q) for t0, t1, r, q in spans if t0 <= mid < t1), None)
                if gt is None:
                    continue
                gt_fam = MMA_TO_Q5.get(gt[1])
                if gt_fam is None:
                    continue
                pred_pc = NOTE_TO_PC.get(label.split(":", 1)[0])
                q5i = P._harte_to_q5idx(label.split(":", 1)[1])
                pred_fam = P._Q5_NAMES[q5i] if q5i is not None else None
                if pred_pc is None or pred_fam is None:
                    continue
                rc = c.get("root_conf")
                raw = c["confidence_raw"] * rc if rc is not None else c["confidence_raw"]
                scores.append(raw)
                correct.append(pred_pc == gt[0] and pred_fam == gt_fam)
    return np.asarray(scores, float), np.asarray(correct, bool)


def ece_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """(bin rows, ECE) — same binning as scripts/plot_calibration.py."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < n_bins - 1 else (conf <= 1.0))
        if m.sum() >= 10:
            rows.append((edges[i], edges[i + 1], conf[m].mean(),
                         correct[m].mean(), int(m.sum())))
            ece += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return rows, ece


def report(name: str, conf: np.ndarray, correct: np.ndarray):
    rows, ece = ece_bins(conf, correct)
    print(f"\n  {name}:  ECE = {ece:.4f}")
    for lo, hi, c, a, n in rows:
        print(f"    [{lo:.1f},{hi:.1f})  conf={c:.3f}  acc={a:.3f}  n={n}")
    return ece


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-start", type=int, default=20)
    ap.add_argument("--fit-n", type=int, default=30)
    ap.add_argument("--test-start", type=int, default=100)
    ap.add_argument("--test-n", type=int, default=12)
    ap.add_argument("--interleave", action="store_true",
                    help="fit=even/test=odd songs from one pool (idx 20..70 + "
                         "96..122, eval set 70..95 excluded). Kills the "
                         "fit/test difficulty shift the block split showed "
                         "(82.4%% vs 74.4%% base acc) while staying "
                         "song-disjoint.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from sklearn.isotonic import IsotonicRegression

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    if args.interleave:
        pool = jz[20:70] + jz[96:122]          # eval split 70..95 excluded
        fit_set = pool[0::2]
        test_set = pool[1::2]
    else:
        fit_set = jz[args.fit_start:args.fit_start + args.fit_n]
        test_set = jz[args.test_start:args.test_start + args.test_n]
    if OUT.exists():
        print(f"NOTE: {OUT.name} already exists — pipeline runs below are "
              f"already calibrated; refit uses confidence_raw, unaffected.")

    print(f"fit split: {len(fit_set)} songs (idx {args.fit_start}..)")
    fs, fc = collect(fit_set, renderer, sf2)
    print(f"  -> {len(fs)} chords, base accuracy {fc.mean():.1%}")
    print(f"test split: {len(test_set)} songs (idx {args.test_start}..)")
    ts, tc = collect(test_set, renderer, sf2)
    print(f"  -> {len(ts)} chords, base accuracy {tc.mean():.1%}")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds="clip").fit(fs, fc.astype(float))
    x, y = iso.X_thresholds_, iso.y_thresholds_

    print("\n=== reliability on the DISJOINT test split ===")
    report("raw fused score (pre-calibration)", ts, tc)
    cal_t = np.interp(ts, x, y)
    ece_cal = report("calibrated (isotonic, fit split only)", cal_t, tc)

    gate = ece_cal < 0.05
    print(f"\nGATE (audit step 1c): calibrated test ECE {ece_cal:.4f} "
          f"{'< 0.05 — PASS' if gate else '>= 0.05 — FAIL'}")

    if args.dry_run:
        print("(dry-run: not saved)")
        return
    np.savez(OUT, x=x, y=y,
             meta=json.dumps({
                 "fitted": "2026-07-13", "n_fit": len(fs), "n_test": len(ts),
                 "split": ("interleave pool 20..70+96..122 (eval 70..95 excluded)"
                           if args.interleave else
                           f"blocks fit {args.fit_start}+{args.fit_n} / "
                           f"test {args.test_start}+{args.test_n}"),
                 "test_ece_calibrated": round(float(ece_cal), 4),
                 "target": "root_pc AND q5 family @ chord midpoint",
                 "score": "confidence_raw * root_conf",
             }))
    print(f"saved -> {OUT}  ({len(x)} breakpoints)")


if __name__ == "__main__":
    main()
