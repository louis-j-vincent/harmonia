"""Mission 3: fit the REAL-audio confidence calibration on the FUSED score.

What this fixes (issue #29)
---------------------------
The current ``real`` map (confidence_calibration_real.npz) was fit on the
QUALITY head alone (``confidence_raw``), so root uncertainty is NOT folded into
the displayed confidence — a regression vs the fused design that #26 shipped for
synth. This script refits the real map on the honest fused score

    fused_raw = confidence_raw * root_conf   (quality max-prob × root posterior)

using Mission 1's NON-circular real-audio benchmark (beat-anchored iReal GT), so
the root-blindness fix also holds on the default real path. It saves
``score_kind="fused"`` into the map; chord_pipeline_v1._get_conf_calibrator then
feeds it the fused raw automatically (no other code change).

Gates / prerequisites
---------------------
- Mission 1: data/real_audio_benchmark/aligned_chords_per_song.json (benchmark).
- Mission 2: retrained quality head on disk (ctx_v2.npz / ctx_v3.npz) — the
  pipeline loads it automatically; nothing here references it directly.
Until both exist this script prints a clear "waiting on Mission 1/2" and exits 0
(dry) or 3 (fit) — it does NOT fabricate a map.

Method
------
Runs the production pipeline on each benchmark song, harvests
(confidence_raw, fused_raw, correct) per output chord (correct = pred root pc AND
q5 family match GT at the chord midpoint), fits isotonic fused_raw→P(correct),
and reports an honest ECE via 5-fold SONG-held-out CV (never fit & score the same
song). Also reports the conf-only map (the #29 regression) as a baseline, so the
root-folding win is visible.

Usage:
    .venv/bin/python scripts/calibrate_quality.py            # fit + save + CV report
    .venv/bin/python scripts/calibrate_quality.py --dry-run  # report only, no save
    .venv/bin/python scripts/calibrate_quality.py --benchmark path/to.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from _calib_common import (  # noqa: E402
    MissingBenchmark, collect_benchmark, ece_bins, load_benchmark,
    report_reliability,
)
from harmonia.models import chord_pipeline_v1 as P  # noqa: E402

OUT = P.CONF_CALIBRATION_REAL_PATH  # data/cache/confidence_calibration_real.npz


def cv_ece(score: np.ndarray, correct: np.ndarray, song_id: np.ndarray,
           n_folds: int = 5) -> tuple[float, np.ndarray]:
    """5-fold SONG-held-out isotonic ECE (honest — no song in both fit & score)."""
    from sklearn.isotonic import IsotonicRegression
    songs = np.array(sorted(set(song_id.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    cal = np.zeros_like(score, dtype=float)
    n_folds = min(n_folds, len(songs))
    for fold in np.array_split(songs, n_folds):
        te = np.isin(song_id, fold)
        if te.all() or not te.any():
            continue
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            score[~te], correct[~te].astype(float))
        cal[te] = iso.predict(score[te])
    return ece_bins(cal, correct)[1], cal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=None,
                    help="Mission-1 aligned-chords JSON (default: "
                         "data/real_audio_benchmark/aligned_chords_per_song.json)")
    ap.add_argument("--dry-run", action="store_true", help="report only, no save")
    args = ap.parse_args()
    from sklearn.isotonic import IsotonicRegression

    try:
        songs = load_benchmark(args.benchmark)
    except MissingBenchmark as e:
        print(f"[waiting on Mission 1] {e}")
        return 0 if args.dry_run else 3

    print(f"benchmark: {len(songs)} songs, "
          f"{sum(len(s['spans']) for s in songs)} GT chords")
    print("running production pipeline (uses the on-disk quality head — "
          "Mission 2's retrain is picked up automatically)...")

    # collect per-chord (conf_raw, fused_raw, correct) + a parallel song_id array
    raws, fused, corr, sid = [], [], [], []
    import tempfile
    with tempfile.TemporaryDirectory() as cd:
        for i, s in enumerate(songs):
            r, f, c = collect_benchmark([s], cache_dir=Path(cd), progress=False)
            print(f"  [{i+1}/{len(songs)}] {s['song_id']}: {len(c)} scored chords",
                  flush=True)
            raws.append(r); fused.append(f); corr.append(c)
            sid.append(np.full(len(c), s["song_id"], dtype=object))
    raws = np.concatenate(raws); fused = np.concatenate(fused)
    corr = np.concatenate(corr); sid = np.concatenate(sid)
    if len(corr) < 30:
        print(f"only {len(corr)} scored chords — too few to calibrate reliably. "
              f"Check benchmark audio paths / GT alignment.")
        return 3
    print(f"\n{len(corr)} scored chords | base accuracy {corr.mean():.1%} | "
          f"mean conf_raw {raws.mean():.3f} | mean fused {fused.mean():.3f}")

    # honest, song-held-out ECE for BOTH scores (fused is the #29 fix)
    ece_conf_cv, _ = cv_ece(raws, corr, sid)
    ece_fused_cv, cal_fused = cv_ece(fused, corr, sid)
    report_reliability("raw fused (pre-cal)", fused, corr)
    report_reliability("calibrated fused (song-held-out CV)", cal_fused, corr)
    print(f"\nCV ECE  conf-only (old #29 regression) = {ece_conf_cv:.4f}")
    print(f"CV ECE  fused (root folded in, #29 fix)  = {ece_fused_cv:.4f}")
    gate = ece_fused_cv < 0.05
    print(f"GATE: calibrated fused CV ECE {ece_fused_cv:.4f} "
          f"{'< 0.05 — PASS' if gate else '>= 0.05 — FAIL'}")

    if args.dry_run:
        print("(dry-run: not saved)")
        return 0
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        fused, corr.astype(float))
    np.savez(
        OUT, x=iso.X_thresholds_, y=iso.y_thresholds_,
        score_kind="fused",
        meta=json.dumps({
            "mission": 3, "fitted_on": "real-audio benchmark (Mission 1, "
            "beat-anchored non-circular iReal GT)",
            "n": int(len(corr)), "n_songs": len(songs),
            "score": "confidence_raw * root_conf (fused; root uncertainty folded in)",
            "target": "root pc AND q5 family @ chord midpoint",
            "base_rate": round(float(corr.mean()), 4),
            "cv_ece_fused": round(float(ece_fused_cv), 4),
            "cv_ece_conf_only": round(float(ece_conf_cv), 4),
            "supersedes": "conf-only real map (issue #29 root-blind regression)",
        }))
    print(f"saved -> {OUT}  ({len(iso.X_thresholds_)} breakpoints, score_kind=fused)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
