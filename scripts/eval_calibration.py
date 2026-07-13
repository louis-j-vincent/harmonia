"""Mission 3: measure confidence-calibration quality (ECE) — real vs synth.

Reports, for the CURRENTLY-SAVED calibration maps, reliability + ECE on:

  REAL  : Mission 1's non-circular real-audio benchmark
          - raw fused (conf × root_conf), pre-calibration
          - conf-only raw (the #29 root-blind regression baseline)
          - calibrated (whatever the saved real map declares: fused or conf)
          Gate: calibrated real ECE < 0.05.

  SYNTH : held-out jazz1460 MMA renders (regression guard)
          - raw fused, pre-calibration
          - calibrated by the synth map
          Gate: synth stays < 0.05 (must not regress).

This is a measurement tool — it fits nothing and saves nothing. Run
scripts/calibrate_quality.py first to (re)fit the real map on the fused score.

Usage:
    .venv/bin/python scripts/eval_calibration.py                  # real + synth
    .venv/bin/python scripts/eval_calibration.py --no-synth       # real only (fast)
    .venv/bin/python scripts/eval_calibration.py --synth-n 20     # bigger synth set
    .venv/bin/python scripts/eval_calibration.py --benchmark p.json
"""
from __future__ import annotations

import argparse
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


def _apply(cal, score: np.ndarray) -> np.ndarray:
    return np.array([cal(float(s)) for s in score]) if cal is not None else score


def eval_real(benchmark: Path | None) -> float | None:
    print("\n" + "=" * 68 + "\n REAL-AUDIO BENCHMARK (Mission 1, non-circular GT)\n" + "=" * 68)
    try:
        songs = load_benchmark(benchmark)
    except MissingBenchmark as e:
        print(f"[waiting on Mission 1] {e}")
        return None
    raws, fused, corr = collect_benchmark(songs)
    if len(corr) < 20:
        print(f"only {len(corr)} scored chords — benchmark too small / audio missing.")
        return None
    report_reliability("raw fused (conf × root_conf, pre-cal)", fused, corr)
    report_reliability("conf-only raw (#29 root-blind baseline)", raws, corr)
    cal = P._get_conf_calibrator("real")
    kind = getattr(cal, "score_kind", "conf")
    cal_in = raws if kind == "conf" else fused
    ece_cal = report_reliability(
        f"calibrated (saved real map, score_kind={kind})", _apply(cal, cal_in), corr)
    print(f"\nGATE: calibrated real ECE {ece_cal:.4f} "
          f"{'< 0.05 — PASS' if ece_cal < 0.05 else '>= 0.05 — FAIL'}")
    if kind == "conf":
        print("NOTE: saved real map is still conf-only (#29 open). Run "
              "scripts/calibrate_quality.py to refit on the fused score.")
    return ece_cal


def eval_synth(n: int, start: int) -> float | None:
    print("\n" + "=" * 68 + "\n SYNTH HOLD-OUT (jazz1460 MMA renders — regression guard)\n" + "=" * 68)
    try:
        import json
        import tempfile

        from build_audio_chord_features import BUCKET_FAMILY
        from fit_confidence_calibration import collect as collect_synth
        from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
    except Exception as e:
        print(f"synth eval unavailable ({e}) — skipping.")
        return None
    DB = REPO / "data" / "accomp_db" / "db.jsonl"
    if not DB.exists():
        print(f"{DB} missing — skipping synth eval.")
        return None
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    # idx 100..112 is fit_confidence_calibration.py's held-out test block
    test = jz[start:start + n]
    print(f"rendering {len(test)} held-out songs (idx {start}..{start+n})...")
    fused, corr = collect_synth(test, renderer, sf2)  # fused = conf × root_conf
    if len(corr) < 20:
        print(f"only {len(corr)} chords — skipping.")
        return None
    report_reliability("raw fused (pre-cal)", fused, corr)
    cal = P._get_conf_calibrator("synth")
    ece_cal = report_reliability("calibrated (saved synth map)", _apply(cal, fused), corr)
    print(f"\nGATE: synth ECE {ece_cal:.4f} "
          f"{'< 0.05 — PASS (no regression)' if ece_cal < 0.05 else '>= 0.05 — REGRESSED'}")
    return ece_cal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=None)
    ap.add_argument("--no-synth", action="store_true", help="skip the synth regression eval")
    ap.add_argument("--synth-n", type=int, default=12)
    ap.add_argument("--synth-start", type=int, default=100)
    args = ap.parse_args()

    real_ece = eval_real(args.benchmark)
    synth_ece = None if args.no_synth else eval_synth(args.synth_n, args.synth_start)

    print("\n" + "=" * 68 + "\n SUMMARY\n" + "=" * 68)
    rr = "n/a (Mission 1 pending)" if real_ece is None else f"{real_ece:.4f}"
    ss = "skipped" if synth_ece is None else f"{synth_ece:.4f}"
    print(f"  real  calibrated ECE = {rr}   (target < 0.05)")
    print(f"  synth calibrated ECE = {ss}   (must stay < 0.05)")
    ok = (real_ece is not None and real_ece < 0.05
          and (synth_ece is None or synth_ece < 0.05))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
