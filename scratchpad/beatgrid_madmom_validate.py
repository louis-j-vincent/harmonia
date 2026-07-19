"""Cross-reference validation of the bestfit beat period against madmom DBN.

Question (known_issues "BAR-GRID vs REAL-MUSIC DRIFT", 2026-07-19): the stock
grid period (librosa tempo scalar) carries a 0.5-2.3% systematic error that
accumulates to multi-bar drift. The staged fix is a whole-song LSQ period
(`_bestfit_beat_period`). With no bar-precise human GT, the best available
independent reference is madmom's RNN+DBN beat tracker (different algorithm,
different features). For each cached real song we compare THREE constant
periods: librosa's scalar, the bestfit over librosa beats, and the bestfit
over madmom beats (the reference), octave-folding madmom to librosa's range.

If bestfit(librosa beats) lands closer to bestfit(madmom beats) than the stock
scalar does, the fix is corroborated by an independent tracker.

Run: .venv/bin/python scratchpad/beatgrid_madmom_validate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parents[1] / "docs" / "audio"
OUT = Path(__file__).with_suffix(".json")


def load_mono(path):
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    return y, sr


def madmom_beats(path):
    from harmonia.models.rhythm import _ensure_madmom_compat

    _ensure_madmom_compat()
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    act = RNNBeatProcessor()(str(path))
    return np.asarray(DBNBeatTrackingProcessor(fps=100)(act), dtype=float)


def fold_to(p, ref):
    """Octave-fold period p (x2 / /2) into the neighbourhood of ref."""
    while p > ref * 1.5:
        p /= 2.0
    while p < ref / 1.5:
        p *= 2.0
    return p


def main():
    import librosa

    rows = []
    for f in sorted(AUDIO_DIR.glob("*.m4a")):
        y, sr = load_mono(f)
        dur = len(y) / sr
        tempo_arr, frames = librosa.beat.beat_track(y=y, sr=sr)
        p_stock = 60.0 / float(np.atleast_1d(tempo_arr)[0])
        lb = librosa.frames_to_time(frames, sr=sr)
        p_best = _bestfit_beat_period(lb, p_stock)

        mb = madmom_beats(f)
        p_mad_med = float(np.median(np.diff(mb)))
        p_ref = _bestfit_beat_period(mb, p_mad_med)
        octave_ratio = p_ref / p_stock
        p_ref_folded = fold_to(p_ref, p_stock)

        n_beats = dur / p_stock
        row = {
            "song": f.stem,
            "dur_s": round(dur, 1),
            "p_stock_ms": round(p_stock * 1e3, 2),
            "p_bestfit_ms": round(p_best * 1e3, 2),
            "p_madmom_ms": round(p_ref * 1e3, 2),
            "p_madmom_folded_ms": round(p_ref_folded * 1e3, 2),
            "octave_ratio_madmom_vs_librosa": round(octave_ratio, 3),
            "err_stock_pct": round(100 * abs(p_stock - p_ref_folded) / p_ref_folded, 3),
            "err_bestfit_pct": round(100 * abs(p_best - p_ref_folded) / p_ref_folded, 3),
            # implied index-matched drift by song end, in 4/4 bars
            "drift_stock_bars": round(abs(p_stock - p_ref_folded) / p_ref_folded * n_beats / 4, 2),
            "drift_bestfit_bars": round(abs(p_best - p_ref_folded) / p_ref_folded * n_beats / 4, 2),
        }
        rows.append(row)
        print(json.dumps(row))

    wins = sum(r["err_bestfit_pct"] < r["err_stock_pct"] - 1e-9 for r in rows)
    ties = sum(abs(r["err_bestfit_pct"] - r["err_stock_pct"]) <= 1e-9 for r in rows)
    summary = {
        "n_songs": len(rows),
        "bestfit_closer_to_madmom": wins,
        "ties": ties,
        "mean_err_stock_pct": round(float(np.mean([r["err_stock_pct"] for r in rows])), 3),
        "mean_err_bestfit_pct": round(float(np.mean([r["err_bestfit_pct"] for r in rows])), 3),
        "mean_drift_stock_bars": round(float(np.mean([r["drift_stock_bars"] for r in rows])), 2),
        "mean_drift_bestfit_bars": round(float(np.mean([r["drift_bestfit_bars"] for r in rows])), 2),
    }
    print("SUMMARY", json.dumps(summary))
    OUT.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
