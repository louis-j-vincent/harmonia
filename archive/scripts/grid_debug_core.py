"""grid_debug: why the SSM is bad on some songs — the UPSTREAM causes.

Prototype for /reports/grid_debug.html. The locked method
(scripts/harmonic_method.py) is imported, never re-implemented; only the GRID
(Beat It) or the SIMILARITY (Sunny, transposition-invariant) changes upstream.

Produces, per experiment, the SSM and the sliding-block curve with the locked
peak rule — the matrices are the judge, numbers come after.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
import scripts.harmonic_method as hm                       # noqa: E402

BEATS_DIR = HERE / "harmonia_min" / "state" / "beats"
AUDIO_DIR = HERE / "docs" / "audio"

STEMS = {
    "beat_it": "michael_jackson_beat_it_official_4k_video",
    "sunny": "bobby_hebb_sunny_official_audio",
    "this_love": "maroon_5_this_love",
    "norah": "norah_jones_don_t_know_why",
}


def load_beats(stem: str) -> dict:
    return json.loads((BEATS_DIR / f"{stem}.json").read_text())


# ── Beat It: repaired grids ─────────────────────────────────────────────────

def downbeat_indices(beats, downbeats) -> np.ndarray:
    b = np.asarray(beats, float)
    return np.array([int(np.abs(b - t).argmin()) for t in downbeats])


def grid_from_downbeats(beats, downbeats, trim_intro: bool = False) -> list[float]:
    """Locally adaptive grid: every tracker downbeat re-anchors the bar walk.

    Bar boundaries are the tracker's downbeat TIMES themselves (deduplicated
    to distinct beat indices).  With `trim_intro`, start at the first downbeat
    from which the next four gaps are all >= 3 beats — drops the pre-lock
    chaos where the tracker emits 1-2-beat "bars" on the intro hits.
    """
    b = np.asarray(beats, float)
    idx = downbeat_indices(beats, downbeats)
    idx = idx[np.concatenate(([True], np.diff(idx) > 0))]
    start = 0
    if trim_intro:
        gaps = np.diff(idx)
        for i in range(len(gaps) - 4):
            if np.all(gaps[i:i + 4] >= 3):
                start = i
                break
    keep = idx[start:]
    grid = [float(b[i]) for i in keep]
    # close the last bar with the median bar length
    med = float(np.median(np.diff(grid))) if len(grid) > 2 else 4 * float(np.median(np.diff(b)))
    grid.append(grid[-1] + med)
    return grid


# ── Sunny: transposition-invariant SSM ──────────────────────────────────────

def ssm_transposition_invariant(V: np.ndarray) -> np.ndarray:
    """max over the 12 chroma rotations of the cosine — a repeat a semitone up
    scores as if it were in the original key.  Mechanically inflates ALL
    similarities (a max over 12 tries), which is why the controls matter."""
    return np.max(np.stack([V @ np.roll(V, r, axis=1).T for r in range(12)]), axis=0)


def run_method_on(S: np.ndarray) -> dict:
    """The LOCKED downstream, unchanged: period, phase, slide, peak rule."""
    L, b0 = hm.period_and_phase(S)
    curve = hm.slide(S, L, b0)
    return {"S": S, "L": L, "b0": b0, "curve": curve,
            "peaks": hm.peaks(curve, L, b0)}


def offdiag_stats(S: np.ndarray, min_lag: int = 2) -> dict:
    i = np.arange(len(S))
    off = S[np.abs(i[:, None] - i[None, :]) >= min_lag]
    return {"mean": float(off.mean()), "p90": float(np.quantile(off, .90))}
