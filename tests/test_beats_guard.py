"""harmonia_min.beats.check_grid — the metre guard after the waltz opening.

2026-08-07 (Louis: quart ET tiers de barre affichables) : a detected metre of
3 is a legitimate waltz and passes; 2 stays refused — it is the half-tempo /
octave-error signature the guard was built for (Georgia On My Mind); low
consistency stays refused whatever the metre.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min.beats import BeatTrackingError, check_grid


def _grid(metre: int, n_bars: int = 40, step: float = 0.5):
    beats = np.arange(n_bars * metre + 1) * step
    downbeats = beats[::metre]
    return beats, downbeats


def test_waltz_metre_3_passes():
    beats, downbeats = _grid(3)
    q = check_grid(beats, downbeats, "waltz.m4a")
    assert q["metre"] == 3 and q["consistency"] > 0.95


def test_metre_4_still_passes():
    beats, downbeats = _grid(4)
    assert check_grid(beats, downbeats, "pop.m4a")["metre"] == 4


def test_metre_2_still_refused():
    beats, downbeats = _grid(2)
    with pytest.raises(BeatTrackingError, match="2 beats per bar"):
        check_grid(beats, downbeats, "georgia.m4a")


def test_unvalidated_metres_still_refused():
    for metre in (5, 6, 7):
        beats, downbeats = _grid(metre)
        with pytest.raises(BeatTrackingError):
            check_grid(beats, downbeats, f"odd{metre}.m4a")


def test_loose_grid_still_refused_even_in_3():
    beats, downbeats = _grid(3, n_bars=40)
    # corrupt 30% of the downbeats so bars stop tiling in 3s
    db = list(downbeats)
    for i in range(3, len(db), 3):
        db[i] += 0.5   # one beat late
    with pytest.raises(BeatTrackingError, match="too loose or rubato"):
        check_grid(beats, np.asarray(db), "loose.m4a")


def test_short_song_lets_through():
    beats, downbeats = _grid(2, n_bars=10)   # under GRID_MIN_BARS: no verdict
    assert check_grid(beats, downbeats, "snippet.m4a")["n_bars"] < 30
