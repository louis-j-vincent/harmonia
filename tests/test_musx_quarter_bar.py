"""harmonia_min.musx.make_beat_arr — the quarter-bar gate (feat/quarter-bar).

The shipped harmonia_min behaviour is HALF-BAR ONLY (Louis, 2026-08-01): every
beat that is neither a downbeat nor the mid-bar beat is zeroed (no transition).
`quarter_beats` opens that level back up, globally ("all") or only at listed
beat indices (the targeted, detector-fed mode). These tests pin all three modes
against the vendored decoder's semantics (0 forbidden / 2 downbeat / 3 mid-bar
/ 4 other beat).

Runs without the music-x-lab clone: make_beat_arr is pure numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min import musx as mm


def _grid(n_beats: int = 16, spb: float = 0.5):
    beats = np.arange(n_beats) * spb
    downbeats = beats[::4]                       # 4/4
    n = int(round((n_beats * spb + spb) / mm.FRAME_DT))
    fr = np.round(beats / mm.FRAME_DT).astype(int)
    return beats, downbeats, n, fr


def test_default_is_half_bar_only():
    beats, downbeats, n, fr = _grid()
    arr = mm.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4)
    for i, f in enumerate(fr):
        expect = {0: 2, 2: 3}.get(i % 4, 0)      # quarter beats forbidden
        assert arr[f] == expect, f"beat {i}: {arr[f]} != {expect}"


def test_all_opens_every_beat_at_grade_4():
    beats, downbeats, n, fr = _grid()
    arr = mm.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4,
                           quarter_beats="all")
    for i, f in enumerate(fr):
        expect = {0: 2, 2: 3}.get(i % 4, 4)
        assert arr[f] == expect, f"beat {i}: {arr[f]} != {expect}"


def test_targeted_opens_only_listed_beat_indices():
    beats, downbeats, n, fr = _grid()
    arr = mm.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4,
                           quarter_beats=[5, 7])
    for i, f in enumerate(fr):
        expect = {0: 2, 2: 3}.get(i % 4, 4 if i in (5, 7) else 0)
        assert arr[f] == expect, f"beat {i}: {arr[f]} != {expect}"
    # a listed index that is a downbeat/mid-bar beat keeps its stronger grade
    arr2 = mm.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4,
                            quarter_beats=[0, 2, 3])
    assert arr2[fr[0]] == 2 and arr2[fr[2]] == 3 and arr2[fr[3]] == 4


def test_targeted_out_of_range_indices_are_ignored():
    beats, downbeats, n, fr = _grid()
    arr = mm.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4,
                           quarter_beats=[-3, 1, 999])
    for i, f in enumerate(fr):
        expect = {0: 2, 2: 3}.get(i % 4, 4 if i == 1 else 0)
        assert arr[f] == expect


def test_unknown_string_mode_raises():
    beats, downbeats, n, _ = _grid()
    with pytest.raises(ValueError):
        mm.make_beat_arr(n, beats, downbeat_times=downbeats,
                         quarter_beats="everywhere")
