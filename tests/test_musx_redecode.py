"""Tests for the default-OFF `musx_redecode` brick.

Everything here runs WITHOUT the music-x-lab clone or its weights: the pieces that
need the clone (`frame_posteriors`, `redecode`) are exercised only through their
pure-python helpers (`make_beat_arr`, `path_loglik`, `_tags_to_lab`), which are the
parts that can silently go wrong.

The `make_beat_arr` tests are written against the SEMANTICS of the vendored
`XHMMDecoder._XHMMDecoder__get_beat_arr` (a value of 0 forbids a transition at that
frame; 1 charges `diff_trans_penalty`; 2/3/4 charge `beat_trans_penalty[0/1/2]`),
so they will fail loudly if the vendored clone is ever swapped for a version with
different conventions.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import musx_redecode as mr


# ── calibration pin: the frame grid (CLAUDE.md #1) ──────────────────────────

def test_frame_rate_matches_the_vendored_settings():
    """22050/512; if the clone's settings.py changes this must fail."""
    assert mr.MUSX_SR == 22050
    assert mr.MUSX_HOP == 512
    assert mr.FRAME_DT == pytest.approx(512 / 22050)
    assert mr.FRAME_DT == pytest.approx(0.023219954648526078, abs=1e-12)
    # 2x finer than the NNLS chroma grid (2048/44100 = 46.44 ms)
    assert mr.FRAME_DT < 2048 / 44100


def test_default_is_off():
    import os
    old = os.environ.pop("HARMONIA_MUSX_REDECODE", None)
    try:
        assert mr.enabled() is False
        os.environ["HARMONIA_MUSX_REDECODE"] = "1"
        assert mr.enabled() is True
    finally:
        os.environ.pop("HARMONIA_MUSX_REDECODE", None)
        if old is not None:
            os.environ["HARMONIA_MUSX_REDECODE"] = old


# ── make_beat_arr ───────────────────────────────────────────────────────────

def test_beat_arr_forbids_transitions_between_beats():
    beats = np.arange(0, 10) * 0.5          # 0.0, 0.5, ... 4.5 s
    n = int(round(5.0 / mr.FRAME_DT))
    arr = mr.make_beat_arr(n, beats)
    fr = np.round(beats / mr.FRAME_DT).astype(int)
    fr = fr[fr < n]
    assert (arr[fr] != 0).all(), "a chord change must be legal ON a beat"
    # every frame strictly between two consecutive beats is forbidden
    for a, b in zip(fr[:-1], fr[1:]):
        assert (arr[a + 1:b] == 0).all()
    # and there must actually BE forbidden frames (otherwise the test is vacuous)
    assert (arr == 0).sum() > 0.8 * n


def test_beat_arr_latency_shifts_the_legal_frames():
    """A latency that is an exact multiple of the frame step must translate the
    legal-transition set by exactly that many frames."""
    beats = np.arange(0, 20) * 0.5
    n = int(round(12.0 / mr.FRAME_DT))
    shift = 7                                    # frames
    L = shift * mr.FRAME_DT
    a0 = mr.make_beat_arr(n, beats, latency=0.0)
    aL = mr.make_beat_arr(n, beats, latency=L)
    legal0 = np.flatnonzero(a0[:len(a0) - shift] != 0)
    legalL = np.flatnonzero(aL != 0)
    assert set((legal0 + shift).tolist()) <= set(legalL.tolist())


def test_beat_arr_downbeat_grading_marks_2_3_4():
    beats = np.arange(0, 16) * 0.5
    downbeats = beats[::4]                   # 4/4
    n = int(round(8.0 / mr.FRAME_DT))
    arr = mr.make_beat_arr(n, beats, downbeat_times=downbeats, beats_per_bar=4)
    fr = np.round(beats / mr.FRAME_DT).astype(int)
    fr = fr[fr < n]
    # within the beat-covered span only 0/2/3/4 occur (the tail after the last
    # beat stays 1, exactly as the vendored __get_beat_arr leaves it)
    inside = set(np.unique(arr[:fr[-1] + 1]).tolist())
    assert inside <= {0, 2, 3, 4}
    assert {2, 3, 4} <= inside
    for i, f in enumerate(fr):
        if i % 4 == 0:
            assert arr[f] == 2, "downbeat must get beat_trans_penalty[0]"
        elif i % 4 == 2:
            assert arr[f] == 3, "mid-bar beat must get beat_trans_penalty[1]"
        else:
            assert arr[f] == 4


def test_beat_arr_degenerate_inputs_are_permissive():
    n = 100
    assert (mr.make_beat_arr(n, []) == 1).all()
    assert (mr.make_beat_arr(n, [0.0]) == 1).all()
    # downbeats too few -> plain beat grid, no grading
    arr = mr.make_beat_arr(n, np.arange(10) * 0.2, downbeat_times=[0.0])
    assert set(np.unique(arr).tolist()) <= {0, 1}


# ── tags -> lab ─────────────────────────────────────────────────────────────

def test_tags_to_lab_runs_and_unshifts():
    tags = ["N"] * 10 + ["C:maj"] * 10 + ["G:7"] * 5
    lab = mr._tags_to_lab(tags, latency=0.0)
    assert [x[2] for x in lab] == ["N", "C:maj", "G:7"]
    assert lab[0][0] == 0.0
    assert lab[1][0] == pytest.approx(10 * mr.FRAME_DT)
    assert lab[-1][1] == pytest.approx(25 * mr.FRAME_DT)
    # a latency of L moves everything EARLIER by exactly L, clipped at 0
    L = 0.1
    labL = mr._tags_to_lab(tags, latency=L)
    assert labL[1][0] == pytest.approx(10 * mr.FRAME_DT - L)
    assert labL[0][0] == 0.0


def test_tags_to_lab_drops_fully_negative_spans():
    # 2 frames of N (~46 ms) then a long C:maj; a 0.2 s latency pushes the whole
    # N span before t=0, so it must be dropped, not emitted with t1 <= 0.
    tags = ["N"] * 2 + ["C:maj"] * 200
    lab = mr._tags_to_lab(tags, latency=0.2)
    assert all(t1 > 0 for _, t1, _ in lab)
    assert [x[2] for x in lab] == ["C:maj"]
    assert lab[0][0] == 0.0


# ── path_loglik: the GT-free latency selector ───────────────────────────────

def _toy(n=60, k=3):
    lp = np.full((n, k), -5.0)
    lp[:30, 0] = -0.01          # chord 0 for the first half
    lp[30:, 1] = -0.01          # chord 1 for the second half
    return lp, ["A", "B", "C"]


def test_path_loglik_prefers_the_true_change_point():
    lp, names = _toy()
    n = lp.shape[0]
    dt = mr.FRAME_DT

    def lab_at(split):
        return [(0.0, split * dt, "A"), (split * dt, n * dt, "B")]

    scores = {s: mr.path_loglik(lp, names, lab_at(s), 1.0, n) for s in (10, 30, 50)}
    assert scores[30] > scores[10] and scores[30] > scores[50]


def test_path_loglik_charges_the_penalty_per_change():
    lp, names = _toy()
    n = lp.shape[0]
    dt = mr.FRAME_DT
    lab = [(0.0, 30 * dt, "A"), (30 * dt, n * dt, "B")]
    a = mr.path_loglik(lp, names, lab, 0.0, n)
    b = mr.path_loglik(lp, names, lab, 7.0, n)
    assert a - b == pytest.approx(7.0)        # exactly one change


def test_path_loglik_rejects_unknown_labels():
    lp, names = _toy()
    n = lp.shape[0]
    lab = [(0.0, n * mr.FRAME_DT, "NOT_A_CHORD")]
    assert mr.path_loglik(lp, names, lab, 1.0, n) == float("-inf")


def test_path_loglik_latency_argument_realigns_frames():
    """A lab produced with latency L must be scored back on the frame grid."""
    lp, names = _toy()
    n = lp.shape[0]
    dt = mr.FRAME_DT
    L = 5 * dt
    lab = [(0.0, 30 * dt - L, "A"), (30 * dt - L, n * dt - L, "B")]
    with_L = mr.path_loglik(lp, names, lab, 0.0, n, latency=L)
    without = mr.path_loglik(lp, names, lab, 0.0, n, latency=0.0)
    assert with_L > without


# ── documented constants stay in sync with the session's measurements ───────

def test_latency_grid_spans_the_measured_range():
    """Measured per-song latency on the frozen benchmark: +46 to +289 ms."""
    g = mr.DEFAULT_LATENCY_GRID
    assert min(g) == 0.0
    assert max(g) >= 0.28
    assert all(b - a == pytest.approx(0.04) for a, b in zip(g[:-1], g[1:]))
