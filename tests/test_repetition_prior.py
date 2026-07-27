"""Tests for the default-OFF repetition_prior brick.

Red-first where a bug was actually found during the session: the self-confirmation
no-op and the degenerate-profile inversion (a constant profile made EVERY beat
expensive and cost blue_bossa 34 segments) both have explicit tests here.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import repetition_prior as RP


# ---------------------------------------------------------------------------
# calibration pins (CLAUDE.md #1)
# ---------------------------------------------------------------------------

def test_default_off():
    import os
    old = os.environ.pop("HARMONIA_REPETITION_PRIOR", None)
    try:
        assert RP.enabled() is False
        os.environ["HARMONIA_REPETITION_PRIOR"] = "1"
        assert RP.enabled() is True
    finally:
        os.environ.pop("HARMONIA_REPETITION_PRIOR", None)
        if old is not None:
            os.environ["HARMONIA_REPETITION_PRIOR"] = old


def test_periods_are_whole_bars():
    """The candidate periods must all be whole 4/4 bars — a period of 6 or 10
    beats would mean the phase index no longer aligns with the bar."""
    assert all(P % 4 == 0 for P in RP.DEFAULT_PERIODS)
    assert min(RP.DEFAULT_PERIODS) == 8      # 2 bars
    assert max(RP.DEFAULT_PERIODS) == 128    # 32 bars


# ---------------------------------------------------------------------------
# change_vector
# ---------------------------------------------------------------------------

def test_change_vector_marks_nearest_beat():
    bt = np.arange(0, 8) * 0.5
    v = RP.change_vector([1.0, 2.5], bt)
    assert v.tolist() == [0, 0, 1, 0, 0, 1, 0, 0]


def test_change_vector_drops_far_changes_instead_of_snapping():
    """A change that falls between beats must be dropped, not snapped — snapping
    invents structure that the grid cannot represent."""
    bt = np.array([0.0, 1.0, 2.0, 3.0])
    assert RP.change_vector([1.5], bt, tol_beats=0.2).sum() == 0
    assert RP.change_vector([1.5], bt, tol_beats=0.6).sum() == 1


def test_change_vector_empty_inputs():
    assert RP.change_vector([], np.arange(4)).sum() == 0
    assert len(RP.change_vector([1.0], np.array([]))) == 0


# ---------------------------------------------------------------------------
# period detection
# ---------------------------------------------------------------------------

def test_detect_period_finds_an_8_beat_form():
    """Changes on beat 0 of every bar plus beat 2 of every OTHER bar: the bar
    model (P=4) cannot express it, an 8-beat period can."""
    n_cycle = 16
    c = np.zeros(8 * n_cycle, np.int8)
    for k in range(n_cycle):
        c[8 * k] = 1          # every bar 1
        c[8 * k + 6] = 1      # beat 3 of the second bar only
    P, gain = RP.detect_period(c)
    assert P == 8
    assert gain > 0.05


def test_detect_period_reports_no_gain_for_pure_metre():
    """A change on every downbeat and nowhere else is the METRICAL base rate, not
    repetition — the gate must not fire on it."""
    c = np.zeros(4 * 40, np.int8)
    c[::4] = 1
    P, gain = RP.detect_period(c)
    assert gain <= RP.DEFAULT_GAIN_GATE


def test_detect_period_degenerate_inputs():
    assert RP.detect_period(np.zeros(64, np.int8))[1] == 0.0
    assert RP.detect_period(np.zeros(3, np.int8))[1] == 0.0


# ---------------------------------------------------------------------------
# leave-one-cycle-out (the fix for self-confirmation)
# ---------------------------------------------------------------------------

def test_loco_excludes_the_beat_itself():
    """A lone change that occurs in exactly ONE cycle must NOT be predicted by the
    profile at its own beat — otherwise the prior echoes the decode it corrects
    (measured failure: 0.000 change on every metric)."""
    c = np.zeros(8 * 10, np.int8)
    c[8 * 3 + 5] = 1                     # a one-off at phase 5
    f = RP.loco_profile(c, 8)
    assert f[8 * 3 + 5] < 0.05
    in_sample = c[np.arange(len(c)) % 8 == 5].mean()
    assert in_sample > f[8 * 3 + 5]


def test_loco_predicts_a_consistent_pattern():
    c = np.zeros(8 * 10, np.int8)
    c[::8] = 1                            # phase 0 in every cycle
    f = RP.loco_profile(c, 8)
    assert f[72] > 0.8                    # its own cycle removed, still ~1
    assert f[73] < 0.2


def test_loco_length_and_no_nan():
    rng = np.random.default_rng(0)
    c = (rng.random(97) < 0.3).astype(np.int8)
    f = RP.loco_profile(c, 16)
    assert len(f) == 97
    assert np.isfinite(f).all()
    assert ((f >= 0) & (f <= 1)).all()


# ---------------------------------------------------------------------------
# beat_change_prior
# ---------------------------------------------------------------------------

def _periodic_changes(bt, period_beats, phases):
    return [bt[i] for i in range(len(bt)) if i % period_beats in phases]


def test_prior_gates_off_on_a_metrical_only_song():
    bt = np.arange(0, 160) * 0.5
    res = RP.beat_change_prior(_periodic_changes(bt, 4, {0}), bt)
    assert res["gated_on"] is False
    assert np.allclose(res["prior"], res["prior"][0])      # flat == no-op


def test_prior_gates_on_and_matches_the_form():
    bt = np.arange(0, 256) * 0.5
    res = RP.beat_change_prior(_periodic_changes(bt, 16, {0, 6, 11}), bt)
    assert res["gated_on"] is True
    assert res["period"] == 16
    p = res["prior"]
    assert p[np.arange(len(bt)) % 16 == 6].mean() > 0.8
    assert p[np.arange(len(bt)) % 16 == 3].mean() < 0.2


def test_prior_phase_index_can_differ_from_the_output_grid():
    """Phase is indexed on the REAL beats; the prior is returned on the (possibly
    uniform) decode grid.  Regression guard for the drift fix."""
    real = np.cumsum(np.r_[0.0, np.full(255, 0.5) + np.linspace(0, .01, 255)])
    uni = np.arange(0, 256) * 0.5
    ch = [real[i] for i in range(len(real)) if i % 16 in (0, 6)]
    res = RP.beat_change_prior(ch, uni, phase_beat_times=real)
    assert res["gated_on"] is True
    assert len(res["prior"]) == len(uni)
    assert np.isfinite(res["prior"]).all()


def test_prior_is_never_degenerate_when_gated_on():
    """The bug that cost blue_bossa 34 segments: a constant profile made every
    beat 'unlikely'.  A gated-ON prior must have real spread."""
    bt = np.arange(0, 256) * 0.5
    res = RP.beat_change_prior(_periodic_changes(bt, 16, {0, 6}), bt)
    assert res["gated_on"]
    assert res["prior"].max() - res["prior"].min() > 0.3


# ---------------------------------------------------------------------------
# candidate ranking
# ---------------------------------------------------------------------------

def test_rank_candidates_skips_existing_cuts_and_orders_by_score():
    prior = np.array([0.9, 0.1, 0.8, 0.7])
    cuts = np.array([True, False, False, False])
    order = RP.rank_candidates(prior, np.arange(4), cuts)
    assert 0 not in order.tolist()
    assert order.tolist() == [2, 3, 1]


def test_rank_candidates_slice_len_targets_merges():
    """With equal prior, the longer slice must be split first — that is what makes
    the ranking target merges rather than already-short slices."""
    prior = np.array([0.5, 0.5, 0.5])
    order = RP.rank_candidates(prior, np.arange(3), np.zeros(3, bool),
                               slice_len=np.array([1.0, 9.0, 4.0]))
    assert order.tolist() == [1, 2, 0]


def test_rank_candidates_drops_zero_scores():
    order = RP.rank_candidates(np.array([0.0, 0.0]), np.arange(2),
                               np.zeros(2, bool))
    assert len(order) == 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
