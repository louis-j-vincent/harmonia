"""Unit tests for the per-beat semi-Markov (explicit-duration) decode."""
import numpy as np
import pytest

from harmonia.models.semi_markov_decode import (
    build_emission,
    build_log_duration,
    semi_markov_decode,
)


def _uniform_dur(D=8):
    return {"pooled": np.ones(D) / D, "per_q5": np.ones((5, D)) / D}


def _peaked_dur(D=8, peak=4):
    p = np.full(D, 1e-4); p[peak - 1] = 1.0; p = p / p.sum()
    return {"pooled": p, "per_q5": np.tile(p, (5, 1))}


def test_degenerate_uniform_duration_equals_per_beat_argmax():
    """dur_weight=0 ⇒ duration term inert ⇒ decode = per-beat root argmax."""
    rng = np.random.default_rng(0)
    T = 20
    bp = rng.random((T, 12)); bp /= bp.sum(1, keepdims=True)
    out = semi_markov_decode(bp, dur_pmf=_uniform_dur(), dur_weight=0.0)
    assert np.array_equal(out["beat_root"], bp.argmax(1))


def test_duration_prior_overrides_single_wrong_beat():
    """A lone 5th-apart spurious beat is merged into the dominant 4-beat span."""
    T = 4
    bp = np.full((T, 12), 0.01)
    bp[:, 0] = 0.9          # every beat favours root 0 ...
    bp[2, 0] = 0.4; bp[2, 7] = 0.55   # ... except beat 2 which (wrongly) peaks at 7
    bp /= bp.sum(1, keepdims=True)
    # per-beat argmax would be [0,0,7,0]; duration prior peaked at d=4 prefers one span
    out = semi_markov_decode(bp, dur_pmf=_peaked_dur(peak=4), dur_weight=1.0)
    assert np.array_equal(out["beat_root"], np.zeros(T, dtype=int))
    assert len(out["segments"]) == 1


def test_known_answer_two_equal_halves():
    """Two 2-beat blocks of distinct roots, duration prior peaked at 2."""
    bp = np.full((4, 12), 0.01)
    bp[0:2, 3] = 0.9; bp[2:4, 8] = 0.9
    bp /= bp.sum(1, keepdims=True)
    out = semi_markov_decode(bp, dur_pmf=_peaked_dur(peak=2), dur_weight=1.0)
    assert np.array_equal(out["beat_root"], np.array([3, 3, 8, 8]))
    assert len(out["segments"]) == 2


def test_transposition_invariance():
    """Rolling the root posterior by s rolls the decoded roots by s (mod 12)."""
    rng = np.random.default_rng(1)
    T = 24
    bp = rng.random((T, 12)); bp /= bp.sum(1, keepdims=True)
    dp = _peaked_dur(peak=2)
    base = semi_markov_decode(bp, dur_pmf=dp, dur_weight=1.0)["beat_root"]
    for s in (1, 5, 7):
        rolled = np.roll(bp, s, axis=1)
        got = semi_markov_decode(rolled, dur_pmf=dp, dur_weight=1.0)["beat_root"]
        assert np.array_equal(got, (base + s) % 12)


def test_log_duration_pooled_is_quality_independent():
    """Default (pooled) duration prior injects zero quality bias."""
    LD = build_log_duration(_peaked_dur(), dur_weight=1.0, per_quality=False)
    LD = LD.reshape(12, 5, -1)
    for r in range(12):
        for q in range(1, 5):
            assert np.allclose(LD[r, q], LD[r, 0])


def test_emission_separable_and_qual_weight_zero_inert():
    T = 10
    bp = np.full((T, 12), 1.0 / 12)
    qp = np.random.default_rng(2).random((T, 5)); qp /= qp.sum(1, keepdims=True)
    E0 = build_emission(bp, qp, qual_weight=0.0)
    # qual_weight=0 ⇒ all 5 q-columns within a root are equal
    E0 = E0.reshape(T, 12, 5)
    assert np.allclose(E0[:, :, 0], E0[:, :, 4])
    E1 = build_emission(bp, qp, qual_weight=1.0).reshape(T, 12, 5)
    assert not np.allclose(E1[:, :, 0], E1[:, :, 4])
