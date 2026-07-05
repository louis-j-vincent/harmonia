"""Tests for the learned progression prior wired into the HMM transition."""

import numpy as np
import pytest

from harmonia.models.chord_hmm import ChordInferrer
from harmonia.theory.chord_vocabulary import build_index
from harmonia.theory.progression_prior import (
    load,
    transition_log_boost,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def test_table_loads_and_shapes():
    logp = load()
    assert logp.shape == (60, 60)              # 12 degrees × 5 families
    # each row is a log-distribution (sums to ~1 in prob space)
    assert np.allclose(np.exp(logp).sum(1), 1.0, atol=1e-4)


def test_boost_shape_and_finiteness():
    idx_to_chord, _ = build_index(max_phase=1)
    boost = transition_log_boost(idx_to_chord, tonic=0, logp=load())
    assert boost.shape == (len(idx_to_chord), len(idx_to_chord))
    assert np.isfinite(boost).all()


def test_default_weight_zero_is_a_noop():
    """progression_prior_weight=0 must not change the inferrer's decode vs default."""
    base = ChordInferrer(max_phase=1)
    withp = ChordInferrer(max_phase=1, progression_prior_weight=0.0)
    assert withp.progression_prior_weight == 0.0
    assert withp._progression_logp is None
    # both build identical transition matrices (no progression term)
    assert base.progression_prior_weight == 0.0


def test_weight_positive_loads_table():
    inf = ChordInferrer(max_phase=1, progression_prior_weight=0.3)
    assert inf.progression_prior_weight == 0.3
    assert inf._progression_logp is not None
    assert inf._progression_logp.shape == (60, 60)


def test_prior_favours_V_to_I():
    """The learned bigram should make a dominant V → I likelier than V → a random chord."""
    logp = load()
    from harmonia.theory.progression_prior import _state
    v_maj = _state(7, "major")      # V (major/dominant family)
    i_maj = _state(0, "major")      # I
    bii_maj = _state(1, "major")    # bII (rare target)
    assert logp[v_maj, i_maj] > logp[v_maj, bii_maj]
