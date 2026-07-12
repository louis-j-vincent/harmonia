"""Tests for the ProgressionEncoder second-pass quality reranker (issue #21).

The reranker refines each segment's coarse 5-class quality (maj/min/dom/hdim/dim)
from its ±6-chord harmonic context.  These tests exercise the plumbing in
harmonia.models.chord_pipeline_v1 (Harte↔q5 maps, tensor windowing, override
logic) and one grammatical-behaviour assertion (a low-confidence dominant in a
ii-V-I context is recovered).
"""

import numpy as np
import pytest

from harmonia.models import chord_pipeline_v1 as P

pytestmark = pytest.mark.filterwarnings("ignore")

_ENC = P._get_progression_encoder()
needs_encoder = pytest.mark.skipif(_ENC is None, reason="progression_encoder.pt unavailable")


def test_harte_to_q5idx_mapping():
    # QUAL5 = [maj, min, dom, hdim, dim]
    assert P._harte_to_q5idx("maj") == 0
    assert P._harte_to_q5idx("maj7") == 0
    assert P._harte_to_q5idx("min7") == 1
    assert P._harte_to_q5idx("7") == 2       # dominant
    assert P._harte_to_q5idx("hdim7") == 3
    assert P._harte_to_q5idx("dim7") == 4
    assert P._harte_to_q5idx("minmaj7") == 1
    assert P._harte_to_q5idx("sus4") == 0    # coarsened into maj family
    # a quality outside the 5-class vocab is skipped (returns None)
    assert P._harte_to_q5idx("weird_quality") is None


@needs_encoder
def test_rerank_preserves_length_and_out_of_vocab():
    roots = [0, 5, 7]
    sevs = ["maj7", "weird_quality", "7"]
    confs = [0.9, 0.9, 0.9]
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=0.5)
    assert len(out) == len(sevs)
    # out-of-vocab quality is left untouched
    assert out[1] == "weird_quality"


@needs_encoder
def test_high_confidence_is_not_overridden():
    """A confident, internally-consistent ii-V-I is left unchanged."""
    roots = [2, 7, 0]                     # Dm7 - G7 - Cmaj7 in C
    sevs = ["min7", "7", "maj7"]
    confs = [0.95, 0.95, 0.95]
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=0.5)
    assert out == sevs


@needs_encoder
def test_low_confidence_dominant_recovered_in_ii_V_I():
    """A low-confidence Gmaj7 sitting between Dm7 and Cmaj7 should flip to G7 (dom).

    This is the encoder's headline lever: dom-vs-maj is a grammatical call the
    IID acoustic classifier under-recalls.
    """
    roots = [2, 7, 0, 2, 7, 0]
    sevs = ["min7", "maj7", "maj7", "min7", "7", "maj7"]  # index 1 mislabelled
    confs = [0.9, 0.35, 0.9, 0.9, 0.9, 0.9]              # low conf on the mislabel
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=1.0)
    assert out[1] == "7"                 # recovered dominant
    # triad-vs-seventh preserved: the flipped chord keeps its seventh form
    assert out[1] in P._SEVENTH_HARTE


@needs_encoder
def test_weight_zero_is_a_noop():
    """weight=0 → acoustic-only → no segment changes."""
    roots = [2, 7, 0]
    sevs = ["min7", "maj7", "maj7"]
    confs = [0.5, 0.3, 0.5]
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=0.0)
    assert out == sevs


def test_empty_sequence_is_safe():
    assert P.rerank_progression_qualities([], [], [], weight=0.5) == []


def test_family_q5_logprobs_sums_to_one_and_splits_correctly():
    """_family_q5_logprobs combines the family (5-class) and b7 posteriors
    into a real q5 distribution — the fix for issue #21's one-hot-gated
    acoustic prior (docs/known_issues.md #21)."""
    # FAMILIES order: major, minor, diminished, augmented, suspended
    p_fam = np.array([0.6, 0.1, 0.2, 0.05, 0.05])
    labels = ["majT", "dom7", "dimT", "m7b5", "minT"]
    # all mass on dom7 among major-family labels -> major goes entirely to q5 dom
    p7 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    q5lp = P._family_q5_logprobs(p_fam, p7, labels)
    q5 = np.exp(q5lp)
    assert q5.shape == (5,)
    assert abs(q5.sum() - 1.0) < 1e-5
    # QUAL5 = [maj, min, dom, hdim, dim]; major mass (0.6) should land on dom
    assert q5[2] == pytest.approx(0.6, abs=1e-3)
    assert q5[0] < 0.1  # maj gets none of the major mass (only aug+sus, tiny)


@needs_encoder
def test_real_logprobs_used_when_provided():
    """When aco_logprobs is supplied it overrides the one-hot fallback:
    a peaked real-logprob prior on a different quality should be able to flip
    the acoustic call even at low scalar confidence, since the greedy q5
    used for confs no longer determines the actual acoustic distribution."""
    roots = [2, 7, 0, 2, 7, 0]
    sevs = ["min7", "maj7", "maj7", "min7", "7", "maj7"]
    confs = [0.9, 0.35, 0.9, 0.9, 0.9, 0.9]
    # real q5 logprobs strongly favouring "7" (dom, idx 2) for segment 1,
    # despite the greedy sev_h being "maj7"
    strong_dom = np.log(np.array([1e-3, 1e-3, 0.996, 1e-3, 1e-3], dtype=np.float32))
    aco = [None] * 6
    aco[1] = strong_dom
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=1.0, aco_logprobs=aco)
    assert out[1] == "7"
