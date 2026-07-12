"""Tests for the ProgressionEncoder second-pass quality reranker (issue #21).

The reranker refines each segment's coarse 5-class quality (maj/min/dom/hdim/dim)
from its ±6-chord harmonic context.  These tests exercise the plumbing in
harmonia.models.chord_pipeline_v1 (Harte↔q5 maps, tensor windowing, override
logic) and one grammatical-behaviour assertion (a low-confidence dominant in a
ii-V-I context is recovered).
"""

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
