"""Tests for the LocalKeySeqGRU diatonic-prior reranker in infer_chords_v1 (#20/#23).

Volet 1: the transpose-equivariant local-key tagger, wired as a second-pass
quality reranker. Pins (a) the Georgia/"Let It Be" fix — a vi chord mis-called
major under uncertain acoustics is snapped to minor by the local key — and (b)
transpose invariance: the same progression in any key gets the same corrections.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _model_or_skip():
    from harmonia.models import chord_pipeline_v1 as P
    if P._get_local_key_seq_model() is None:
        pytest.skip("local_key_seq_gru.pt not available")
    return P


def test_let_it_be_vi_maj_to_min():
    """I-V-vi-IV (C G Am F) with the vi mis-called A major → corrected to minor.

    This is the exact family error (A major where La mineur is expected) that
    motivated the whole local-key line of work."""
    P = _model_or_skip()
    roots = [0, 7, 9, 5] * 3           # C G A F, three loops
    sev = ["maj"] * 12                 # vi (A) wrongly major
    confs = [0.5] * 12                 # uncertain acoustics → prior may fire
    out = P.rerank_local_key_qualities(roots, sev, confs, global_tonic=0, boost=4.0)
    assert out[2] == "min", "vi chord (A) should be pulled to minor by C-major key"


def test_confident_acoustics_not_overridden():
    """A confident acoustic call is left alone even if non-diatonic."""
    P = _model_or_skip()
    roots = [0, 7, 9, 5] * 3
    sev = ["maj"] * 12
    confs = [0.95] * 12                # >= threshold_chromatic → prior skipped
    out = P.rerank_local_key_qualities(roots, sev, confs, global_tonic=0, boost=4.0)
    assert out == sev, "confident acoustic quality must survive the prior"


def test_rerank_local_key_transpose_invariant():
    """Same progression transposed by any interval → same corrections."""
    P = _model_or_skip()
    roots0 = [0, 7, 9, 5] * 3
    sev = ["maj"] * 12
    confs = [0.5] * 12
    base = P.rerank_local_key_qualities(roots0, sev, confs, 0, boost=4.0)
    assert base[2] == "min"
    for semis in (1, 4, 6, 10):
        roots = [(r + semis) % 12 for r in roots0]
        got = P.rerank_local_key_qualities(roots, sev, confs, semis % 12, boost=4.0)
        assert got == base, f"transpose +{semis} changed local-key corrections"
