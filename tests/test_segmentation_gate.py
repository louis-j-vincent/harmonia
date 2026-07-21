"""Load-bearing sanity checks for the confidence-gated segmentation brick.

CLAUDE.md #1: unit-test the most basic load-bearing assumption. Here that is:
the gate is an EXACT passthrough to the shipped baseline at margin 0.0, so wiring
it in cannot change behaviour until deliberately enabled.
"""
import numpy as np
import pytest

from harmonia.models.chord_pipeline_v1 import _root_change_segs
from harmonia.models.segmentation_gate import (
    confidence_gated_segs,
    gated_root_change_segs,
)


def _rand_proba(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 12))
    return x / x.sum(1, keepdims=True)


@pytest.mark.parametrize("seed", range(6))
def test_passthrough_matches_baseline(seed):
    bp = _rand_proba(40, seed)
    assert confidence_gated_segs(bp, margin_thresh=0.0) == _root_change_segs(bp)
    # env-driven drop-in is also a no-op with the flag unset
    assert gated_root_change_segs(bp) == _root_change_segs(bp)


def test_empty_input():
    assert confidence_gated_segs(np.zeros((0, 12))) == []


def test_segs_cover_all_beats_contiguously():
    bp = _rand_proba(50, 1)
    for m in (0.0, 0.2, 0.5, 0.9):
        segs = confidence_gated_segs(bp, margin_thresh=m)
        assert segs[0][0] == 0
        assert segs[-1][1] == len(bp)
        for (s, e), (s2, _) in zip(segs, segs[1:]):
            assert e == s2 and e > s


def test_higher_margin_never_increases_segments():
    bp = _rand_proba(120, 3)
    counts = [len(confidence_gated_segs(bp, margin_thresh=m))
              for m in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)]
    assert counts == sorted(counts, reverse=True)  # monotonically non-increasing


def test_protect_strong_requires_phase():
    bp = _rand_proba(20, 2)
    with pytest.raises(ValueError):
        confidence_gated_segs(bp, margin_thresh=0.5, protect_strong=True)


def test_protect_strong_keeps_at_least_strong_flips():
    bp = _rand_proba(80, 4)
    phase = np.array([i % 4 for i in range(len(bp))])
    free = confidence_gated_segs(bp, margin_thresh=0.9, protect_strong=False)
    prot = confidence_gated_segs(bp, margin_thresh=0.9, phase=phase, protect_strong=True)
    # protecting strong-beat flips can only ADD cuts back vs the grid-free gate
    assert len(prot) >= len(free)
