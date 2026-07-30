"""tests/test_span_rescore.py — pure-numpy unit tests for the lock-propagation
lattice (harmonia/models/span_rescore.py). No server, no audio, no acoustic
backend — acoustic_logp and the context scorer are hand-built so the DP's
correctness is isolated from any model.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.span_rescore import (
    N_CANDIDATES, QUAL5, idx_of, lattice_rescore, token_of,
)

MAJ, MIN, DOM, HDIM, DIM = (QUAL5.index(q) for q in ("maj", "min", "dom", "hdim", "dim"))


class _SharpCtx:
    """Fires a strong preference for ``prefer_idx`` ONLY when the neighbours
    are exactly (prefer_prev, prefer_next); uniform otherwise (including any
    call involving a None/end token) — so it never perturbs an unrelated
    span's decision."""

    def __init__(self, prefer_prev, prefer_next, prefer_idx, boost=0.95):
        self.pp, self.pn, self.pi, self.boost = prefer_prev, prefer_next, prefer_idx, boost

    def score_candidates(self, prev_token, next_token):
        if prev_token == self.pp and next_token == self.pn:
            vec = np.full(N_CANDIDATES, (1.0 - self.boost) / (N_CANDIDATES - 1))
            vec[self.pi] = self.boost
            return vec
        return np.full(N_CANDIDATES, 1.0 / N_CANDIDATES)


class _UniformCtx:
    def score_candidates(self, prev_token, next_token):
        return np.full(N_CANDIDATES, 1.0 / N_CANDIDATES)


def _base_acoustic(n):
    return np.full((n, N_CANDIDATES), -50.0)


def _ii_v_i_case():
    """Dm(ii) - ?(G major or G7) - C(i, LOCKED) — the design doc's own
    worked example (docs/design_chord_context_prior.md 'Representation').

    Span 0 (Dm): acoustic overwhelmingly favours (D, min) — a confident
    neighbour, not itself locked.
    Span 1 (the ambiguous middle): acoustic favours G MAJOR (idx A) slightly
    over G DOMINANT7 (idx B) — a real but weak preference.
    Span 2 (C, LOCKED): acoustic actively prefers something else entirely;
    the lock must win regardless.
    """
    acoustic = _base_acoustic(3)
    d_min = idx_of(2, MIN)
    acoustic[0, d_min] = -0.1

    g_maj = idx_of(7, MAJ)   # "A" — the acoustic (and displayed) favourite
    g_dom = idx_of(7, DOM)   # "B" — the context-preferred resolution (V7)
    acoustic[1, g_maj] = -1.0
    acoustic[1, g_dom] = -1.2

    c_maj = idx_of(0, MAJ)
    off_pick = idx_of(9, MIN)          # acoustic's (wrong) favourite for span 2
    acoustic[2, c_maj] = -50.0          # the lock itself: acoustically UNsupported
    acoustic[2, off_pick] = -0.1

    displayed = [(2, MIN), (7, MAJ), (0, MAJ)]
    locks = [None, None, (0, MAJ)]
    ctx = _SharpCtx(prefer_prev=(2, MIN), prefer_next=(0, MAJ), prefer_idx=g_dom)
    return acoustic, displayed, locks, ctx, g_maj, g_dom


def test_sharp_context_flips_ambiguous_middle_when_locked_neighbour_present():
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()
    chosen, margins = lattice_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)

    assert len(chosen) == 3
    assert chosen[0] == (2, MIN)                 # unambiguous neighbour, untouched
    assert chosen[1] == token_of(g_dom)           # flipped to the context-preferred V7
    assert chosen[2] == (0, MAJ)                  # the lock, unconditionally


def test_lam_zero_keeps_pure_acoustic_choice():
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()
    chosen, margins = lattice_rescore(acoustic, displayed, locks, ctx, lam=0.0, K=6)

    assert chosen[1] == token_of(g_maj)           # no context pull at lam=0
    assert chosen[2] == (0, MAJ)                  # still the lock


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0, 5.0])
def test_locked_span_always_returned_as_locked_even_against_strong_acoustics(lam):
    """A lock's candidate set is EXACTLY {lock} — acoustics can't override it
    no matter how lopsided they are, and no matter lam."""
    acoustic, displayed, locks, ctx, *_ = _ii_v_i_case()
    chosen, _ = lattice_rescore(acoustic, displayed, locks, ctx, lam=lam, K=6)
    assert chosen[2] == (0, MAJ)


def test_boundary_count_unchanged():
    """lattice_rescore never adds/drops spans — one (root,qual5) per input span."""
    acoustic, displayed, locks, ctx, *_ = _ii_v_i_case()
    chosen, margins = lattice_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)
    assert len(chosen) == len(acoustic) == 3
    assert len(margins) == 3
    for r, q in chosen:
        assert 0 <= r < 12
        assert 0 <= q < len(QUAL5)


def test_single_span_end_scoring_does_not_crash():
    acoustic = _base_acoustic(1)
    c_maj = idx_of(0, MAJ)
    acoustic[0, c_maj] = -0.1
    chosen, margins = lattice_rescore(acoustic, [(0, MAJ)], [None], _UniformCtx(), lam=1.0, K=6)
    assert chosen == [(0, MAJ)]
    assert len(margins) == 1


def test_two_span_both_ends_do_not_crash():
    acoustic = _base_acoustic(2)
    acoustic[0, idx_of(0, MAJ)] = -0.1
    acoustic[1, idx_of(7, DOM)] = -0.1
    chosen, margins = lattice_rescore(
        acoustic, [(0, MAJ), (7, DOM)], [None, None], _UniformCtx(), lam=1.0, K=6)
    assert chosen == [(0, MAJ), (7, DOM)]
    assert len(margins) == 2


def test_all_spans_locked_returns_locks_verbatim():
    acoustic = _base_acoustic(3)   # zero acoustic signal anywhere
    locks = [(0, MAJ), (2, MIN), (7, DOM)]
    chosen, margins = lattice_rescore(
        acoustic, locks, locks, _UniformCtx(), lam=1.0, K=6)
    assert chosen == locks
    assert len(margins) == 3


def test_empty_lattice_returns_empty():
    chosen, margins = lattice_rescore(
        np.zeros((0, N_CANDIDATES)), [], [], _UniformCtx(), lam=1.0, K=6)
    assert chosen == []
    assert margins == []


def test_idx_token_roundtrip():
    for r in range(12):
        for q in range(len(QUAL5)):
            assert token_of(idx_of(r, q)) == (r, q)
