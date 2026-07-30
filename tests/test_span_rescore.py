"""tests/test_span_rescore.py — pure-numpy unit tests for the lock-propagation
lattice (harmonia/models/span_rescore.py). No server, no audio, no acoustic
backend — acoustic_logp and the context scorer are hand-built so the DP's
correctness is isolated from any model.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.span_rescore import (
    N_CANDIDATES, QUAL5, differential_rescore, idx_of, lattice_rescore, token_of,
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


# ── incumbent-stickiness bonus (delta), task 2 of the lock-propagation
# tuning brief: a NO-lock call must change ~0 spans, not silently flip every
# near-tie to the acoustic argmax.

def test_delta_default_is_zero():
    import inspect
    assert inspect.signature(lattice_rescore).parameters["delta"].default == 0.0


def test_delta_zero_flips_a_near_tie_away_from_displayed():
    """Baseline churn case this feature exists to fix: with delta=0, a span
    whose acoustic evidence marginally prefers something other than what's
    displayed flips — even with no lock anywhere."""
    acoustic = _base_acoustic(1)
    c_maj, g_maj = idx_of(0, MAJ), idx_of(7, MAJ)
    acoustic[0, c_maj] = -1.05   # displayed, marginally worse
    acoustic[0, g_maj] = -1.00   # marginally better acoustic pick
    chosen, _ = lattice_rescore(acoustic, [(0, MAJ)], [None], _UniformCtx(),
                                lam=1.0, K=6, delta=0.0)
    assert chosen[0] == (7, MAJ)


def test_incumbent_bonus_breaks_near_tie_toward_displayed():
    """Same near-tie as above, but with a delta bigger than the acoustic gap:
    the displayed (incumbent) label must win, with zero locks anywhere."""
    acoustic = _base_acoustic(1)
    c_maj, g_maj = idx_of(0, MAJ), idx_of(7, MAJ)
    acoustic[0, c_maj] = -1.05
    acoustic[0, g_maj] = -1.00
    chosen, _ = lattice_rescore(acoustic, [(0, MAJ)], [None], _UniformCtx(),
                                lam=1.0, K=6, delta=0.1)
    assert chosen[0] == (0, MAJ)


def test_incumbent_bonus_multi_span_no_lock_is_a_no_op():
    """The realistic no-lock-call shape: several unlocked spans, each with a
    small acoustic edge toward a DIFFERENT candidate than what's displayed —
    a delta bigger than every gap must reproduce the displayed chart exactly."""
    n = 4
    acoustic = _base_acoustic(n)
    displayed = [(0, MAJ), (2, MIN), (7, DOM), (5, MAJ)]
    rivals = [(7, MAJ), (9, MIN), (0, MAJ), (2, MIN)]
    for i, (d, riv) in enumerate(zip(displayed, rivals)):
        acoustic[i, idx_of(*d)] = -1.05
        acoustic[i, idx_of(*riv)] = -1.00
    locks = [None] * n
    chosen0, _ = lattice_rescore(acoustic, displayed, locks, _UniformCtx(),
                                 lam=1.0, K=6, delta=0.0)
    assert chosen0 != displayed   # delta=0 baseline: some spans DO churn
    chosen1, _ = lattice_rescore(acoustic, displayed, locks, _UniformCtx(),
                                 lam=1.0, K=6, delta=0.5)
    assert chosen1 == displayed  # delta=0.5 (>> the 0.05 gap): a true no-op


def test_delta_does_not_change_a_locked_span():
    """Locked spans have candidate set == {lock} regardless of delta — the
    incumbent bonus must never move, or even matter for, a lock."""
    acoustic, displayed, locks, ctx, *_ = _ii_v_i_case()
    chosen0, _ = lattice_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6, delta=0.0)
    chosen5, _ = lattice_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6, delta=5.0)
    assert chosen0[2] == chosen5[2] == (0, MAJ)


def test_delta_does_not_block_lock_driven_propagation():
    """A modest incumbent bonus must not swamp genuine lock-driven propagation
    — the whole point of the feature is that locks still win when the
    combined acoustic+context evidence clears delta."""
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()
    chosen, _ = lattice_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6, delta=0.5)
    assert chosen[1] == token_of(g_dom)   # still flips despite the displayed-label bonus
    assert chosen[2] == (0, MAJ)          # the lock itself, unconditionally


# ── differential_rescore: lock-ATTRIBUTABLE changes only, 2026-07-30 redesign
# (docs/lock_propagation_tuning.md "Differential re-analysis"). The bug this
# fixes: comparing a locked rescore straight against `displayed` conflates
# genuine lock effects with the lattice's own unconditional disagreement with
# the display (real, present even with zero locks). The fix mirrors the OLD,
# WORKING /api/reinfer: decode base (no constraints) and cons (with
# constraints) under IDENTICAL settings, diff cons vs base, never vs display.

def test_differential_rescore_zero_locks_is_always_a_no_op():
    """No locks anywhere -> baseline-vs-baseline by construction -> nothing
    ever changes, regardless of lam/delta/context. This is what solves task
    2's no-lock-churn problem STRUCTURALLY (not by tuning delta)."""
    acoustic, displayed, _locks, ctx, *_ = _ii_v_i_case()
    n = len(displayed)
    final, changed, margins = differential_rescore(
        acoustic, displayed, [None] * n, ctx, lam=1.0, K=6, delta=0.0)
    assert final == displayed
    assert changed == [False] * n
    assert len(margins) == n


def test_differential_rescore_reports_genuine_lock_attributable_propagation():
    """The design doc's own ii-V-I worked example: the ambiguous middle span
    flips to the context-preferred V7 ONLY because of the lock (the baseline,
    same context scorer, no lock anywhere, would NOT flip it that way -- see
    the uniform-context fallback in _SharpCtx). That flip must survive
    differential_rescore and be marked `changed`."""
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()
    final, changed, _ = differential_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)
    assert final[0] == (2, MIN)                # unambiguous neighbour, never moves
    assert final[1] == token_of(g_dom)          # genuinely flipped BY the lock
    assert changed[1] is True
    assert final[2] == (0, MAJ)                 # the lock itself


def test_differential_rescore_suppresses_baseline_only_disagreement():
    """THE regression test for the redesign: a span whose acoustic evidence
    disagrees with what's displayed REGARDLESS of any lock (uniform context
    -> no coupling between spans, so locking span 1 cannot affect span 0's
    own decision either way) must keep its DISPLAYED value. Sections 1-7's
    original (non-differential) harness would have called this span
    "corrupted" or "fixed" even though the lock had nothing to do with it --
    exactly the bug this function exists to fix."""
    acoustic = _base_acoustic(3)
    acoustic[0, idx_of(7, MAJ)] = -0.1     # span 0's OWN acoustic strongly prefers G major...
    displayed = [(0, MAJ), (2, MIN), (9, MIN)]   # ...but C major is displayed there
    locks = [None, (2, MIN), None]               # lock span 1 (uniform ctx -> no coupling)
    ctx = _UniformCtx()

    # confirm the premise: the BASELINE (no locks at all) really would flip
    # span 0 away from displayed -- otherwise this test proves nothing.
    baseline_chosen, _ = lattice_rescore(acoustic, displayed, [None, None, None], ctx, lam=1.0, K=6)
    assert baseline_chosen[0] == (7, MAJ)
    assert baseline_chosen[0] != displayed[0]

    final, changed, _ = differential_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)
    assert final[0] == (0, MAJ)     # kept displayed: the disagreement isn't lock-caused
    assert changed[0] is False


def test_differential_rescore_changed_matches_final_vs_displayed():
    """`changed[i]` is exactly `final[i] != displayed[i]` -- a convenience,
    not an independent computation; pin the contract."""
    acoustic, displayed, locks, ctx, *_ = _ii_v_i_case()
    final, changed, _ = differential_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)
    assert changed == [f != d for f, d in zip(final, displayed)]


def test_differential_rescore_empty_lattice():
    final, changed, margins = differential_rescore(
        np.zeros((0, N_CANDIDATES)), [], [], _UniformCtx(), lam=1.0, K=6)
    assert final == [] and changed == [] and margins == []


# ── margin gate (2026-07-30, docs/lock_propagation_tuning.md "Margin gate"):
# a PROPAGATED change only survives if the locked run's own score for its
# choice beats the baseline's candidate by >= margin_gate nats, evaluated
# inside the locked run's actual neighbours. The locked span itself is never
# gated (checked directly below too).

def test_margin_gate_suppresses_a_sub_margin_propagated_flip():
    """Same ii-V-I fixture as the genuine-propagation test above: the natural
    margin for span 1's flip (measured empirically) is between 5 and 10 nats.
    A gate ABOVE that (10.0) must suppress the flip back to displayed; a gate
    below it (2.0) must let it through, unchanged from margin_gate=0."""
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()

    final_low, changed_low, _ = differential_rescore(
        acoustic, displayed, locks, ctx, lam=1.0, K=6, margin_gate=2.0)
    assert final_low[1] == token_of(g_dom)
    assert changed_low[1] is True

    final_high, changed_high, _ = differential_rescore(
        acoustic, displayed, locks, ctx, lam=1.0, K=6, margin_gate=10.0)
    assert final_high[1] == displayed[1]          # suppressed -> reverted to displayed
    assert changed_high[1] is False


def test_margin_gate_never_suppresses_the_locked_span_itself():
    """The user asked for this span's value explicitly -- no margin, however
    large, may override a lock."""
    acoustic, displayed, locks, ctx, *_ = _ii_v_i_case()
    for margin_gate in (0.0, 1.0, 100.0, 10_000.0):
        final, changed, _ = differential_rescore(
            acoustic, displayed, locks, ctx, lam=1.0, K=6, margin_gate=margin_gate)
        assert final[2] == (0, MAJ)   # the lock, unconditionally regardless of gate


def test_margin_gate_zero_matches_ungated_behaviour():
    """margin_gate=0.0 (the default) must reproduce the exact pre-gate
    behaviour -- a regression pin for the gate's off-switch."""
    acoustic, displayed, locks, ctx, g_maj, g_dom = _ii_v_i_case()
    final0, changed0, _ = differential_rescore(acoustic, displayed, locks, ctx, lam=1.0, K=6)
    final_gate0, changed_gate0, _ = differential_rescore(
        acoustic, displayed, locks, ctx, lam=1.0, K=6, margin_gate=0.0)
    assert final0 == final_gate0
    assert changed0 == changed_gate0
