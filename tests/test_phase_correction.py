"""Unit tests for section-phase correction (issue #22 cycle-shift bug).

The concrete instance: Let It Be's C-G-Am-F loop (I-V-vi-IV in C major) came out
of the real-audio pipeline phase-shifted so the tonic C, which opens each 4-bar
cycle, landed *last*.  correct_section_phase must recover the phase that reopens
each period on the tonic, using progression likelihood alone (no downbeat GT).
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.section_structure import (
    apply_phase_shift,
    build_progression_model,
    correct_section_phase,
    estimate_base_period_bars,
    build_chord_ssm,
)

BPB = 4

# q5 family indices (harmonia.models.section_structure._Q5): maj=0 min=1 dom=2 hdim=3
MAJ, MIN, DOM, HDIM = 0, 1, 2, 3

# One C-G-Am-F cycle, key-relative to C (tonic pc 0), as per-bar (root_rel, q5).
LOOP = [(0, MAJ), (7, MAJ), (9, MIN), (5, MAJ)]  # C  G  Am  F


@pytest.fixture(scope="module")
def model():
    m = build_progression_model()
    if m is None:
        pytest.skip("iReal corpus unavailable — cannot build progression model")
    return m


def _beats(bar_seq: list[tuple[int, int]], reps: int, bpb: int = BPB):
    """Expand a per-bar cycle into a per-beat (root_rel, q5) sequence, repeated."""
    out: list[tuple[int, int]] = []
    for _ in range(reps):
        for root, q in bar_seq:
            out.extend([(root, q)] * bpb)
    return out


def test_start_distribution_peaks_on_tonic(model):
    # The BOS distribution must prefer opening on the tonic major (root_rel 0).
    start = model["start"]
    assert int(start.argmax()) // 5 == 0  # argmax root-relative pitch class is 0
    assert start[0, MAJ] == start.max()


@pytest.mark.parametrize("wrong_start", [1, 2, 3])
def test_recovers_shift_to_tonic_first(model, wrong_start):
    # Build the loop starting on the WRONG bar (rotate LOOP left by wrong_start),
    # repeated 4x.  correct_section_phase must return the shift that rotates it
    # back so each period opens on C (the tonic).
    misphased = LOOP[wrong_start:] + LOOP[:wrong_start]
    seq = _beats(misphased, reps=4)
    shift = correct_section_phase(seq, period_bars=4, beats_per_bar=BPB, model=model)
    # rotating the misphased loop forward by `shift` bars must land C (0,maj) first
    recovered = misphased[shift:] + misphased[:shift]
    assert recovered[0] == (0, MAJ), (wrong_start, shift, recovered)


def test_already_phased_loop_returns_zero(model):
    # A loop already opening on the tonic needs no shift.
    seq = _beats(LOOP, reps=4)
    assert correct_section_phase(seq, period_bars=4, beats_per_bar=BPB, model=model) == 0


def test_too_short_returns_zero(model):
    # Fewer than two full periods -> no phase decision.
    seq = _beats(LOOP, reps=1)
    assert correct_section_phase(seq, period_bars=4, beats_per_bar=BPB, model=model) == 0


def test_estimate_base_period_finds_four_bar_loop():
    # A pure 4-bar loop's base period should be detected as 4 bars.
    seq = [((r) % 12, q) for r, q in _beats(LOOP, reps=6)]
    ssm = build_chord_ssm(seq)
    assert estimate_base_period_bars(ssm, beats_per_bar=BPB) == 4


def test_apply_phase_shift_slides_and_inserts_boundary():
    # 32 beats, boundary at 16; shift +3 bars (12 beats) -> {12, 28}.
    out = apply_phase_shift([16], shift_bars=3, beats_per_bar=BPB, n_beats=32)
    assert out == [12, 28]
    # zero shift is a no-op
    assert apply_phase_shift([16], 0, BPB, 32) == [16]
    # boundaries at/after n_beats are dropped
    assert apply_phase_shift([16], 5, BPB, 32) == [20]  # 20; 16+20=36 dropped


# ── trigram-beats-bigram cases (issue #22 upgrade, was bigram) ─────────────────
#
# Both loops embed a functional ii–V–I whose resolution the bigram cannot
# localise (its transition/pair statistics are nearly rotation-symmetric, so the
# tonic-BOS prior alone leaves several phases in a near-tie and picks the WRONG
# one), while the trigram scores the intact ii–V–I triple and recovers the phase
# that opens each period on the tonic.  A bigram-only view of the same model
# (``tri`` stripped) reproduces the old behaviour, so these tests pin the *delta*
# the trigram upgrade buys, not just an absolute answer.

# C major, 8-bar loop:  Dm7 G7 | Cmaj7 Em7 | Am7 D7 | Dm7 G7  (ii–V–I at bars 1-3)
# Correct section head = bar index 2 (the tonic Cmaj7).
JAZZ_MAJ = [(2, MIN), (7, DOM), (0, MAJ), (4, MIN),
            (9, MIN), (2, DOM), (2, MIN), (7, DOM)]
# A minor, 8-bar loop:  Am F | Bm7b5 E7 | Am Dm | Bm7b5 E7  (ii°–V–i at bars 1-3),
# key-relative to the A-minor tonic (pc 0).  Correct head = bar index 2 (i = Am).
JAZZ_MIN = [(0, MIN), (8, MAJ), (2, HDIM), (7, DOM),
            (0, MIN), (5, MIN), (2, HDIM), (7, DOM)]


def _beats8(loop, reps=2):
    out = []
    for _ in range(reps):
        for rq in loop:
            out.extend([rq] * BPB)
    return out


@pytest.mark.parametrize("loop,target", [(JAZZ_MAJ, 2), (JAZZ_MIN, 2)])
def test_trigram_recovers_iiVI_phase_bigram_misses(model, loop, target):
    seq = _beats8(loop, reps=2)
    bigram_only = {"start": model["start"], "trans": model["trans"]}  # no 'tri'

    tri_shift = correct_section_phase(seq, period_bars=8, beats_per_bar=BPB, model=model)
    bi_shift = correct_section_phase(seq, period_bars=8, beats_per_bar=BPB, model=bigram_only)

    # trigram lands the section head on the tonic (the intact ii–V–I resolution)…
    assert tri_shift == target, (tri_shift, target)
    # …and it does so by *correcting* a phase the bigram gets wrong.
    assert bi_shift != target, (bi_shift, target)


def test_trigram_margin_dominates_bigram_near_tie(model):
    # Quantify the demonstration on JAZZ_MAJ: the bigram leaves the correct phase
    # in a near-tie with its own argmax (< 0.5 nats per 2 periods), while the
    # trigram separates it by a clear margin (> 1 nat).
    from harmonia.models.section_structure import _period_logprob

    loop, target, P = JAZZ_MAJ, 2, 8
    bars = loop * 2
    bigram_only = {"start": model["start"], "trans": model["trans"]}

    def phase_score(mdl, shift):
        rot = bars[shift:] + bars[:shift]
        return sum(_period_logprob(rot[j * P:(j + 1) * P], mdl) for j in range(2))

    bi = [phase_score(bigram_only, s) for s in range(P)]
    tri = [phase_score(model, s) for s in range(P)]
    bi_best = max(range(P), key=lambda s: bi[s])
    tri_second = max((s for s in range(P) if s != target), key=lambda s: tri[s])

    assert bi_best != target                       # bigram argmax is wrong
    assert bi[bi_best] - bi[target] < 0.5          # …but only by a whisker
    assert tri[target] - tri[tri_second] > 1.0     # trigram wins target clearly
