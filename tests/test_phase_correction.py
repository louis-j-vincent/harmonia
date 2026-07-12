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

# q5 family indices (harmonia.models.section_structure._Q5): maj=0, min=1, dom=2
MAJ, MIN, DOM = 0, 1, 2

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
