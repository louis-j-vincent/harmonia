"""Minimal, audio-free pins for the Phase-2 independent-GT beat harness.

Mirrors harmonia.eval.beat_alignment_gt --selftest but as pytest. No render /
no beat tracker needed — these only pin the metric calibration and the GT
octave (CLAUDE.md #1: unit-test the load-bearing assumption against an external
reference).
"""
import numpy as np
import pytest

from harmonia.eval.beat_alignment_gt import (
    octave_class, beat_f, load_pop909_gt, POP909_DIR,
)


def test_octave_class_fires_on_song002_scenario():
    # librosa locks the 2x-fast octave on POP909 002 (GT ~64) -> must flag.
    assert octave_class(128.0, 64.0) == "octave"
    # the literal CLAUDE.md "63 vs 129" pair (either labelling) -> octave.
    assert octave_class(63.0, 129.0) == "octave"
    # correct octave -> ok; a ~1.5x error -> 'other', not 'octave'.
    assert octave_class(64.0, 64.0) == "ok"
    assert octave_class(95.0, 64.0) == "other"


def test_beat_f_perfect_and_halfrate():
    ref = np.arange(0.0, 60.0, 0.5)          # 120 bpm, 60 s
    assert abs(beat_f(ref, ref.copy()) - 1.0) < 1e-6
    assert beat_f(ref, ref[::2]) < 0.8       # half-rate est cannot saturate


@pytest.mark.skipif(not (POP909_DIR / "002" / "beat_midi.txt").exists(),
                    reason="POP909 not present")
def test_pop909_002_gt_octave_is_64_not_129():
    # Three POP909 annotations agree song 002 is ~64 bpm; CLAUDE.md's "129 GT"
    # is the octave the tracker mistakenly locks to, not the true GT.
    _, downs, tempo = load_pop909_gt("002")
    assert 60 <= tempo <= 68
    assert len(downs) > 10
