"""harmonia.beats.apply_tempo_octave — Louis's ÷2/×2 manual override for a
tracker-wide tempo-octave error (2026-09-17, Can't Take My Eyes Off You:
Beat This! read ~125 BPM where the real quarter-note pulse is ~62.5 BPM).

`check_grid` cannot see this class of bug on its own: it is invariant to the
absolute tempo (see the block comment in beats.py above `apply_tempo_octave`).
The real song's raw cache (`state/cache/beats/J36z7AnhvOM__3269035.json`)
showed the failure mode this file pins: downbeats fall every 4 of the
too-dense beats (bpb=4, direct≈1.0) — a grid that looks perfectly healthy to
`check_grid`, just twice too fast. `_octave_test_grid` reproduces that shape
synthetically so the test doesn't depend on a cached fixture.
"""
from __future__ import annotations

import pytest

from harmonia.beats import (BeatTrackingError, apply_tempo_octave,
                            check_grid, grid_quality)


def _octave_locked_grid(n_bars: int = 60, beat_step: float = 0.48, bpb: int = 4):
    """A grid exactly like the real one: beats twice too dense, downbeats
    every `bpb` of THOSE beats — self-consistent, so `check_grid` accepts it
    as-is even though it is twice too fast."""
    beats = [round(i * beat_step, 4) for i in range(n_bars * bpb + 1)]
    downbeats = beats[0::bpb]
    return beats, downbeats


def test_raw_octave_lock_is_invisible_to_check_grid():
    """Ground the premise: the un-corrected grid passes the guard cleanly —
    proving the bug needs a human override, not a smarter guard."""
    beats, downbeats = _octave_locked_grid()
    r = check_grid(beats, downbeats, "locked.m4a")
    assert r["metre"] == 4
    assert r["coverage"] > 0.99


def test_halve_recovers_a_valid_metre_4_grid():
    beats, downbeats = _octave_locked_grid()
    d = {"beats": beats, "downbeats": downbeats, "bpm": 125.0}
    fixed = apply_tempo_octave(d, 0.5)
    assert fixed["bpm"] == pytest.approx(62.5, abs=1.0)
    assert len(fixed["beats"]) == pytest.approx(len(beats) / 2, abs=1)
    # the invariant `track()` promises: every downbeat is a real beat
    beat_set = set(fixed["beats"])
    assert all(t in beat_set for t in fixed["downbeats"])
    r = check_grid(fixed["beats"], fixed["downbeats"], "fixed.m4a")
    assert r["metre"] == 4
    assert r["coverage"] > 0.99


def test_halving_only_beats_would_have_read_as_a_half_tempo_lock():
    """Negative control: dropping every other BEAT while leaving downbeats
    untouched — the naive fix — reproduces the disallowed metre-2 signature
    instead of solving anything. Pins why `_halve_beats` must also thin
    `downbeats`, not just filter them against the halved beats."""
    beats, downbeats = _octave_locked_grid()
    naive_beats = beats[0::2]
    with pytest.raises(BeatTrackingError):
        check_grid(naive_beats, downbeats, "naive.m4a")


def test_double_recovers_a_valid_metre_4_grid_from_a_too_slow_lock():
    """Mirror case: the tracker locked onto HALF the true tempo (too few
    beats, too few downbeats — the opposite octave error)."""
    beats, downbeats = _octave_locked_grid(beat_step=1.92)  # half-tempo lock
    d = {"beats": beats, "downbeats": downbeats, "bpm": 31.25}
    fixed = apply_tempo_octave(d, 2.0)
    assert fixed["bpm"] == pytest.approx(62.5, abs=2.0)
    beat_set = set(fixed["beats"])
    assert all(t in beat_set for t in fixed["downbeats"])
    r = check_grid(fixed["beats"], fixed["downbeats"], "fixed.m4a")
    assert r["metre"] == 4
    assert r["coverage"] > 0.99


def test_corrections_stack_and_cancel():
    """The server route treats the factor as a NET, accumulated correction
    (÷2 twice = ÷4, ÷2 then ×2 = back to start) — this pins the arithmetic
    `apply_tempo_octave` must get right for that to hold."""
    beats, downbeats = _octave_locked_grid()
    d = {"beats": beats, "downbeats": downbeats, "bpm": 125.0}
    quartered = apply_tempo_octave(d, 0.25)
    assert quartered["bpm"] == pytest.approx(31.25, abs=1.0)
    back = apply_tempo_octave(apply_tempo_octave(d, 0.5), 2.0)
    assert back["bpm"] == pytest.approx(d["bpm"], rel=0.05)


def test_none_and_unity_are_no_ops():
    beats, downbeats = _octave_locked_grid()
    d = {"beats": beats, "downbeats": downbeats, "bpm": 125.0}
    assert apply_tempo_octave(d, None) is d
    assert apply_tempo_octave(d, 1.0) is d


@pytest.mark.parametrize("bad_factor", [0.75, 3, -2, 0])
def test_non_power_of_two_rejected(bad_factor):
    """The UI only ever sends 0.5/2.0 (and the route only ever multiplies
    those into the stored net); anything else reaching here is a caller bug,
    not user input to absorb silently."""
    beats, downbeats = _octave_locked_grid()
    d = {"beats": beats, "downbeats": downbeats, "bpm": 125.0}
    with pytest.raises(ValueError):
        apply_tempo_octave(d, bad_factor)
