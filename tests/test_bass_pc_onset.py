"""Tests for `nnls_features.bass_pc_onset` — the onset-windowed sounding-bass
reader (2026-09-15 finding: reading only the attack instead of pooling a
chord's whole span went 6/11 -> 11/11 on "Ready" bars 1-8; see
docs/known_issues.md for the measurement and its scope, n=1 song).

Run with:
    .venv/bin/python -m pytest tests/test_bass_pc_onset.py -v
"""
import numpy as np
import pytest

from harmonia.nnls_features import bass_pc_onset, _ROLL_TO_C


def _bothchroma_with_bass_at(pc_before: int, pc_after: int, switch_t: float,
                             n_frames: int = 200, dt: float = 0.05):
    """A synthetic (arr, times) bothchroma: all-bass energy at `pc_before`
    (absolute pitch class, index 0 = A per the module's own convention)
    until `switch_t`, then at `pc_after`. Treble half left at zero — only the
    bass half is under test.
    """
    times = np.arange(n_frames) * dt
    arr = np.zeros((n_frames, 24), dtype=np.float32)
    for i, t in enumerate(times):
        pc = pc_before if t < switch_t else pc_after
        # module convention: arr index 0 = A, C-rolled by _ROLL_TO_C to land
        # C at index 0 downstream — so write the RAW (A=0) index that rolls
        # to `pc` under np.roll(., _ROLL_TO_C).
        raw_idx = (pc - _ROLL_TO_C) % 12
        arr[i, raw_idx] = 1.0
    return arr, times


def test_onset_reads_only_the_attack_not_the_sustain():
    """The bass note at t=0 is C(0); it changes to G(7) at t=0.5s. A 150ms
    onset window starting at t=0 must report C, not the note it changes to."""
    arr, times = _bothchroma_with_bass_at(pc_before=0, pc_after=7, switch_t=0.5)
    dist = bass_pc_onset(arr, times, t0=0.0, t1=2.0, window_s=0.15)
    assert int(np.argmax(dist)) == 0, f"expected C(0) at the attack, got pc={int(np.argmax(dist))}"


def test_pooling_the_whole_span_would_see_the_later_note_too():
    """Sanity check on the fixture itself: without windowing (window_s >=
    span length), both notes are in view and the later, longer-held one
    (G, held 1.5s vs C's 0.5s) dominates -- this is the failure mode the
    onset window exists to avoid."""
    arr, times = _bothchroma_with_bass_at(pc_before=0, pc_after=7, switch_t=0.5)
    dist = bass_pc_onset(arr, times, t0=0.0, t1=2.0, window_s=2.0)
    assert int(np.argmax(dist)) == 7, "fixture sanity check failed: expected the longer-held note to win"


def test_output_is_a_probability_like_distribution():
    arr, times = _bothchroma_with_bass_at(pc_before=3, pc_after=3, switch_t=0.0)
    dist = bass_pc_onset(arr, times, t0=0.0, t1=1.0)
    assert dist.shape == (12,)
    assert np.all(dist >= 0)
    assert abs(float(dist.sum()) - 1.0) < 1e-5


def test_empty_window_falls_back_to_nearest_frame():
    """A span shorter than one frame (or past the end of the data) must not
    crash -- same empty-interval rule as `pool_beats`."""
    arr, times = _bothchroma_with_bass_at(pc_before=5, pc_after=5, switch_t=0.0, n_frames=10, dt=0.05)
    dist = bass_pc_onset(arr, times, t0=100.0, t1=100.01)  # far past the data
    assert dist.shape == (12,)
    assert abs(float(dist.sum()) - 1.0) < 1e-5


def test_window_never_reads_past_t1():
    """A span shorter than `window_s` must not pull in frames beyond its own
    t1 -- window_s is a CEILING on the read, not a fixed-size read past the
    span's real end."""
    arr, times = _bothchroma_with_bass_at(pc_before=2, pc_after=9, switch_t=0.1)
    # span itself ends at 0.1s (right at the note change) -- a naive
    # window_s=0.15 read must not spill into the pc=9 region beyond t1
    dist = bass_pc_onset(arr, times, t0=0.0, t1=0.1, window_s=0.15)
    assert int(np.argmax(dist)) == 2


@pytest.mark.parametrize("pc", list(range(12)))
def test_c_roll_convention_matches_pool_beats(pc):
    """bass_pc_onset must place pitch class `pc` at index `pc` after the
    roll -- the same convention `pool_beats` uses (C at index 0)."""
    arr, times = _bothchroma_with_bass_at(pc_before=pc, pc_after=pc, switch_t=0.0)
    dist = bass_pc_onset(arr, times, t0=0.0, t1=0.5)
    assert int(np.argmax(dist)) == pc
