"""
Unit tests for structural segmentation's chroma aggregation.

No audio/model needed — beat_probs are synthetic (B, 88) arrays.
"""

import numpy as np
import pytest

from harmonia.models.structure import Segmenter
from harmonia.theory.key_profiles import infer_key


def _synthetic_beat_probs(n_beats: int, active_midi: list[int], amplitude: float) -> np.ndarray:
    """(n_beats, 88) array with only `active_midi` keys firing, at `amplitude`."""
    beat_probs = np.zeros((n_beats, 88), dtype=np.float32)
    for midi in active_midi:
        beat_probs[:, midi - 21] = amplitude
    return beat_probs


class TestMakeSegmentChroma:

    def test_chroma_is_raw_not_normalised(self):
        # 20 beats, two active keys at amplitude 0.5 each => each beat
        # contributes 1.0 total activation; raw chroma should sum to 20.0,
        # not 1.0 (see docs/known_issues.md #0 — a pre-normalised chroma
        # destroys the magnitude information infer_key() needs to calibrate
        # its posterior).
        beat_probs = _synthetic_beat_probs(n_beats=20, active_midi=[60, 64], amplitude=0.5)
        beat_times = np.arange(21, dtype=float)  # 21 boundaries for 20 beats
        seg = Segmenter()._make_segment(beat_probs, beat_times, 0, 20)
        assert seg.chroma.sum() == pytest.approx(20.0)

    def test_chroma_scales_with_segment_length(self):
        # A segment covering twice as many beats of identical content should
        # have twice the raw chroma magnitude -- this is exactly the signal
        # infer_key() uses to be more confident about longer segments.
        short = _synthetic_beat_probs(n_beats=10, active_midi=[60], amplitude=1.0)
        long = _synthetic_beat_probs(n_beats=40, active_midi=[60], amplitude=1.0)
        seg_short = Segmenter()._make_segment(short, np.arange(11, dtype=float), 0, 10)
        seg_long = Segmenter()._make_segment(long, np.arange(41, dtype=float), 0, 40)
        assert seg_long.chroma.sum() == pytest.approx(4 * seg_short.chroma.sum())


class TestSyntheticUnambiguousKey:
    """
    Layer-2 foundation: synthetic, unambiguous cases for scale-fitting,
    exercised through the real segment -> infer_key glue (not just infer_key
    in isolation, which tests/test_theory.py already covers).
    """

    def test_pure_c_major_triad_segment_infers_c_major_confidently(self):
        # C4, E4, G4 (MIDI 60, 64, 67) held for 32 beats -- a C major triad
        # with no other pitch content at all. About as unambiguous as real
        # segment-level chroma can get.
        beat_probs = _synthetic_beat_probs(n_beats=32, active_midi=[60, 64, 67], amplitude=0.8)
        beat_times = np.arange(33, dtype=float)
        seg = Segmenter()._make_segment(beat_probs, beat_times, 0, 32)
        kp = infer_key(seg.chroma)
        assert kp.tonic == 0
        assert kp.mode == "major"
        assert kp.confidence > 0.5

    def test_pure_a_minor_triad_segment_infers_a_minor_confidently(self):
        # A3, C4, E4 (MIDI 57, 60, 64) -- A minor triad.
        beat_probs = _synthetic_beat_probs(n_beats=32, active_midi=[57, 60, 64], amplitude=0.8)
        beat_times = np.arange(33, dtype=float)
        seg = Segmenter()._make_segment(beat_probs, beat_times, 0, 32)
        kp = infer_key(seg.chroma)
        assert kp.tonic == 9
        assert kp.mode == "minor"
        assert kp.confidence > 0.5

    def test_longer_segment_more_confident_than_shorter_same_content(self):
        # Same triad, same relative shape, different segment length. More
        # observed evidence (more beats, same content) should sharpen the
        # posterior -- the exact behaviour the old bug made impossible
        # (chroma was pre-normalised to sum 1 regardless of segment length).
        short_probs = _synthetic_beat_probs(n_beats=4, active_midi=[60, 64, 67], amplitude=0.8)
        long_probs = _synthetic_beat_probs(n_beats=32, active_midi=[60, 64, 67], amplitude=0.8)
        seg_short = Segmenter()._make_segment(short_probs, np.arange(5, dtype=float), 0, 4)
        seg_long = Segmenter()._make_segment(long_probs, np.arange(33, dtype=float), 0, 32)
        kp_short = infer_key(seg_short.chroma)
        kp_long = infer_key(seg_long.chroma)
        assert kp_long.confidence > kp_short.confidence
