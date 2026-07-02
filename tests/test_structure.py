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
        # 20 beats, two active keys at amplitude 0.5 each (each beat's raw
        # activation already sums to 1.0). Segment chroma sums beats without
        # any further normalisation, so it should total 20.0, not 1.0 (see
        # docs/known_issues.md #0 — a segment-level-normalised chroma
        # destroys the magnitude information infer_key() needs to calibrate
        # its posterior).
        beat_probs = _synthetic_beat_probs(n_beats=20, active_midi=[60, 64], amplitude=0.5)
        beat_times = np.arange(21, dtype=float)  # 21 boundaries for 20 beats
        seg = Segmenter()._make_segment(beat_probs, beat_times, 0, 20)
        assert seg.chroma.sum() == pytest.approx(20.0)

    def test_chroma_invariant_to_per_beat_amplitude(self):
        # A beat's raw activation-probability *amplitude* isn't a genuine
        # independent-trial count (many pitch classes can co-sound within
        # one beat, inflating it arbitrarily) -- so each beat is
        # L1-normalised before being summed into the segment chroma. Two
        # segments with identical relative pitch-class shape per beat but
        # very different absolute amplitude must produce the same chroma
        # magnitude: only the *number* of beats is real evidence, not how
        # loud any one of them happens to be. This is what prevents
        # infer_key()'s posterior from saturating to a meaningless bit-exact
        # 1.0 confidence on every real segment regardless of length.
        quiet = _synthetic_beat_probs(n_beats=16, active_midi=[60, 64, 67], amplitude=0.1)
        loud = _synthetic_beat_probs(n_beats=16, active_midi=[60, 64, 67], amplitude=50.0)
        seg_quiet = Segmenter()._make_segment(quiet, np.arange(17, dtype=float), 0, 16)
        seg_loud = Segmenter()._make_segment(loud, np.arange(17, dtype=float), 0, 16)
        np.testing.assert_allclose(seg_quiet.chroma, seg_loud.chroma, rtol=1e-5)

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

    def test_confidence_does_not_saturate_for_realistic_segment(self):
        # A 35-beat segment (a genuinely long real segment, see the song 001
        # example in docs/handoff_2026-07-02_key_inference.md §4) with noisy,
        # multi-note-per-beat activation, still not perfectly clean (small
        # activation on a few off-triad notes too). Confidence should land
        # confidently but genuinely below 1.0 -- if this saturates to
        # bit-exact 1.0, per-beat evidence normalisation has regressed and
        # every segment's confidence is uninformative again (just at the
        # opposite, equally-useless extreme from the original bug).
        beat_probs = _synthetic_beat_probs(n_beats=35, active_midi=[60, 64, 67], amplitude=0.6)
        beat_probs[:, 61 - 21] = 0.05   # small unrelated activation (C#)
        beat_probs[:, 66 - 21] = 0.05   # small unrelated activation (F#)
        seg = Segmenter()._make_segment(beat_probs, np.arange(36, dtype=float), 0, 35)
        kp = infer_key(seg.chroma)
        assert kp.tonic == 0
        assert kp.mode == "major"
        assert 0.5 < kp.confidence < 0.999
