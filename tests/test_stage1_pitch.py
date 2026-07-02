"""
Unit tests for harmonia/models/stage1_pitch.py.

Starts with the simplest possible check for the most basic, load-bearing
assumption in this module — the frame rate constant — rather than jumping
straight to audio/model-level integration tests. This is exactly the kind
of test that should have existed from the start: BASIC_PITCH_FRAME_RATE was
found to be off by exactly 2x (43.066 Hz instead of the correct
86.1328125 Hz), silently corrupting frame-to-beat alignment for every song
ever processed, and nothing caught it because nothing checked this constant
against its own source of truth. See docs/known_issues.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harmonia.models.stage1_pitch import BASIC_PITCH_FRAME_RATE

_SAMPLE_WAV = (
    Path(__file__).parent.parent / "data" / "renders" / "pop909" / "001"
    / "001_v005_musescoregeneral.wav"
)


class TestFrameRateConstant:

    def test_matches_basic_pitch_own_constants(self):
        """The simplest possible check: does our frame rate assumption
        match Basic Pitch's own documented constants? No audio, no model
        inference needed — this is a pure constants comparison and should
        always be the first thing checked when something about frame-to-time
        conversion looks wrong."""
        from basic_pitch import constants

        expected = constants.AUDIO_SAMPLE_RATE / constants.FFT_HOP
        assert BASIC_PITCH_FRAME_RATE == expected

    def test_matches_documented_annotations_fps(self):
        """Cross-check against Basic Pitch's separately-documented FPS
        constant too, in case the sample-rate/hop-based derivation and the
        documented FPS ever drift apart."""
        from basic_pitch import constants

        assert abs(BASIC_PITCH_FRAME_RATE - constants.ANNOTATIONS_FPS) < 1.0


@pytest.mark.skipif(not _SAMPLE_WAV.exists(), reason="no rendered POP909 audio locally")
class TestComputedDurationMatchesRealAudio:
    """One level up in complexity from the pure-constants check: does the
    computed duration actually match reality for one real file? This is the
    check that would have caught the frame-rate bug directly (computed
    duration was silently 2x the real audio's duration) even without
    knowing to suspect the frame-rate constant specifically."""

    def test_duration_matches_soundfile(self):
        import soundfile as sf

        from harmonia.models.stage1_pitch import PitchExtractor

        info = sf.info(str(_SAMPLE_WAV))
        real_duration_s = info.frames / info.samplerate

        extractor = PitchExtractor(cache_dir=None)
        activations = extractor.extract(_SAMPLE_WAV, use_cache=False)

        # generous tolerance: frame quantisation, not exact-sample alignment
        assert activations.duration_s == pytest.approx(real_duration_s, abs=2.0)
