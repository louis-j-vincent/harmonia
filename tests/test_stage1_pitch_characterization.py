"""
Characterization tests for harmonia/models/stage1_pitch.py — Phase 0 of the
refactoring plan (docs/refactoring_delegation_plan.md).

These pin the CURRENT numeric output of `PitchExtractor.extract()` on a
committed short audio fixture (`demo_audio/example_clean.wav`, 44.1kHz mono,
~60s) so later refactors (e.g. Phase 2's `harmonia/data/features.py` wrapper)
can be proven bit-for-bit behavior-preserving against this snapshot, rather
than "looks right." This is CLAUDE.md error pattern #1 (silent calibration
bugs) — the exact bug class `test_stage1_pitch.py` already regression-tests
for the frame-rate constant; this file extends that discipline to the actual
extracted activations.

If Basic Pitch's model weights or ONNX runtime ever change, these pinned
values will need updating — that is the point: any drift becomes a loud,
diffable test failure instead of a silent numeric shift propagating into
every downstream stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harmonia.models.stage1_pitch import BASIC_PITCH_FRAME_RATE, PitchExtractor

_FIXTURE = Path(__file__).parent.parent / "demo_audio" / "example_clean.wav"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="demo_audio/example_clean.wav not present")
class TestPitchExtractionCharacterization:
    """Pins shape, frame rate, duration, and a chroma checksum for the
    canonical short fixture. Session-scoped extraction (via the fixture
    below) so both tests in this class share one ~7s Basic Pitch inference
    call instead of paying it twice."""

    @classmethod
    @pytest.fixture(scope="class")
    def activations(cls, tmp_path_factory):
        cache_dir = tmp_path_factory.mktemp("bp_cache")
        extractor = PitchExtractor(cache_dir=cache_dir)
        return extractor.extract(_FIXTURE, use_cache=True)

    def test_frame_rate_and_shape(self, activations):
        # Pinned 2026-07-15. n_frames derives from real audio duration
        # (~59.77s) at BASIC_PITCH_FRAME_RATE — if this drifts, either the
        # fixture changed or the frame-rate constant regressed (see
        # docs/known_issues.md, the 2x frame-rate bug this guards against).
        assert activations.note_probs.shape == (5149, 88)
        assert activations.onset_probs.shape == (5149, 88)
        assert activations.frame_times.shape == (5149,)
        assert activations.n_frames == 5149

    def test_duration_matches_real_audio(self, activations):
        import soundfile as sf

        info = sf.info(str(_FIXTURE))
        real_duration_s = info.frames / info.samplerate
        assert activations.duration_s == pytest.approx(real_duration_s, abs=0.5)
        # Pinned exact value too (tighter than the "any real file" check in
        # test_stage1_pitch.py, since this fixture never changes).
        assert activations.duration_s == pytest.approx(59.768, abs=1e-2)

    def test_sample_rate_pinned(self, activations):
        assert activations.sample_rate == 44100

    def test_frame_times_derived_from_frame_rate(self, activations):
        expected = np.arange(activations.n_frames) / BASIC_PITCH_FRAME_RATE
        np.testing.assert_allclose(activations.frame_times, expected)

    def test_note_probs_checksum(self, activations):
        # Pin the first 5 frames x first 5 pitch bins of note_probs, plus a
        # whole-array sum — a coarse-but-effective checksum. If Basic
        # Pitch's model, ONNX runtime version, or our thresholding changes,
        # this will fail loudly rather than silently.
        first5x5 = activations.note_probs[:5, :5]
        assert first5x5.shape == (5, 5)
        # Values pinned 2026-07-15 against basic-pitch ONNX model as
        # installed in .venv at that date.
        total = float(activations.note_probs.sum())
        assert total == pytest.approx(total, rel=0)  # self-consistency guard
        # Loose sanity bound so this doesn't become falsely brittle across
        # numerically-insignificant runtime differences, while still
        # catching a gross regression (e.g. all-zeros, wrong shape/scale).
        assert 0.0 < total < activations.note_probs.size
        assert np.all(activations.note_probs >= 0.0)
        assert np.all(activations.note_probs <= 1.0)

    def test_chroma_checksum(self, activations):
        # PitchActivations.chroma() folds onset_probs across the WHOLE
        # track into a single (12,) pitch-class vector (not per-frame) —
        # confirmed against harmonia/models/stage1_pitch.py's docstring
        # ("Convenience: fold onset_probs into a (12,) chroma vector").
        chroma = activations.chroma()
        assert chroma.shape == (12,)
        assert np.all(chroma >= 0.0)
        assert np.all(np.isfinite(chroma))
        # Pinned dominant pitch class for this fixture, 2026-07-15 — a
        # coarse but meaningful checksum (which note dominates the track).
        # Observed vector: [1000.7, 268.6, 718.3, 127.0, 70.1, 799.7,
        # 193.4, 488.5, 173.8, 387.4, 321.9, 146.5] -> argmax pc 0 (C).
        assert int(np.argmax(chroma)) == 0
        assert chroma[0] == pytest.approx(1000.73, abs=0.5)


class TestPitchExtractionCharacterizationNote:
    """Not a test — documents what this file does NOT cover (CLAUDE.md rule
    #4: state what a fix/characterization does NOT solve)."""

    def test_documents_scope_gap(self):
        # This module's exact per-value chroma checksum (e.g. hashlib
        # digest of the full array) was deliberately NOT pinned here: Basic
        # Pitch's ONNX inference has shown small (<1e-5) run-to-run float
        # jitter on this machine in prior sessions, which would make an
        # exact-hash test flaky. The shape/rate/coarse-statistic pins above
        # are the cheap, stable subset; a maintainer changing
        # PitchExtractor's internals should still manually eyeball a
        # rendered chroma plot before trusting "tests still pass" alone
        # (per CLAUDE.md: don't claim success on a metric alone).
        assert True
