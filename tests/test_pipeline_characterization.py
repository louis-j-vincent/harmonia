"""
Characterization test for harmonia/pipeline.py::HarmoniaPipeline — Phase 0
of the refactoring plan (docs/refactoring_delegation_plan.md).

This is the "shippable pipeline still runs end-to-end" gate: it pins the
full audio -> chord chart output on `demo_audio/example_clean.wav` (default
`HarmoniaPipeline()` construction, i.e. exactly what a user gets running the
pipeline out of the box) so any later refactor phase can be checked for
behavior parity against a real run, not just "the individual stage tests
pass." Verified deterministic across two independent runs during this
session (same tempo/key/chord sequence both times) before pinning.

Runs real Basic Pitch (ONNX) + beat tracking + the full HMM chord inference
— this is the slowest test in the Phase-0 set (~15s uncached, faster with
the pitch-extraction cache warm from other tests in the same session) but
is exactly the path that must not silently break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harmonia.pipeline import ChordChart, HarmoniaPipeline

_FIXTURE = Path(__file__).parent.parent / "demo_audio" / "example_clean.wav"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="demo_audio/example_clean.wav not present")
class TestPipelineEndToEndCharacterization:

    @classmethod
    @pytest.fixture(scope="class")
    def chart(cls, tmp_path_factory) -> ChordChart:
        cache_dir = tmp_path_factory.mktemp("pipeline_bp_cache")
        pipeline = HarmoniaPipeline(cache_dir=cache_dir)
        return pipeline.run(_FIXTURE)

    def test_returns_chord_chart(self, chart):
        assert isinstance(chart, ChordChart)

    def test_tempo_pinned(self, chart):
        # librosa beat tracking on this fixture (madmom not exercised here
        # if unavailable/falls back) — pinned 2026-07-15.
        assert chart.tempo_bpm == pytest.approx(139.5, abs=0.1)

    def test_time_signature_pinned(self, chart):
        assert chart.time_signature == "4/4"

    def test_global_key_pinned(self, chart):
        assert chart.global_key == "F major"
        assert chart.global_key_confidence == pytest.approx(1.0, abs=1e-3)

    def test_style_pinned(self, chart):
        assert chart.style == "jazz_medium_swing"

    def test_chord_count_pinned(self, chart):
        assert len(chart.chords) == 10

    def test_segment_count_pinned(self, chart):
        assert len(chart.segments) == 7

    def test_chord_label_sequence_pinned(self, chart):
        labels = [c["label"] for c in chart.chords]
        assert labels == [
            "Dmin7", "Gmin7", "C7", "Dmin7", "Dmin7",
            "Cmaj", "Fmaj", "Dmin7", "Gmin7", "Cmaj7",
        ]

    def test_first_and_last_chord_timing_pinned(self, chart):
        first, last = chart.chords[0], chart.chords[-1]
        assert first["start_s"] == pytest.approx(0.01, abs=1e-2)
        assert last["end_s"] == pytest.approx(chart.duration_s, abs=0.5)

    def test_duration_matches_real_audio(self, chart):
        import soundfile as sf

        info = sf.info(str(_FIXTURE))
        real_duration_s = info.frames / info.samplerate
        assert chart.duration_s == pytest.approx(real_duration_s, abs=0.5)


class TestPipelineCharacterizationNote:
    """Documents scope gaps (CLAUDE.md rule #4)."""

    def test_documents_scope_gap(self):
        # NOT covered by this characterization: (1) `prefer_madmom=True` is
        # the pipeline default, but whether madmom is actually installed
        # and used vs. the librosa fallback is an environment fact, not
        # pinned here — a future environment where madmom becomes available
        # could legitimately change tempo/beat output without that being a
        # "regression." (2) Confidence values on individual chord events
        # were not pinned (they were ~0.01-0.02 in this run, i.e. near the
        # HMM's raw-probability floor) — they're sensitive to many
        # continuous hyperparameters and pinning them tightly would make
        # this test brittle for the wrong reasons; the *label sequence* is
        # the load-bearing invariant. (3) `sections` (issue #22 boundaries)
        # were empty on this pipeline's default config and are not
        # exercised here.
        assert True
