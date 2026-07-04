"""
Unit tests for harmonia/models/rhythm.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harmonia.models.rhythm import RhythmAnalyser

_POP909_MIDI = (
    Path(__file__).parent.parent
    / "data" / "pop909" / "POP909" / "001" / "001.mid"
)


class TestAnalyseFromMidi:
    """Regression test for docs/known_issues.md #8: analyse_from_midi()
    called pm.get_tempo_change_times(), which is not a pretty_midi API
    (get_tempo_changes() returns the (times, tempi) tuple directly) —
    crashed unconditionally, with no callers ever exercising the path."""

    @pytest.mark.skipif(not _POP909_MIDI.exists(), reason="no local POP909 dataset")
    def test_runs_without_crashing_on_real_midi(self):
        grid = RhythmAnalyser().analyse_from_midi(_POP909_MIDI)
        assert grid.n_beats > 0
        assert grid.tempo_bpm > 0
        assert grid.backend == "midi"

    def test_runs_on_synthetic_midi_with_no_tempo_events(self, tmp_path):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=2.0))
        pm.instruments.append(inst)
        midi_path = tmp_path / "synthetic.mid"
        pm.write(str(midi_path))

        grid = RhythmAnalyser().analyse_from_midi(midi_path, default_tempo=100.0)
        assert grid.n_beats > 0
        assert grid.beat_times[0] == 0.0
