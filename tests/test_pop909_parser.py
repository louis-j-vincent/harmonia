"""
Unit tests for POP909Parser's key_audio.txt ground-truth loading.

Skipped if the (gitignored, locally-downloaded) POP909 dataset isn't
present — same pattern as tests/test_stage1_pitch.py.
"""

from pathlib import Path

import pytest

from harmonia.data.pop909_parser import POP909Parser

_POP909_DIR = Path(__file__).parent.parent / "data" / "pop909" / "POP909"


@pytest.mark.skipif(not _POP909_DIR.exists(), reason="no local POP909 dataset")
class TestKeyGroundTruth:

    def test_song001_key_events_loaded(self):
        song = POP909Parser(_POP909_DIR).parse_song("001")
        assert song is not None
        assert len(song.key_events) >= 1

    def test_song001_key_is_fsharp_major(self):
        # data/pop909/POP909/001/key_audio.txt: "... Gb:maj" — Gb == F#
        # enharmonically, pitch class 6.
        song = POP909Parser(_POP909_DIR).parse_song("001")
        ev = song.key_events[0]
        assert ev.tonic == 6
        assert ev.mode == "major"

    def test_key_at_time_returns_none_outside_range(self):
        song = POP909Parser(_POP909_DIR).parse_song("001")
        assert song.key_at_time(-1.0) is None

    def test_key_at_time_returns_event_inside_range(self):
        song = POP909Parser(_POP909_DIR).parse_song("001")
        ev = song.key_at_time(50.0)
        assert ev is not None
        assert ev.tonic == 6


@pytest.mark.skipif(not _POP909_DIR.exists(), reason="no local POP909 dataset")
class TestDownbeatGroundTruth:
    """Regression tests for docs/known_issues.md #7: beat_midi.txt's third
    column is a ground-truth downbeat flag (spacing exactly 4 in 4/4), but
    _parse_beat_file() used to keep only column 0, silently discarding it —
    while scripts grew private loaders for the same column."""

    def test_song001_downbeats_loaded(self):
        song = POP909Parser(_POP909_DIR).parse_song("001")
        assert song.is_downbeat.shape == song.beat_times.shape
        assert song.is_downbeat.dtype == bool
        assert song.is_downbeat.sum() > 0

    def test_song001_downbeats_every_four_beats(self):
        import numpy as np
        song = POP909Parser(_POP909_DIR).parse_song("001")
        idxs = np.where(song.is_downbeat)[0]
        assert len(idxs) == 73
        assert set(np.diff(idxs)) == {4}

    def test_downbeat_times_are_subset_of_beat_times(self):
        import numpy as np
        song = POP909Parser(_POP909_DIR).parse_song("001")
        assert np.all(np.isin(song.downbeat_times, song.beat_times))
        assert len(song.downbeat_times) == int(song.is_downbeat.sum())
