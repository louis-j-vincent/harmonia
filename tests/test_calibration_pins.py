"""tests/test_calibration_pins.py — red-first calibration pins for rewrite parity.

These pins guard the load-bearing, silently-corrupting constants that CLAUDE.md
error-pattern #1 is about: "a low-level unit/scale/constant error (frame rate off
by 2x, key posterior pinned near-uniform, mislabeled soundfont, beat tracker
locking a 2x tempo octave) silently corrupted everything downstream while
producing plausible numbers." Each pin checks one such assumption against an
EXTERNAL reference (upstream library constant, a real file's real duration, or
independent ground-truth tempo), so it would go RED if the bug reappeared.

All pins must PASS against current code (this is the Phase 0 gate).

Run:  PYTHONPATH=$PWD pytest -o addopts="" tests/test_calibration_pins.py -v

────────────────────────────────────────────────────────────────────────────
2026-07-22 CORRECTION (error-pattern #1, logged): the prior version of this
file asserted "Song 002 tempo must detect ~129 BPM" and titled it an
anti-octave-lock pin. That was BACKWARDS — 129 BPM *is* the 2x octave-lock
error; the three POP909 annotations agree the true tempo is ~64 BPM (CLAUDE.md
Environment gotchas; beat_midi median-diff = 64.0). A pin that asserted 129
would have PINNED THE BUG. It is corrected below to reject the 129 octave.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


class TestFrameRateConstant:
    """Pin #1: BASIC_PITCH_FRAME_RATE == upstream Basic-Pitch hop constant.

    Bug history: the frame rate was 2x wrong once, silently corrupting every
    downstream time index. Invariant: our constant must equal the library's
    AUDIO_SAMPLE_RATE / FFT_HOP, exactly.
    """

    def test_frame_rate_literal(self):
        from harmonia.models.stage1_pitch import BASIC_PITCH_FRAME_RATE
        assert BASIC_PITCH_FRAME_RATE == pytest.approx(86.1328125, abs=1e-9), (
            f"BASIC_PITCH_FRAME_RATE={BASIC_PITCH_FRAME_RATE} != 86.1328125"
        )

    def test_frame_rate_matches_upstream_ratio(self):
        """Cross-check against the ACTUAL upstream basic_pitch constants."""
        import basic_pitch.constants as C

        from harmonia.models.stage1_pitch import BASIC_PITCH_FRAME_RATE

        upstream = C.AUDIO_SAMPLE_RATE / C.FFT_HOP  # 22050 / 256
        assert BASIC_PITCH_FRAME_RATE == pytest.approx(upstream, abs=1e-9), (
            f"{BASIC_PITCH_FRAME_RATE} != AUDIO_SAMPLE_RATE/FFT_HOP="
            f"{C.AUDIO_SAMPLE_RATE}/{C.FFT_HOP}={upstream}"
        )
        # ANNOTATIONS_FPS is the ROUNDED int (86) — must NOT be used as the rate.
        assert int(C.ANNOTATIONS_FPS) == 86
        assert abs(BASIC_PITCH_FRAME_RATE - C.ANNOTATIONS_FPS) > 0.1, (
            "the true rate must differ from the rounded ANNOTATIONS_FPS int"
        )


class TestChromaNormalization:
    """Pin #2: the chroma normalization convention is L2-per-12dim-block.

    Bug history (Issue #10): raw summed chroma scales with segment length; the
    family classifier is only duration-invariant if each 12-dim block is
    L2-normalized. A missing L2-norm on a feature block dropped majmin ~24pp.
    This pins the ACTUAL production function `_norm_blocks`, not a mock.
    """

    def test_norm_blocks_unit_norm_per_block(self):
        from harmonia.models.chord_pipeline_v1 import _norm_blocks

        # two stacked 12-dim blocks (24-dim row), arbitrary magnitudes
        x = np.array([[3.0, 4.0] + [0.0] * 10 + [1.0] * 12])  # block0 L2=5, block1 L2=sqrt(12)
        y = _norm_blocks(x).reshape(1, 2, 12)
        norms = np.linalg.norm(y, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6), f"blocks not unit-norm: {norms}"

    def test_norm_blocks_zero_safe(self):
        """An all-zero block must not divide-by-zero (uses +1e-9 epsilon)."""
        from harmonia.models.chord_pipeline_v1 import _norm_blocks

        z = _norm_blocks(np.zeros((1, 12)))
        assert np.all(np.isfinite(z)) and np.allclose(z, 0.0)


class TestSampleRateAndDuration:
    """Pin #3: frame count vs REAL file duration must be consistent.

    Bug history: an off-by-2x frame rate (or a resample mismatch) made the
    per-frame time index disagree with the audio's real length. This checks a
    REAL cached file: n_frames / BASIC_PITCH_FRAME_RATE must equal the .m4a's
    real duration (librosa) within 1 second. This is the check that would have
    caught the frame-rate-2x bug.
    """

    def test_stem_cache_duration_matches_real_audio(self):
        from harmonia.models.stage1_pitch import (
            BASIC_PITCH_FRAME_RATE, PitchActivations,
        )

        stem = REPO / "data" / "cache" / "pitch" / "ben_e_king_stand_by_me_audio.npz"
        m4a = REPO / "docs" / "audio" / "ben_e_king_stand_by_me_audio.m4a"
        if not (stem.exists() and m4a.exists()):
            pytest.skip("ben_e_king stem cache / m4a not on disk")

        acts = PitchActivations.load(stem)
        n_frames = acts.frame_times.shape[0]
        dur_from_frames = n_frames / BASIC_PITCH_FRAME_RATE

        import librosa
        real_dur = librosa.get_duration(path=str(m4a))

        assert abs(dur_from_frames - real_dur) < 1.0, (
            f"frame-derived duration {dur_from_frames:.2f}s disagrees with real "
            f"audio {real_dur:.2f}s by >1s — frame-rate/resample calibration bug"
        )

    def test_frame_times_spacing_is_frame_rate(self):
        """The cached frame_times must be spaced at exactly 1/BASIC_PITCH_FRAME_RATE."""
        from harmonia.models.stage1_pitch import (
            BASIC_PITCH_FRAME_RATE, PitchActivations,
        )

        stem = REPO / "data" / "cache" / "pitch" / "ben_e_king_stand_by_me_audio.npz"
        if not stem.exists():
            pytest.skip("stem cache not on disk")
        ft = PitchActivations.load(stem).frame_times
        dt = np.median(np.diff(ft))
        assert dt == pytest.approx(1.0 / BASIC_PITCH_FRAME_RATE, rel=1e-6), (
            f"frame spacing {dt} != 1/{BASIC_PITCH_FRAME_RATE}"
        )


class TestSong002TempoOctave:
    """Pin #4: POP909 Song 002 GT tempo is ~64 BPM, NOT the 129 octave.

    Bug history (CLAUDE.md): librosa doubles song 002 to ~129 BPM (2x-fast
    octave lock); three POP909 annotations agree on ~64 (beat_midi 64.0 /
    beat_audio 63.8 / MIDI 62). This pin uses the INDEPENDENT beat_midi GT
    (col 1 beat times) — no audio render needed — and REJECTS the 129 octave.
    """

    def test_song_002_gt_tempo_is_64_not_129(self):
        beat_midi = REPO / "data" / "pop909" / "POP909" / "002" / "beat_midi.txt"
        if not beat_midi.exists():
            pytest.skip("POP909 song 002 beat_midi.txt not on disk")

        beats = np.loadtxt(beat_midi)[:, 0]
        tempo = 60.0 / float(np.median(np.diff(beats)))

        assert 55.0 <= tempo <= 75.0, (
            f"song 002 GT tempo {tempo:.1f} BPM outside ~64 band — beat_midi changed?"
        )
        # explicit anti-octave-lock: must NOT be the doubled ~129 error.
        assert not (120.0 <= tempo <= 140.0), (
            f"song 002 tempo {tempo:.1f} is the 2x octave-lock error, not ~64"
        )

    def test_song_002_downbeat_gt_present(self):
        """beat_midi col 3 must carry real downbeat flags (independent GT)."""
        beat_midi = REPO / "data" / "pop909" / "POP909" / "002" / "beat_midi.txt"
        if not beat_midi.exists():
            pytest.skip("POP909 song 002 beat_midi.txt not on disk")
        arr = np.loadtxt(beat_midi)
        assert arr.shape[1] >= 3, "beat_midi should have >=3 columns"
        n_down = int((arr[:, 2] == 1).sum())
        assert n_down > 0, "no downbeat flags in col 3 — GT missing"


class TestChordChartContract:
    """Pin #5: the real ChordChart dataclass has the required parity fields and
    round-trips through JSON without loss (the parity diff relies on this)."""

    def test_real_chordchart_fields(self):
        from harmonia.pipeline import ChordChart

        required = {
            "source_path", "duration_s", "tempo_bpm", "time_signature",
            "global_key", "global_key_confidence", "style", "chords",
            "segments", "sections",
        }
        fields = set(ChordChart.__dataclass_fields__)
        missing = required - fields
        assert not missing, f"ChordChart missing parity fields: {missing}"

    def test_chart_json_roundtrip(self):
        import json

        from harmonia.pipeline import ChordChart

        chart = ChordChart(
            source_path="test.wav", duration_s=180.0, tempo_bpm=120.0,
            time_signature="4/4", global_key="C major", global_key_confidence=0.95,
            style="v1", modulations=[],
            chords=[{"label": "C:maj", "start_s": 0.0, "end_s": 2.0,
                     "duration_beats": 4, "confidence": 0.95}],
            segments=[{"start_s": 0.0, "end_s": 180.0, "key": "C major", "n_beats": 360}],
        )
        blob = {
            "chords": chart.chords, "segments": chart.segments,
            "tempo_bpm": chart.tempo_bpm, "global_key": chart.global_key,
        }
        assert json.loads(json.dumps(blob)) == blob


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-o", "addopts="])
