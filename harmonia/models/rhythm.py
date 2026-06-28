"""
Rhythm analysis: beat tracking, tempo estimation, beat-grid quantisation.

Primary backend: madmom (RNN + DBN — state of the art).
Fallback backend: librosa (good enough, always available).

The beat grid is the temporal backbone of the entire pipeline:
    audio → beats → quantise note activations per beat
                  → segment into bars/sections
                  → infer key per section
                  → infer chords per beat

Why madmom over librosa?
    madmom's DBN beat tracker is substantially better on jazz and music
    with complex rhythms, syncopation, and tempo changes. librosa's
    default dynamic programming approach struggles with swing.
    We try madmom first and fall back gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class TimeSignature(str, Enum):
    FOUR_FOUR  = "4/4"
    THREE_FOUR = "3/4"
    SIX_EIGHT  = "6/8"
    FIVE_FOUR  = "5/4"
    SEVEN_FOUR = "7/4"
    UNKNOWN    = "?"


@dataclass
class BeatGrid:
    """
    Output of the rhythm analyser.

    beat_times:      (B,) seconds — time of each detected beat.
    downbeat_times:  (D,) seconds — subset of beats that are downbeats (beat 1).
                     Empty if downbeat detection was skipped.
    tempo_bpm:       estimated global tempo (BPM).
    tempo_curve:     (B-1,) local BPM between consecutive beats.
                     Useful for detecting rubato / ritardando.
    time_signature:  inferred metre.
    backend:         "madmom" or "librosa" — which engine produced this.
    """
    beat_times: np.ndarray          # (B,)
    downbeat_times: np.ndarray      # (D,) — may be empty
    tempo_bpm: float
    tempo_curve: np.ndarray         # (B-1,)
    time_signature: TimeSignature
    backend: str

    @property
    def n_beats(self) -> int:
        return len(self.beat_times)

    @property
    def beat_duration_s(self) -> float:
        """Median inter-beat interval in seconds."""
        if len(self.beat_times) < 2:
            return 60.0 / self.tempo_bpm
        return float(np.median(np.diff(self.beat_times)))

    def beats_per_bar(self) -> int:
        """Integer beats per bar based on inferred time signature."""
        mapping = {
            TimeSignature.FOUR_FOUR:  4,
            TimeSignature.THREE_FOUR: 3,
            TimeSignature.SIX_EIGHT:  6,
            TimeSignature.FIVE_FOUR:  5,
            TimeSignature.SEVEN_FOUR: 7,
        }
        return mapping.get(self.time_signature, 4)

    def beat_index_at(self, time_s: float) -> int:
        """Return index of the nearest beat to the given time."""
        return int(np.argmin(np.abs(self.beat_times - time_s)))

    def quantise_frames(
        self,
        frame_times: np.ndarray,
        note_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate frame-level note probabilities into beat-level summaries.

        For each beat interval [beat_t, beat_{t+1}), sum the note activation
        probabilities over all frames in that interval.

        Args:
            frame_times:  (F,) — time in seconds of each frame (from Stage 1).
            note_probs:   (F, 88) — per-frame note probabilities.

        Returns:
            beat_probs: (B, 88) — per-beat summed note probabilities.
                        Unnormalised — sum reflects how long a note was active.
        """
        n_beats = len(self.beat_times)
        beat_probs = np.zeros((n_beats, note_probs.shape[1]), dtype=np.float32)

        for b in range(n_beats):
            t_start = self.beat_times[b]
            t_end = (
                self.beat_times[b + 1]
                if b + 1 < n_beats
                else t_start + self.beat_duration_s
            )
            mask = (frame_times >= t_start) & (frame_times < t_end)
            if mask.any():
                beat_probs[b] = note_probs[mask].sum(axis=0)

        return beat_probs


# ---------------------------------------------------------------------------
# Time signature inference
# ---------------------------------------------------------------------------

def _infer_time_signature(
    beat_times: np.ndarray,
    downbeat_times: np.ndarray,
) -> TimeSignature:
    """
    Infer time signature from beat and downbeat times.
    Falls back to 4/4 if downbeats are unavailable.
    """
    if len(downbeat_times) < 2 or len(beat_times) < 4:
        return TimeSignature.FOUR_FOUR

    # Count beats between consecutive downbeats
    beats_per_bar_counts = []
    for i in range(len(downbeat_times) - 1):
        t0, t1 = downbeat_times[i], downbeat_times[i + 1]
        n_beats = np.sum((beat_times >= t0) & (beat_times < t1))
        if n_beats > 0:
            beats_per_bar_counts.append(n_beats)

    if not beats_per_bar_counts:
        return TimeSignature.FOUR_FOUR

    # Modal count
    values, counts = np.unique(beats_per_bar_counts, return_counts=True)
    modal_bpb = int(values[np.argmax(counts)])

    mapping = {4: TimeSignature.FOUR_FOUR, 3: TimeSignature.THREE_FOUR,
               6: TimeSignature.SIX_EIGHT, 5: TimeSignature.FIVE_FOUR,
               7: TimeSignature.SEVEN_FOUR}
    return mapping.get(modal_bpb, TimeSignature.UNKNOWN)


# ---------------------------------------------------------------------------
# madmom backend
# ---------------------------------------------------------------------------

def _track_beats_madmom(audio_path: str) -> BeatGrid:
    """
    Beat tracking via madmom's RNN + DBN.

    Uses:
      - RNNBeatProcessor: bi-directional RNN producing beat activation function
      - DBNBeatTrackingProcessor: dynamic Bayesian network for tempo/beat decoding
      - RNNDownBeatProcessor + DBNDownBeatTrackingProcessor: downbeat detection
    """
    import madmom.features.beats as mb
    import madmom.features.downbeats as md

    logger.info("Running madmom RNN beat tracker...")

    # Beat tracking
    beat_proc = mb.DBNBeatTrackingProcessor(fps=100)
    beat_act = mb.RNNBeatProcessor()(audio_path)
    beat_times = beat_proc(beat_act)

    # Downbeat detection (determines time signature)
    try:
        db_proc = md.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        db_act = md.RNNDownBeatProcessor()(audio_path)
        downbeats_raw = db_proc(db_act)
        # downbeats_raw: (N, 2) — columns: [beat_time, beat_position_in_bar]
        downbeat_times = downbeats_raw[downbeats_raw[:, 1] == 1, 0]
    except Exception as e:
        logger.warning(f"Downbeat detection failed: {e}. Using beat times only.")
        downbeat_times = np.array([])

    tempo_curve = 60.0 / np.diff(beat_times) if len(beat_times) > 1 else np.array([120.0])
    tempo_bpm = float(np.median(tempo_curve))
    time_sig = _infer_time_signature(beat_times, downbeat_times)

    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        tempo_bpm=tempo_bpm,
        tempo_curve=tempo_curve,
        time_signature=time_sig,
        backend="madmom",
    )


# ---------------------------------------------------------------------------
# librosa fallback backend
# ---------------------------------------------------------------------------

def _track_beats_librosa(audio_path: str) -> BeatGrid:
    """
    Beat tracking via librosa's dynamic programming approach.
    Less accurate on jazz/swing but always available.
    """
    import librosa

    logger.info("Running librosa beat tracker (madmom unavailable)...")

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    tempo_bpm = float(np.atleast_1d(tempo)[0])
    tempo_curve = np.full(max(len(beat_times) - 1, 1), tempo_bpm)

    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=np.array([]),
        tempo_bpm=tempo_bpm,
        tempo_curve=tempo_curve,
        time_signature=TimeSignature.FOUR_FOUR,
        backend="librosa",
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class RhythmAnalyser:
    """
    Tempo and beat tracking with automatic backend selection.

    Usage:
        analyser = RhythmAnalyser()
        grid = analyser.analyse("song.wav")
        print(f"Tempo: {grid.tempo_bpm:.1f} BPM, {grid.n_beats} beats")
        beat_probs = grid.quantise_frames(frame_times, note_probs)
    """

    def __init__(self, prefer_madmom: bool = True):
        self.prefer_madmom = prefer_madmom
        self._madmom_available: bool | None = None

    def _check_madmom(self) -> bool:
        if self._madmom_available is None:
            try:
                import madmom  # noqa: F401
                self._madmom_available = True
            except ImportError:
                self._madmom_available = False
                logger.warning(
                    "madmom not installed — falling back to librosa beat tracker. "
                    "For best results on jazz: pip install madmom"
                )
        return self._madmom_available

    def analyse(self, audio_path: str | "Path") -> BeatGrid:
        """
        Run beat tracking on an audio file.

        Args:
            audio_path: path to .wav / .mp3 / .flac

        Returns:
            BeatGrid with beat times, tempo, and time signature.
        """
        path = str(audio_path)

        if self.prefer_madmom and self._check_madmom():
            try:
                return _track_beats_madmom(path)
            except Exception as e:
                logger.warning(f"madmom failed ({e}), falling back to librosa")

        return _track_beats_librosa(path)

    def analyse_from_midi(
        self,
        midi_path: str | "Path",
        default_tempo: float = 120.0,
    ) -> BeatGrid:
        """
        Construct a beat grid directly from a MIDI file's tempo map.
        Useful during data pipeline — MIDI has exact beat information.

        Args:
            midi_path:      path to .mid file
            default_tempo:  BPM to use if MIDI has no tempo events

        Returns:
            BeatGrid derived from MIDI tempo map — perfectly accurate.
        """
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        end_time = pm.get_end_time()

        # Get tempo changes: list of (time, tempo_bpm)
        tempo_change_times, tempos = pm.get_tempo_change_times(), pm.get_tempo_changes()
        if len(tempos) == 0:
            tempos = np.array([default_tempo])
            tempo_change_times = np.array([0.0])

        # Build beat grid by advancing through tempo map
        beat_times: list[float] = []
        t = 0.0
        tempo_idx = 0

        while t <= end_time:
            beat_times.append(t)
            # Find current tempo
            while (tempo_idx + 1 < len(tempo_change_times) and
                   tempo_change_times[tempo_idx + 1] <= t):
                tempo_idx += 1
            current_bpm = float(tempos[tempo_idx])
            t += 60.0 / current_bpm

        beat_arr = np.array(beat_times)
        tempo_curve = 60.0 / np.diff(beat_arr) if len(beat_arr) > 1 else np.array([tempos[0]])
        tempo_bpm = float(np.median(tempo_curve))

        return BeatGrid(
            beat_times=beat_arr,
            downbeat_times=np.array([]),
            tempo_bpm=tempo_bpm,
            tempo_curve=tempo_curve,
            time_signature=TimeSignature.FOUR_FOUR,
            backend="midi",
        )
