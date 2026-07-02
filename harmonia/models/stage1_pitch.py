"""
Stage 1: Audio → Probabilistic Note Activations.

Wraps Spotify's Basic Pitch model to extract frame-level note salience
as a (frames, 88) float32 tensor — the "soft MIDI" representation.

Unlike hard MIDI transcription (note on/off), we preserve the full
probability distribution over piano keys at every frame. This is critical
for Stage 2: the Bayesian chord inference model uses these soft activations
as its likelihood rather than thresholded binary note events, which would
discard harmonic ambiguity that the model can reason about.

Output tensors are saved to disk as compressed .npz for reuse.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Basic Pitch output frame rate (fixed by the model)
BASIC_PITCH_FRAME_RATE = 43.066  # Hz  (~23.2 ms per frame)
N_PIANO_KEYS = 88
MIDI_START = 21  # A0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PitchActivations:
    """
    Stage 1 output for a single audio file.

    note_probs:   (frames, 88) float32 — P(key active | frame).
                  Key index 0 = MIDI 21 (A0), index 87 = MIDI 108 (C8).
    onset_probs:  (frames, 88) float32 — P(note onset | frame).
                  Useful for rhythm inference.
    frame_times:  (frames,) float64 — time in seconds of each frame centre.
    sample_rate:  original audio sample rate.
    duration_s:   audio duration in seconds.
    source_path:  path to the source audio file.
    """
    note_probs: np.ndarray       # (F, 88) float32
    onset_probs: np.ndarray      # (F, 88) float32
    frame_times: np.ndarray      # (F,) float64
    sample_rate: int
    duration_s: float
    source_path: Path

    @property
    def n_frames(self) -> int:
        return self.note_probs.shape[0]

    def chroma(self, weight_by_octave: bool = True) -> np.ndarray:
        """Convenience: fold onset_probs into a (12,) chroma vector.

        Uses onset_probs, not note_probs — note_probs is a near-constant
        sustain signal (see PitchExtractor docstring) and carries almost no
        pitch-class information.
        """
        from harmonia.theory.key_profiles import activations_to_chroma
        return activations_to_chroma(self.onset_probs, weight_by_octave)

    def save(self, path: Path) -> None:
        """Save to compressed .npz."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            note_probs=self.note_probs,
            onset_probs=self.onset_probs,
            frame_times=self.frame_times,
            sample_rate=np.array(self.sample_rate),
            duration_s=np.array(self.duration_s),
            source_path=np.array(str(self.source_path)),
        )

    @classmethod
    def load(cls, path: Path) -> "PitchActivations":
        """Load from .npz."""
        data = np.load(path, allow_pickle=True)
        return cls(
            note_probs=data["note_probs"],
            onset_probs=data["onset_probs"],
            frame_times=data["frame_times"],
            sample_rate=int(data["sample_rate"]),
            duration_s=float(data["duration_s"]),
            source_path=Path(str(data["source_path"])),
        )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PitchExtractor:
    """
    Wraps Basic Pitch to extract soft note activations from audio files.

    Basic Pitch (Spotify, 2022) is a polyphonic pitch estimator based on
    a CNN trained on a large dataset of aligned audio and MIDI.
    It outputs per-frame note salience maps rather than quantised MIDI events,
    making it ideal as Stage 1 of our Bayesian pipeline.

    Paper: Bitteur et al., "A Lightweight Instrument-Agnostic Model for
           Polyphonic Note Transcription and Multipitch Estimation" (ICASSP 2022).
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Args:
            cache_dir: if provided, .npz activations are cached here keyed
                       by a hash of the audio file path + mtime. Subsequent
                       calls on the same file skip inference entirely.
        """
        self.cache_dir = cache_dir
        self._model = None  # lazy-loaded on first call

    def _load_model(self) -> None:
        """Lazy-load Basic Pitch to avoid import overhead at module level."""
        try:
            from basic_pitch.inference import run_inference
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from pathlib import Path as _Path
            self._run_inference = run_inference
            # Prefer ONNX — avoids TF 2.21 SavedModel breakage, faster on M4.
            onnx_path = _Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
            self._model_path = onnx_path if onnx_path.exists() else ICASSP_2022_MODEL_PATH
            logger.info(f"Basic Pitch model loaded ({self._model_path.suffix or 'SavedModel'}).")
        except ImportError as e:
            raise ImportError(
                "basic-pitch is not installed. Run: pip install basic-pitch"
            ) from e

    def _cache_key(
        self,
        audio_path: Path,
        onset_threshold: float,
        frame_threshold: float,
        onset_percentile: float | None,
    ) -> str:
        stat = audio_path.stat()
        raw = (
            f"{audio_path.resolve()}:{stat.st_mtime}:{stat.st_size}:"
            f"{onset_threshold}:{frame_threshold}:{onset_percentile}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def raw_activations(self, audio_path: Path) -> dict:
        """
        Run Basic Pitch on an audio file and return the raw, pre-threshold
        sigmoid outputs plus timing metadata — no thresholding, no caching.

        Exists so callers that want to try several threshold/normalization
        choices on the same audio (e.g. an A/B experiment harness) don't
        have to re-run ONNX inference for each variant; `extract()` is a
        thin thresholding wrapper around this for the normal cached path.

        Returns:
            dict with keys "note", "onset" (both (F, 88) raw sigmoid
            outputs, NOT thresholded), "frame_times" (F,), "sample_rate",
            "duration_s".
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        if self._model is None:
            self._load_model()

        logger.info(f"Running Basic Pitch on {audio_path.name} ...")

        # Run inference — use run_inference directly to get raw tensors,
        # bypassing predict()'s note_creation step (which uses removed
        # scipy.signal.gaussian in scipy>=1.14).
        model_output = self._run_inference(
            audio_path=str(audio_path),
            model_or_model_path=self._model_path,
        )

        # model_output["note"] shape: (frames, 88) — note salience
        # model_output["onset"] shape: (frames, 88) — onset salience
        note_raw = model_output["note"].astype(np.float32)
        onset_raw = model_output["onset"].astype(np.float32)

        n_frames = note_raw.shape[0]
        frame_times = np.arange(n_frames) / BASIC_PITCH_FRAME_RATE
        duration_s = float(frame_times[-1]) if n_frames > 0 else 0.0

        # Get sample rate from librosa
        import librosa
        _, sr = librosa.load(str(audio_path), sr=None, duration=0.01)

        return {
            "note": note_raw,
            "onset": onset_raw,
            "frame_times": frame_times,
            "sample_rate": int(sr),
            "duration_s": duration_s,
        }

    def extract(
        self,
        audio_path: Path,
        onset_threshold: float = 0.3,
        frame_threshold: float = 0.3,
        onset_percentile: float | None = None,
        use_cache: bool = True,
    ) -> PitchActivations:
        """
        Run Basic Pitch on an audio file and return soft note activations.

        Args:
            audio_path:        path to .wav / .mp3 / .flac file.
            onset_threshold:   Basic Pitch onset detection threshold.
                               Lower = more sensitive (more false positives).
                               Ignored if onset_percentile is set.
            frame_threshold:   Basic Pitch frame threshold. We intentionally
                               keep this LOW (0.3) to preserve soft activations
                               rather than hard binary decisions.
            onset_percentile:  if set, the onset threshold is computed per-song
                               as this percentile of that song's own raw onset
                               distribution, instead of using the fixed
                               onset_threshold. Adapts to per-song loudness.
            use_cache:         if True and cache_dir is set, skip inference
                               on previously processed files.

        Returns:
            PitchActivations with (F, 88) note and onset probability tensors.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        # Cache lookup
        if use_cache and self.cache_dir is not None:
            cache_path = self.cache_dir / (
                f"{self._cache_key(audio_path, onset_threshold, frame_threshold, onset_percentile)}.npz"
            )
            if cache_path.exists():
                logger.debug(f"Cache hit: {audio_path.name}")
                return PitchActivations.load(cache_path)

        raw = self.raw_activations(audio_path)

        # Raw sigmoid outputs have a high floor (~0.1–0.2) for inactive keys.
        # Subtract the threshold so only genuinely active notes stay positive.
        effective_onset_threshold = (
            float(np.percentile(raw["onset"], onset_percentile))
            if onset_percentile is not None else onset_threshold
        )
        note_probs = np.clip(raw["note"] - frame_threshold, 0.0, None)
        onset_probs = np.clip(raw["onset"] - effective_onset_threshold, 0.0, None)

        result = PitchActivations(
            note_probs=note_probs,
            onset_probs=onset_probs,
            frame_times=raw["frame_times"],
            sample_rate=raw["sample_rate"],
            duration_s=raw["duration_s"],
            source_path=audio_path,
        )

        # Cache write
        if self.cache_dir is not None:
            cache_path = self.cache_dir / (
                f"{self._cache_key(audio_path, onset_threshold, frame_threshold, onset_percentile)}.npz"
            )
            result.save(cache_path)
            logger.debug(f"Cached activations → {cache_path}")

        return result

    def extract_batch(
        self,
        audio_paths: list[Path],
        **kwargs,
    ) -> list[PitchActivations]:
        """Process a list of audio files. Uses cache when available."""
        results = []
        for i, p in enumerate(audio_paths):
            logger.info(f"[{i+1}/{len(audio_paths)}] {p.name}")
            try:
                results.append(self.extract(p, **kwargs))
            except Exception as e:
                logger.error(f"Failed on {p}: {e}")
        return results
