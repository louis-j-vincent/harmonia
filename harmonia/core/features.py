"""harmonia/core/features.py — canonical feature extraction abstraction (Phase 1.2 PORT).

Pluggable interface for swapping between BasicPitch48, NNLS24, and Musx
feature extractors without rewriting downstream logic. Preserves all caching
strategies and semantic differences (soft activations vs. chord labels).

Public API:
    from harmonia.core.features import FeatureExtractor

    extractor = FeatureExtractor.create("bp48")  # or "nnls24", "musx"
    result = extractor.extract(audio_path)

    if isinstance(result, ActivationResult):
        beat_features = result.pool_to_beats(beat_times)
        # (n_beats, n_features) ready for downstream
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureExtractionResult:
    """Base class for all feature extraction results."""

    extractor_name: str  # "bp48", "nnls24", "musx"

    def pool_to_beats(self, beat_times: np.ndarray, **kwargs) -> np.ndarray:
        """Pool to beat intervals. Subclasses must override if applicable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support pool_to_beats. "
            "This result type may need special handling."
        )


@dataclass
class ActivationResult(FeatureExtractionResult):
    """Per-frame soft activations or predictions (BP48, NNLS24)."""

    activations: np.ndarray  # (frames, n_features)
    onsets: np.ndarray | None = None  # (frames, n_features) or None
    frame_times: np.ndarray | None = None  # (frames,) in seconds
    # Optional provenance/metadata, populated by BP48; None for extractors that
    # don't supply it (e.g. NNLS24). Ported from the legacy
    # stage1_pitch.PitchActivations so rerouted call-sites keep access to
    # .n_frames / .duration_s / .sample_rate / .source_path / .chroma() / .save().
    sample_rate: int | None = None
    duration_s: float | None = None
    source_path: Path | None = None

    def pool_to_beats(
        self,
        beat_times: np.ndarray,
        pool_strategy: Literal["sum", "mean"] = "sum",
    ) -> np.ndarray:
        """Pool activations per beat interval.

        Args:
            beat_times: (n_beats+1,) beat boundaries in seconds [t0, t1, t2, ...]
            pool_strategy: "sum" (default, for BP48) or "mean" (for NNLS)

        Returns:
            (n_beats, n_features) pooled activations

        Raises:
            ValueError: if frame_times is not set
        """
        if self.frame_times is None:
            raise ValueError(
                "Cannot pool activations: frame_times is not set. "
                "Ensure the extractor provides frame timing information."
            )

        pooled = []
        for i in range(len(beat_times) - 1):
            t_start, t_end = beat_times[i], beat_times[i + 1]

            # Find frames in this beat interval [t_start, t_end)
            mask = (self.frame_times >= t_start) & (self.frame_times < t_end)
            feat = self.activations[mask]

            # Handle empty intervals: use nearest frame
            if feat.shape[0] == 0:
                j = int(
                    np.argmin(
                        np.abs(self.frame_times - 0.5 * (t_start + t_end))
                    )
                )
                feat = self.activations[j : j + 1]

            # Aggregate
            if pool_strategy == "sum":
                pooled.append(feat.sum(axis=0))
            elif pool_strategy == "mean":
                pooled.append(feat.mean(axis=0))
            else:
                raise ValueError(
                    f"Unknown pool_strategy: {pool_strategy}. "
                    "Choose 'sum' or 'mean'."
                )

        return np.asarray(pooled, dtype=np.float32)

    # ── Legacy stage1_pitch.PitchActivations conveniences (faithful port) ──
    # These mirror the old dataclass exactly so feature-reroute call-sites that
    # used PitchActivations members keep working with ActivationResult.
    @property
    def n_frames(self) -> int:
        """Number of frames (rows of the activation matrix)."""
        return self.activations.shape[0]

    def chroma(self, weight_by_octave: bool = True) -> np.ndarray:
        """Fold onsets into a (12,) chroma vector (bp48 only).

        Uses onsets, NOT activations — the sustain/note channel is a
        near-constant signal carrying little pitch-class information (see the
        legacy PitchActivations.chroma docstring). Equivalent to the old
        PitchActivations.chroma (onsets is the same array as onset_probs).
        """
        from harmonia.theory.key_profiles import activations_to_chroma
        return activations_to_chroma(self.onsets, weight_by_octave)

    def save(self, path: Path) -> None:
        """Save to compressed .npz, using the legacy PitchActivations key names
        (note_probs / onset_probs) so artifacts are interchangeable."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            note_probs=self.activations,
            onset_probs=self.onsets,
            frame_times=self.frame_times,
            sample_rate=np.array(self.sample_rate),
            duration_s=np.array(self.duration_s),
            source_path=np.array(str(self.source_path)),
        )

    @classmethod
    def load(cls, path: Path) -> "ActivationResult":
        """Load an .npz written by save() (or legacy PitchActivations.save)."""
        data = np.load(Path(path), allow_pickle=True)
        return cls(
            extractor_name="bp48",
            activations=data["note_probs"],
            onsets=data["onset_probs"],
            frame_times=data["frame_times"],
            sample_rate=int(data["sample_rate"]),
            duration_s=float(data["duration_s"]),
            source_path=Path(str(data["source_path"])),
        )


@dataclass
class NNLS24ActivationResult(ActivationResult):
    """NNLS-24 bothchroma with custom pooling (L2-norm per half, C-frame rolling)."""

    def pool_to_beats(
        self,
        beat_times: np.ndarray,
        **kwargs,  # Ignore pool_strategy, etc. — use NNLS-specific logic
    ) -> np.ndarray:
        """Pool bothchroma per beat using NNLS-specific strategy.

        Applies L2-norm per half and C-frame rolling per the training feature spec.

        Args:
            beat_times: (n_beats+1,) beat boundaries in seconds
            **kwargs: ignored (strategy is fixed per NNLS)

        Returns:
            (n_beats, 24) pooled bothchroma, C-frame, L2-per-half

        Raises:
            ImportError: if harmonia.models.nnls_features is not available
        """
        try:
            from harmonia.models.nnls_features import pool_beats as _pool_nnls
        except ImportError as e:
            raise ImportError(
                "NNLS-24 pooling requires harmonia.models.nnls_features"
            ) from e

        return _pool_nnls(self.activations, self.frame_times, beat_times)


@dataclass
class ChordLabelResult(FeatureExtractionResult):
    """Chord labels with optional extracted features (Musx)."""

    chord_labels: list[str]  # Harte notation strings, e.g. ['C:maj', 'G:7', ...]
    times: np.ndarray  # (n_chords, 2) [start_s, end_s] for each chord
    bass_pcs: np.ndarray | None = None  # (n_chords,) or None, pitch classes 0-11
    root_pcs: np.ndarray | None = None  # (n_chords,) or None


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ─────────────────────────────────────────────────────────────────────────────


class FeatureExtractor(ABC):
    """Abstract base for all feature extractors."""

    @abstractmethod
    def extract(self, audio_path: Path | str) -> FeatureExtractionResult:
        """Extract features from audio.

        Args:
            audio_path: Path to audio file (wav/mp3/flac)

        Returns:
            FeatureExtractionResult subclass (ActivationResult or ChordLabelResult)

        Raises:
            FileNotFoundError: if audio file does not exist
            RuntimeError: if required dependencies (VAMP, musx, etc.) are unavailable
        """
        pass

    @classmethod
    def create(cls, name: str, **kwargs) -> FeatureExtractor:
        """Factory: instantiate extractor by name.

        Args:
            name: "bp48" (Basic Pitch), "nnls24" (NNLS-Chroma), or "musx" (music-x-lab)
            **kwargs: passed to the extractor's __init__ (e.g., cache_dir for BP48)

        Returns:
            Configured FeatureExtractor instance

        Raises:
            ValueError: if name is not recognized

        Example:
            extractor = FeatureExtractor.create("bp48", cache_dir=Path("data/cache"))
            result = extractor.extract(Path("song.wav"))
        """
        name_lower = name.lower().strip()

        if name_lower == "bp48":
            return BasicPitch48Extractor(**kwargs)
        elif name_lower == "nnls24":
            return NNLS24Extractor(**kwargs)
        elif name_lower == "musx":
            return MusxExtractor(**kwargs)
        else:
            raise ValueError(
                f"Unknown feature extractor: {name!r}. "
                "Choose from: 'bp48', 'nnls24', 'musx'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Implementations
# ─────────────────────────────────────────────────────────────────────────────


class BasicPitch48Extractor(FeatureExtractor):
    """Wraps PitchExtractor (Basic Pitch via ONNX/SavedModel).

    Outputs: (F, 88) note and onset soft activations.
    Caching: per-file with hash(path + mtime + thresholds).

    This is the current default backend. Preserves all existing caching
    behavior and threshold handling from stage1_pitch.PitchExtractor.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Args:
            cache_dir: Directory to cache .npz activations. If None, no caching.
                       Keyed on hash(audio_path + mtime + threshold parameters).
        """
        self.cache_dir = cache_dir
        self._pitch_extractor = None

    def _ensure_loaded(self) -> None:
        """Lazy-load PitchExtractor on first use."""
        if self._pitch_extractor is None:
            try:
                from harmonia.models.stage1_pitch import PitchExtractor
            except ImportError as e:
                raise ImportError(
                    "Basic Pitch extractor requires harmonia.models.stage1_pitch"
                ) from e
            self._pitch_extractor = PitchExtractor(cache_dir=self.cache_dir)

    def extract(
        self,
        audio_path: Path | str,
        *,
        onset_threshold: float = 0.3,
        frame_threshold: float = 0.3,
        onset_percentile: float | None = None,
        use_cache: bool = True,
    ) -> ActivationResult:
        """Extract note/onset activations from audio via Basic Pitch.

        Args:
            audio_path: Path to audio file
            onset_threshold: Basic Pitch onset detection threshold.
            frame_threshold: Basic Pitch frame threshold.
            onset_percentile: if set, onset threshold is a per-song percentile.
            use_cache: if True and cache_dir is set, reuse cached activations.

        The keyword-only args are forwarded verbatim to the underlying
        stage1_pitch.PitchExtractor.extract; every default reproduces that
        method's own default, so a path-only call `.extract(path)` is
        BYTE-IDENTICAL to the pre-extension behavior.

        Returns:
            ActivationResult with note_probs and onset_probs (F, 88)

        Raises:
            FileNotFoundError: if audio file does not exist
            ImportError: if basic-pitch is not installed
        """
        self._ensure_loaded()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Run extraction (uses cache if cache_dir is set). Kwargs default to
        # PitchExtractor.extract's own defaults → path-only call is unchanged.
        pitch_acts = self._pitch_extractor.extract(
            audio_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            onset_percentile=onset_percentile,
            use_cache=use_cache,
        )

        # Wrap in our result type
        return ActivationResult(
            extractor_name="bp48",
            activations=pitch_acts.note_probs,  # (frames, 88)
            onsets=pitch_acts.onset_probs,  # (frames, 88)
            frame_times=pitch_acts.frame_times,  # (frames,)
            sample_rate=pitch_acts.sample_rate,
            duration_s=pitch_acts.duration_s,
            source_path=pitch_acts.source_path,
        )


class NNLS24Extractor(FeatureExtractor):
    """Wraps NNLS-Chroma VAMP plugin.

    Outputs: (T, 24) bothchroma (bass|treble) per frame.
    Caching: per-file with stem (NOT mtime). Keyed on audio_path.stem.

    Requires the native NNLS-Chroma VAMP plugin and the python vamp module.
    This is an opt-in advanced backend with superior acoustic modeling
    (+17.3pp root, +20.0pp quality vs BP48 on RWC).
    """

    def __init__(self, cache_dir: Path | None = None):
        """Args: cache_dir is ignored (NNLS uses data/cache/nnls_infer/ internally)."""
        pass

    def extract(self, audio_path: Path | str) -> NNLS24ActivationResult:
        """Extract NNLS-24 bothchroma activations (raw, pre-pooling).

        Args:
            audio_path: Path to audio file

        Returns:
            NNLS24ActivationResult with bothchroma (T, 24)

        Raises:
            FileNotFoundError: if audio file does not exist
            ImportError: if VAMP plugin or vamp module is unavailable
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            from harmonia.models.nnls_features import extract_bothchroma
        except ImportError as e:
            raise ImportError(
                "NNLS-24 extractor requires harmonia.models.nnls_features. "
                "Also requires the VAMP plugin and python vamp module."
            ) from e

        # Run extraction (cached on stem, not mtime)
        arr, times = extract_bothchroma(audio_path, use_cache=True)

        return NNLS24ActivationResult(
            extractor_name="nnls24",
            activations=arr,  # (T, 24) bothchroma
            frame_times=times,  # (T,)
            onsets=None,  # NNLS doesn't return onsets
        )


class MusxExtractor(FeatureExtractor):
    """Wraps music-x-lab chord recognition (subprocess-based).

    Outputs: Harte chord labels + extracted sounding bass pitch classes.
    Caching: per-file with stem (NOT mtime). Keyed on audio_path.stem.

    This is an opt-in backend that provides the strongest sounding-bass
    estimates (0.900 all / 0.744 inversions on RWC). Requires the music-x-lab
    clone.

    Unlike BP48 and NNLS24, this returns structured chord labels rather
    than soft activations, requiring different downstream handling.
    """

    def __init__(self, cache_dir: Path | None = None):
        """Args: cache_dir is ignored (musx uses data/cache/musx_infer/ internally)."""
        pass

    def extract(self, audio_path: Path | str) -> ChordLabelResult:
        """Extract chord labels and sounding-bass from music-x-lab.

        Args:
            audio_path: Path to audio file

        Returns:
            ChordLabelResult with chord_labels (Harte) and bass_pcs

        Raises:
            FileNotFoundError: if audio file does not exist
            ImportError: if music-x-lab clone is unavailable
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            from harmonia.models.musx_bass import musx_labels
        except ImportError as e:
            raise ImportError(
                "Musx extractor requires harmonia.models.musx_bass. "
                "Also requires the music-x-lab clone."
            ) from e

        # Fetch chord labels (cached on stem)
        chords_raw, times = musx_labels(audio_path)

        return ChordLabelResult(
            extractor_name="musx",
            chord_labels=chords_raw,  # ['C:maj', 'G:7/3', 'N', ...]
            times=times,  # (n_chords, 2)
        )
