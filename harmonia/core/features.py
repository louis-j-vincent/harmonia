"""harmonia/core/features.py — canonical feature extraction entry point.

Phase 0 stub: defines the interface, will port implementations in Phase 1.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np


class FeatureExtractor(ABC):
    """Base class for feature extractors (chroma + bass tracking)."""

    @abstractmethod
    def extract_chroma(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Extract chroma features.

        Returns:
            (chroma: (n_frames, 12), sample_rate: int)
        """
        pass

    @abstractmethod
    def extract_bass(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Extract bass tracking.

        Returns:
            (bass_salience: (n_frames, 12), sample_rate: int)
        """
        pass


class NNLS24Extractor(FeatureExtractor):
    """NNLS-Chroma 24-dimensional feature extractor (Phase 1 PORT)."""

    def extract_chroma(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port NNLS-24")

    def extract_bass(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port NNLS-24 bass")


class BasicPitch48Extractor(FeatureExtractor):
    """Basic Pitch 48-dimensional feature extractor (alternative, Phase 1 PORT)."""

    def extract_chroma(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port BasicPitch-48")

    def extract_bass(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port BasicPitch-48 bass")


class MusxExtractor(FeatureExtractor):
    """musx-native feature extractor (Phase 1 PORT)."""

    def extract_chroma(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port musx frontend")

    def extract_bass(self, audio_path: Path) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Phase 1: port musx bass")


def create_extractor(
    frontend: Literal["nnls24", "bp48", "musx"] = "nnls24",
) -> FeatureExtractor:
    """Factory for feature extractors.

    Args:
        frontend: which frontend to use

    Returns:
        FeatureExtractor instance
    """
    if frontend == "nnls24":
        return NNLS24Extractor()
    elif frontend == "bp48":
        return BasicPitch48Extractor()
    elif frontend == "musx":
        return MusxExtractor()
    else:
        raise ValueError(f"Unknown frontend: {frontend}")
