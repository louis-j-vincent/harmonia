"""harmonia/stages/chords.py — chord root/quality/bass inference.

Phase 0 stub: will port root/quality/bass heads from existing code in Phase 3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ChordPrediction:
    """Single chord prediction."""
    root: int  # pitch class 0-11
    quality: str  # "maj", "min", "dom", "dim", "aug", "sus", "hdim"
    bass: int | None = None  # sounding bass pitch class, if available
    confidence: float = 1.0


class ChordHead(ABC):
    """Base class for chord quality/root predictors."""

    @abstractmethod
    def predict_root(self, chroma: np.ndarray) -> int:
        """Predict chord root from chroma.

        Args:
            chroma: 12-dimensional chroma vector or longer chord features

        Returns:
            root pitch class (0-11)
        """
        pass

    @abstractmethod
    def predict_quality(self, chroma: np.ndarray, root: int) -> str:
        """Predict chord quality given root.

        Args:
            chroma: features
            root: predicted root pitch class

        Returns:
            quality string ("maj", "min", "dom", etc.)
        """
        pass

    @abstractmethod
    def predict_bass(self, chroma: np.ndarray, root: int) -> int | None:
        """Predict sounding bass note.

        Args:
            chroma: features
            root: predicted root pitch class

        Returns:
            bass pitch class (0-11) or None if not available
        """
        pass


class NNLS24ChordHead(ChordHead):
    """NNLS-24 based chord predictor (Phase 3 PORT)."""

    def predict_root(self, chroma: np.ndarray) -> int:
        raise NotImplementedError("Phase 3: port NNLS-24 root head")

    def predict_quality(self, chroma: np.ndarray, root: int) -> str:
        raise NotImplementedError("Phase 3: port NNLS-24 quality head")

    def predict_bass(self, chroma: np.ndarray, root: int) -> int | None:
        raise NotImplementedError("Phase 3: port NNLS-24 bass head")


class MusxChordHead(ChordHead):
    """musx-based chord predictor (Phase 3 PORT)."""

    def predict_root(self, chroma: np.ndarray) -> int:
        raise NotImplementedError("Phase 3: port musx root head")

    def predict_quality(self, chroma: np.ndarray, root: int) -> str:
        raise NotImplementedError("Phase 3: port musx quality head")

    def predict_bass(self, chroma: np.ndarray, root: int) -> int | None:
        raise NotImplementedError("Phase 3: port musx bass head")
