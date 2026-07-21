"""harmonia/core/audio.py — canonical audio loading and resampling.

Phase 0 stub: imports from old code, will be ported in Phase 1.
"""

from pathlib import Path
import numpy as np


def load_audio(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load and resample audio to canonical sample rate.

    Args:
        path: audio file path
        sr: target sample rate (default 22050 Hz)

    Returns:
        (audio_samples, sample_rate)
    """
    # Phase 1 PORT: extract from current pipeline
    # For now, import and re-export from old location
    raise NotImplementedError("Phase 1: port audio loader")


def get_duration_s(path: Path) -> float:
    """Get audio duration in seconds.

    Args:
        path: audio file path

    Returns:
        duration in seconds
    """
    # Phase 1 PORT
    raise NotImplementedError("Phase 1: port duration calculator")
