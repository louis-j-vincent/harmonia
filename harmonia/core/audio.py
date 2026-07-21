"""harmonia/core/audio.py — canonical audio loading (Phase 1 PORT).

Single canonical place for audio I/O constants and functions. Loaded audio is
always mono, float32. Duration is computed from sample count / sample rate.
No forced resampling; loads at native sample rate (preserves Phase 0 behavior).
"""

from pathlib import Path
import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32.

    Args:
        path: audio file path (any format soundfile supports)

    Returns:
        (audio_samples, sample_rate) where audio is mono float32 array
    """
    path = Path(path)
    y, sr = sf.read(path)
    # Convert to mono if stereo/multi-channel
    y = (y.mean(axis=1) if y.ndim > 1 else y).astype("float32")
    return y, sr


def get_duration_s(path: Path) -> float:
    """Get audio duration in seconds without loading full audio.

    Args:
        path: audio file path

    Returns:
        duration in seconds (from metadata, fast)
    """
    path = Path(path)
    info = sf.info(path)
    return info.duration


def compute_duration_from_samples(n_samples: int, sr: int) -> float:
    """Compute duration from sample count and sample rate.

    Args:
        n_samples: number of audio samples
        sr: sample rate in Hz

    Returns:
        duration in seconds
    """
    return n_samples / float(sr)
