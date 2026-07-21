"""harmonia/stages/beat_grid.py — beat tracking and phase estimation.

Phase 0 stub: will port from existing beat_grid.py in Phase 2.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BeatGrid:
    """Beat tracking output."""
    beat_times: np.ndarray  # beat onset times in seconds
    beat_frames: np.ndarray  # beat positions in frames
    tempo_bpm: float  # detected tempo
    downbeat_times: np.ndarray | None = None  # downbeat positions
    downbeat_frames: np.ndarray | None = None


def estimate_beat_grid(
    audio: np.ndarray,
    sr: int,
    method: str = "beatthis",
) -> BeatGrid:
    """Estimate beat grid from audio.

    Args:
        audio: audio samples
        sr: sample rate
        method: beat tracking method ("beatthis" = default)

    Returns:
        BeatGrid with beat times and tempo
    """
    # Phase 2 REDESIGN: will implement proper beat tracking
    raise NotImplementedError("Phase 2: port beat_grid")
