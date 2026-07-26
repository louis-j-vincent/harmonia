"""harmonia/core/chroma.py — canonical librosa CQT-chroma brick (Phase 7).

ONE source of truth for the inline `librosa.feature.chroma_cqt(...)` +
`librosa.frames_to_time(...)` (+ LTAS normalisation) block that was copy-pasted
verbatim across ~20 scripts. Faithful to that behavior by construction: same
librosa calls, same defaults, same 1e-9 LTAS floor.

Scope / relationship to `harmonia.core.features`:
    `features.FeatureExtractor` is the *path -> cached model activations*
    abstraction (BP48 / NNLS24 / musx: audio file in, ActivationResult out).
    This module is the complementary *in-memory signal -> chroma* brick: it
    takes an already-loaded `(y, sr)` waveform (typically a freshly rendered
    temp wav) and returns plain arrays. Different responsibility, so it is its
    own small module rather than another FeatureExtractor backend.

Public API:
    raw, times = chroma_cqt(y, sr, hop_length=512)       # (12, T) + (T,)
    norm       = ltas_normalize(raw)                     # (12, T), row mean ~1
    chroma, t  = chroma_cqt_ltas(y, sr, hop_length=512)  # the common composite

librosa defaults pinned here (verified against librosa 0.11.0): chroma_cqt
`bins_per_octave=36`, `hop_length=512`; frames_to_time `hop_length=512`. The
scripts that passed `bins_per_octave=36` explicitly and those that passed
nothing were therefore always computing the SAME geometry — this brick keeps
that single default rather than perpetuating two spellings of it.
"""

from __future__ import annotations

import numpy as np


# LTAS floor: pitch classes whose long-term average is essentially zero are
# left unscaled (divide-by-~0 guard). Value is load-bearing — it is the exact
# constant every inline copy used.
_LTAS_FLOOR = 1e-9


def chroma_cqt(
    y: np.ndarray,
    sr: int | float,
    *,
    hop_length: int = 512,
    bins_per_octave: int = 36,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw CQT chroma plus the frame-time axis.

    Args:
        y: mono waveform.
        sr: sample rate.
        hop_length: CQT hop, in samples. Also used for the time axis, so the
            two can never drift apart (they were separate literals inline).
        bins_per_octave: CQT resolution (36 = 3 bins/semitone, librosa's own
            default for chroma_cqt).
        transpose: if True return (T, 12) instead of (12, T). Several
            call-sites wanted frame-major; this flag replaces their `.T`.

    Returns:
        (chroma, times): chroma is (12, T) — or (T, 12) if `transpose` —
        and times is (T,) frame-centre times in seconds.
    """
    import librosa

    raw = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave
    )
    times = librosa.frames_to_time(
        np.arange(raw.shape[1]), sr=sr, hop_length=hop_length
    )
    return (raw.T if transpose else raw), times


def ltas_normalize(raw: np.ndarray) -> np.ndarray:
    """Divide each pitch-class row by its long-term average (LTAS).

    Each of the 12 rows ends with mean ~1, which preserves *local* dynamics
    while removing the per-song spectral tilt that otherwise makes some pitch
    classes systematically louder. Rows whose average is below `_LTAS_FLOOR`
    are left unscaled instead of exploding.

    Args:
        raw: (12, T) chroma, as returned by `chroma_cqt`.

    Returns:
        (12, T) LTAS-normalised chroma.
    """
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < _LTAS_FLOOR, 1.0, ltas)
    return raw / ltas


def chroma_cqt_ltas(
    y: np.ndarray,
    sr: int | float,
    *,
    hop_length: int = 512,
    bins_per_octave: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """`chroma_cqt` followed by `ltas_normalize` — the common composite.

    This is the exact block the ~16 LTAS call-sites had inline. Use
    `chroma_cqt` + `ltas_normalize` separately only when the raw chroma is
    also needed (e.g. to additionally L2-normalise it for a plot).

    Returns:
        (chroma, times): (12, T) LTAS-normalised chroma and (T,) frame times.
    """
    raw, times = chroma_cqt(
        y, sr, hop_length=hop_length, bins_per_octave=bins_per_octave
    )
    return ltas_normalize(raw), times
