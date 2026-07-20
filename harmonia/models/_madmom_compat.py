"""madmom Py3.12 compatibility shim + DeepChroma novelty helper.

madmom (a pinned project dependency) does not import on Python 3.12: it uses
``from collections import MutableSequence`` (moved to ``collections.abc`` in 3.10)
and numpy aliases removed in NumPy 1.24 (``np.int``/``np.float``). This module
installs the minimal aliases BEFORE importing madmom, isolated here so the rest of
the codebase never touches the broken import path.

Provides ``deepchroma_novelty(audio_path)`` — the Korzeniowski & Widmer 2016 deep
chroma extractor's frame chroma projected to Harte & Sandler 2006 tonal-centroid
space, differenced to a 1-D harmonic-change novelty. Measured (change-timing session
2026-07-20) to give ~3.6× the change-detection F1@150ms of the raw NNLS-chroma L2
flux against music-x-lab change times on the matched set. Result cached per audio
stem+mtime so a re-analyse does not re-run the NN.

Kept OFF by default in the pipeline (kill-switch ``HARMONIA_FLUX_NOVELTY``): it adds
a NN forward pass per analyse and depends on this shim, so it is opt-in pending a
proper madmom-py312 fix. See docs/research_sessions/chord_change_timing_2026-07-20.md.
"""
from __future__ import annotations

import collections
import collections.abc
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SHIM_DONE = False


def _install_shim() -> None:
    global _SHIM_DONE
    if _SHIM_DONE:
        return
    for _n in ("MutableSequence", "MutableMapping", "Sequence", "Mapping",
               "Callable", "Iterable"):
        if not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    for _a, _t in (("float", float), ("int", int), ("bool", bool),
                   ("object", object), ("complex", complex), ("str", str)):
        if not hasattr(np, _a):
            setattr(np, _a, _t)
    _SHIM_DONE = True


# Harte & Sandler 2006 tonal-centroid projection (6, 12), C-first ordering. TCS
# distance is invariant to a global pitch-class rotation, so the exact chroma
# ordering does not matter for a frame-to-frame CHANGE novelty.
_r12 = np.arange(12, dtype=np.float32)
_TCS12 = np.stack([
    np.sin(_r12 * 7 * np.pi / 6), np.cos(_r12 * 7 * np.pi / 6),
    np.sin(_r12 * 3 * np.pi / 2), np.cos(_r12 * 3 * np.pi / 2),
    np.sin(_r12 * 2 * np.pi / 3), np.cos(_r12 * 2 * np.pi / 3),
], axis=0).astype(np.float32)
del _r12

_DCP = None
_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def _get_processor():
    global _DCP
    if _DCP is None:
        _install_shim()
        from madmom.audio.chroma import DeepChromaProcessor  # noqa: E402
        _DCP = DeepChromaProcessor()
    return _DCP


def deepchroma_novelty(audio_path: Path, *, tcs: bool = True,
                       fps: float = 10.0) -> "tuple[np.ndarray, float]":
    """(1-D harmonic-change novelty d, fps) from madmom DeepChroma.

    ``tcs``: project to tonal-centroid space before differencing (matches the raw
    L2 flux when False; on clean deep chroma the two are ~tied, the win is the
    chroma quality). Cached by audio stem+mtime.
    """
    key = f"{audio_path.stem}:{int(audio_path.stat().st_mtime)}:{int(tcs)}"
    if key in _CACHE:
        return _CACHE[key]
    dc = np.asarray(_get_processor()(str(audio_path)))       # (frames, 12) @ ~10 fps
    if tcs:
        s = dc.sum(1, keepdims=True)
        p = dc / np.where(s > 1e-9, s, 1.0)
        feat = p @ _TCS12.T
    else:
        feat = dc
    d = np.sqrt((np.diff(feat, axis=0) ** 2).sum(1))
    d = np.concatenate([[0.0], d]).astype(np.float64)
    _CACHE[key] = (d, fps)
    return d, fps
