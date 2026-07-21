"""Confidence-gated chord-change segmentation (kill-switched brick, default OFF).

WHY THIS EXISTS
---------------
The nnls24 segmentation path cuts a new chord segment at EVERY per-beat root-argmax
flip (``chord_pipeline_v1._root_change_segs``) with no stay-cost — unlike the BP48
path, which has an HMM stay-cost. On POP909 (N=40, cached NNLS features, GT beats)
this over-segments: ~30% of its flip boundaries are spurious, and ~77% of those
spurious flips sit on weak (off-downbeat) beats.

Requiring a minimum posterior MARGIN before accepting a flip suppresses the low-
confidence spurious flips. Adjacent over-segments that would have carried the same
label coalesce away for free, so this is pure denoising. Measured lift over the raw
per-beat-flip baseline (POP909-NNLS, pure NNLS labels, root acc at GT midpoints):

    grid-free gate  (protect_strong=False), margin 0.5:  +3.05pp root / +2.1pp qual7
    downbeat-aware  (protect_strong=True),  margin 0.8:  +2.75pp mid / +4.07pp dense

See docs/research_sessions/chord_beat_grid_unification_2026-07-22.md +
docs/plots/grid_unification_negative_2026-07-22.png.

WHAT THIS IS *NOT* (CLAUDE.md #4 — state the unsolved remainder)
---------------------------------------------------------------
- NOT a fix for the ~28% of GT changes the argmax MISSES (under-segmentation); the
  gate is a precision play and, if anything, lowers recall of real changes.
- NOT "unification onto the downbeat/bar grid." The downbeat grid (``phase``) helps
  ONLY as an asymmetric safety-rail (protect strong-beat flips) worth <1pp beyond the
  grid-free gate, and only under boundary-sensitive scoring. HARD grid-snapping
  (per-bar / half-bar / strong-only pooling) is NEGATIVE (-1.6 to -20.6pp root).
- NOT yet validated on: music-x-lab labels (only pure NNLS labels tested), DETECTED
  (vs GT) beats/downbeats, or real (vs POP909-rendered, functional-root) audio. The
  optimal margin is corpus-dependent and POP909's coarse harmonic rhythm biases it
  HIGH; fast jazz ii-V changes will want a LOWER margin. Validate before wiring ON.

INTEGRATION HOOK (default OFF; do NOT flip without the validation above)
-----------------------------------------------------------------------
In ``chord_pipeline_v1._infer_nnls24`` replace the single line::

    segs = _root_change_segs(beat_proba)

with::

    from harmonia.models.segmentation_gate import gated_root_change_segs
    segs = gated_root_change_segs(beat_proba)   # env HARMONIA_FLIP_MARGIN, off by default

``gated_root_change_segs`` returns EXACTLY ``_root_change_segs(beat_proba)`` when the
env flag is unset/0.0, so wiring it in is a behavioural no-op until enabled.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["confidence_gated_segs", "gated_root_change_segs"]


def confidence_gated_segs(
    beat_proba: np.ndarray,
    margin_thresh: float = 0.0,
    phase: np.ndarray | None = None,
    protect_strong: bool = False,
) -> list[tuple[int, int]]:
    """Per-beat root-argmax segmentation with a posterior-margin flip gate.

    Args:
        beat_proba:     (n_beats, 12) per-beat root posteriors (need not be
                        normalised — normalised internally per beat).
        margin_thresh:  accept a flip at beat b only if, at THAT beat, the new
                        root's normalised posterior exceeds the current root's by
                        >= this margin. 0.0 == accept every flip (== baseline
                        ``_root_change_segs``). Range [0, 1).
        phase:          (n_beats,) beat-within-bar index (0 = downbeat). Only used
                        when ``protect_strong`` is True.
        protect_strong: if True, flips on strong beats (phase in {0, 2}) bypass the
                        margin gate (downbeat-aware safety-rail). Requires ``phase``.

    Returns:
        [(start_beat, end_beat), ...] covering [0, n_beats); [] if empty.

    Reference behaviour is bit-for-bit identical to
    ``chord_pipeline_v1._root_change_segs`` when margin_thresh == 0.0.
    """
    beat_proba = np.asarray(beat_proba)
    pred = beat_proba.argmax(1)
    n = len(pred)
    if n == 0:
        return []
    if margin_thresh <= 0.0:
        cuts = [0] + [b for b in range(1, n) if pred[b] != pred[b - 1]] + [n]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    if protect_strong and phase is None:
        raise ValueError("protect_strong=True requires phase")
    p = beat_proba / np.clip(beat_proba.sum(1, keepdims=True), 1e-9, None)
    cuts = [0]
    cur = int(pred[0])
    for b in range(1, n):
        nb = int(pred[b])
        if nb == cur:
            continue
        if protect_strong and int(phase[b]) in (0, 2):
            cuts.append(b)
            cur = nb
            continue
        if (p[b, nb] - p[b, cur]) >= margin_thresh:
            cuts.append(b)
            cur = nb
        # else: reject the flip; beat b stays in the current segment
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def gated_root_change_segs(
    beat_proba: np.ndarray,
    phase: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Drop-in for ``_root_change_segs`` controlled by env vars (default OFF).

    - ``HARMONIA_FLIP_MARGIN`` (float, default 0.0): flip-margin threshold. 0.0
      reproduces the baseline exactly.
    - ``HARMONIA_FLIP_PROTECT_STRONG`` ("1" to enable): use the downbeat-aware
      safety-rail (needs ``phase``; silently falls back to grid-free if phase is None).
    """
    try:
        margin = float(os.environ.get("HARMONIA_FLIP_MARGIN", "0.0"))
    except ValueError:
        margin = 0.0
    protect = os.environ.get("HARMONIA_FLIP_PROTECT_STRONG", "") == "1" and phase is not None
    return confidence_gated_segs(beat_proba, margin, phase, protect)
