"""Chord template scoring utilities.

Provides dot-product-based log-likelihood of a chroma vector under a chord
hypothesis (root + family), using normalised templates.
"""
from __future__ import annotations

import numpy as np


def chord_log_likelihood(
    chroma_12: np.ndarray,
    root_pc: int,
    fam_idx: int,
    templates: list,
) -> float:
    """Score a chord hypothesis against a 12-d chroma vector.

    Returns the dot product of the L2-normalised chroma against the matching
    chord template (higher = better fit, range [0, 1] for non-negative chroma).

    Parameters
    ----------
    chroma_12 : array of shape (12,)
        Chroma observation (will be L2-normalised internally).
    root_pc : int 0-11
        Root pitch class.
    fam_idx : int 0-4
        Family index (major/minor/dim/aug/sus).
    templates : list of (root_pc, template_vector) tuples
        The 60 chord templates (12 roots x 5 families), ordered as
        [(r0,t0), (r0,t1), ..., (r0,t4), (r1,t0), ...].

    Returns
    -------
    float : dot-product score in [0, 1].
    """
    c = np.asarray(chroma_12, dtype=np.float64)
    n = np.linalg.norm(c)
    if n < 1e-9:
        return 0.0
    c = c / n

    # Template index: 5 families per root
    idx = root_pc * 5 + fam_idx
    _, tmpl = templates[idx]
    return float(c @ tmpl)


def best_hypothesis(
    chroma_12: np.ndarray,
    templates: list,
) -> tuple[int, int, float]:
    """Find the globally best-fitting chord template for a chroma vector.

    Parameters
    ----------
    chroma_12 : array of shape (12,)
    templates : list of (root_pc, template_vector) tuples (length 60)

    Returns
    -------
    (root_pc, fam_idx, score) for the best match.
    """
    c = np.asarray(chroma_12, dtype=np.float64)
    n = np.linalg.norm(c)
    if n < 1e-9:
        return (0, 0, 0.0)
    c = c / n

    best_score = -1.0
    best_root = 0
    best_fam = 0
    for i, (r, t) in enumerate(templates):
        s = float(c @ t)
        if s > best_score:
            best_score = s
            best_root = r
            best_fam = i % 5
    return (best_root, best_fam, best_score)
