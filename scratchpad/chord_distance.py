"""chord_distance.py — chord-tone-based similarity, 4 variants, per user spec
(2026-07-18): a bar (or block) is a SUM of per-beat chord vectors; bar-to-bar
similarity is the dot product of these sums, which by bilinearity decomposes
into the sum of pairwise chord-to-chord dot products — exactly the worked
example given (F-major bar vs [Dmin,Dmin,Bbmin,Bbmin] bar).

The atomic building block is chord_vector(root_pc, qual) -> 12-dim (or complex
TIV) vector; everything else (bar vectors, block vectors, similarity, cluster)
falls out of that by linearity + cosine similarity.

V1 BINARY   — root/3rd/5th(/7th) membership, weight 1 each. Literally "shared
              notes". Reuses this project's qbucket() interval families.
V2 WEIGHTED — same membership, root/fifth weighted above the third, per
              Lerdahl's Tonal Pitch Space intuition (root is structurally
              more central than the third). Weights are a documented choice,
              not tuned/validated — a hyperparameter to sweep, not a citation.
V3 TIV      — DFT of the (weighted) 12-dim chroma, consonance-weighted per
              coefficient, INSPIRED BY Bernardes/Chew/Amiot's Tonal Interval
              Space. NOTE: the coefficient weights below are a simplified,
              roughly-consonance-ordered approximation, not a verified
              reproduction of the original paper's exact published weights —
              flagged explicitly so this isn't mistaken for a faithful TIV
              implementation.
V4 CHROMA   — real NNLS chroma dot product (no idealized triad templates at
              all) — for REAL AUDIO only; on clean iReal symbolic data this
              necessarily reduces to V1 (no real chroma exists for iReal).
"""
from __future__ import annotations
import numpy as np

# quality-bucket -> chord-tone intervals (matches symstruct.py's qbucket() families)
QBUCKET_INTERVALS = {
    0: [0, 4, 7],       # major
    1: [0, 3, 7],       # minor
    2: [0, 4, 7, 10],   # dominant/7th-family (treat generically as dom7)
    3: [0, 3, 6],       # diminished
    4: [0, 4, 8],       # augmented
    5: [0, 5, 7],       # sus4
}

# V2: role weight by position within QBUCKET_INTERVALS (root, 3rd, 5th, 7th)
ROLE_WEIGHTS = [3.0, 1.0, 2.0, 1.2]

# V3: simplified consonance-ish weights per DFT coefficient index 1..6
# (index 5 ~ circle-of-fifths content, index 3/4 ~ major/minor-third content —
# this ordering is the qualitative part that's well established; the exact
# numeric weights here are a reasonable, NOT paper-verified, approximation)
TIV_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.8, 1.1, 0.3])


def chord_vector_binary(root_pc: int, qual: int) -> np.ndarray:
    v = np.zeros(12)
    if root_pc is None or root_pc < 0 or qual not in QBUCKET_INTERVALS:
        return v
    for iv in QBUCKET_INTERVALS[qual]:
        v[(root_pc + iv) % 12] = 1.0
    return v


def chord_vector_weighted(root_pc: int, qual: int) -> np.ndarray:
    v = np.zeros(12)
    if root_pc is None or root_pc < 0 or qual not in QBUCKET_INTERVALS:
        return v
    ivs = QBUCKET_INTERVALS[qual]
    for k, iv in enumerate(ivs):
        w = ROLE_WEIGHTS[k] if k < len(ROLE_WEIGHTS) else 1.0
        v[(root_pc + iv) % 12] = w
    return v


def chroma_to_tiv(chroma12: np.ndarray) -> np.ndarray:
    """12-dim (real, nonneg) chroma -> 6 complex TIV coefficients (indices 1..6
    of the DFT, consonance-weighted). Similarity = real part of the inner
    product of two such vectors (standard TIV cosine-style comparison)."""
    F = np.fft.fft(chroma12)
    coeffs = F[1:7] * TIV_WEIGHTS
    return coeffs


def chord_vector_tiv(root_pc: int, qual: int) -> np.ndarray:
    return chroma_to_tiv(chord_vector_binary(root_pc, qual))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return float(np.real(np.vdot(a, b)) / (na * nb))
    return float(np.dot(a, b) / (na * nb))
