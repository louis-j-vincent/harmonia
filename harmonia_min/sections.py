"""harmonia_min/sections.py — section boundaries + A/B/C labels from the RAW
NNLS chroma at HALF-BAR granularity. Label strip only: NO folding (milestone
scoped with Louis 2026-07-31 — detection first, repli later).

v2 (Louis's corrections, 2026-07-31, after the This Love diagnostic):
  * v1 built the SSM from DECODED CHORD TONES per BAR. That erased the
    structure: every This Love bar collapsed to the same C-minor-family
    vector, the SSM turned into a uniform fine checkerboard, and the novelty
    fired on turnarounds. "Le SSM a une structure, mais tu la lis mal."
  * v2 reads the structure the way it is actually written: the substrate is
    the raw NNLS bothchroma pooled per HALF-BAR (texture: voicings, bass
    movement, harmonic rhythm — not just which chord family), and the
    blurred checkerboard runs on THAT. The blur at half-bar resolution pools
    the fast alternation into section-scale blocks.
  * Cuts are detected at half-bar resolution, then snapped to bar lines
    (ChartModel sections are bar-ranged).

Letters: average-linkage agglomeration of segment-mean chroma features.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# constants — few, documented, not a config surface. Grain is HALF-BARS.
KERNEL_HB = 16         # checkerboard half-width (half-bars) = 8 bars context
BLUR_SIGMA = 1.5       # gaussian blur (half-bars) applied to the SSM
PEAK_FRAC = 0.5        # keep peaks >= this fraction of the strongest one
MIN_SEG_BARS = 2       # refuse degenerate slivers (NOT an 8-bar prior)
LABEL_COS = 0.96       # cross/self block ratio above this = same letter
                       # (This Love measured: same-type pairs 0.98-1.00,
                       #  verse-vs-chorus 0.91-0.94 — the gap is real)


def halfbar_features(grid: list[float], arr, times) -> np.ndarray:
    """(2*n_bars, 24) mean raw NNLS bothchroma per half-bar, L2-normalised
    per half (bass and treble each unit-norm, so neither half dominates)."""
    edges = []
    for b in range(len(grid) - 1):
        mid = 0.5 * (grid[b] + grid[b + 1])
        edges += [(grid[b], mid), (mid, grid[b + 1])]
    F = np.zeros((len(edges), 24))
    for i, (t0, t1) in enumerate(edges):
        sel = (times >= t0) & (times < t1)
        if not sel.any():
            j = int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))
            v = arr[j]
        else:
            v = arr[sel].mean(0)
        for h in (slice(0, 12), slice(12, 24)):
            n = np.linalg.norm(v[h])
            F[i, h] = v[h] / n if n > 1e-9 else 0.0
    return F / np.sqrt(2.0)          # whole row ~unit norm when both halves live


def _blur(S: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, S)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def _novelty(S: np.ndarray, kw: int) -> np.ndarray:
    """Checkerboard-kernel novelty along the diagonal."""
    n = len(S)
    kernel = np.zeros((2 * kw, 2 * kw))
    kernel[:kw, :kw] = kernel[kw:, kw:] = 1.0
    kernel[:kw, kw:] = kernel[kw:, :kw] = -1.0
    g = np.exp(-0.5 * ((np.arange(2 * kw) - (kw - 0.5)) / (0.6 * kw)) ** 2)
    kernel *= np.outer(g, g)
    nov = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - kw), min(n, i + kw)
        sub = S[a:b, a:b]
        ka, kb = kw - (i - a), kw + (b - i)
        nov[i] = float((sub * kernel[ka:kb, ka:kb]).sum())
    return nov


def detect_sections(grid: list[float], arr, times) -> list[dict]:
    """[{b0, b1, label}] over BAR indices — contiguous, covering, unfolded.

    Detection runs at half-bar grain on the raw-chroma SSM; each accepted cut
    is snapped to its nearest bar line.
    """
    n_bars = len(grid) - 1
    if n_bars < 2 * MIN_SEG_BARS:
        return [{"b0": 0, "b1": n_bars - 1, "label": "A"}]
    F = halfbar_features(grid, arr, times)
    S = F @ F.T
    n = len(S)                                   # = 2 * n_bars

    nov = _novelty(_blur(S, BLUR_SIGMA), KERNEL_HB)
    # Edge half-bars see a truncated, unbalanced kernel — their values are
    # artifacts and inflated the old mean+z·σ threshold past every real peak
    # (This Love: threshold 20.8 over real section peaks at 11–13, found
    # nothing). Mask them, then keep interior local maxima that reach
    # PEAK_FRAC of the strongest — a per-song-adaptive threshold — with a
    # weak absolute floor so a structureless song doesn't get noise cuts.
    kw = KERNEL_HB
    interior = nov[kw:n - kw]
    if len(interior) == 0:
        return [{"b0": 0, "b1": n_bars - 1, "label": "A"}]
    floor = interior.mean() + 0.5 * interior.std()
    cand = [i for i in range(kw, n - kw)
            if nov[i] == max(nov[max(0, i - 3):i + 4])
            and nov[i] >= max(PEAK_FRAC * interior.max(), floor)]

    cuts = []
    for h in cand:
        b = int(round(h / 2.0))                  # snap half-bar cut → bar line
        if 0 < b < n_bars and (not cuts or b - cuts[-1] >= MIN_SEG_BARS):
            cuts.append(b)
    bounds = [0] + cuts + [n_bars]
    segs = [{"b0": a, "b1": b - 1} for a, b in zip(bounds, bounds[1:])]

    # ── letters from the OFF-DIAGONAL repetition blocks ──────────────────
    # Segment-mean chroma cosine failed (every This Love segment is C-minor
    # material → one letter for the whole song). What actually says "these
    # two segments are the same section type" is the cross-block of the
    # blurred SSM — the red off-diagonal blocks in the diagnostic plot. Two
    # segments share a letter when their cross-block similarity comes close
    # to their own internal similarity (a correlation-style ratio).
    Sb = _blur(S, BLUR_SIGMA)
    k = len(segs)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ri = slice(2 * segs[i]["b0"], 2 * (segs[i]["b1"] + 1))
            rj = slice(2 * segs[j]["b0"], 2 * (segs[j]["b1"] + 1))
            M[i, j] = float(Sb[ri, rj].mean())
    groups: list[list[int]] = []
    for i in range(k):
        best, best_r = None, LABEL_COS
        for gi, g in enumerate(groups):
            r = float(np.mean([M[i, j] / np.sqrt(max(M[i, i] * M[j, j], 1e-12))
                               for j in g]))
            if r > best_r:
                best, best_r = gi, r
        if best is None:
            groups.append([i])
        else:
            groups[best].append(i)
    for gi, g in enumerate(groups):
        letter = chr(ord("A") + gi) if gi < 26 else f"S{gi}"
        for i in g:
            segs[i]["label"] = letter
    logger.info("sections v2: %d segments, %d letter groups (half-bar grain)",
                len(segs), len(groups))
    return segs
