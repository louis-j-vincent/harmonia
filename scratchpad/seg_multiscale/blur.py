"""(b) The blurred-SSM prior: where do the sections change?

Gaussian-blur the chord-tone SSM, run Foote checkerboard novelty on the blurred
matrix, and read the peaks as CHANGE ZONES — coarse, ±σ, deliberately blind to
anything shorter than the kernel.

Everything is at SLOT resolution (`SLOTS_PER_BAR` slots per bar) and every σ /
kernel is stated in BARS, converted here. Getting that conversion wrong is the
classic silent scale bug (CLAUDE.md #1), so it happens in exactly one place.
"""
from __future__ import annotations

import numpy as np

from harmonia.models.section_vocab import SLOTS_PER_BAR


def gaussian_blur(S: np.ndarray, sigma_bars: float) -> np.ndarray:
    """Separable Gaussian blur of the SSM, σ given in BARS."""
    s = float(sigma_bars) * SLOTS_PER_BAR
    if s <= 0:
        return S.astype(np.float64)
    r = max(1, int(round(3 * s)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= k.sum()
    n = S.shape[0]
    pad = np.pad(S.astype(np.float64), ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(S, dtype=np.float64)
    for i in range(n):
        out[i] = np.convolve(pad[:, i], k, mode="valid")
    pad = np.pad(out, ((0, 0), (r, r)), mode="edge")
    for i in range(n):
        out[i] = np.convolve(pad[i], k, mode="valid")
    return out


def checkerboard(L: int, taper: bool = True) -> np.ndarray:
    """Foote's ``2L x 2L`` checkerboard kernel, Gaussian-tapered."""
    g = np.ones((2 * L, 2 * L))
    g[:L, L:] = -1.0
    g[L:, :L] = -1.0
    if taper:
        x = np.arange(-L, L) + 0.5
        w = np.exp(-0.5 * (x / (L / 2.0)) ** 2)
        g *= np.outer(w, w)
    return g / np.abs(g).sum()


def novelty(S: np.ndarray, kernel_bars: float) -> np.ndarray:
    """Foote novelty along the diagonal, one value per SLOT. Kernel half-width in
    bars. Ends are zero (the kernel does not fit) rather than reflected — a
    reflected end manufactures a boundary at bar 0, which is never information."""
    L = max(2, int(round(float(kernel_bars) * SLOTS_PER_BAR)))
    n = S.shape[0]
    C = checkerboard(L)
    out = np.zeros(n)
    if n < 2 * L:
        return out
    for t in range(L, n - L):
        out[t] = float((S[t - L:t + L, t - L:t + L] * C).sum())
    return out


def peaks(nov: np.ndarray, min_gap_bars: float, frac: float = 0.25) -> list[int]:
    """Local maxima of a novelty curve, in BAR units, ≥ ``frac`` of the max and at
    least ``min_gap_bars`` apart (strongest first, greedy)."""
    gap = max(1, int(round(min_gap_bars * SLOTS_PER_BAR)))
    hi = float(nov.max()) if nov.size else 0.0
    if hi <= 0:
        return []
    cand = [t for t in range(1, len(nov) - 1)
            if nov[t] >= nov[t - 1] and nov[t] >= nov[t + 1] and nov[t] >= frac * hi]
    cand.sort(key=lambda t: -nov[t])
    out: list[int] = []
    for t in cand:
        if all(abs(t - u) >= gap for u in out):
            out.append(t)
    # report on the BAR grid — a boundary is a bar line, never a half-bar
    return sorted({int(round(t / SLOTS_PER_BAR)) for t in out})


def change_zones(S: np.ndarray, sigma_bars: float = 4.0,
                 kernel_bars: float | None = None,
                 min_gap_bars: float | None = None) -> list[int]:
    """Coarse boundary bars from the BLURRED matrix. Kernel and minimum spacing
    default to the blur width — a coarse pass must not resolve finer than it sees."""
    kb = sigma_bars if kernel_bars is None else kernel_bars
    mg = sigma_bars if min_gap_bars is None else min_gap_bars
    return peaks(novelty(gaussian_blur(S, sigma_bars), kb), mg)


# ── the sharp pass: pin the boundary inside the zone ─────────────────────────

def pin(S: np.ndarray, zone_bar: int, n_bars: int, radius_bars: float,
        kernel_bars: float = 2.0) -> int:
    """Sharp-SSM refinement of one coarse zone: the bar within ``±radius`` whose
    SHARP checkerboard novelty is largest. The blur says roughly where; the sharp
    matrix says exactly which bar line."""
    nov = novelty(S, kernel_bars)
    r = max(1, int(round(radius_bars)))
    lo, hi = max(0, zone_bar - r), min(n_bars, zone_bar + r + 1)
    best, bv = zone_bar, -np.inf
    for b in range(lo, hi):
        t = b * SLOTS_PER_BAR
        if 0 <= t < len(nov) and nov[t] > bv:
            best, bv = b, float(nov[t])
    return best


# ── periodicity: the lag the song actually repeats at ────────────────────────

def lag_profile(S: np.ndarray, known: list[bool] | None = None) -> np.ndarray:
    """``p[d] = mean_i S[i, i+d]`` — how well the song matches itself d slots later.
    Slots the decoder could not name are abstained from, as everywhere else."""
    n = S.shape[0]
    out = np.zeros(n)
    for d in range(1, n):
        idx = [i for i in range(n - d)
               if known is None or (known[i] and known[i + d])]
        out[d] = float(np.mean([S[i, i + d] for i in idx])) if len(idx) >= 4 else 0.0
    return out


def dominant_period_bars(S, known=None, lo_bars=2, hi_bars=32, margin=0.02):
    """The SMALLEST whole-bar lag whose self-match is within ``margin`` of the best
    in range — smallest, because a multiple of the true period always scores at
    least as well (the same trap `minimal_period` documents)."""
    p = lag_profile(S, known)
    n_bars = S.shape[0] // SLOTS_PER_BAR
    cands = {}
    for b in range(lo_bars, min(hi_bars, n_bars - 1) + 1):
        d = b * SLOTS_PER_BAR
        if d < len(p):
            cands[b] = p[d]
    if not cands:
        return None, {}
    top = max(cands.values())
    best = next(b for b in sorted(cands) if cands[b] >= top - margin)
    return best, cands
