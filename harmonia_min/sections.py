"""harmonia_min/sections.py — section boundaries + A/B/C labels from the
decoded chord chain. Label strip only: NO folding (milestone scoped with
Louis 2026-07-31 — detection first, repli later, under-fold doctrine applies
at folding time, not here).

Method (Louis's rules baked in):
  * The bar feature is CHORD-TONE mass, never root-only — Bb(=Bb,D,F) must
    sit closer to Gm(=G,Bb,D) than to F(=F,A,C). Duration-weighted over the
    bar; a held bar carries the chord sounding through it.
  * Boundaries by blur-then-refine (Louis's approach that beat the fixed
    8-bar minimum): checkerboard novelty on a lightly blurred SSM finds the
    coarse cuts, then each cut is refined ±1 bar on the raw SSM. No fixed
    section-length prior.
  * Labels by average-linkage agglomeration of segment features; letters in
    order of first appearance. An all-silent segment is tagged "NC", not
    lettered.

The detector returns bar indices; pipeline.py turns them into ChartModel
sections (reps=1, spans/barRanges/barSpans sliced from the existing grid).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia_min.labels import chord_pcs

logger = logging.getLogger(__name__)

# blur-then-refine constants — small and few, documented, not a config surface
KERNEL_BARS = 4        # checkerboard half-width (bars); the coarse question is
                       # "do the 4 bars behind me sound like the 4 ahead?"
BLUR_SIGMA = 1.0       # gaussian blur (bars) applied to the SSM before novelty
PEAK_Z = 0.8           # novelty peak threshold, in std devs above the mean
MIN_SEG_BARS = 2       # refuse only degenerate 1-bar slivers (NOT an 8-bar prior)
LABEL_COS = 0.75       # segments closer than this (cosine) share a letter


def bar_features(bars: list[list[dict]]) -> np.ndarray:
    """(n_bars, 12) duration-weighted chord-tone mass per bar, L2-normalised.

    Held bars inherit the previous sounding chord (it IS the bar's harmony).
    N.C. spans contribute nothing — an all-silent bar stays a zero vector.
    """
    n = len(bars)
    F = np.zeros((n, 12))
    prev = None
    for b, bar in enumerate(bars):
        chs = bar or ([prev] if prev is not None else [])
        for c in chs:
            if c is None or c.get("nc"):
                continue
            w = max(0.25, float(c["t1"]) - float(c["t0"]))
            for pc in chord_pcs(c["root"], c["q"]):
                F[b, pc] += w
            if c.get("bass", -1) >= 0:
                F[b, c["bass"] % 12] += 0.5 * w      # sounding bass counts
        if bar:
            last = [c for c in bar if not c.get("nc")]
            prev = last[-1] if last else prev
        norm = np.linalg.norm(F[b])
        if norm > 1e-9:
            F[b] /= norm
    return F


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


def detect_sections(bars: list[list[dict]]) -> list[dict]:
    """[{b0, b1, label}] over bar indices — contiguous, covering, unfolded.

    MEASURED LIMIT (screen run, 2026-07-31, 4 songs): harmony-only novelty
    cuts where the harmony changes — Close to You's modulation lands to the
    bar — and CANNOT cut a song whose progression never changes (Let It Be
    loops C-G-Am-F through every section: one segment). A cheap arrangement
    cue (raw chroma-mass delta, 50/50 blend) was tried and REFUTED: it
    fragmented 3/4 songs and lost Close's modulation cut. Cutting constant-
    harmony forms needs real arrangement features — the section-detection
    branch's territory, not this brick's.
    """
    n = len(bars)
    if n < 2 * MIN_SEG_BARS:
        return [{"b0": 0, "b1": n - 1, "label": "A"}]
    F = bar_features(bars)
    S = F @ F.T                                     # cosine (rows are unit/zero)

    # ── blur then coarse novelty peaks ──
    nov = _novelty(_blur(S, BLUR_SIGMA), KERNEL_BARS)
    thr = nov.mean() + PEAK_Z * nov.std()
    cand = [i for i in range(1, n - 1)
            if nov[i] >= thr and nov[i] == max(nov[max(0, i - 2):i + 3])]

    # ── refine each cut ±1 bar on the RAW ssm novelty ──
    raw_nov = _novelty(S, KERNEL_BARS)
    cuts = []
    for i in cand:
        j = max(1, min(n - 1, i - 1 + int(np.argmax(raw_nov[i - 1:i + 2]))))
        if not cuts or j - cuts[-1] >= MIN_SEG_BARS:
            cuts.append(j)
    bounds = [0] + cuts + [n]
    segs = [{"b0": a, "b1": b - 1} for a, b in zip(bounds, bounds[1:])]

    # ── labels: average-linkage agglomeration on segment-mean features ──
    means = []
    for s in segs:
        m = F[s["b0"]:s["b1"] + 1].mean(0)
        nrm = np.linalg.norm(m)
        means.append(m / nrm if nrm > 1e-9 else m)
    groups: list[list[int]] = []
    for i, m in enumerate(means):
        if np.linalg.norm(means[i]) < 1e-9:          # all-silent segment
            segs[i]["label"] = "NC"
            continue
        best, best_cos = None, LABEL_COS
        for gi, g in enumerate(groups):
            cos = float(np.mean([means[j] @ m for j in g]))
            if cos > best_cos:
                best, best_cos = gi, cos
        if best is None:
            groups.append([i])
        else:
            groups[best].append(i)
    for gi, g in enumerate(groups):
        letter = chr(ord("A") + gi) if gi < 26 else f"S{gi}"
        for i in g:
            segs[i]["label"] = letter
    logger.info("sections: %d segments, %d letter groups",
                len(segs), len(groups))
    return segs
