"""Louis's idea (2026-08-05), implemented and measured:

  « créer une nouvelle matrice SSM qui encode la similarité de ces BI-BARRES
    entre elles, mais avec le PRODUIT SCALAIRE BINAIRE défini entre les
    bi-barres → une matrice de uns et de zéros, et avec ça on peut segmenter
    en sous-sections en lisant les régions où il y a des SÉRIES DE 1 dans les
    SOUS-DIAGONALES. »

Four decisions, each an explicit knob below:

  (a) UNIT — the bi-bar (2 bars). Taken with **stride 1** (bi-bar i covers bars
      i and i+1), not stride 2. Stride 2 would halve the resolution and could
      not answer "which BAR does the section start on", which is the metric.
  (b) BINARISATION — several, swept: top-k chroma bins per half; per-bin
      per-song quantile; chord-tone membership; chord-symbol identity.
  (c) THRESHOLD — the binary dot product is normalised by the (constant) number
      of ones, so `tau` is a fraction of shared active bins.
  (d) READING THE SUB-DIAGONALS — M[i, i-l] as a function of i is the lag-l
      sub-diagonal. A run of 1s there = "this passage is repeating what
      happened l bars ago". A section START is the LEFT EDGE of such a run.

`run_edge_score` is the whole method in six lines. Everything else is scoring.

    python scripts/bibar_binary_ssm.py            # the full sweep
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── (b) BINARISATION ────────────────────────────────────────────────────────
def binarise(Cb: np.ndarray, mode: str, kb: int = 2, kt: int = 4) -> np.ndarray:
    """(n_bars, 24) chroma -> (n_bars, 24) bool. Bass and treble halves are
    binarised SEPARATELY so the (much louder) treble cannot decide the bass."""
    nb = len(Cb)
    B = np.zeros((nb, 24), bool)
    for h, k in ((slice(0, 12), kb), (slice(12, 24), kt)):
        V = Cb[:, h]
        if mode == "topk":
            # k strongest bins of THIS bar — a fixed number of ones per bar,
            # so the dot product has a constant maximum and `tau` is a clean
            # fraction. Invariant to how LOUD the bar is.
            idx = np.argsort(-V, axis=1)[:, :k]
            for i in range(nb):
                B[i, h][idx[i]] = True
        elif mode == "quantile":
            # a bin is ON when it is loud FOR THIS SONG (per-bin 70th pct).
            # Variable number of ones per bar -> normalise by the geometric
            # mean of the two counts at compare time.
            thr = np.quantile(V, 0.70, axis=0)
            B[:, h] = V > thr[None, :]
        elif mode == "relmax":
            # a bin is ON when it reaches 50% of the loudest bin of that bar.
            B[:, h] = V >= 0.5 * V.max(axis=1, keepdims=True)
        else:
            raise ValueError(mode)
    return B


PC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
_FLAT = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4,
         "E#": 5, "B#": 0}
_TONES = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def chordtone_binary(sym: np.ndarray) -> np.ndarray:
    """(n_bars,) majmin symbols -> (n_bars, 24) bool = [root one-hot | tones]."""
    B = np.zeros((len(sym), 24), bool)
    for i, s in enumerate(sym):
        if not s or s == "N" or ":" not in s:
            continue
        r, q = s.split(":", 1)
        p = _FLAT.get(r, PC.get(r))
        if p is None:
            continue
        B[i, p] = True                                    # bass/root half
        for iv in _TONES.get(q, (0, 4, 7)):
            B[i, 12 + (p + iv) % 12] = True               # treble half
    return B


def bibar(B: np.ndarray, join: str = "concat") -> np.ndarray:
    """(n_bars, D) bool -> (n_bars-1, 2D or D) bool, stride 1.

    `concat` keeps the ORDER inside the bi-bar ("Bb then Eb" != "Eb then Bb");
    `union` is order-blind and is the cheaper reading of Louis's phrase.
    """
    if join == "concat":
        return np.concatenate([B[:-1], B[1:]], axis=1)
    return B[:-1] | B[1:]


# ── (c) THE BINARY SSM ──────────────────────────────────────────────────────
def bin_ssm(X: np.ndarray, tau: float) -> np.ndarray:
    """Binary dot product, normalised by the geometric mean of the row sums
    (= cosine on 0/1 vectors), thresholded -> a matrix of ones and zeros."""
    Xf = X.astype(np.float32)
    D = Xf @ Xf.T
    n = np.sqrt(np.maximum(np.diag(D), 1e-9))
    return (D / np.outer(n, n)) >= tau


def _rot(X: np.ndarray, r: int) -> np.ndarray:
    """Transpose the whole feature UP by r semitones: every 12-wide block
    (bass, treble, and each bar of a `concat`) is rolled by the same r."""
    out = X.copy()
    for c in range(0, X.shape[1], 12):
        out[:, c:c + 12] = np.roll(X[:, c:c + 12], r, axis=1)
    return out


def bin_ssm_rot(X: np.ndarray, tau: float) -> np.ndarray:
    """Transposition-INVARIANT binary SSM: max over the 12 semitone rotations.

    Motivation (measured on Bobby Hebb's *Sunny*, 2026-08-05): raw chroma is
    key-dependent, so a section that returns a semitone higher scores 0.797 and
    misses `LABEL_COS`; rotated it scores 0.973. Every modulating song splits
    its own repeats into fresh letters without this.
    WARNING, and this is why it is measured and not assumed: taking a max over
    12 rotations gives every pair twelve chances to clear the threshold, so it
    must inflate false merges. See `letter_merge_prf`.
    """
    Xf = X.astype(np.float32)
    n = np.sqrt(np.maximum((Xf * Xf).sum(1), 1e-9))
    best = np.full((len(Xf), len(Xf)), -1.0, np.float32)
    for r in range(12):
        np.maximum(best, (Xf @ _rot(Xf, r).T) / np.outer(n, n), out=best)
    return best >= tau


def cont_ssm(Cb: np.ndarray, join: str, tau: float) -> np.ndarray:
    """CONTROL: the identical pipeline on the CONTINUOUS chroma. Isolating the
    effect of binarising is the whole point — everything else is held fixed."""
    V = Cb / np.maximum(np.linalg.norm(Cb, axis=1, keepdims=True), 1e-9)
    X = np.concatenate([V[:-1], V[1:]], 1) if join == "concat" else V[:-1] + V[1:]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return (X @ X.T) >= tau


# ── (d) READING THE SUB-DIAGONALS ───────────────────────────────────────────
def lag_fwd_bwd(M: np.ndarray, min_lag: int = 4):
    """fwd[i, l] / bwd[i, l] on every sub-diagonal, for BOTH signs of the lag.

    Along sub-diagonal l we look at M[i, i-l] as a function of i. `fwd` = how
    far the run of 1s extends forward from i; `bwd` = how far it extends
    backward from i-1. A repeat is symmetric in time, so its left edge exists
    at bar i (the later occurrence, lag +l) AND at bar i-l (the earlier one,
    lag -l); both are folded in, which is what lets one repeat vote twice.
    """
    n = len(M)
    F = np.zeros((n, 2 * n), np.int32)
    B = np.zeros((n, 2 * n), np.int32)
    for l in range(min_lag, n):
        d = M[np.arange(l, n), np.arange(0, n - l)]
        m = len(d)
        f = np.zeros(m, np.int32)
        for i in range(m - 1, -1, -1):
            f[i] = (f[i + 1] + 1) if (d[i] and i + 1 < m) else int(d[i])
        b = np.zeros(m, np.int32)
        for i in range(1, m):
            b[i] = (b[i - 1] + 1) if d[i - 1] else 0
        F[l:, l], B[l:, l] = f, b            # later occurrence  (bar i)
        F[:m, n + l], B[:m, n + l] = f, b    # earlier occurrence (bar i-l)
    return F, B


def run_edge_score(M: np.ndarray, reader: str = "edgesum", L: int = 8,
                   min_lag: int = 4) -> np.ndarray:
    """score[i] = "a repetition STARTS at bi-bar i". Three readings of the
    same sub-diagonal runs; `edgesum` is the one that survived measurement.

      diff    max_l  min(fwd,L) - min(bwd,L)          — the obvious one.
                     Integer and capped, so within a +-4 bar window several
                     positions tie at L and the metric decides them by coin
                     flip. Measured: this alone costs ~10 pp.
      edgelen max_l  fwd  if bwd == 0 else 0          — strict left edges,
                     scored by HOW LONG the repeat is.
      edgesum sum_l  min(fwd,L) if bwd == 0 else 0    — every lag that starts a
                     run here adds its vote. A chorus heard four times produces
                     edges at three different lags at once, and they agree only
                     at the true start. This is RefraiD's "cross-check between
                     different repeat-pairs", read on a binary matrix.
    """
    F, B = lag_fwd_bwd(M, min_lag)
    if reader == "diff":
        return (np.minimum(F, L) - np.minimum(B, L)).max(1).astype(float)
    edge = (B == 0) & (F > 0)
    if reader == "edgelen":
        return np.where(edge, F, 0).max(1).astype(float)
    return np.where(edge, np.minimum(F, L), 0).sum(1).astype(float)


def lag_runs(M: np.ndarray, min_run: int = 4, min_lag: int = 4) -> list[tuple]:
    """All maximal runs of 1s on the sub-diagonals: (start_bar, length, lag)."""
    n = len(M)
    out = []
    for l in range(min_lag, n):
        d = M[np.arange(l, n), np.arange(0, n - l)]
        i = 0
        while i < len(d):
            if d[i]:
                j = i
                while j + 1 < len(d) and d[j + 1]:
                    j += 1
                if j - i + 1 >= min_run:
                    out.append((i + l, j - i + 1, l))
                i = j + 1
            else:
                i += 1
    return out


# ── the shipped cue, reimplemented here so the comparison is same-code ──────
def novelty_score(Hb: np.ndarray, kw: int = 16, sigma: float = 1.5) -> np.ndarray:
    """`harmonia_min/sections.py`'s checkerboard novelty on its OWN grain —
    REAL half-bar pooled chroma, L2-normalised per half exactly as
    `sections.halfbar_features` does. This is the 34.7% reference cue."""
    from harmonia_min.sections import _blur, _novelty
    F = Hb.copy()
    for h in (slice(0, 12), slice(12, 24)):
        nrm = np.linalg.norm(F[:, h], axis=1, keepdims=True)
        F[:, h] = F[:, h] / np.maximum(nrm, 1e-9)
    F /= np.sqrt(2.0)
    nov = _novelty(_blur(F @ F.T, sigma), kw)
    return nov[::2]                                # -> per BAR (bar downbeat)


def chordstring_score(sym: np.ndarray, W: int = 8) -> np.ndarray:
    """The lit review's 75.2% cue: does the W-bar chord string starting here
    occur again elsewhere in the song?"""
    n = len(sym)
    sc = np.full(n, -9.0)
    for c in range(n - W):
        A = sym[c:c + W]
        best = 0.0
        for o in range(0, n - W):
            if abs(o - c) < 4:
                continue
            m = float((A == sym[o:o + W]).mean())
            if m > best:
                best = m
        sc[c] = best
    return sc


# ── scoring ─────────────────────────────────────────────────────────────────
_RNG = np.random.default_rng(0)


def placement(score: np.ndarray, starts, nb, win=4):
    """The lit review's metric: for each GT start, argmax the cue over a
    +-`win` bar window. Returns the signed bar error of each start.

    TIES ARE BROKEN AT RANDOM, deliberately. A distance tie-break (`prefer the
    smaller |d|`) silently hands 100% to any cue whose score is CONSTANT in the
    window — and a binary SSM saturated to all-ones is exactly that. That is
    error-pattern #1 (a calibration artefact producing a plausible number);
    caught here on the first run, when topk/union/tau0.5 scored 85%.
    """
    errs = []
    for sb in starts:
        if sb < 8 or sb > nb - 9:
            continue
        v = np.array([score[c] if 0 <= c < len(score) else -9.0
                      for c in range(sb - win, sb + win + 1)], float)
        top = np.flatnonzero(v >= v.max() - 1e-9)
        errs.append(int(_RNG.choice(top)) - win)
    return errs


def density(M: np.ndarray) -> float:
    """Fraction of ones off the main band. A saturated matrix carries no
    information and wins the placement metric only through tie-breaking —
    always report this next to an accuracy."""
    n = len(M)
    off = ~np.eye(n, dtype=bool)
    for k in range(1, 4):
        off &= ~np.eye(n, k=k, dtype=bool) & ~np.eye(n, k=-k, dtype=bool)
    return float(M[off].mean())


def boundary_prf(pred, gt, tol=0):
    """Bar-tolerance boundary P/R/F (mir_eval `detection` shape, in BARS)."""
    pred, gt = sorted(set(pred)), sorted(set(gt))
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    used = set()
    hit = 0
    for p in pred:
        cands = [i for i, g in enumerate(gt)
                 if abs(g - p) <= tol and i not in used]
        if cands:
            used.add(min(cands, key=lambda i: abs(gt[i] - p)))
            hit += 1
    P, R = hit / len(pred), hit / len(gt)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)
