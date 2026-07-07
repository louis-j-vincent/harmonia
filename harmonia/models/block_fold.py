"""
Symbolic block folding: recover repeated song structure from the *decoded*
chord stream, then fold it.

Motivation
----------
`periodicity.py` folds beat-level *evidence* at one global period for SNR.
`demo_infer_song.py` folds *decoded chords* by certainty-weighted voting —
but it groups occurrences using ground-truth section labels
(``rec["section_per_bar"]``), so the reported ``+fold`` gain leaks GT
structure. This module infers that grouping from the chords themselves.

Two layers (see the two-SSM design):
  1. ABSOLUTE  — literal repeats. Two 8-bar A-sections match note-for-note,
     so we can VOTE across the copies and fix the weaker one. Drives folding.
  2. RELATIVE  — transposition-invariant motif shape. Every ii-V looks alike
     regardless of key; used only to *tag* recurring motifs for display, not
     to vote (transposed copies are genuinely different chords).

Unit cell is the bar (rhythm-changes bridge = 1 chord / 2 bars, A = 2
chords / bar — the bar is the natural granularity).

Repeat detection searches over phase: fixed bar-tiling misses real repeats
(on Anthropology a fixed 8-bar tile scored block0~block2 at 0%, while the
phase-aligned comparison scored 75%).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    idx: int                       # bar index within the song
    roots: list[int]               # pitch-class root per chord in the bar (0-11)
    quals: list[str]               # quality token per chord ("m7", "7", "maj7"...)
    conf: list[float]              # model confidence per chord

    @property
    def n(self) -> int:
        return len(self.roots)


@dataclass
class Block:
    label: str                     # "A", "B", ... assigned by cluster identity
    start_bar: int                 # inclusive
    end_bar: int                   # exclusive
    cluster_id: int                # windows sharing a cluster_id are "the same"


@dataclass
class FoldResult:
    blocks: list[Block]
    bar_cluster: list[int]         # per-bar cluster id (for eval vs GT sections)
    period_bars: int               # inferred section length
    phase_bars: int                # inferred phase (offset of first full window)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

def to_bars(chords: list[dict], *, root_key="root", qual_key="qual",
            conf_key="conf", bar_key="bar") -> list[Bar]:
    """Collapse a decoded per-chord stream into per-bar cells.

    Expects each chord dict to expose a pitch-class root, a quality token, a
    confidence and a bar index. Bars with no decoded chord become empty cells
    so bar indexing stays aligned with the song's bar grid.
    """
    if not chords:
        return []
    by_bar: dict[int, list[dict]] = {}
    for c in chords:
        by_bar.setdefault(int(c[bar_key]), []).append(c)
    last = max(by_bar)
    bars: list[Bar] = []
    for b in range(last + 1):
        cs = by_bar.get(b, [])
        bars.append(Bar(
            idx=b,
            roots=[int(c[root_key]) % 12 for c in cs],
            quals=[str(c[qual_key]) for c in cs],
            conf=[float(c.get(conf_key, 1.0)) for c in cs],
        ))
    return bars


# ---------------------------------------------------------------------------
# Bar similarity primitives
# ---------------------------------------------------------------------------

def _seq_match(a: list, b: list) -> float:
    """Position-wise agreement of two equal-role token lists in [0,1].

    Empty-vs-empty is treated as a match (both bars silent); empty-vs-nonempty
    as a mismatch. Unequal lengths are compared over the shorter length and
    penalised by the length ratio.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    agree = sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n
    return agree * (n / max(len(a), len(b)))


def bar_similarity(a: Bar, b: Bar, *, level="root", transpose=False) -> float:
    """Similarity of two bars in [0,1].

    level="root"    : compare roots (optionally + quality if level="seventh").
    transpose=True  : compare interval shape relative to each bar's first root
                      (motif layer) instead of absolute pitch.
    """
    if a.n == 0 or b.n == 0:
        return 1.0 if a.n == b.n else 0.0

    if transpose:
        ra = [(r - a.roots[0]) % 12 for r in a.roots]
        rb = [(r - b.roots[0]) % 12 for r in b.roots]
    else:
        ra, rb = a.roots, b.roots

    s = _seq_match(ra, rb)
    if level == "seventh":
        s = 0.5 * s + 0.5 * _seq_match(a.quals, b.quals)
    return s


def build_chord_ssm(bars: list[Bar], *, level="root", transpose=False) -> np.ndarray:
    """(n_bars, n_bars) symbolic self-similarity over bars."""
    n = len(bars)
    ssm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = bar_similarity(bars[i], bars[j], level=level, transpose=transpose)
            ssm[i, j] = ssm[j, i] = s
    return ssm


# ---------------------------------------------------------------------------
# Structure: phase-searched window matching
# ---------------------------------------------------------------------------

def _window_sim(ssm: np.ndarray, a: int, b: int, S: int) -> float:
    """Aligned similarity of the S-bar window at a vs the one at b."""
    n = ssm.shape[0]
    m = min(S, n - a, n - b)
    if m <= 0:
        return 0.0
    return float(np.mean([ssm[a + k, b + k] for k in range(m)]))


def _find_repeats(ssm: np.ndarray, *, min_len: int, sim_thresh: float
                  ) -> list[tuple[float, int, int, int]]:
    """Repeated segments as diagonal stripes of the SSM.

    A run of >= min_len consecutive bars whose lag-k diagonal stays above
    sim_thresh means segment [i, i+L) recurs at [i+k, i+k+L). Returns
    (score, a, b, L) with score = mean-similarity x length, so the longest,
    cleanest repeats sort first. This is variable-length and anchored to where
    repeats actually land, so it survives odd section lengths (A15 B8 A8) that
    fixed-period tiling straddles.
    """
    n = ssm.shape[0]
    reps: list[tuple[float, int, int, int]] = []
    for k in range(min_len, n):
        diag = np.array([ssm[i, i + k] for i in range(n - k)])
        i = 0
        while i < len(diag):
            if diag[i] >= sim_thresh:
                j = i
                while j < len(diag) and diag[j] >= sim_thresh:
                    j += 1
                if j - i >= min_len:
                    reps.append((float(diag[i:j].mean()) * (j - i), i, i + k, j - i))
                i = j
            else:
                i += 1
    reps.sort(key=lambda r: -r[0])
    return reps


def _cluster_windows(starts: list[int], S: int, ssm: np.ndarray,
                     sim_thresh: float) -> dict[int, int]:
    """Greedy nearest-exemplar clustering of S-bar windows by aligned sim."""
    exemplars: list[int] = []
    cluster_of: dict[int, int] = {}
    for a in starts:
        best_c, best_s = -1, sim_thresh
        for cid, ex in enumerate(exemplars):
            s = _window_sim(ssm, a, ex, S)
            if s >= best_s:
                best_c, best_s = cid, s
        if best_c < 0:
            best_c = len(exemplars)
            exemplars.append(a)
        cluster_of[a] = best_c
    return cluster_of


def _tiling_quality(starts: list[int], S: int, ssm: np.ndarray,
                    cluster_of: dict[int, int]) -> float:
    """Silhouette-like score: mean within-cluster minus between-cluster window
    similarity. High when repeated blocks are internally consistent and
    distinct from the others — i.e. the tiling carves real sections."""
    intra, inter = [], []
    for i, a in enumerate(starts):
        for b in starts[i + 1:]:
            s = _window_sim(ssm, a, b, S)
            (intra if cluster_of[a] == cluster_of[b] else inter).append(s)
    if not intra:
        return -1.0
    q = float(np.mean(intra))
    if inter:
        q -= float(np.mean(inter))
    return q


def detect_blocks(ssm: np.ndarray, *, candidate_lens=(4, 8, 16, 32),
                  sim_thresh=0.8, min_len=4) -> tuple[list[Block], int, int]:
    """Infer block length, phase, and per-window cluster letters.

    Jazz/pop forms are overwhelmingly built from regular 4/8/16/32-bar units,
    so we search block length S over those regular candidates (not raw stripe
    lengths, which let spurious odd periods win) and, for each, search the
    phase. The (S, phase) chosen maximises tiling quality — mean within-cluster
    minus between-cluster window similarity — so the grid lands where repeated
    sections actually are. Windows are then clustered into letters A,B,C...

    Returns (blocks, period_bars, phase_bars).
    """
    n = ssm.shape[0]
    if n < 2 * min_len:
        return [Block("A", 0, n, 0)], n, 0

    cands = [S for S in candidate_lens if min_len <= S <= n // 2]
    if not cands:
        return [Block("A", 0, n, 0)], n, 0

    best = None                       # (score, S, phase, cluster_of, starts)
    for S in cands:
        for phase in range(S):
            starts = list(range(phase, n - S + 1, S))
            if len(starts) < 2:
                continue
            cluster_of = _cluster_windows(starts, S, ssm, sim_thresh)
            q = _tiling_quality(starts, S, ssm, cluster_of)
            coverage = len(starts) * S / n          # prefer grids with little leftover
            # Bias toward larger blocks: a pure silhouette prefers 4-bar phrases,
            # but a lead-sheet wants the section-length repeat unit (8/16 bars),
            # not the phrase. Without this, a 16-bar "A" splits into A B A C.
            score = q + 0.1 * coverage + 1.0 * S / n
            if best is None or score > best[0]:
                best = (score, S, phase, cluster_of, starts)

    if best is None:
        return [Block("A", 0, n, 0)], n, 0

    _, S, phase, cluster_of, starts = best
    order = {cid: i for i, cid in enumerate(dict.fromkeys(cluster_of[a] for a in starts))}
    letters = [chr(ord("A") + i) for i in range(len(order))]
    blocks: list[Block] = []
    if phase > 0:
        blocks.append(Block("_", 0, phase, -1))
    for a in starts:
        cid = order[cluster_of[a]]
        blocks.append(Block(letters[cid], a, min(a + S, n), cid))
    tail = starts[-1] + S if starts else phase
    if tail < n:
        blocks.append(Block("_", tail, n, -1))
    return blocks, S, phase


# ---------------------------------------------------------------------------
# Top-level fold
# ---------------------------------------------------------------------------

def fold_structure(bars: list[Bar], *, level="root", sim_thresh=0.8,
                   candidate_lens=(4, 8, 12, 16, 32)) -> FoldResult:
    """Infer block structure from decoded bars (absolute layer)."""
    n = len(bars)
    if n == 0:
        return FoldResult([], [], 0, 0)
    ssm = build_chord_ssm(bars, level=level)
    blocks, S, phase = detect_blocks(ssm, candidate_lens=candidate_lens,
                                     sim_thresh=sim_thresh)
    bar_cluster = [-1] * n
    for blk in blocks:
        for b in range(blk.start_bar, blk.end_bar):
            bar_cluster[b] = blk.cluster_id
    n_unique = len({b.cluster_id for b in blocks if b.cluster_id >= 0})
    return FoldResult(
        blocks=blocks, bar_cluster=bar_cluster, period_bars=S, phase_bars=phase,
        stats={"n_bars": n, "n_blocks": len(blocks), "n_unique": n_unique,
               "compression": len(blocks) / max(n_unique, 1)},
    )
