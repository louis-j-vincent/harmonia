"""
tab_aligner.py — align a guitar-tab chord sequence against a Harmonia chart.

Two problems to solve:
  1. Key offset: the tab author may have published in a different key (capo,
     different standard tuning, or just a different arrangement key).  We find
     the chromatic transposition (0–11) that minimises DTW cost.

  2. Sequence alignment: tab chords are one-per-section-change; chart chords
     are one-per-beat-event.  DTW with a simple quality-distance metric aligns
     them so each chart chord gets a tab annotation (or a gap marker).

Output per chart chord:
    tab_chord   (str | None)  — transposed tab chord label that aligns here
    match       (str)         — "exact" | "family" | "mismatch" | "gap"
    tab_conf    (float)       — 0.0–1.0 suggested confidence boost

Design notes
------------
- "Exact" means root-class AND quality family both match after transposing.
- "Family" means only root-class matches.
- Quality distance uses a 4-bucket taxonomy (maj/min/dom/dim-aug-sus) so
  Cmaj7 vs C are "exact family" and count as 0 distance.
- A high-rated tab (≥4.5, ≥100 votes) with an exact match boosts confidence
  to min(original_conf + 0.25, 0.95).  Family match boosts by 0.10.
  Mismatch does nothing (we don't want to penalise—tabs can be wrong too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ── Chord quality families ──────────────────────────────────────────────────
# Bucket any quality token into a coarse 5-class family for distance purposes.

_MAJ_TOKENS  = {"", "maj", "M", "^7", "maj7", "6", "maj9", "^9", "maj11", "^11",
                "maj13", "^13", "2", "add9", "add2"}
_MIN_TOKENS  = {"-", "m", "min", "-7", "m7", "min7", "-9", "m9", "min9",
                "-11", "m11", "min11", "-13", "m13", "min13", "-6", "m6",
                "-^7", "mM7", "minMaj7"}
_DOM_TOKENS  = {"7", "9", "11", "13", "7b9", "7#9", "7alt", "+7", "aug7",
                "7sus", "7sus4", "9sus", "9sus4"}
_DIM_TOKENS  = {"o", "dim", "o7", "dim7", "°7"}
_HDIM_TOKENS = {"h7", "m7b5", "ø7", "ø"}
_AUG_TOKENS  = {"+", "aug", "+^7", "augMaj7"}
_SUS_TOKENS  = {"sus", "sus4", "sus2"}

def _family(q: str) -> str:
    q = q.strip()
    if q in _MAJ_TOKENS:   return "maj"
    if q in _MIN_TOKENS:   return "min"
    if q in _DOM_TOKENS:   return "dom"
    if q in _DIM_TOKENS:   return "dim"
    if q in _HDIM_TOKENS:  return "hdim"
    if q in _AUG_TOKENS:   return "aug"
    if q in _SUS_TOKENS:   return "sus"
    # fallback: look at first character
    if q.startswith("-") or q.startswith("m"): return "min"
    if q.startswith("o") or q.startswith("°"): return "dim"
    if q.startswith("+"):                       return "aug"
    if q.startswith("h"):                       return "hdim"
    return "maj"

# Coarse distance between two quality families
_FAM_DIST: dict[tuple[str,str], float] = {}
for _a in ("maj","min","dom","dim","hdim","aug","sus"):
    for _b in ("maj","min","dom","dim","hdim","aug","sus"):
        _FAM_DIST[(_a,_b)] = 0.0 if _a==_b else (
            0.3 if {_a,_b} in ({"maj","dom"},{"maj","sus"},{"min","hdim"},{"dim","hdim"})
            else 0.6 if {_a,_b} in ({"min","dom"},{"maj","min"},{"dom","sus"})
            else 1.0
        )

def _quality_dist(qa: str, qb: str) -> float:
    return _FAM_DIST.get((_family(qa), _family(qb)), 1.0)


# ── Chord token parser (iReal format) ────────────────────────────────────────

_ROOTS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_FLAT_MAP = {"Db":"C#","Eb":"D#","Fb":"E","Gb":"F#","Ab":"G#","Bb":"A#","Cb":"B"}
_SHARP_TO_PC = {r: i for i, r in enumerate(_ROOTS)}


def _parse_ireal(token: str) -> tuple[int, str]:
    """Parse an iReal token like 'F#-7' or 'Bbmaj7' → (root_pc, quality_tail).
    Returns (-1, '') on failure."""
    if not token or token in ("N.C.", "N", "NC"):
        return -1, "N"
    t = token.strip()
    # 2-char root first
    root_str = t[:2] if len(t) >= 2 else t[:1]
    if root_str in _FLAT_MAP:
        root_str = _FLAT_MAP[root_str]
    if root_str in _SHARP_TO_PC:
        return _SHARP_TO_PC[root_str], t[2:]
    root_str = t[:1]
    if root_str in _SHARP_TO_PC:
        return _SHARP_TO_PC[root_str], t[1:]
    return -1, t


def _transpose_ireal(token: str, semitones: int, prefer_flats: bool = False) -> str:
    """Transpose an iReal token by semitones."""
    pc, q = _parse_ireal(token)
    if pc < 0:
        return token
    new_pc = (pc + semitones) % 12
    if prefer_flats:
        flat_names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
        return flat_names[new_pc] + q
    return _ROOTS[new_pc] + q


# ── UG-style chord name → iReal token ────────────────────────────────────────
# Tab chords use slightly different notation than iReal (e.g. "Am7" vs "A-7")

_UG_QUAL_MAP = {
    "maj7":"^7","M7":"^7","Δ7":"^7","△7":"^7","^7":"^7",
    "maj9":"^9","M9":"^9",
    "m7":"-7","min7":"-7",
    "m":"-","min":"-",
    "dim":"o","dim7":"o7","°":"o","°7":"o7",
    "aug":"+","aug7":"+7",
    "m7b5":"h7","ø":"h7","ø7":"h7","half-dim":"h7",
    "sus":"sus","sus4":"sus","sus2":"sus2",
    "7sus4":"7sus","7sus":"7sus",
    "mMaj7":"-^7","mM7":"-^7","minMaj7":"-^7",
    "add9":"add9","2":"add9","6":"6","-6":"-6","9":"9","-9":"-9",
}

def _ug_to_ireal(ug_chord: str) -> str:
    """Convert a UG-style chord name to an iReal token (best-effort)."""
    ug_chord = ug_chord.strip()
    # Strip slash bass
    ug_chord = ug_chord.split("/")[0]
    if not ug_chord:
        return ""
    # Root
    root = ug_chord[:2] if len(ug_chord) >= 2 and ug_chord[1] in "#b" else ug_chord[:1]
    qual_raw = ug_chord[len(root):]
    # Normalise flat root
    if root in _FLAT_MAP:
        root = _FLAT_MAP[root]
    # Map quality; fall back to raw if not found
    qual = _UG_QUAL_MAP.get(qual_raw, qual_raw)
    return root + qual


# ── DTW alignment ─────────────────────────────────────────────────────────────

def _chord_dist(pc_a: int, q_a: str, pc_b: int, q_b: str) -> float:
    """Distance in [0,2] between two chord tokens."""
    if pc_a < 0 or pc_b < 0:
        return 1.5       # N.C. vs anything
    root_d = 0.0 if pc_a == pc_b else (0.5 if abs(pc_a - pc_b) in (1, 11) else 1.0)
    qual_d = _quality_dist(q_a, q_b)
    return root_d + qual_d


def _dtw(seq_a: list[tuple[int,str]], seq_b: list[tuple[int,str]]) -> tuple[float, list[tuple[int,int]]]:
    """Standard DTW between two sequences of (root_pc, quality) pairs.
    Returns (cost, path) where path is list of (i,j) index pairs (0-based)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf"), []

    INF = float("inf")
    # cost matrix
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _chord_dist(seq_a[i-1][0], seq_a[i-1][1],
                            seq_b[j-1][0], seq_b[j-1][1])
            dp[i][j] = d + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        diag = dp[i-1][j-1]
        left = dp[i][j-1]
        up   = dp[i-1][j]
        best = min(diag, left, up)
        if best == diag:
            i -= 1; j -= 1
        elif best == left:
            j -= 1
        else:
            i -= 1
    path.reverse()
    return dp[n][m] / max(n, m), path


# ── Key offset detection ──────────────────────────────────────────────────────

def _best_transpose(
    chart_chords: list[tuple[int,str]],
    tab_chords: list[tuple[int,str]],
) -> tuple[int, float]:
    """Try all 12 transpositions of the tab and return the one with lowest DTW cost.
    Returns (best_semitones, best_cost)."""
    best_offset, best_cost = 0, float("inf")
    for offset in range(12):
        transposed = [((pc + offset) % 12, q) for pc, q in tab_chords]
        cost, _ = _dtw(chart_chords, transposed)
        if cost < best_cost:
            best_cost, best_offset = cost, offset
    return best_offset, best_cost


# ── Per-chord annotation ──────────────────────────────────────────────────────

@dataclass
class ChordAnnotation:
    chord_idx: int          # index into the chart's P.chords array
    tab_chord: str          # transposed tab chord (iReal token)
    tab_pc: int             # root pitch class of transposed tab chord
    tab_q: str              # quality tail of transposed tab chord
    match: str              # "exact" | "family" | "mismatch" | "gap"
    tab_conf_boost: float   # how much to add to chart chord's confidence


def _match_grade(
    chart_pc: int, chart_q: str,
    tab_pc: int,   tab_q: str,
) -> tuple[str, float]:
    """Return (match_label, conf_boost_amount)."""
    if chart_pc < 0 or tab_pc < 0:
        return "gap", 0.0
    if chart_pc != tab_pc:
        return "mismatch", 0.0
    chart_fam = _family(chart_q)
    tab_fam   = _family(tab_q)
    if chart_fam == tab_fam:
        return "exact", 0.25
    return "family", 0.10


@dataclass
class AlignmentResult:
    transpose_semitones: int    # semitones added to tab to match chart key
    dtw_cost: float             # normalised DTW cost (lower = better match)
    annotations: list[ChordAnnotation]


def align_tab_to_chart(
    chart_chords_payload: list[dict],
    tab_chord_tokens: list[str],
    tab_rating: float = 0.0,
    tab_votes: int = 0,
) -> AlignmentResult:
    """
    Align a list of UG tab chord tokens against chart chord payload dicts.

    chart_chords_payload: list of P.chords entries (dicts with 'root', 'lv')
    tab_chord_tokens: raw UG chord names in order, e.g. ["Am", "G", "F", "E7"]
    tab_rating / tab_votes: used to scale confidence boost (high-quality tabs
        boost more; low-vote tabs get half boost).

    Returns AlignmentResult with per-chart-chord annotations.
    """
    # Parse chart chords
    chart_seq: list[tuple[int,str]] = []
    for c in chart_chords_payload:
        pc  = c.get("root", -1)
        q   = c.get("lv", {}).get("seventh", {}).get("q", "")
        chart_seq.append((pc, q))

    # Parse and deduplicate consecutive identical tab chords (tab authors often
    # rewrite the same chord for every bar repeat)
    raw_tab: list[tuple[int,str]] = []
    for tok in tab_chord_tokens:
        ireal = _ug_to_ireal(tok)
        pc, q = _parse_ireal(ireal)
        # Skip exact duplicates of the previous chord
        if raw_tab and raw_tab[-1] == (pc, q):
            continue
        raw_tab.append((pc, q))

    if not raw_tab or not chart_seq:
        return AlignmentResult(0, float("inf"), [])

    # Find best transposition
    offset, cost = _best_transpose(chart_seq, raw_tab)
    transposed_tab: list[tuple[int,str]] = [
        ((pc + offset) % 12 if pc >= 0 else -1, q)
        for pc, q in raw_tab
    ]

    # Quality scale factor: high-rating + high-vote tabs boost more
    vote_factor  = min(1.0, math.log2(tab_votes + 2) / math.log2(502))  # 500 votes → 1.0
    rat_factor   = max(0.0, (tab_rating - 3.5) / 1.5)                   # 5.0 → 1.0, 3.5 → 0.0
    quality_mult = 0.3 + 0.7 * vote_factor * rat_factor

    # DTW alignment
    _, path = _dtw(chart_seq, transposed_tab)
    # path: list of (chart_idx, tab_idx)
    # Build a mapping chart_idx → tab_idx (last winner if multi-mapped)
    chart_to_tab: dict[int, int] = {}
    for ci, ti in path:
        chart_to_tab[ci] = ti

    annotations: list[ChordAnnotation] = []
    for ci, (chart_pc, chart_q) in enumerate(chart_seq):
        ti = chart_to_tab.get(ci)
        if ti is None:
            annotations.append(ChordAnnotation(
                chord_idx=ci, tab_chord="", tab_pc=-1, tab_q="",
                match="gap", tab_conf_boost=0.0,
            ))
            continue

        tab_pc, tab_q = transposed_tab[ti]
        # Reconstruct readable label
        tab_label = (_ROOTS[tab_pc] if 0 <= tab_pc < 12 else "N") + tab_q

        match, base_boost = _match_grade(chart_pc, chart_q, tab_pc, tab_q)
        boost = round(base_boost * quality_mult, 3)

        annotations.append(ChordAnnotation(
            chord_idx=ci,
            tab_chord=tab_label,
            tab_pc=tab_pc,
            tab_q=tab_q,
            match=match,
            tab_conf_boost=boost,
        ))

    return AlignmentResult(
        transpose_semitones=offset,
        dtw_cost=round(cost, 4),
        annotations=annotations,
    )
