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
from dataclasses import dataclass, field
from typing import Optional

# ── Pitch-class set chord distance ──────────────────────────────────────────
# Each chord is described by its constituent pitch classes (relative to C=0).
# Distance = 1 - Jaccard(tones_a, tones_b), so chords sharing many tones
# are close regardless of root name.  Example:
#   Gm  = {G, Bb, D} = {7, 10, 2}
#   Bb  = {Bb, D, F} = {10, 2, 5}   → Jaccard = 2/4 = 0.5  → dist = 0.5
#   Ab  = {Ab, C, Eb}= {8, 0, 3}    → Jaccard = 0/6 = 0    → dist = 1.0
#
# Intervals (semitones from root) for each quality token:

_INTERVALS: dict[str, tuple[int, ...]] = {
    # triads
    "":      (0, 4, 7),     # major
    "m":     (0, 3, 7),     # minor
    "-":     (0, 3, 7),
    "o":     (0, 3, 6),     # diminished
    "dim":   (0, 3, 6),
    "+":     (0, 4, 8),     # augmented
    "aug":   (0, 4, 8),
    "sus4":  (0, 5, 7),
    "sus":   (0, 5, 7),
    "sus2":  (0, 2, 7),
    "5":     (0, 7),        # power chord
    # sixths
    "6":     (0, 4, 7, 9),
    "m6":    (0, 3, 7, 9),
    "-6":    (0, 3, 7, 9),
    "69":    (0, 2, 4, 7, 9),
    # major sevenths
    "^7":    (0, 4, 7, 11),
    "maj7":  (0, 4, 7, 11),
    "M7":    (0, 4, 7, 11),
    "^9":    (0, 2, 4, 7, 11),
    "^13":   (0, 2, 4, 7, 9, 11),
    # dominant sevenths
    "7":     (0, 4, 7, 10),
    "9":     (0, 2, 4, 7, 10),
    "11":    (0, 2, 4, 5, 7, 10),
    "13":    (0, 2, 4, 7, 9, 10),
    "7b9":   (0, 1, 4, 7, 10),
    "7#9":   (0, 3, 4, 7, 10),
    "7#11":  (0, 4, 6, 7, 10),
    "7alt":  (0, 1, 3, 4, 6, 10),
    "7sus4": (0, 5, 7, 10),
    "7sus":  (0, 5, 7, 10),
    "9sus4": (0, 2, 5, 7, 10),
    # minor sevenths
    "-7":    (0, 3, 7, 10),
    "m7":    (0, 3, 7, 10),
    "-9":    (0, 2, 3, 7, 10),
    "-11":   (0, 2, 3, 5, 7, 10),
    "-^7":   (0, 3, 7, 11),   # minMaj7
    "mM7":   (0, 3, 7, 11),
    # half-diminished
    "h7":    (0, 3, 6, 10),
    "m7b5":  (0, 3, 6, 10),
    # diminished seventh
    "o7":    (0, 3, 6, 9),
    "dim7":  (0, 3, 6, 9),
    # additions
    "add9":  (0, 2, 4, 7),
    "2":     (0, 2, 4, 7),
}

def _chord_tones(root_pc: int, quality: str) -> frozenset[int]:
    """Pitch classes present in the chord (all in 0–11)."""
    if root_pc < 0:
        return frozenset()
    ivs = _INTERVALS.get(quality.strip(), _INTERVALS[""])
    return frozenset((root_pc + iv) % 12 for iv in ivs)


def _pc_set_dist(a: frozenset[int], b: frozenset[int]) -> float:
    """1 − Jaccard similarity. Range [0, 1].  Empty sets → 1.0."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union


# Keep family classification for the match-grade label (exact / family / mismatch)
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
    if q.startswith("-") or q.startswith("m"): return "min"
    if q.startswith("o") or q.startswith("°"): return "dim"
    if q.startswith("+"):                       return "aug"
    if q.startswith("h"):                       return "hdim"
    return "maj"


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
    """Pitch-class-set Jaccard distance in [0, 1].

    Gm vs Bb  → Jaccard({G,Bb,D},{Bb,D,F}) = 2/4 = 0.5   → dist = 0.5
    Gm vs Ab  → Jaccard({G,Bb,D},{Ab,C,Eb})= 0/6 = 0     → dist = 1.0
    N.C. vs X → 1.0
    """
    if pc_a < 0 or pc_b < 0:
        return 1.0
    return _pc_set_dist(_chord_tones(pc_a, q_a), _chord_tones(pc_b, q_b))


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
    dist: float = 0.0       # pitch-class-set Jaccard distance in [0, 1]


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
        dist  = round(_chord_dist(chart_pc, chart_q, tab_pc, tab_q), 3)

        annotations.append(ChordAnnotation(
            chord_idx=ci,
            tab_chord=tab_label,
            tab_pc=tab_pc,
            tab_q=tab_q,
            match=match,
            tab_conf_boost=boost,
            dist=dist,
        ))

    return AlignmentResult(
        transpose_semitones=offset,
        dtw_cost=round(cost, 4),
        annotations=annotations,
    )


# ── Audio-grounded alignment ──────────────────────────────────────────────────
# Instead of aligning the tab directly to the iReal sequence, align it to
# Harmonia's inferred chord sequence from the actual audio.  This gives each
# tab chord a real timestamp and bar/beat position.

@dataclass
class AudioChord:
    """One chord event from Harmonia's pipeline output."""
    pc: int
    quality: str
    label: str
    start_s: float
    end_s: float
    bar: int        # 1-indexed, computed from cumulative beats
    beat: int       # 0-indexed within bar


@dataclass
class TabPlacement:
    """A tab chord pinned to the audio grid."""
    tab_chord: str        # original UG token
    tab_pc: int
    tab_q: str
    audio: AudioChord     # the audio chord this was aligned to
    dist: float           # Jaccard distance tab vs audio chord


@dataclass
class AudioAlignmentResult:
    transpose_semitones: int
    dtw_cost: float
    placements: list[TabPlacement]


def audio_chart_to_sequence(chart) -> list[AudioChord]:
    """Convert a HarmoniaPipeline ChordChart to a list of AudioChord.

    Bar/beat positions are derived from cumulative beat count and
    chart.time_signature (e.g. '4/4').
    """
    bpb = int(chart.time_signature.split("/")[0])
    result: list[AudioChord] = []
    cum_beat = 0.0
    for ch in chart.chords:
        pc, quality = _parse_ireal(ch["label"])
        bar  = int(cum_beat / bpb) + 1
        beat = int(cum_beat) % bpb
        result.append(AudioChord(
            pc=pc, quality=quality, label=ch["label"],
            start_s=ch["start_s"], end_s=ch["end_s"],
            bar=bar, beat=beat,
        ))
        cum_beat += ch["duration_beats"]
    return result


def align_tab_to_audio(
    audio_seq: list[AudioChord],
    tab_chord_tokens: list[str],
) -> AudioAlignmentResult:
    """DTW-align a tab chord list to an audio-inferred chord sequence.

    Tries all 12 transpositions (tab may be in a different key or capo).
    Returns a TabPlacement per (deduplicated) tab chord giving it a
    timestamp and bar/beat position from the audio grid.
    """
    audio_pairs = [(a.pc, a.quality) for a in audio_seq]

    # Parse + deduplicate consecutive identical tab chords
    raw_tab: list[tuple[int, str, str]] = []  # (pc, quality, original_token)
    for tok in tab_chord_tokens:
        ireal = _ug_to_ireal(tok)
        pc, q = _parse_ireal(ireal)
        if raw_tab and raw_tab[-1][:2] == (pc, q):
            continue
        raw_tab.append((pc, q, tok))

    if not raw_tab or not audio_pairs:
        return AudioAlignmentResult(0, float("inf"), [])

    tab_pairs = [(pc, q) for pc, q, _ in raw_tab]

    # Best transposition
    offset, cost = _best_transpose(audio_pairs, tab_pairs)
    transposed = [((pc + offset) % 12 if pc >= 0 else -1, q) for pc, q, _ in raw_tab]

    # DTW tab (query) against audio (reference) — we want tab_idx → audio_idx
    _, path = _dtw(audio_pairs, transposed)
    # path: (audio_idx, tab_idx); build tab_idx → audio_idx mapping
    tab_to_audio: dict[int, int] = {}
    for ai, ti in path:
        tab_to_audio[ti] = ai   # last audio chord mapped to this tab chord

    placements: list[TabPlacement] = []
    for ti, (tab_pc, tab_q, orig_tok) in enumerate(raw_tab):
        t_pc = (tab_pc + offset) % 12 if tab_pc >= 0 else -1
        ai = tab_to_audio.get(ti)
        if ai is None:
            continue
        audio_ch = audio_seq[ai]
        tab_label = (_ROOTS[t_pc] if 0 <= t_pc < 12 else "N") + tab_q
        dist = round(_chord_dist(audio_ch.pc, audio_ch.quality, t_pc, tab_q), 3)
        placements.append(TabPlacement(
            tab_chord=tab_label,
            tab_pc=t_pc,
            tab_q=tab_q,
            audio=audio_ch,
            dist=dist,
        ))

    return AudioAlignmentResult(
        transpose_semitones=offset,
        dtw_cost=round(cost, 4),
        placements=placements,
    )
