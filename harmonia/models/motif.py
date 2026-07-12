"""
Motif stacking: find short chord patterns that recur through a tune and
collapse the song onto its handful of unique motifs.

Two views of "the same pattern":
  EXACT  — the literal chords recur: ``Cm7 F7`` appears six times.
  SHAPE  — transposition-invariant. ``Cm7 F7``, ``Dm7 G7`` and ``Fm7 Bb7`` are
           the *same shape* (a ii-V: a min7 followed by a dom7 a fourth up) in
           three different keys. This is the "stacking" that compresses a tune:
           a rhythm-changes A-section is really just one ii-V shape reused.

A motif is a *contiguous* run of chords. We score a candidate by how many
chord-slots it saves if we factor it out — length x (occurrences - 1) — and
greedily tile the song with the most compressive motifs, longest first, so the
reduction reads like a musician's shorthand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Chord:
    root: int            # pitch class 0-11
    qual: str            # quality token, e.g. "m7", "7", "maj7"
    label: str           # display label, e.g. "Cm7"
    bar: int = 0         # bar index (motifs are constrained to whole bars)


@dataclass
class Motif:
    kind: str                        # "exact" or "shape"
    key: tuple                       # hashable identity (exact labels, or shape)
    length: int                      # chords in the motif
    occurrences: list[int]           # start indices into the chord sequence
    display: str                     # human-readable, e.g. "Cm7 F7" or "ii-V"
    keys: list[str] = field(default_factory=list)  # shape only: root of each occ

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def saving(self) -> int:
        return self.length * (self.count - 1)


# ---------------------------------------------------------------------------
# Shape encoding (transposition-invariant)
# ---------------------------------------------------------------------------

def _shape_key(chords: list[Chord]) -> tuple:
    """Transposition-invariant identity of a chord run: the first chord's
    quality, then (interval-from-previous-root, quality) for each following
    chord. Independent of absolute key."""
    out = [(chords[0].qual,)]
    for a, b in zip(chords, chords[1:]):
        out.append(((b.root - a.root) % 12, b.qual))
    return tuple(out)


def _root_shape_key(chords: list[Chord]) -> tuple:
    """Root-movement-only identity: intervals between successive roots, no
    quality. Cm7→F7 and Cmaj7→Fmaj7 share the same key (both move +5)."""
    out: list = [()]
    for a, b in zip(chords, chords[1:]):
        out.append(((b.root - a.root) % 12,))
    return tuple(out)


# quality-aware shapes → friendly names.
# Quality tokens match QUALITY_MAP buckets: min7, dom7, maj7, 6, min, …
_SHAPE_NAMES: dict[tuple, str] = {
    # cadential ii-V family
    (("min7",), (5, "dom7")): "ii-V",
    (("min7",), (5, "dom7"), (5, "maj7")): "ii-V-I",
    (("min7",), (5, "dom7"), (5, "6")): "ii-V-I",
    (("min7",), (5, "dom7"), (5, "min7"), (5, "dom7")): "ii-V ii-V",
    # dominant resolution
    (("dom7",), (5, "maj7")): "V-I",
    (("dom7",), (5, "6")): "V-I",
    # subdominant approach
    (("maj7",), (5, "maj7")): "I-IV",
    (("6",), (5, "maj7")): "I-IV",
    (("6",), (5, "6")): "I-IV",
    # turnaround / back-cycle
    (("maj7",), (9, "dom7")): "I-VI",
    (("6",), (9, "dom7")): "I-VI",
    # diatonic fourth chains
    (("min7",), (5, "min7")): "vi-ii",
    # secondary dominant
    (("dom7",), (5, "dom7")): "V/V-V",
    (("dom7",), (5, "dom7"), (5, "dom7")): "dom-cycle",
}

# root-movement-only shapes → friendly names (intervals only, no quality)
_ROOT_SHAPE_NAMES: dict[tuple, str] = {
    ((), (5,)): "+4th",
    ((), (7,)): "+5th",
    ((), (2,)): "+2nd",
    ((), (10,)): "-2nd",
    ((), (3,)): "+m3rd",
    ((), (9,)): "+M3rd",
    ((), (6,)): "tritone",
    ((), (1,)): "+½step",
    ((), (11,)): "-½step",
    ((), (5,), (5,)): "+4th chain",
    ((), (5,), (5,), (5,)): "4th cycle",
}


# Same chord-symbol convention the chart itself uses (e.g. "F△7", "Gm7") —
# an *unnamed* shape's fallback display used to show raw internal quality
# bucket names ("dom7", "min7"), which read as a different, inconsistent
# vocabulary next to named shapes like "ii-V". Translate to real symbols.
_QUAL_SYMBOL: dict[str, str] = {
    "maj": "", "min": "m", "dom7": "7", "maj7": "△7", "min7": "m7",
    "m7b5": "ø7", "dim7": "°7", "dim": "°", "6": "6", "sus4": "sus4", "aug": "+",
}


def _shape_display(shape: tuple, chords: list[Chord], *, root_only: bool = False) -> str:
    names = _ROOT_SHAPE_NAMES if root_only else _SHAPE_NAMES
    if shape in names:
        return names[shape]
    if root_only:
        ivs = " ".join(f"+{iv}" for (iv,) in shape[1:])
        return f"[{ivs}]"
    head = _QUAL_SYMBOL.get(chords[0].qual, chords[0].qual)
    # space before the quality symbol — "+11 7" not "+117", which misreads
    # as the single number "117" once the symbol itself is numeric ("7")
    tail = " ".join(f"+{iv} {_QUAL_SYMBOL.get(q, q)}" for iv, q in (s for s in shape[1:]))
    return f"[{head} {tail}]".strip()


# ---------------------------------------------------------------------------
# Motif finding
# ---------------------------------------------------------------------------

def _all_ngrams(chords: list[Chord], min_len: int, max_len: int, *,
                shape: bool, root_only: bool = False):
    """Return {key: [start indices]} for every length in [min_len, max_len].

    Motifs are constrained to whole bars — a run may only start on the first
    chord of a bar and end on the last chord of a bar — so patterns align to
    bar lines ("Cm7 F7") instead of straddling them ("G7 Cm7 F7")."""
    n = len(chords)
    bar_start = [i == 0 or chords[i].bar != chords[i - 1].bar for i in range(n)]
    bar_end = [i == n - 1 or chords[i].bar != chords[i + 1].bar for i in range(n)]
    index: dict[tuple, list[int]] = {}
    for L in range(min_len, max_len + 1):
        for i in range(n - L + 1):
            if not (bar_start[i] and bar_end[i + L - 1]):
                continue
            run = chords[i:i + L]
            if root_only:
                key = _root_shape_key(run)
            elif shape:
                key = _shape_key(run)
            else:
                key = tuple(c.label for c in run)
            index.setdefault((L, key), []).append(i)
    return index


def find_motifs(chords: list[Chord], *, shape: bool, root_only: bool = False,
                min_len=2, max_len=4, min_count=2) -> list[Motif]:
    """All recurring motifs (count >= min_count), most-compressive first.

    root_only=True: match only on interval sequence, ignoring chord quality.
    Requires shape=True (root_only implies transposition-invariance).
    """
    index = _all_ngrams(chords, min_len, max_len, shape=shape, root_only=root_only)
    motifs: list[Motif] = []
    for (L, key), occ in index.items():
        if len(occ) < min_count:
            continue
        run = chords[occ[0]:occ[0] + L]
        if shape or root_only:
            disp = _shape_display(key, run, root_only=root_only)
            keys = [PC_NAMES[chords[o].root] for o in occ]
        else:
            disp = " ".join(c.label for c in run)
            keys = []
        motifs.append(Motif("shape" if shape else "exact", key, L, occ, disp, keys))
    motifs.sort(key=lambda m: (-m.saving, -m.length))
    return motifs


def reduce_song(chords: list[Chord], *, shape: bool, root_only: bool = False,
                min_len=2, max_len=4, min_count=2,
                _motifs: list[Motif] | None = None) -> tuple[list, list[Motif]]:
    """Greedily tile the song with the most compressive non-overlapping motifs.

    Returns (timeline, legend):
      timeline — ordered list of ("motif", Motif, start) or ("chord", Chord, i)
                 covering every slot exactly once.
      legend   — the distinct motifs actually used, in first-appearance order.

    Pass _motifs to use a pre-computed (and optionally pre-filtered) motif list.
    """
    n = len(chords)
    motifs = (_motifs if _motifs is not None
              else find_motifs(chords, shape=shape, root_only=root_only,
                               min_len=min_len, max_len=max_len, min_count=min_count))
    covered = [False] * n
    placed: dict[int, Motif] = {}        # start index -> motif placed there
    for m in motifs:                     # greedy: biggest saving first
        for s in m.occurrences:
            if any(covered[s:s + m.length]):
                continue
            placed[s] = m
            for k in range(s, s + m.length):
                covered[k] = True

    timeline, used, seen = [], [], set()
    i = 0
    while i < n:
        if i in placed:
            m = placed[i]
            timeline.append(("motif", m, i))
            if m.key not in seen:
                seen.add(m.key)
                used.append(m)
            i += m.length
        else:
            timeline.append(("chord", chords[i], i))
            i += 1
    return timeline, used
