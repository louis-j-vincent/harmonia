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


# common shapes → friendly names (extend freely). Quality tokens are the
# bucket names from analyze_accomp_emission.QUALITY_MAP (min7, dom7, maj7, 6…).
_SHAPE_NAMES = {
    (("min7",), (5, "dom7")): "ii-V",
    (("min7",), (5, "dom7"), (5, "maj7")): "ii-V-I",
    (("min7",), (5, "dom7"), (5, "6")): "ii-V-I",
    (("min7",), (5, "dom7"), (5, "min7"), (5, "dom7")): "ii-V ii-V",
    (("maj7",), (9, "dom7")): "I-VI",
    (("6",), (9, "dom7")): "I-VI",
    (("dom7",), (5, "dom7")): "V/V-V",
    (("dom7",), (5, "dom7"), (5, "dom7")): "dom-cycle",
}


def _shape_display(shape: tuple, chords: list[Chord]) -> str:
    if shape in _SHAPE_NAMES:
        return _SHAPE_NAMES[shape]
    # fall back to the interval/quality signature
    head = chords[0].qual
    tail = " ".join(f"+{iv}{q}" for iv, q in (s for s in shape[1:]))
    return f"[{head} {tail}]".strip()


# ---------------------------------------------------------------------------
# Motif finding
# ---------------------------------------------------------------------------

def _all_ngrams(chords: list[Chord], min_len: int, max_len: int, *, shape: bool):
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
            key = _shape_key(run) if shape else tuple(c.label for c in run)
            index.setdefault((L, key), []).append(i)
    return index


def find_motifs(chords: list[Chord], *, shape: bool, min_len=2, max_len=4,
                min_count=2) -> list[Motif]:
    """All recurring motifs (count >= min_count), most-compressive first."""
    index = _all_ngrams(chords, min_len, max_len, shape=shape)
    motifs: list[Motif] = []
    for (L, key), occ in index.items():
        if len(occ) < min_count:
            continue
        run = chords[occ[0]:occ[0] + L]
        if shape:
            disp = _shape_display(key, run)
            keys = [PC_NAMES[chords[o].root] for o in occ]
        else:
            disp = " ".join(c.label for c in run)
            keys = []
        motifs.append(Motif("shape" if shape else "exact", key, L, occ, disp, keys))
    motifs.sort(key=lambda m: (-m.saving, -m.length))
    return motifs


def reduce_song(chords: list[Chord], *, shape: bool, min_len=2, max_len=4,
                min_count=2) -> tuple[list, list[Motif]]:
    """Greedily tile the song with the most compressive non-overlapping motifs.

    Returns (timeline, legend):
      timeline — ordered list of ("motif", Motif, start) or ("chord", Chord, i)
                 covering every slot exactly once.
      legend   — the distinct motifs actually used, in first-appearance order.
    """
    n = len(chords)
    motifs = find_motifs(chords, shape=shape, min_len=min_len, max_len=max_len,
                         min_count=min_count)
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
