"""harmonia_min/labels.py — musx chord labels → the app UI's chord fields.

The re-decode emits labels from the clone's submission_chord_list.txt: 25
C-rooted templates × 12 roots + "N". Verified against the actual file
(harmonia/third_party/.../data/submission_chord_list.txt, 2026-07-30):

    min/b7 min/2 maj/b7 maj/2 sus4(b7) sus2 sus4 13 11 min9 9 maj9
    dim7 hdim7 min7 7 maj7 min/5 min/b3 maj/5 maj/3 dim aug min maj

The app UI (app_shell.html) consumes iReal quality *tails* (its TOK table:
"" - 7 ^7 -7 o o7 h7 + sus4 9 -9 ^9 13 …) plus an optional sounding-bass
pitch class for slash chords. This module is that one mapping, nothing else.
"""
from __future__ import annotations

_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# musx quality → (iReal tail, bass interval in semitones above root or None).
# Inversions keep the PARENT quality as the tail; the bass renders as "/X".
_QUAL = {
    "maj":      ("",     None),
    "min":      ("-",    None),
    "7":        ("7",    None),
    "maj7":     ("^7",   None),
    "min7":     ("-7",   None),
    "dim":      ("o",    None),
    "dim7":     ("o7",   None),
    "hdim7":    ("h7",   None),
    "aug":      ("+",    None),
    "sus2":     ("sus2", None),
    "sus4":     ("sus4", None),
    "sus4(b7)": ("7sus4", None),
    "9":        ("9",    None),
    "min9":     ("-9",   None),
    "maj9":     ("^9",   None),
    "11":       ("7sus4", None),   # C11 sounds as C9sus — nearest iReal tail
    "13":       ("13",   None),
    "maj/3":    ("",     4),
    "maj/5":    ("",     7),
    "maj/b7":   ("7",    10),      # maj triad over its b7 ≈ dominant inversion
    "maj/2":    ("",     2),
    "min/b3":   ("-",    3),
    "min/5":    ("-",    7),
    "min/b7":   ("-7",   10),
    "min/2":    ("-",    2),
}


def parse_root(sym: str) -> int:
    """'Bb' -> 10, 'F#' -> 6, 'C' -> 0."""
    pc = _NOTE_PC[sym[0].upper()]
    for ch in sym[1:]:
        if ch in "#s":
            pc += 1
        elif ch in "b!":
            pc -= 1
    return pc % 12


def to_chord(label: str) -> dict | None:
    """musx label -> {root, q, bass} (bass=-1 when root position), or None for N.

    Unknown qualities raise: the vocabulary is closed (26 entries, verified),
    so an unknown here means the decoder changed under us — fail loudly, never
    default to maj (the pop909 parser did exactly that, silently, until
    2026-07-30).
    """
    if label in ("N", "X", ""):
        return None
    root_s, _, qual = label.partition(":")
    root = parse_root(root_s)
    if qual not in _QUAL:
        raise ValueError(f"harmonia_min.labels: unknown musx quality {label!r}")
    tail, bass_iv = _QUAL[qual]
    return {"root": root,
            "q": tail,
            "bass": -1 if bass_iv is None else (root + bass_iv) % 12}


# Pitch classes each quality sounds (for key inference from decoded chords).
_TRIAD_PCS = {
    "": (0, 4, 7), "-": (0, 3, 7), "o": (0, 3, 6), "+": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}
_TAIL_PCS = {
    "7": (0, 4, 7, 10), "^7": (0, 4, 7, 11), "-7": (0, 3, 7, 10),
    "o7": (0, 3, 6, 9), "h7": (0, 3, 6, 10), "7sus4": (0, 5, 7, 10),
    "9": (0, 4, 7, 10, 2), "-9": (0, 3, 7, 10, 2), "^9": (0, 4, 7, 11, 2),
    "13": (0, 4, 7, 10, 2, 9),
}


def chord_pcs(root: int, q: str) -> tuple[int, ...]:
    """Sounding pitch classes of a chord, for chroma-style accumulation."""
    rel = _TAIL_PCS.get(q) or _TRIAD_PCS.get(q) or (0, 4, 7)
    return tuple((root + r) % 12 for r in rel)
