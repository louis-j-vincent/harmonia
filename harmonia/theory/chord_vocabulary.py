"""
Chord vocabulary, templates, and hierarchy.

Chords are represented as interval sets relative to the root (in semitones).
Soft weights encode the salience of each interval in the acoustic spectrum —
the 3rd and 7th carry the most harmonic identity, the 5th is often weak/absent.

Vocabulary phases mirror the model extension roadmap:
  Phase 1: triads + 7ths + sus          (~121 chord types)
  Phase 2: + 9ths / altered dominants   (~181 chord types)
  Phase 3: + 11ths                      (~217 chord types)
  Phase 4: + 13ths                      (~241 chord types)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

SEMITONE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_class(semitones: int) -> int:
    return semitones % 12


# ---------------------------------------------------------------------------
# Chord quality taxonomy
# ---------------------------------------------------------------------------

class ChordQuality(str, Enum):
    # --- Triads ---
    MAJOR       = "maj"
    MINOR       = "min"
    DIMINISHED  = "dim"
    AUGMENTED   = "aug"
    # --- Suspended ---
    SUS2        = "sus2"
    SUS4        = "sus4"
    # --- Seventh chords ---
    MAJ7        = "maj7"
    MIN7        = "min7"
    DOM7        = "7"
    MIN_MAJ7    = "mMaj7"    # minor-major 7th (e.g. melodic minor I)
    HALF_DIM7   = "ø7"       # m7b5
    DIM7        = "°7"       # fully diminished
    AUG_MAJ7    = "augMaj7"
    AUG7        = "aug7"     # 7#5
    DOM7SUS4    = "7sus4"
    # --- Phase 2: Ninths ---
    MAJ9        = "maj9"
    MIN9        = "min9"
    DOM9        = "9"
    DOM7B9      = "7b9"
    DOM7S9      = "7#9"      # Hendrix chord
    DOM9SUS4    = "9sus4"
    # --- Phase 3: Elevenths ---
    MAJ9S11     = "maj9#11"  # Lydian tonic
    MIN11       = "min11"
    DOM7S11     = "7#11"     # Lydian dominant
    DOM7B9S11   = "7b9#11"
    # --- Phase 4: Thirteenths ---
    MAJ13       = "maj13"
    MIN13       = "min13"
    DOM13       = "13"
    DOM13B9     = "13b9"
    # --- Special ---
    NO_CHORD    = "N"


# ---------------------------------------------------------------------------
# Chord template
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChordTemplate:
    """
    Defines the acoustic fingerprint of a chord quality.

    intervals: frozenset of semitone offsets from the root that are present.
    weights:   dict mapping each interval to its salience weight ∈ (0, 1].
               Used as emission likelihood P(note_activated | chord).
               The 3rd and 7th are the highest-weight intervals — they define
               chord quality. The 5th is often omitted in jazz and weighted lower.
    phase:     vocabulary phase this quality belongs to.
    """
    quality: ChordQuality
    intervals: FrozenSet[int]
    weights: dict[int, float]
    phase: int = 1
    description: str = ""

    def to_weight_vector(self, n_pitch_classes: int = 12) -> list[float]:
        """Return a 12-element chroma weight vector (root = index 0)."""
        vec = [0.0] * n_pitch_classes
        for interval, w in self.weights.items():
            vec[interval % n_pitch_classes] = w
        return vec


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------
# Weight conventions:
#   1.0  — essential, almost always audible
#   0.85 — strongly present
#   0.6  — present, harmonic colour
#   0.35 — optional / sometimes omitted (e.g. 5th in jazz)
#   0.15 — upper extension, soft presence

_T = ChordTemplate  # alias for brevity

CHORD_TEMPLATES: dict[ChordQuality, ChordTemplate] = {

    # ---- Triads (Phase 1) ----
    ChordQuality.MAJOR: _T(
        quality=ChordQuality.MAJOR,
        intervals=frozenset({0, 4, 7}),
        weights={0: 1.0, 4: 0.85, 7: 0.35},
        phase=1, description="Major triad",
    ),
    ChordQuality.MINOR: _T(
        quality=ChordQuality.MINOR,
        intervals=frozenset({0, 3, 7}),
        weights={0: 1.0, 3: 0.85, 7: 0.35},
        phase=1, description="Minor triad",
    ),
    ChordQuality.DIMINISHED: _T(
        quality=ChordQuality.DIMINISHED,
        intervals=frozenset({0, 3, 6}),
        weights={0: 1.0, 3: 0.85, 6: 0.85},
        phase=1, description="Diminished triad",
    ),
    ChordQuality.AUGMENTED: _T(
        quality=ChordQuality.AUGMENTED,
        intervals=frozenset({0, 4, 8}),
        weights={0: 1.0, 4: 0.85, 8: 0.85},
        phase=1, description="Augmented triad",
    ),

    # ---- Suspended (Phase 1) ----
    ChordQuality.SUS2: _T(
        quality=ChordQuality.SUS2,
        intervals=frozenset({0, 2, 7}),
        weights={0: 1.0, 2: 0.85, 7: 0.35},
        phase=1, description="Suspended 2nd",
    ),
    ChordQuality.SUS4: _T(
        quality=ChordQuality.SUS4,
        intervals=frozenset({0, 5, 7}),
        weights={0: 1.0, 5: 0.85, 7: 0.35},
        phase=1, description="Suspended 4th",
    ),

    # ---- 7th chords (Phase 1) ----
    ChordQuality.MAJ7: _T(
        quality=ChordQuality.MAJ7,
        intervals=frozenset({0, 4, 7, 11}),
        weights={0: 1.0, 4: 0.85, 7: 0.3, 11: 0.85},
        phase=1, description="Major 7th — Ionian / Lydian tonic",
    ),
    ChordQuality.MIN7: _T(
        quality=ChordQuality.MIN7,
        intervals=frozenset({0, 3, 7, 10}),
        weights={0: 1.0, 3: 0.85, 7: 0.3, 10: 0.85},
        phase=1, description="Minor 7th — Dorian / Aeolian ii or vi",
    ),
    ChordQuality.DOM7: _T(
        quality=ChordQuality.DOM7,
        intervals=frozenset({0, 4, 7, 10}),
        weights={0: 1.0, 4: 0.85, 7: 0.3, 10: 0.85},
        phase=1, description="Dominant 7th — V7, strong resolution pull",
    ),
    ChordQuality.MIN_MAJ7: _T(
        quality=ChordQuality.MIN_MAJ7,
        intervals=frozenset({0, 3, 7, 11}),
        weights={0: 1.0, 3: 0.85, 7: 0.3, 11: 0.85},
        phase=1, description="Minor-major 7th — melodic minor I",
    ),
    ChordQuality.HALF_DIM7: _T(
        quality=ChordQuality.HALF_DIM7,
        intervals=frozenset({0, 3, 6, 10}),
        weights={0: 1.0, 3: 0.85, 6: 0.85, 10: 0.85},
        phase=1, description="Half-diminished (m7b5) — viiø in major, iiø in minor",
    ),
    ChordQuality.DIM7: _T(
        quality=ChordQuality.DIM7,
        intervals=frozenset({0, 3, 6, 9}),
        weights={0: 1.0, 3: 0.85, 6: 0.85, 9: 0.85},
        phase=1, description="Fully diminished 7th — symmetric, 3 inversions are enharmonic",
    ),
    ChordQuality.AUG_MAJ7: _T(
        quality=ChordQuality.AUG_MAJ7,
        intervals=frozenset({0, 4, 8, 11}),
        weights={0: 1.0, 4: 0.85, 8: 0.85, 11: 0.85},
        phase=1, description="Augmented major 7th — melodic minor III",
    ),
    ChordQuality.AUG7: _T(
        quality=ChordQuality.AUG7,
        intervals=frozenset({0, 4, 8, 10}),
        weights={0: 1.0, 4: 0.85, 8: 0.85, 10: 0.85},
        phase=1, description="Augmented dominant 7th (7#5)",
    ),
    ChordQuality.DOM7SUS4: _T(
        quality=ChordQuality.DOM7SUS4,
        intervals=frozenset({0, 5, 7, 10}),
        weights={0: 1.0, 5: 0.85, 7: 0.3, 10: 0.85},
        phase=1, description="Dominant 7th sus4 — unresolved suspension",
    ),

    # ---- 9ths (Phase 2) ----
    ChordQuality.MAJ9: _T(
        quality=ChordQuality.MAJ9,
        intervals=frozenset({0, 4, 7, 11, 14}),
        weights={0: 1.0, 4: 0.85, 7: 0.25, 11: 0.85, 14: 0.6},
        phase=2, description="Major 9th",
    ),
    ChordQuality.MIN9: _T(
        quality=ChordQuality.MIN9,
        intervals=frozenset({0, 3, 7, 10, 14}),
        weights={0: 1.0, 3: 0.85, 7: 0.25, 10: 0.85, 14: 0.6},
        phase=2, description="Minor 9th",
    ),
    ChordQuality.DOM9: _T(
        quality=ChordQuality.DOM9,
        intervals=frozenset({0, 4, 7, 10, 14}),
        weights={0: 1.0, 4: 0.85, 7: 0.25, 10: 0.85, 14: 0.6},
        phase=2, description="Dominant 9th",
    ),
    ChordQuality.DOM7B9: _T(
        quality=ChordQuality.DOM7B9,
        intervals=frozenset({0, 4, 7, 10, 13}),
        weights={0: 1.0, 4: 0.85, 7: 0.25, 10: 0.85, 13: 0.6},
        phase=2, description="Dominant 7b9 — strong tension, resolves to minor tonic",
    ),
    ChordQuality.DOM7S9: _T(
        quality=ChordQuality.DOM7S9,
        intervals=frozenset({0, 4, 7, 10, 15}),
        weights={0: 1.0, 4: 0.85, 7: 0.25, 10: 0.85, 15: 0.6},
        phase=2, description="Dominant 7#9 — Hendrix chord, blues/funk dominant",
    ),
    ChordQuality.DOM9SUS4: _T(
        quality=ChordQuality.DOM9SUS4,
        intervals=frozenset({0, 5, 7, 10, 14}),
        weights={0: 1.0, 5: 0.85, 7: 0.25, 10: 0.85, 14: 0.6},
        phase=2, description="9sus4 — very common in modern jazz / neo-soul",
    ),

    # ---- 11ths (Phase 3) ----
    ChordQuality.MAJ9S11: _T(
        quality=ChordQuality.MAJ9S11,
        intervals=frozenset({0, 4, 7, 11, 14, 18}),
        weights={0: 1.0, 4: 0.85, 7: 0.2, 11: 0.85, 14: 0.5, 18: 0.5},
        phase=3, description="Major 9#11 — Lydian tonic, signature Herbie Hancock sound",
    ),
    ChordQuality.MIN11: _T(
        quality=ChordQuality.MIN11,
        intervals=frozenset({0, 3, 7, 10, 14, 17}),
        weights={0: 1.0, 3: 0.85, 7: 0.2, 10: 0.85, 14: 0.5, 17: 0.5},
        phase=3, description="Minor 11th — Dorian ii or vi",
    ),
    ChordQuality.DOM7S11: _T(
        quality=ChordQuality.DOM7S11,
        intervals=frozenset({0, 4, 7, 10, 14, 18}),
        weights={0: 1.0, 4: 0.85, 7: 0.2, 10: 0.85, 14: 0.5, 18: 0.5},
        phase=3, description="Lydian dominant (7#11) — tritone sub territory",
    ),
    ChordQuality.DOM7B9S11: _T(
        quality=ChordQuality.DOM7B9S11,
        intervals=frozenset({0, 4, 7, 10, 13, 18}),
        weights={0: 1.0, 4: 0.85, 7: 0.2, 10: 0.85, 13: 0.5, 18: 0.4},
        phase=3, description="7b9#11 — altered dominant, maximum tension",
    ),

    # ---- 13ths (Phase 4) ----
    ChordQuality.MAJ13: _T(
        quality=ChordQuality.MAJ13,
        intervals=frozenset({0, 4, 7, 11, 14, 18, 21}),
        weights={0: 1.0, 4: 0.85, 7: 0.15, 11: 0.85, 14: 0.4, 18: 0.4, 21: 0.5},
        phase=4, description="Major 13th",
    ),
    ChordQuality.MIN13: _T(
        quality=ChordQuality.MIN13,
        intervals=frozenset({0, 3, 7, 10, 14, 17, 21}),
        weights={0: 1.0, 3: 0.85, 7: 0.15, 10: 0.85, 14: 0.4, 17: 0.4, 21: 0.5},
        phase=4, description="Minor 13th",
    ),
    ChordQuality.DOM13: _T(
        quality=ChordQuality.DOM13,
        intervals=frozenset({0, 4, 7, 10, 14, 21}),
        weights={0: 1.0, 4: 0.85, 7: 0.15, 10: 0.85, 14: 0.4, 21: 0.5},
        phase=4, description="Dominant 13th",
    ),
    ChordQuality.DOM13B9: _T(
        quality=ChordQuality.DOM13B9,
        intervals=frozenset({0, 4, 7, 10, 13, 21}),
        weights={0: 1.0, 4: 0.85, 7: 0.15, 10: 0.85, 13: 0.5, 21: 0.4},
        phase=4, description="13b9 — altered dominant with added 13th colour",
    ),

    # ---- No chord ----
    ChordQuality.NO_CHORD: _T(
        quality=ChordQuality.NO_CHORD,
        intervals=frozenset(),
        weights={},
        phase=1, description="Silence / no harmonic content",
    ),
}


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def get_vocabulary(max_phase: int = 1) -> list[ChordQuality]:
    """Return all chord qualities up to and including the given phase."""
    return [q for q, t in CHORD_TEMPLATES.items() if t.phase <= max_phase]


def get_template(quality: ChordQuality) -> ChordTemplate:
    return CHORD_TEMPLATES[quality]


def chord_label(root: int, quality: ChordQuality) -> str:
    """Human-readable chord label, e.g. 'Gmaj7', 'Bbø7'."""
    if quality == ChordQuality.NO_CHORD:
        return "N"
    return f"{SEMITONE_NAMES[root % 12]}{quality.value}"


def n_chords(max_phase: int = 1) -> int:
    """Total number of (root, quality) pairs in vocabulary, plus N."""
    qualities = [q for q in get_vocabulary(max_phase) if q != ChordQuality.NO_CHORD]
    return 12 * len(qualities) + 1  # +1 for NO_CHORD


# ---------------------------------------------------------------------------
# Chord index <-> (root, quality) bijection
# ---------------------------------------------------------------------------

def build_index(max_phase: int = 1) -> tuple[list[tuple[int, ChordQuality]], dict]:
    """
    Returns:
        idx_to_chord: list mapping integer index → (root, quality)
        chord_to_idx: dict mapping (root, quality) → integer index
    """
    qualities = [q for q in get_vocabulary(max_phase) if q != ChordQuality.NO_CHORD]
    idx_to_chord: list[tuple[int, ChordQuality]] = []
    for root in range(12):
        for q in qualities:
            idx_to_chord.append((root, q))
    idx_to_chord.append((-1, ChordQuality.NO_CHORD))  # last index = N
    chord_to_idx = {c: i for i, c in enumerate(idx_to_chord)}
    return idx_to_chord, chord_to_idx
