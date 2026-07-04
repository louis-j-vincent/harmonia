"""Hierarchical chord tree — report a chord at the level the evidence supports.

Motivation (validated 2026-07-04, docs/chord_tree_2026-07-04.md): from real
audio, the chord *family* (major/minor/dim/aug/sus) is recovered ~81% of the
time, but the exact chord only ~53%. So the honest thing is to report the
family by default and only descend to the exact quality when the observation
clearly supports it — rather than always committing to one exact chord and
being wrong about half the time.

The tree has three levels:

    Level 1  FAMILY   — decided by the third + fifth:
                        major / minor / diminished / augmented / suspended
    Level 2  SEVENTH  — the base triad-or-seventh chord (extensions stripped):
                        maj, maj7, dom7(7), min7, mMaj7, ø7, °7, sus, 7sus4, …
    Level 3  EXACT    — the full quality incl. 9ths/11ths/13ths/alterations.

A C major triad is the parent of Cmaj7 / C7 / C6; C7 is the parent of C9 /
C7b9. Deeper nodes specify more (quieter) notes.

This module is pure music theory (no data dependency): the maps below are
deterministic. The evidence-gated *reporting* logic lives in
`HierarchicalReporter`, which turns a per-beat chord posterior into a
report-at-the-right-depth label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from harmonia.theory.chord_vocabulary import (
    SEMITONE_NAMES,
    ChordQuality,
    chord_label,
)


class Family(str, Enum):
    """Level-1 chord family, decided by the third and fifth."""

    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUSPENDED = "sus"
    NO_CHORD = "N"


# ── Level 1: quality → family ─────────────────────────────────────────────────
_FAMILY: dict[ChordQuality, Family] = {
    # major third + perfect fifth
    ChordQuality.MAJOR: Family.MAJOR,
    ChordQuality.MAJ7: Family.MAJOR,
    ChordQuality.DOM7: Family.MAJOR,
    ChordQuality.MAJ9: Family.MAJOR,
    ChordQuality.DOM9: Family.MAJOR,
    ChordQuality.DOM7B9: Family.MAJOR,
    ChordQuality.DOM7S9: Family.MAJOR,
    ChordQuality.MAJ9S11: Family.MAJOR,
    ChordQuality.DOM7S11: Family.MAJOR,
    ChordQuality.DOM7B9S11: Family.MAJOR,
    ChordQuality.MAJ13: Family.MAJOR,
    ChordQuality.DOM13: Family.MAJOR,
    ChordQuality.DOM13B9: Family.MAJOR,
    # minor third + perfect fifth
    ChordQuality.MINOR: Family.MINOR,
    ChordQuality.MIN7: Family.MINOR,
    ChordQuality.MIN_MAJ7: Family.MINOR,
    ChordQuality.MIN9: Family.MINOR,
    ChordQuality.MIN11: Family.MINOR,
    ChordQuality.MIN13: Family.MINOR,
    # minor third + diminished fifth
    ChordQuality.DIMINISHED: Family.DIMINISHED,
    ChordQuality.HALF_DIM7: Family.DIMINISHED,
    ChordQuality.DIM7: Family.DIMINISHED,
    # major third + augmented fifth
    ChordQuality.AUGMENTED: Family.AUGMENTED,
    ChordQuality.AUG_MAJ7: Family.AUGMENTED,
    ChordQuality.AUG7: Family.AUGMENTED,
    # no third (2nd/4th instead)
    ChordQuality.SUS2: Family.SUSPENDED,
    ChordQuality.SUS4: Family.SUSPENDED,
    ChordQuality.DOM7SUS4: Family.SUSPENDED,
    ChordQuality.DOM9SUS4: Family.SUSPENDED,
    # special
    ChordQuality.NO_CHORD: Family.NO_CHORD,
}

# ── Level 2: quality → base triad-or-seventh chord (extensions stripped) ──────
_BASE_SEVENTH: dict[ChordQuality, ChordQuality] = {
    # major family
    ChordQuality.MAJOR: ChordQuality.MAJOR,
    ChordQuality.MAJ7: ChordQuality.MAJ7,
    ChordQuality.MAJ9: ChordQuality.MAJ7,
    ChordQuality.MAJ9S11: ChordQuality.MAJ7,
    ChordQuality.MAJ13: ChordQuality.MAJ7,
    ChordQuality.DOM7: ChordQuality.DOM7,
    ChordQuality.DOM9: ChordQuality.DOM7,
    ChordQuality.DOM7B9: ChordQuality.DOM7,
    ChordQuality.DOM7S9: ChordQuality.DOM7,
    ChordQuality.DOM7S11: ChordQuality.DOM7,
    ChordQuality.DOM7B9S11: ChordQuality.DOM7,
    ChordQuality.DOM13: ChordQuality.DOM7,
    ChordQuality.DOM13B9: ChordQuality.DOM7,
    # minor family
    ChordQuality.MINOR: ChordQuality.MINOR,
    ChordQuality.MIN7: ChordQuality.MIN7,
    ChordQuality.MIN9: ChordQuality.MIN7,
    ChordQuality.MIN11: ChordQuality.MIN7,
    ChordQuality.MIN13: ChordQuality.MIN7,
    ChordQuality.MIN_MAJ7: ChordQuality.MIN_MAJ7,
    # diminished family
    ChordQuality.DIMINISHED: ChordQuality.DIMINISHED,
    ChordQuality.HALF_DIM7: ChordQuality.HALF_DIM7,
    ChordQuality.DIM7: ChordQuality.DIM7,
    # augmented family
    ChordQuality.AUGMENTED: ChordQuality.AUGMENTED,
    ChordQuality.AUG7: ChordQuality.AUG7,
    ChordQuality.AUG_MAJ7: ChordQuality.AUG_MAJ7,
    # suspended family
    ChordQuality.SUS2: ChordQuality.SUS4,
    ChordQuality.SUS4: ChordQuality.SUS4,
    ChordQuality.DOM7SUS4: ChordQuality.DOM7SUS4,
    ChordQuality.DOM9SUS4: ChordQuality.DOM7SUS4,
    # special
    ChordQuality.NO_CHORD: ChordQuality.NO_CHORD,
}

# label shown at family (level-1) granularity, per family
_FAMILY_SUFFIX: dict[Family, str] = {
    Family.MAJOR: "",       # "C"
    Family.MINOR: "m",      # "Cm"
    Family.DIMINISHED: "dim",
    Family.AUGMENTED: "aug",
    Family.SUSPENDED: "sus",
    Family.NO_CHORD: "",
}


def family_of(quality: ChordQuality) -> Family:
    """Level-1 family of a chord quality."""
    return _FAMILY[quality]


def base_seventh_of(quality: ChordQuality) -> ChordQuality:
    """Level-2 base triad-or-seventh chord (9ths/11ths/13ths/alterations stripped)."""
    return _BASE_SEVENTH[quality]


def family_label(root: int, quality: ChordQuality) -> str:
    """Human-readable label at family (level-1) granularity, e.g. 'C', 'Am', 'Bdim'."""
    fam = family_of(quality)
    if fam == Family.NO_CHORD:
        return "N"
    return f"{SEMITONE_NAMES[root % 12]}{_FAMILY_SUFFIX[fam]}"


def label_at_depth(root: int, quality: ChordQuality, depth: int) -> str:
    """Chord label generalized to a tree depth: 1=family, 2=base seventh, 3=exact."""
    if depth <= 1:
        return family_label(root, quality)
    if depth == 2:
        return chord_label(root, base_seventh_of(quality))
    return chord_label(root, quality)


@dataclass
class HierarchicalReport:
    depth: int          # 1 (family), 2 (seventh), or 3 (exact)
    label: str          # the reported label at that depth
    family: str         # always the level-1 family label


class HierarchicalReporter:
    """Turns a per-beat chord posterior into a report at the confident depth.

    Given the decoded chord and how the observation's probability mass is
    distributed over the vocabulary, descend the tree only while the child
    node holds at least `confidence` of its parent's mass. Family is always the
    floor — we never report less than the family.

    `confidence` ∈ (0, 1]: fraction of the parent level's posterior mass the
    child must retain to justify descending. 0.5 = "the specific seventh must
    account for at least half the family's evidence". Higher = more
    conservative (shallower reports). 1.0 effectively pins output to family.
    """

    def __init__(self, idx_to_chord: list[tuple[int, ChordQuality]],
                 confidence: float = 0.5):
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        self.confidence = confidence
        self._idx_to_chord = idx_to_chord
        # group id per vocab index, keyed on (root, family) and (root, base-seventh)
        self._family_key = np.array(
            [hash((r, family_of(q))) for r, q in idx_to_chord]
        )
        self._seventh_key = np.array(
            [hash((r, base_seventh_of(q))) for r, q in idx_to_chord]
        )

    def report(self, chord_idx: int, mean_posterior: np.ndarray) -> HierarchicalReport:
        """
        Args:
            chord_idx:       decoded vocabulary index for this run.
            mean_posterior:  (C,) mean per-beat emission posterior over the run.
        """
        root, quality = self._idx_to_chord[chord_idx]
        family = family_label(root, quality)
        if quality == ChordQuality.NO_CHORD:
            return HierarchicalReport(depth=1, label="N", family="N")

        fam_mass = float(mean_posterior[self._family_key == self._family_key[chord_idx]].sum())
        sev_mass = float(mean_posterior[self._seventh_key == self._seventh_key[chord_idx]].sum())
        exact_mass = float(mean_posterior[chord_idx])

        depth = 1
        # descend to the seventh if it holds enough of the family's mass
        if fam_mass > 1e-9 and sev_mass / fam_mass >= self.confidence:
            depth = 2
            # descend to the exact chord if it holds enough of the seventh's mass
            if sev_mass > 1e-9 and exact_mass / sev_mass >= self.confidence:
                depth = 3
        return HierarchicalReport(
            depth=depth,
            label=label_at_depth(root, quality, depth),
            family=family,
        )
