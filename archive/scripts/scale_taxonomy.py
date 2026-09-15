"""
Atomic scale taxonomy: 2026-07-03 follow-up to the bigram/canonicalization
work in plot_structure_proposal_illustrations.py.

The core idea (from discussion, not derived independently): the 7 so-called
"church modes" (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian,
Locrian) are NOT seven different scales. They are the same 7-note pitch-class
COLLECTION, just with a different member of that collection picked out as
"home". "D Dorian" and "C Ionian" (=C major) are literally the same 7 notes.
So the only genuinely atomic, transposition-covering objects are:

  1. A small number of distinct 7-note (or symmetric 6/8-note) COLLECTIONS,
     each with 12 transpositions (or fewer, for symmetric ones) -- the
     "family". A family is defined once, by its interval pattern from an
     arbitrary reference point (we use the family's own Ionian-equivalent
     "mode 1" as that reference, by convention, not because it's privileged).
  2. Within a family, a MODAL CENTRE -- which member of the collection is
     currently felt as tonic. This is a separate, softer, more slowly-varying
     piece of state from the collection itself.

Consequence used throughout this module: "is this chord diatonic" is a
property of (which family, which transposition of that family) alone --
it does NOT depend on the modal centre. The by-scale-degree triad-quality
table only needs to be built ONCE per family (anchored at that family's own
mode-1 reference), then re-indexed for any transposition. Natural minor
being "diatonic" to the relative major and harmonic minor's raised-7th V
chord being diatonic to the SAME-tonic harmonic-minor collection both fall
out of this for free, rather than needing separately hand-maintained tables
(which is what the first pass at this analysis did, and which -- checked
directly, see the module docstring bottom -- turns out to have been
numerically equivalent, just duplicated).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Families: interval pattern (semitones from that family's own mode-1
# reference tonic), plus the standard mode names for each rotation.
# ---------------------------------------------------------------------------

MAJOR_FAMILY = [0, 2, 4, 5, 7, 9, 11]
MAJOR_FAMILY_MODES = ["Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"]
# Aeolian (natural minor) = mode 6, i.e. tonic at MAJOR_FAMILY[5] = 9
# semitones above the family's own mode-1 (Ionian/major) reference tonic --
# this is exactly the "relative major is +3" relationship, seen from the
# other direction (minor tonic = major tonic + 9, equivalently major tonic
# = minor tonic + 3).

HARMONIC_MINOR_FAMILY = [0, 2, 3, 5, 7, 8, 11]
HARMONIC_MINOR_FAMILY_MODES = [
    "Harmonic minor", "Locrian #6", "Ionian #5", "Dorian #4",
    "Phrygian dominant", "Lydian #2", "Superlocrian bb7",
]
# Mode 1 (Harmonic minor itself) tonic = the family's own reference tonic,
# no shift -- unlike the major family, harmonic minor's most commonly used
# mode (as "the" minor scale for cadential/harmonic purposes, mode 1) sits
# AT the reference point, not offset from it. Mode 5 (Phrygian dominant,
# tonic at HARMONIC_MINOR_FAMILY[4] = 7 above the reference) is the one that
# matters most for pop/tonal harmony: it's what makes "the harmonic-minor
# V chord" (major triad on the 5th degree of an otherwise-minor scale) a
# genuinely diatonic member of this family, not a chromatic alteration.

MELODIC_MINOR_FAMILY = [0, 2, 3, 5, 7, 9, 11]
MELODIC_MINOR_FAMILY_MODES = [
    "Melodic minor", "Dorian b2", "Lydian augmented", "Lydian dominant",
    "Mixolydian b6", "Locrian #2 (half-diminished)", "Altered (Superlocrian)",
]
# Documented, NOT wired into classify_membership() yet -- mode 4 (Lydian
# dominant) and mode 7 (Altered) are real, common jazz sounds, but pop
# music (POP909) is expected to draw on this rarely; deferred rather than
# guessed at without checking prevalence first (see "Not yet implemented"
# below).

WHOLE_TONE_FAMILY = [0, 2, 4, 6, 8, 10]
# 6 notes, symmetric under transposition by 2 semitones -- only 2 distinct
# transpositions exist (even-rooted vs odd-rooted). No triads in the normal
# major/minor/dim/aug sense build cleanly from stacked major thirds (every
# triad you can stack within it is augmented). Documented, not implemented.

OCTATONIC_WH_FAMILY = [0, 2, 3, 5, 6, 8, 9, 11]  # whole-half diminished
OCTATONIC_HW_FAMILY = [0, 1, 3, 4, 6, 7, 9, 10]  # half-whole diminished
# 8 notes, symmetric under transposition by 3 semitones -- only 3 distinct
# transpositions of each. Documented, not implemented. (The half-whole form
# is the other rotation, included for completeness -- whole-half and
# half-whole are different pitch-class sets, not modes of one another,
# since the pattern isn't a simple rotation for this one.)


# ---------------------------------------------------------------------------
# Diatonic triad-quality-by-position tables, derived by stacking thirds --
# computed once per family, not per mode (see module docstring for why).
# ---------------------------------------------------------------------------

def precise_triad_quality(quality) -> str:
    """maj / min / dim / aug / sus / other, checking third AND fifth (not
    just third) so diminished triads/7ths are never confused with plain
    minor ones. 7th/extension tones are ignored -- classification is by the
    underlying TRIAD (e.g. dom7 -> maj, min7 -> min, hdim7 -> dim)."""
    from harmonia.theory.chord_vocabulary import get_template
    t = get_template(quality)
    has = t.intervals
    if 4 in has and 7 in has:
        return "maj"
    if 3 in has and 7 in has:
        return "min"
    if 3 in has and 6 in has:
        return "dim"
    if 4 in has and 8 in has:
        return "aug"
    if 7 in has and 3 not in has and 4 not in has:
        return "sus"
    return "other"


def _stack_thirds_diatonic_triads(scale: list[int]) -> dict[int, str]:
    """{interval_from_the_family's_own_mode1_reference_tonic: triad_quality},
    by stacking thirds on each scale degree -- computed, not
    hand-transcribed, to avoid exactly the kind of sign error that's easy to
    make by hand here."""
    n = len(scale)
    table = {}
    for i in range(n):
        root = scale[i]
        third = scale[(i + 2) % n] + (12 if (i + 2) >= n else 0)
        fifth = scale[(i + 4) % n] + (12 if (i + 4) >= n else 0)
        rel_third, rel_fifth = (third - root) % 12, (fifth - root) % 12
        if rel_third == 4 and rel_fifth == 7:
            q = "maj"
        elif rel_third == 3 and rel_fifth == 7:
            q = "min"
        elif rel_third == 3 and rel_fifth == 6:
            q = "dim"
        elif rel_third == 4 and rel_fifth == 8:
            q = "aug"
        else:
            q = "other"
        table[root] = q
    return table


DIATONIC_MAJOR_FAMILY = _stack_thirds_diatonic_triads(MAJOR_FAMILY)
# {0:maj, 2:min, 4:min, 5:maj, 7:maj, 9:min, 11:dim} -- I=maj,ii=min,iii=min,
# IV=maj,V=maj,vi=min,vii°=dim. Re-indexing this table at reference+9 gives
# exactly the natural-minor-by-its-own-tonic table used in the first pass
# of this analysis (verified: DIATONIC_MAJOR_FAMILY[(x-3)%12] for x in the
# old hand-built DIATONIC_MINOR table matches it at every position).

DIATONIC_HARMONIC_MINOR_FAMILY = _stack_thirds_diatonic_triads(HARMONIC_MINOR_FAMILY)
# {0:min, 2:dim, 3:other(aug), 5:min, 7:maj, 8:maj, 11:dim} -- i=min,
# ii°=dim, III+=aug, iv=min, V=maj (the raised-leading-tone dominant),
# VI=maj, vii°=dim.


# ---------------------------------------------------------------------------
# Membership classification
# ---------------------------------------------------------------------------

def classify_membership(interval: int, quality, song_mode: str = "major") -> str:
    """
    Where does this (interval-from-the-SONG's-OWN-annotated-tonic, quality)
    chord actually belong? song_mode is "major" or "minor" (the song's own
    annotated mode from key_audio.txt) -- used only to decide which anchor
    counts as "own" vs "borrowed", not to gate which tables are checked.

    Returns one of:
      "diatonic_own"        -- diatonic to the song's own annotated mode
                                (major-family at the song's own tonic if
                                major-annotated; major-family at tonic+3,
                                i.e. natural minor, if minor-annotated)
      "parallel_borrow"      -- diatonic to the PARALLEL other mode at the
                                SAME tonic (major-family at tonic+3 for a
                                major-annotated song = borrowing from the
                                parallel natural minor, e.g. bVI/bVII/iv;
                                major-family at tonic for a minor-annotated
                                song = borrowing the parallel major, e.g.
                                a major IV or bIII)
      "harmonic_minor_borrow" -- diatonic to the harmonic-minor family
                                anchored at the SAME tonic (no transposition
                                -- harmonic minor's mode-1 tonic IS the
                                song's own tonic either way). Chiefly
                                catches the raised-leading-tone V chord
                                (major triad on the 5th degree) that a plain
                                natural-minor table would call "chromatic".
      "sus"                  -- suspended (no third) -- doesn't structurally
                                conflict with anything, not counted as
                                chromatic.
      "chromatic"             -- matches none of the above -- secondary
                                dominants, melodic-minor colour, tritone
                                subs, or a real key change/error.
    """
    tq = precise_triad_quality(quality)
    if tq == "sus":
        return "sus"

    own_anchor = 0 if song_mode == "major" else 3
    other_anchor = 3 if song_mode == "major" else 0

    if DIATONIC_MAJOR_FAMILY.get((interval - own_anchor) % 12) == tq:
        return "diatonic_own"
    if DIATONIC_MAJOR_FAMILY.get((interval - other_anchor) % 12) == tq:
        return "parallel_borrow"
    if DIATONIC_HARMONIC_MINOR_FAMILY.get(interval) == tq:
        return "harmonic_minor_borrow"
    return "chromatic"


# ---------------------------------------------------------------------------
# Sanity checks (run directly: `python scripts/scale_taxonomy.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from harmonia.theory.chord_vocabulary import ChordQuality

    # Old hand-built DIATONIC_MINOR table (session-earlier version) should
    # be recoverable exactly by re-indexing DIATONIC_MAJOR_FAMILY at +3.
    OLD_DIATONIC_MINOR = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}
    for interval, q in OLD_DIATONIC_MINOR.items():
        derived = DIATONIC_MAJOR_FAMILY.get((interval - 3) % 12)
        assert derived == q, f"mismatch at {interval}: old={q} derived={derived}"
    print("OK: old hand-built natural-minor table == DIATONIC_MAJOR_FAMILY re-indexed at +3")

    print("classify_membership(5, MINOR, song_mode='major') ->",
          classify_membership(5, ChordQuality.MINOR, "major"), "(expect parallel_borrow: iv)")
    print("classify_membership(4, MAJOR, song_mode='major') ->",
          classify_membership(4, ChordQuality.MAJOR, "major"), "(expect chromatic: V/vi)")
    print("classify_membership(7, MAJOR, song_mode='minor') ->",
          classify_membership(7, ChordQuality.MAJOR, "minor"),
          "(expect harmonic_minor_borrow: the raised-7th V)")
    print("classify_membership(8, MAJOR, song_mode='major') ->",
          classify_membership(8, ChordQuality.MAJOR, "major"), "(expect parallel_borrow: bVI)")
