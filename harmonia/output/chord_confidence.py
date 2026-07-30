"""Per-chord display confidence from how often the song repeats that chord.

**Why this replaced the model's own confidence (2026-07-30).** The number the
app used to show was the NNLS-24 quality head's score mapped through an isotonic
calibration. Audited on the 7 verified Brick-0 songs (603 chords, shipped
config, partial-credit target = root + parent family):

  * it read 0.465 where the chords were 0.827 right — 36 pp too low, and the
    calibration map was what pushed it down, not what corrected it;
  * worse, it had **AUC 0.480** — it did not rank right chords above wrong ones
    at all, so no monotone recalibration could rescue it.

Thirteen candidate replacements were measured (`scratchpad/conf_score_search*.py`).
Every model-self-confidence score failed on at least one song, including
music-x-lab's own posterior margin (mean per-song AUC 0.593, worst song 0.24) and
the genuinely independent NNLS root head (0.503). The signal that survived is not
a model score at all:

    accuracy by contiguous length (rows) x times the chord appears (cols)
                  once    2-3x    4-7x     8+x
      < 1.5 s      0%     26%     35%     93%
      1.5-3 s     22%     47%     55%     87%
      > 3 s         -     54%     77%     92%

**Repetition dominates length.** A chord held under 1.5 s that recurs 8+ times is
more reliable (93%) than a chord held over 3 s that appears 2-3 times (54%). A
chord the song plays exactly ONCE is right 14% of the time (1 of 12) and that
held in every song containing one: 0/3, 0/1, 1/5, 0/3.

Musically it is the simplicity principle with a number on it: a chord that never
recurs is usually the decoder inventing something rather than the song modulating.

**Validation.** Leave-one-song-out (fit the table on 6 songs, score the 7th):
ECE 0.051 and mean per-song AUC 0.716, against the deployed number's ECE 0.363
and AUC 0.439. The table below is then re-fitted on all 7 songs, which is what
ships.

**What this does NOT solve.**
  * Fitted on 7 songs, and the "appears once" bucket is n=12. Treat the exact
    probabilities as provisional; the ORDERING is the robust part.
  * It says nothing about WHICH chord is right — only how much to trust the one
    shown. A song that genuinely modulates once will have its real chords marked
    uncertain, correctly by this rule and wrongly by ear.
  * It cannot see a chord that is consistently wrong every time it repeats
    (a systematically mis-decoded vamp reads as maximally confident).
  * A short chart (few chords total) pushes every count down; the counts are raw
    occurrences, not a rate.
"""
from __future__ import annotations

from collections import Counter

from harmonia.tab_aligner import _family as _ireal_family

# Occurrence count -> P(root + parent family correct). Fitted on the 7 verified
# Brick-0 songs, duration-weighted; see the module docstring for the LOSO check.
# (upper bound inclusive, probability)
REPETITION_P_CORRECT: tuple[tuple[int, float], ...] = (
    (1, 0.14),          # the song plays this chord exactly once
    (3, 0.48),
    (7, 0.64),
    (10 ** 9, 0.91),    # 8 or more
)

# Below this, the app marks the chord as doubtful rather than merely tinting it.
DOUBTFUL_BELOW = 0.50

NO_CHORD_CONFIDENCE = 0.0


def chord_key(root_pc: int, quality: str) -> tuple[int, str]:
    """What a reader means by "the same chord": same root, same parent family.

    Cm and Cm7 are the same chord for this purpose — you play either over the
    other — which is also the partial-credit target the table was fitted on.
    """
    return int(root_pc) % 12, _ireal_family(quality or "")


def repetition_counts(chords) -> Counter:
    """How many times each (root, family) is played across the whole song.

    ``chords`` is any iterable of ``(root_pc, quality, is_no_chord)`` triples.
    No-chord cells are not counted and never contribute to another chord's count.
    """
    counts: Counter = Counter()
    for root_pc, quality, is_nc in chords:
        if is_nc or root_pc is None:
            continue
        counts[chord_key(root_pc, quality)] += 1
    return counts


def confidence_for_count(n: int) -> float:
    """P(the chord shown is right) given how often the song repeats it."""
    for upper, p in REPETITION_P_CORRECT:
        if n <= upper:
            return p
    return REPETITION_P_CORRECT[-1][1]


def confidence(root_pc: int | None, quality: str, is_nc: bool,
               counts: Counter) -> float:
    """Display confidence for one chord cell, given the song's counts."""
    if is_nc or root_pc is None:
        return NO_CHORD_CONFIDENCE
    return confidence_for_count(counts.get(chord_key(root_pc, quality), 0))
