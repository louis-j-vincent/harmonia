"""Repetition-based display confidence (harmonia/output/chord_confidence.py).

The behaviour these lock down is the ORDERING — a chord the song repeats is
trusted more than one it plays once — plus the two ways the old number was
wrong, so neither can come back silently.
"""
from collections import Counter

import pytest

from harmonia.output.chord_confidence import (DOUBTFUL_BELOW,
                                              NO_CHORD_CONFIDENCE,
                                              REPETITION_P_CORRECT, chord_key,
                                              confidence, confidence_for_count,
                                              repetition_counts)


def test_more_repetitions_never_lowers_confidence():
    """Monotone by construction — the whole premise of the table."""
    vals = [confidence_for_count(n) for n in range(0, 40)]
    assert vals == sorted(vals)


def test_a_chord_played_once_is_doubtful_and_a_repeated_one_is_not():
    assert confidence_for_count(1) < DOUBTFUL_BELOW
    assert confidence_for_count(8) > 0.85
    # the measured headline: once-only is far below the song average (~0.83)
    assert confidence_for_count(1) < 0.25


def test_same_root_different_seventh_counts_as_the_same_chord():
    """Cm and Cm7 are one chord for the reader — and for the fitted target."""
    assert chord_key(0, "-") == chord_key(0, "-7")
    assert chord_key(0, "") == chord_key(0, "^7")
    assert chord_key(0, "-") != chord_key(0, "")
    counts = repetition_counts([(0, "-", False), (0, "-7", False), (0, "-9", False)])
    assert counts[chord_key(0, "-")] == 3


def test_different_roots_are_different_chords():
    counts = repetition_counts([(0, "", False), (7, "", False)])
    assert counts[chord_key(0, "")] == 1
    assert counts[chord_key(7, "")] == 1


def test_no_chord_cells_are_not_counted_and_score_zero():
    counts = repetition_counts([(0, "", False), (None, "", True), (0, "", True)])
    assert counts[chord_key(0, "")] == 1
    assert confidence(None, "", True, counts) == NO_CHORD_CONFIDENCE
    assert confidence(0, "", True, counts) == NO_CHORD_CONFIDENCE


def test_confidence_reflects_the_whole_song_not_the_single_cell():
    """The point of the signal: an identical cell scores differently depending
    on whether the rest of the song ever plays that chord again."""
    lonely = repetition_counts([(0, "", False)] + [(5, "", False)] * 9)
    common = repetition_counts([(0, "", False)] * 9 + [(5, "", False)])
    assert confidence(0, "", False, lonely) < DOUBTFUL_BELOW
    assert confidence(0, "", False, common) > 0.85


def test_unseen_chord_scores_as_rare_rather_than_crashing():
    assert confidence(3, "-", False, Counter()) == confidence_for_count(0)


@pytest.mark.parametrize("upper,p", REPETITION_P_CORRECT)
def test_table_is_a_probability(upper, p):
    assert 0.0 <= p <= 1.0
    assert upper >= 1


def test_the_two_defects_of_the_old_number_are_gone():
    """Regression guard, red before this module existed.

    The deployed confidence read 0.465 mean where accuracy was 0.827 (36 pp low)
    and had AUC 0.480 (it did not rank right above wrong). Both are structural
    here: the table's mass sits near the measured base rate, and it is strictly
    increasing so its ranking cannot be inverted.
    """
    ps = [p for _, p in REPETITION_P_CORRECT]
    assert max(ps) > 0.83, "the top bucket must not under-report like the old map"
    assert ps == sorted(ps) and len(set(ps)) == len(ps), "must be strictly ranked"
