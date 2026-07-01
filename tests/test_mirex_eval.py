"""
Unit tests for harmonia/eval/mirex_eval.py.

Regression coverage for a real bug found while validating the pipeline on
POP909: `_label_to_mireval` checked a generic "7" suffix before more
specific suffixes ("mMaj7", "°7", "ø7"), so minor-major-7th / diminished-7th
/ half-diminished-7th chords were mangled into invalid Harte strings (e.g.
"EmMaj7" -> "EmMaj:7"), which crashed mir_eval and silently zeroed out the
score for the entire song (3 of 5 POP909 songs tested hit this).
"""

from __future__ import annotations

import pytest

from harmonia.eval.mirex_eval import _label_to_mireval
from harmonia.theory.chord_vocabulary import ChordQuality, chord_label, get_vocabulary


class TestLabelToMireval:

    def test_no_chord(self):
        assert _label_to_mireval("N") == "N"

    @pytest.mark.parametrize("label,expected", [
        ("Cmaj7", "C:maj7"),
        ("G7", "G:7"),
        ("A#min7", "A#:min7"),
        ("EmMaj7", "E:minmaj7"),   # previously crashed: -> "EmMaj:7"
        ("Dø7", "D:hdim7"),        # previously crashed: -> "Dø:7"
        ("C°7", "C:dim7"),         # previously crashed: -> "C°:7"
        ("F#sus4", "F#:sus4"),
        ("C", "C:maj"),
    ])
    def test_known_labels(self, label, expected):
        assert _label_to_mireval(label) == expected

    def test_every_vocabulary_label_encodes_without_crashing(self):
        """Every (root, quality) label the HMM can actually emit must convert
        to a label mir_eval.chord.encode() accepts — this is exactly what
        mir_eval.chord.evaluate() calls internally, and an invalid label
        used to crash evaluation for the whole song."""
        import mir_eval.chord as mec

        for max_phase in (1, 2, 3, 4):
            for quality in get_vocabulary(max_phase):
                if quality == ChordQuality.NO_CHORD:
                    continue
                for root in range(12):
                    label = chord_label(root, quality)
                    mireval_label = _label_to_mireval(label)
                    mec.encode(mireval_label)  # raises on invalid input
