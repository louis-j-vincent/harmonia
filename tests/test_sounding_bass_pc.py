"""Unit tests for the sounding-bass pitch-class resolver (2026-07-16 target
redefinition: functional root -> sounding bass pc). Red-first discipline is
moot here (new function), but the edge cases below are the ones CLAUDE.md
rule #3/#4 care about: both bass conventions, root position, and no-chord.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonia.data.corpus_schema import sounding_bass_pc

# note-name -> pc, for readable assertions
C, Cs, D, Eb, E, F, Fs, G, Ab, A, Bb, B = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11


def test_root_position_returns_root():
    # No slash -> bass == functional root (unchanged from old behaviour).
    assert sounding_bass_pc("C:maj", C) == C
    assert sounding_bass_pc("Eb:min7", Eb) == Eb
    assert sounding_bass_pc("A", A) == A  # bare root, no quality


def test_numeric_scale_degree_bass():
    # RWC/JAAH convention: bass is a scale degree relative to the root.
    assert sounding_bass_pc("C:maj/3", C) == E     # 0 + 4
    assert sounding_bass_pc("C:maj/5", C) == G      # 0 + 7
    assert sounding_bass_pc("A:maj/3", A) == Cs     # 9 + 4 = 13 -> 1 (Db/C#)
    assert sounding_bass_pc("Ab:min7/b7", Ab) == Fs # 8 + 10 = 18 -> 6 (Gb/F#)
    assert sounding_bass_pc("Eb:7/3", Eb) == G      # 3 + 4
    assert sounding_bass_pc("F:maj/5", F) == C       # 5 + 7 = 12 -> 0


def test_flat_and_sharp_degree_tokens():
    assert sounding_bass_pc("C:min/b3", C) == Eb    # 0 + 3
    assert sounding_bass_pc("C:7/b7", C) == Bb       # 0 + 10
    assert sounding_bass_pc("C:maj/2", C) == D       # 0 + 2


def test_literal_note_letter_bass():
    # Billboard/other convention: bass is an absolute note name, root-independent.
    assert sounding_bass_pc("C:maj/D", C) == D
    assert sounding_bass_pc("C:maj/Bb", C) == Bb
    assert sounding_bass_pc("G:7/F#", G) == Fs
    # note-letter resolves independent of the root value passed
    assert sounding_bass_pc("C:maj/E", 99 % 12) == E


def test_no_chord_and_unknown():
    assert sounding_bass_pc("N", None) is None
    assert sounding_bass_pc("X", None) is None
    assert sounding_bass_pc("", None) is None
    assert sounding_bass_pc("N", 0) is None  # N wins over a stray root


def test_unrecognized_bass_falls_back_to_root():
    # Garbage bass token must never invent a pc; fall back to the root.
    assert sounding_bass_pc("C:maj/???", C) == C
    assert sounding_bass_pc("C:maj/", C) == C


def test_none_root_with_degree_returns_none():
    # Degree bass needs a root; without one we return None, not a guess.
    assert sounding_bass_pc("C:maj/3", None) is None


def test_all_rwc_degree_tokens_resolve():
    # The exact token multiset verified present in RWC-Popular (known_issues).
    for tok, expected_off in [("3", 4), ("5", 7), ("b3", 3), ("b7", 10),
                              ("2", 2), ("4", 5), ("7", 11), ("6", 9),
                              ("b6", 8), ("b5", 6)]:
        assert sounding_bass_pc(f"C:maj/{tok}", C) == expected_off % 12


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
