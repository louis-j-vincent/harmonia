"""Unit tests for harmonia/theory/local_key.py.

The load-bearing assumptions of the section-key-highlight and transpose
features: that an iReal token parses to the right root pitch class + quality,
that transposition shifts the root correctly with sensible spelling, and that
the chord-tone Krumhansl match recovers an obvious key.
"""

from __future__ import annotations

from harmonia.theory.local_key import (chord_pcs, estimate_key, key_name,
                                        parse_token, prefer_flats, transpose_token)


def test_parse_token_roots_and_bass():
    assert parse_token("C")[0] == 0
    assert parse_token("Ab^7")[0] == 8
    assert parse_token("F#-7")[0] == 6
    assert parse_token("Bb7")[:2] == (10, "7")
    # slash bass
    root, qual, bass = parse_token("D-7/G")
    assert (root, qual, bass) == (2, "-7", 7)


def test_chord_pcs_triads():
    # C major triad → {C, E, G}; C minor → {C, Eb, G}
    assert {0, 4, 7} <= set(chord_pcs("C"))
    assert {0, 3, 7} <= set(chord_pcs("C-"))
    assert {0, 3, 6} <= set(chord_pcs("Co"))          # diminished
    assert {0, 4, 8} <= set(chord_pcs("C+"))          # augmented


def test_transpose_token():
    # up a perfect 4th (5 semitones), flat spelling: C7 → F7, quality kept
    assert transpose_token("C7", 5, flats=True) == "F7"
    assert transpose_token("Ab^7", 2, flats=True) == "Bb^7"
    assert transpose_token("D-7/G", 3, flats=True) == "F-7/Bb"
    # sharp spelling
    assert transpose_token("C7", 6, flats=False) == "F#7"


def test_prefer_flats_and_key_name():
    assert prefer_flats(5, "major") is True          # F major → flats
    assert prefer_flats(7, "major") is False         # G major → sharps
    assert key_name(8, "major") == "Ab major"        # not "G# major"
    assert key_name(10, "minor") == "Bb minor"


def test_estimate_key_recovers_obvious_key():
    # a plain ii-V-I in C major must come back as C major
    est = estimate_key(["D-7", "G7", "C^7"])
    assert (est["tonic"], est["mode"]) == (0, "major")
    # relative minor cadence resolves to A minor
    est = estimate_key(["B-7b5", "E7", "A-"])
    assert (est["tonic"], est["mode"]) == (9, "minor")
