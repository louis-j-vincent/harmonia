"""Tests for the symbolic section-key dataset / oracle (issue #23)."""
from __future__ import annotations

from harmonia.models.local_key_data import (
    key_to_idx, idx_to_key, parse_global_key, token_to_q5,
    oracle_section_key, section_instances,
)
from harmonia.models.local_key_model import transpose_example


def test_parse_global_key():
    assert parse_global_key("C") == (0, "major")
    assert parse_global_key("Bb") == (10, "major")
    assert parse_global_key("F#-") == (6, "minor")
    assert parse_global_key("E-") == (4, "minor")
    assert parse_global_key("garbage") is None


def test_key_idx_roundtrip():
    for idx in range(24):
        assert key_to_idx(*idx_to_key(idx)) == idx
    assert key_to_idx(0, "major") == 0
    assert key_to_idx(0, "minor") == 12


def test_token_to_q5():
    # maj/min/dom/hdim/dim families
    assert token_to_q5("C") is not None          # maj
    assert token_to_q5("C-7") == token_to_q5("D-7")  # both min family
    assert token_to_q5("G7") == token_to_q5("A7")    # both dom family
    # half-dim and dim distinct from each other
    assert token_to_q5("Bh7") != token_to_q5("Bo7")


def test_oracle_holds_global_when_unforced():
    # a clean C-major ii-V-I: oracle must NOT modulate away from C major.
    toks = [("D-7", 4.0), ("G7", 4.0), ("C^7", 8.0)]
    g = key_to_idx(0, "major")
    idx, modulated = oracle_section_key(toks, g, margin=6.0)
    assert not modulated
    assert idx == g


def test_oracle_detects_clear_modulation():
    # a section that is decisively an Eb-major ii-V-I while the song is in C.
    toks = [("F-7", 4.0), ("Bb7", 4.0), ("Eb^7", 8.0)]
    g = key_to_idx(0, "major")  # song global = C major
    idx, modulated = oracle_section_key(toks, g, margin=6.0)
    assert modulated
    assert idx == key_to_idx(3, "major")  # Eb major


def test_transpose_equivariance():
    seq = [(2, 1), (7, 2), (0, 0)]  # roots D, G, C
    y = key_to_idx(0, "major")      # C major
    s2, y2 = transpose_example(seq, y, 2)  # up a whole tone
    assert s2 == [(4, 1), (9, 2), (2, 0)]
    assert y2 == key_to_idx(2, "major")    # D major
    # minor mode preserved under transpose
    _, ym = transpose_example(seq, key_to_idx(9, "minor"), 3)
    assert ym == key_to_idx(0, "minor")


def test_section_instances_split_by_label():
    rec = {
        "section_per_bar": ["A", "A", "B", "B"],
        "beats_per_bar": 4,
        "n_bars": 4,
        "chord_timeline": [
            {"bar": 1, "beat": 0, "ireal": "C"},
            {"bar": 3, "beat": 0, "ireal": "G7"},
        ],
    }
    secs = section_instances(rec)
    assert [s["label"] for s in secs] == ["A", "B"]
    assert secs[0]["bar_lo"] == 1 and secs[0]["bar_hi"] == 2
