"""Tests for the ported client-side continuity key heuristic baseline (#23).

Covers the section-level reduction, a pure-diatonic sanity check (the tracker
must NOT modulate on in-key chords), and 1:1 alignment of the heuristic example
set with the oracle example set the GRU is scored on (same val split).
"""
from __future__ import annotations

import pytest

from harmonia.models.local_key_data import (
    DEFAULT_DB, build_examples, split_examples,
)
from harmonia.models.local_key_heuristic import (
    build_heuristic_examples, evaluate_heuristic, section_pred_from_track,
)

# 0..23 key indices: 0 = C major, 12 = C minor, 21 = A minor.
C_MAJ, A_MIN, G_MAJ = 0, 21, 7


def test_section_pred_duration_weighted_vote():
    # Two chords in C major (long) vs one in G major (short) -> C major wins.
    scales = [
        {"tonic": 0, "mode": "major"},
        {"tonic": 0, "mode": "major"},
        {"tonic": 7, "mode": "major"},
    ]
    assert section_pred_from_track(scales, [4.0, 4.0, 1.0]) == C_MAJ
    # Flip the durations -> the single G bar now dominates.
    assert section_pred_from_track(scales, [0.5, 0.5, 8.0]) == G_MAJ


def test_section_pred_empty_defaults_to_c_major():
    assert section_pred_from_track([], []) == C_MAJ


def test_pure_diatonic_section_does_not_modulate():
    # A pure I-IV-V-I in C: every chord is in the C-major collection, so the
    # continuity tracker must stay on C major for the whole run (no spurious jump).
    from harmonia.theory.local_key import continuity_scale_track
    toks = ["C", "F", "G7", "C"]
    track = continuity_scale_track(toks, home_tonic=0, home_mode="major")
    assert section_pred_from_track(track, [4.0] * len(toks)) == C_MAJ


@pytest.fixture(scope="module")
def _has_db():
    if not DEFAULT_DB.exists():
        pytest.skip("iReal corpus unavailable")


def test_heuristic_examples_align_with_oracle_split(_has_db):
    # Restrict to one small corpus for speed; the heuristic and oracle example
    # sets must be row-for-row aligned so split_examples yields the same val set.
    corp = {"blues50"}
    oracle = build_examples(DEFAULT_DB, corpora=corp)
    heur = build_heuristic_examples(DEFAULT_DB, corpora=corp)
    assert len(oracle) == len(heur) > 0
    for o, h in zip(oracle, heur):
        assert (o["song_idx"], o["label"]) == (h["song_idx"], h["label"])
        assert o["y"] == h["y"] and o["modulated"] == h["modulated"]
    # every prediction is a valid 0..23 key index
    assert all(0 <= h["pred"] < 24 for h in heur)


def test_evaluate_heuristic_reports_recall(_has_db):
    heur = build_heuristic_examples(DEFAULT_DB, corpora={"blues50"})
    _, hval = split_examples(heur)
    res = evaluate_heuristic(hval)
    assert res["n"] == len(hval)
    assert 0.0 <= res["acc"] <= 1.0
    assert 0.0 <= res["mod_acc"] <= 1.0
