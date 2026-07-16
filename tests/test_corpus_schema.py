"""
Tests for harmonia/data/corpus_schema.py — Phase 1 of the refactoring plan
(docs/refactoring_delegation_plan.md), fixing docs/refactoring_suggestions.md
§2a: the corpus `match`-quality field was an unvalidated free string, and an
unrecognized value (`"billboard_gt"`) was silently filtered to zero rows by
every trainer's hardcoded `match == "exact"` gate instead of raising.

The single most important property under test: `load_corpus` RAISES on an
unknown `match` value rather than silently dropping it.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.data.corpus_schema import (
    MatchQuality,
    UnknownMatchValueError,
    filter_by_match,
    load_corpus,
    match_level,
    save_corpus,
)


# --- match_level / MatchQuality -------------------------------------------------


def test_match_level_known_values():
    assert match_level("none") == MatchQuality.NONE
    assert match_level("mismatch") == MatchQuality.MISMATCH
    assert match_level("family") == MatchQuality.FAMILY
    assert match_level("exact") == MatchQuality.EXACT


def test_billboard_gt_aliases_to_exact():
    """The documented judgment call: billboard_gt == EXACT trust tier."""
    assert match_level("billboard_gt") == MatchQuality.EXACT


def test_match_level_unknown_raises():
    with pytest.raises(UnknownMatchValueError):
        match_level("totally_bogus_value")


def test_match_level_ordering_supports_minimum_filtering():
    assert MatchQuality.NONE < MatchQuality.MISMATCH < MatchQuality.FAMILY < MatchQuality.EXACT


# --- filter_by_match --------------------------------------------------------------


def test_filter_by_match_default_minimum_exact():
    match = ["exact", "family", "none", "mismatch", "billboard_gt"]
    mask = filter_by_match(match)  # default minimum=EXACT
    # exact and billboard_gt (aliased to EXACT) pass; family/none/mismatch don't.
    assert mask.tolist() == [True, False, False, False, True]


def test_filter_by_match_minimum_family_includes_exact_and_family():
    match = ["exact", "family", "none", "mismatch", "billboard_gt"]
    mask = filter_by_match(match, minimum=MatchQuality.FAMILY)
    assert mask.tolist() == [True, True, False, False, True]


def test_filter_by_match_unknown_value_raises():
    match = ["exact", "some_new_unrecognized_tag"]
    with pytest.raises(UnknownMatchValueError):
        filter_by_match(match)


# --- save_corpus / load_corpus round-trip -----------------------------------------


def _tiny_corpus(match_values):
    n = len(match_values)
    return dict(
        feat48=np.random.rand(n, 48).astype(np.float32),
        feat48_abs=np.random.rand(n, 48).astype(np.float32),
        root=np.arange(n, dtype=np.int32) % 12,
        quality_idx=np.zeros(n, dtype=np.int32),
        quality=np.array(["maj"] * n),
        labels=np.array([f"C:maj_{i}" for i in range(n)]),
        match=np.array(match_values),
        t0=np.arange(n, dtype=np.float64),
        t1=np.arange(n, dtype=np.float64) + 1.0,
        song_id=np.array(["song_0"] * n),
        qualities=np.array(["maj", "min", "dom", "hdim", "dim", "aug", "sus"]),
    )


def test_round_trip_preserves_keys_and_values(tmp_path):
    corpus = _tiny_corpus(["exact", "family", "none"])
    path = tmp_path / "corpus.npz"
    save_corpus(path, **corpus)
    loaded = load_corpus(path)

    assert set(loaded.keys()) == set(corpus.keys())
    for key, arr in corpus.items():
        np.testing.assert_array_equal(loaded[key], arr)


def test_round_trip_with_billboard_gt_values(tmp_path):
    """The exact bug scenario: a corpus containing billboard_gt records
    must round-trip cleanly (save_corpus accepts it, load_corpus accepts it,
    because it's a recognized alias) rather than erroring or vanishing."""
    corpus = _tiny_corpus(["billboard_gt", "billboard_gt", "exact"])
    path = tmp_path / "bb_corpus.npz"
    save_corpus(path, **corpus)
    loaded = load_corpus(path)

    np.testing.assert_array_equal(loaded["match"], corpus["match"])
    # And it's usable downstream via filter_by_match without special-casing.
    mask = filter_by_match(loaded["match"])
    assert mask.tolist() == [True, True, True]


def test_load_corpus_raises_on_unknown_match_value(tmp_path):
    """The core regression test for the §2a bug: an unrecognized match
    value must raise at load time, not be silently filtered to nothing."""
    n = 3
    bad = dict(
        feat48=np.random.rand(n, 48).astype(np.float32),
        feat48_abs=np.random.rand(n, 48).astype(np.float32),
        root=np.zeros(n, dtype=np.int32),
        quality_idx=np.zeros(n, dtype=np.int32),
        quality=np.array(["maj"] * n),
        labels=np.array(["C:maj"] * n),
        match=np.array(["exact", "some_brand_new_tag_no_trainer_knows", "exact"]),
        t0=np.zeros(n, dtype=np.float64),
        t1=np.ones(n, dtype=np.float64),
        song_id=np.array(["song_0"] * n),
        qualities=np.array(["maj", "min", "dom", "hdim", "dim", "aug", "sus"]),
    )
    path = tmp_path / "bad_corpus.npz"
    # save_corpus itself validates match values up front (fail fast at write time).
    with pytest.raises(UnknownMatchValueError):
        save_corpus(path, **bad)

    # Simulate a corpus written by some OTHER path (bypassing save_corpus,
    # e.g. a bare np.savez as flagged in §2c) that reaches disk with a bad
    # match value anyway: load_corpus must still catch it.
    np.savez(path, **bad)
    with pytest.raises(UnknownMatchValueError):
        load_corpus(path)


def test_save_corpus_warns_on_missing_required_keys(tmp_path):
    path = tmp_path / "incomplete.npz"
    with pytest.warns(UserWarning, match="missing expected keys"):
        save_corpus(path, feat48=np.zeros((2, 48), dtype=np.float32), match=np.array(["exact", "exact"]))
    # Still writes and is still loadable.
    loaded = load_corpus(path)
    assert "feat48" in loaded
