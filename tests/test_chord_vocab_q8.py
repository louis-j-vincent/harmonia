"""The extended (Q8) quality vocabulary, and the guard that keeps loaders honest.

The bug this file exists for (2026-08-06): the pooled corpus has three loaders.
Two went through `parse_harte_lite`; `_load_accomp_db` called
`progression_encoder.fine_to_q5` directly, a hard-wired QUAL5 map. The screen
that chose the vocabulary switched alphabets by monkeypatching
`parse_harte_lite`, so 26.8% of the corpus (92,268 tokens — jazz1460 + pop400 +
blues50, the most seventh-rich source of the three) went on emitting QUAL5
indices into a stream the other loaders were filling with 8- and 18-class
indices. Index 2 meant `dom` from one loader and `dim` from another. Nothing
raised; the per-class table it produced read 8.4% augmented chords.
"""
from __future__ import annotations

import numpy as np
import pytest

import harmonia_min.chord_context_prior as cp


@pytest.fixture(autouse=True)
def _restore_vocab():
    name, _ = cp.active_vocab()
    yield
    cp.set_vocab(name)


# ── the vocabulary itself ───────────────────────────────────────────────────
def test_q8_refines_qual5_and_never_disagrees_with_it():
    """Backing off from a Q8 class must land on the QUAL5 class the same label
    would have had, or the backoff chain mixes two different answers."""
    for label, q8 in cp.HARTE_QUAL_TO_QUAL8.items():
        assert cp.QUAL8_TO_QUAL5[q8] == cp.HARTE_QUAL_TO_QUAL5[label], label


def test_both_tables_accept_exactly_the_same_labels():
    """Otherwise the arms are trained on different corpora and the comparison
    between them measures the label domain, not the granularity."""
    assert set(cp.HARTE_QUAL_TO_QUAL8) == set(cp.HARTE_QUAL_TO_QUAL5)


def test_sevenths_survive_the_round_trip():
    cp.set_vocab("q8")
    _, classes = cp.active_vocab()
    for label, expect in (("C:min7", "min7"), ("C:maj7", "maj7"),
                          ("C:7", "dom7"), ("C:hdim7", "hdim7")):
        root, q = cp.parse_harte_lite(label)
        assert root == 0 and classes[q] == expect, label


def test_qual5_still_flattens_them():
    cp.set_vocab("q5")
    _, classes = cp.active_vocab()
    for label in ("C:min7", "C:min", "C:min9"):
        _root, q = cp.parse_harte_lite(label)
        assert classes[q] == "min"


def test_dim7_folds_into_dim_but_hdim7_stays():
    """0.41% + 0.24% does not buy two classes; hdim7 at 0.28% keeps its own
    because it already ships and is the ii of a minor ii-V."""
    cp.set_vocab("q8")
    _, classes = cp.active_vocab()
    assert classes[cp.parse_harte_lite("C:dim7")[1]] == "dim"
    assert classes[cp.parse_harte_lite("C:dim")[1]] == "dim"
    assert classes[cp.parse_harte_lite("C:hdim7")[1]] == "hdim7"


def test_the_fine_mma_table_targets_the_same_classes():
    """accomp_db speaks MMA bucket names, not Harte — but must land in the same
    alphabet, which is the entire point of the fix."""
    assert set(cp.FINE_TO_QUAL8.values()) <= set(cp.QUAL8)
    cp.set_vocab("q8")
    _, classes = cp.active_vocab()
    for fine, expect in (("dom7", "dom7"), ("min7", "min7"), ("maj7", "maj7"),
                         ("m7b5", "hdim7"), ("7sus4", "sus"), ("dim7", "dim")):
        assert classes[cp.fine_to_active(fine)] == expect, fine


def test_unknown_quality_is_still_never_silently_major():
    cp.set_vocab("q8")
    assert cp.parse_harte_lite("C:notachord") is None
    assert cp.fine_to_active("notabucket") is None


# ── shapes follow the vocabulary ────────────────────────────────────────────
def test_shapes_track_the_active_vocabulary():
    cp.set_vocab("q5")
    assert cp.n_qual() == 5 and cp.n_cand() == 60
    cp.set_vocab("q8")
    assert cp.n_qual() == 8 and cp.n_cand() == 96


def test_cand_index_round_trips_in_both_vocabularies():
    for name, nq in (("q5", 5), ("q8", 8)):
        cp.set_vocab(name)
        for root in range(12):
            for q in range(nq):
                assert cp.cand_decode(cp.cand_index(root, q)) == (root, q)


def test_set_vocab_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        cp.set_vocab("q42")


# ── the guard ───────────────────────────────────────────────────────────────
def test_guard_catches_a_loader_that_kept_its_own_alphabet(monkeypatch):
    """The regression test for the actual bug: a loader emitting an index
    outside the active vocabulary must fail loudly, naming itself."""
    cp.set_vocab("q5")

    def fake_choco(*_a, **_k):
        # index 7 is legal in Q8, out of range in QUAL5 — exactly the shape of
        # the accomp_db bug, with the roles reversed so the test is cheap
        return {"choco:rogue": [[(0, 0), (2, 7), (4, 1), (5, 0)]]}, {}

    monkeypatch.setattr(cp, "_load_choco", fake_choco)
    with pytest.raises(AssertionError) as e:
        cp.load_all_corpus_sequences(sources=("choco",))
    assert "choco" in str(e.value)


def test_every_loader_lands_in_range_on_the_real_corpus():
    """The live version of the same check, on the actual data."""
    for name, nq in (("q5", 5), ("q8", 8)):
        cp.set_vocab(name)
        _songs, stats = cp.load_all_corpus_sequences()
        per_source = stats["vocab"]["max_index_per_source"]
        assert len(per_source) >= 3, f"expected 3 loaders, got {per_source}"
        assert all(m < nq for m in per_source.values()), per_source
        # and the extended arm must actually USE the extra classes, in every
        # source — a loader stuck at QUAL5 would show max index 4 here
        if name == "q8":
            assert all(m > 4 for m in per_source.values()), (
                f"a loader never emitted an extended class: {per_source}")


def test_extended_vocabulary_recovers_the_chord_changes_qual5_deleted():
    """Consecutive identical tokens are de-duplicated, so under QUAL5 a real
    `C -> C7` move was erased as "same chord". Measured: +12,704 tokens."""
    counts = {}
    for name in ("q5", "q8"):
        cp.set_vocab(name)
        songs, _ = cp.load_all_corpus_sequences()
        counts[name] = sum(len(s) for seqs in songs.values() for s in seqs)
    assert counts["q8"] > counts["q5"] + 10_000, counts
