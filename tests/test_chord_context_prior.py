"""Tests for harmonia/models/chord_context_prior.py (Phase 1, issue: chord
context prior for lock propagation, docs/design_chord_context_prior.md)."""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.chord_context_prior import (
    CORPUS_SOURCES,
    N_CAND,
    N_Q5,
    N_ROOT,
    ContextPriorModel,
    _default_cache_path_for,
    build_context_prior,
    cand_decode,
    cand_index,
    chord_name,
    load_all_corpus_sequences,
    load_context_prior,
    parse_harte_lite,
    score_candidates,
    top_candidates,
    trigram_key,
)
from harmonia.models.progression_encoder import QUAL5_IDX

MAJ, MIN, DOM, HDIM, DIM = (QUAL5_IDX[q] for q in ("maj", "min", "dom", "hdim", "dim"))

pytestmark = pytest.mark.filterwarnings("ignore")


# ── representation arithmetic ───────────────────────────────────────────────

def test_worked_example_dm7_g7_cmaj():
    """Design doc worked example: Dm7 G7 Cmaj, scoring candidate G7.

    Roots: D=2, G=7, C=0. Expect Δp=(2-7)%12=7, q_p=min; Δn=(0-7)%12=5,
    q_n=maj; q_c=dom.
    """
    prev = (2, MIN)   # Dm7
    cand = (7, DOM)   # G7
    nxt = (0, MAJ)    # Cmaj
    qc, dp, qp, dn, qn = trigram_key(prev, cand, nxt)
    assert (qc, dp, qp, dn, qn) == (DOM, 7, MIN, 5, MAJ)


def test_tritone_sub_gives_a_distinct_key():
    """Db7 in the same slot (design doc): a distinct, learnable pattern."""
    prev = (2, MIN)   # Dm7
    cand = (1, DOM)   # Db7
    nxt = (0, MAJ)    # Cmaj
    key_db7 = trigram_key(prev, cand, nxt)
    key_g7 = trigram_key(prev, (7, DOM), nxt)
    assert key_db7 != key_g7
    assert key_db7[0] == DOM  # same candidate quality, different root geometry


def test_transposition_invariance():
    """The same progression transposed by 3 semitones -> identical key."""
    prev, cand, nxt = (2, MIN), (7, DOM), (0, MAJ)
    key = trigram_key(prev, cand, nxt)
    t = 3
    prev_t = ((prev[0] + t) % 12, prev[1])
    cand_t = ((cand[0] + t) % 12, cand[1])
    nxt_t = ((nxt[0] + t) % 12, nxt[1])
    key_t = trigram_key(prev_t, cand_t, nxt_t)
    assert key == key_t


def test_cand_index_roundtrip():
    for root in range(N_ROOT):
        for qi in range(N_Q5):
            idx = cand_index(root, qi)
            assert cand_decode(idx) == (root, qi)
    assert N_CAND == N_ROOT * N_Q5 == 60


def test_chord_name_readable():
    assert chord_name(7, DOM) == "G7"
    assert chord_name(0, MAJ) == "Cmaj"
    assert chord_name(2, MIN) == "Dm7"


# ── Harte-lite parsing ───────────────────────────────────────────────────────

def test_parse_harte_lite_basic():
    assert parse_harte_lite("C:maj") == (0, MAJ)
    assert parse_harte_lite("G:7") == (7, DOM)
    assert parse_harte_lite("A:min7") == (9, MIN)
    assert parse_harte_lite("Bb:hdim7") == (10, HDIM)
    assert parse_harte_lite("F#:dim7") == (6, DIM)


def test_parse_harte_lite_no_chord_and_ambiguous():
    assert parse_harte_lite("N") is None
    assert parse_harte_lite("X") is None


def test_parse_harte_lite_bare_root_implies_major():
    # Harte convention: quality omitted entirely (no colon) -> major triad.
    assert parse_harte_lite("A") == (9, MAJ)
    assert parse_harte_lite("D/5") == (2, MAJ)  # bass-degree slash, no colon


def test_parse_harte_lite_interval_list_and_power_chord_are_unmapped():
    # Regression test for a real bug caught mid-build: an earlier draft
    # silently defaulted these to "maj" via an empty-string fallthrough.
    assert parse_harte_lite("C:(3,5,b7,b9)") is None
    assert parse_harte_lite("G:5") is None
    assert parse_harte_lite("E:1/1") is None


def test_parse_harte_lite_aug_and_sus_fold_to_maj():
    assert parse_harte_lite("C:aug") == (0, MAJ)
    assert parse_harte_lite("C:sus4") == (0, MAJ)
    assert parse_harte_lite("C:sus2") == (0, MAJ)


# ── smoothing sanity ─────────────────────────────────────────────────────────

def _empty_model() -> dict:
    return {
        "tri": np.zeros((N_Q5, N_ROOT, N_Q5, N_ROOT, N_Q5), dtype=np.float64),
        "prevbi": np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float64),
        "nextbi": np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float64),
        "uni": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),  # flat, nonzero
    }


def test_unseen_trigram_falls_back_without_crashing():
    model = _empty_model()
    probs = score_candidates((2, MIN), (0, MAJ), model=model)
    assert probs.shape == (N_CAND,)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    assert abs(probs.sum() - 1.0) < 1e-8
    # zero evidence everywhere -> uniform (all roots/quals equally unweighted)
    assert np.allclose(probs, probs[0])


def test_missing_prev_falls_back_to_next_bigram():
    model = _empty_model()
    model["nextbi"][DOM, 5, MAJ] = 10.0  # some (Δn=5, q_n=maj) -> dom evidence
    probs = score_candidates(None, (0, MAJ), model=model)
    assert probs.shape == (N_CAND,)
    assert abs(probs.sum() - 1.0) < 1e-8
    assert np.isfinite(probs).all()
    # root 7 (G), quality dom should now be favoured over an unseen root
    assert probs[cand_index(7, DOM)] > probs[cand_index(3, DOM)]


def test_missing_next_falls_back_to_prev_bigram():
    model = _empty_model()
    model["prevbi"][DOM, 7, MIN] = 10.0  # (Δp=7, q_p=min) -> dom evidence
    probs = score_candidates((2, MIN), None, model=model)
    assert probs.shape == (N_CAND,)
    assert abs(probs.sum() - 1.0) < 1e-8
    assert probs[cand_index(7, DOM)] > probs[cand_index(3, DOM)]


def test_both_missing_is_uniform_over_roots():
    model = _empty_model()
    probs = score_candidates(None, None, model=model)
    assert abs(probs.sum() - 1.0) < 1e-8
    assert np.allclose(probs, 1.0 / N_CAND)


def test_ii_v_i_evidence_beats_unseen_candidate():
    """A model with real ii-V-I trigram counts must rank the V7 candidate
    above an otherwise-unattested candidate root."""
    model = _empty_model()
    prev, nxt = (2, MIN), (0, MAJ)
    qc, dp, qp, dn, qn = trigram_key(prev, (7, DOM), nxt)
    model["tri"][qc, dp, qp, dn, qn] = 20.0
    model["prevbi"][qc, dp, qp] = 20.0
    model["nextbi"][qc, dn, qn] = 20.0
    probs = score_candidates(prev, nxt, model=model)
    top = top_candidates(prev, nxt, k=1, model=model)
    assert top[0]["root"] == 7 and top[0]["q5"] == "dom"
    assert probs[cand_index(7, DOM)] == probs.max()


# ── build -> save -> load -> score round trip ──────────────────────────────

def test_build_save_load_roundtrip(tmp_path):
    cache = tmp_path / "ctx_prior_test.npz"
    model = _empty_model()
    prev, cand, nxt = (2, MIN), (7, DOM), (0, MAJ)
    qc, dp, qp, dn, qn = trigram_key(prev, cand, nxt)
    model["tri"][qc, dp, qp, dn, qn] = 5.0
    model["prevbi"][qc, dp, qp] = 5.0
    model["nextbi"][qc, dn, qn] = 5.0

    import json

    import numpy as _np
    _np.savez_compressed(
        cache,
        tri=model["tri"].astype(_np.float32),
        prevbi=model["prevbi"].astype(_np.float32),
        nextbi=model["nextbi"].astype(_np.float32),
        uni=model["uni"].astype(_np.float32),
        train_ids=_np.array(["a:1", "a:2"], dtype=object),
        heldout_ids=_np.array(["a:3"], dtype=object),
        stats_json=_np.array(json.dumps({"ok": True})),
    )

    loaded = load_context_prior(cache_path=cache)
    assert loaded["train_ids"] == ["a:1", "a:2"]
    assert loaded["heldout_ids"] == ["a:3"]
    assert loaded["stats"] == {"ok": True}

    probs_orig = score_candidates(prev, nxt, model=model)
    probs_loaded = score_candidates(prev, nxt, model=loaded)
    assert np.allclose(probs_orig, probs_loaded, atol=1e-5)


# ── genre-arm: corpus= parameter (jazz/pop/pooled tables) ──────────────────
# Added 2026-07-30, genre-arm experiment (docs/context_prior_phase1_results.md
# "Genre-arm experiment"): build_context_prior/load_context_prior gained a
# corpus= arg so tables can be built from a corpus SUBSET (jazz = accomp_db
# only, pop = POP909+ChoCo only) while the zero-arg pooled call/cache stays
# exactly as Phase 1 left it.

def test_corpus_sources_mapping():
    assert CORPUS_SOURCES["pooled"] == ("accomp_db", "pop909", "choco")
    assert CORPUS_SOURCES["jazz"] == ("accomp_db",)
    assert CORPUS_SOURCES["pop"] == ("pop909", "choco")
    # pooled must be exactly the union of jazz + pop, no source double-counted
    assert set(CORPUS_SOURCES["jazz"]) | set(CORPUS_SOURCES["pop"]) == set(CORPUS_SOURCES["pooled"])
    assert not (set(CORPUS_SOURCES["jazz"]) & set(CORPUS_SOURCES["pop"]))


def test_default_cache_path_for():
    assert _default_cache_path_for("pooled").name == "chord_context_prior.npz"
    assert _default_cache_path_for("jazz").name == "chord_context_prior_jazz.npz"
    assert _default_cache_path_for("pop").name == "chord_context_prior_pop.npz"


def test_load_all_corpus_sequences_sources_filter(tmp_path):
    """Restricting `sources` must skip both the load AND the stats entry for
    the excluded corpora — not just fail to find missing files silently."""
    songs, stats = load_all_corpus_sequences(
        accomp_db_path=tmp_path / "missing.jsonl",
        pop909_dir=tmp_path / "missing_pop909",
        choco_dir=tmp_path / "missing_choco",
        sources=("accomp_db",),
    )
    assert songs == {}
    assert "accomp_db" in stats and "error" in stats["accomp_db"]
    assert "pop909" not in stats
    assert "choco" not in stats


def test_load_all_corpus_sequences_default_sources_is_all_three():
    import inspect
    sig = inspect.signature(load_all_corpus_sequences)
    assert sig.parameters["sources"].default == ("accomp_db", "pop909", "choco")


def test_build_context_prior_invalid_corpus_raises():
    with pytest.raises(ValueError, match="corpus"):
        build_context_prior(corpus="rock")


def test_load_context_prior_invalid_corpus_raises():
    with pytest.raises(ValueError, match="corpus"):
        load_context_prior(corpus="rock")


def test_jazz_arm_scoped_to_accomp_db_only(tmp_path):
    """corpus='jazz' must pull ONLY accomp_db, even with real data on disk —
    every train/held-out id must be accomp:-prefixed, and pop909/choco must
    not appear in stats at all."""
    model = build_context_prior(cache_path=tmp_path / "jazz.npz", corpus="jazz")
    assert model["corpus"] == "jazz"
    assert len(model["train_ids"]) > 0
    assert all(sid.startswith("accomp:") for sid in model["train_ids"])
    assert all(sid.startswith("accomp:") for sid in model["heldout_ids"])
    assert "accomp_db" in model["stats"]
    assert "pop909" not in model["stats"]
    assert "choco" not in model["stats"]


def test_pop_arm_scoped_to_pop909_and_choco_only(tmp_path):
    model = build_context_prior(cache_path=tmp_path / "pop.npz", corpus="pop")
    assert model["corpus"] == "pop"
    assert len(model["train_ids"]) > 0
    assert all(sid.startswith(("pop909:", "choco:")) for sid in model["train_ids"])
    assert all(sid.startswith(("pop909:", "choco:")) for sid in model["heldout_ids"])
    assert "pop909" in model["stats"] or "choco" in model["stats"]
    assert "accomp_db" not in model["stats"]


def test_zero_arg_load_context_prior_is_still_pooled():
    """The genre-arm brief's hard constraint: another agent's scaffold calls
    load_context_prior() with no args and expects the Phase-1 pooled table —
    that contract must survive the corpus= extension."""
    model = load_context_prior()
    assert model.get("corpus", "pooled") == "pooled"
    assert len(model["train_ids"]) > 3000  # Phase 1 reported 3826


# ── genre-arm: ContextPriorModel object API (span_rescore.py contract) ─────
# Added 2026-07-30: harmonia/models/span_rescore.py's Phase 2 scaffold calls
# `context_scorer.score_candidates(prev, next)` as a bound method on whatever
# load_context_prior() returns. A plain dict has no such method (the scaffold
# was silently degrading to its uniform lam-inert stub). load_context_prior/
# build_context_prior now return a ContextPriorModel (dict subclass).

def test_context_prior_model_is_still_a_plain_dict():
    """Every existing dict-style access must keep working unchanged."""
    m = ContextPriorModel(_empty_model())
    assert isinstance(m, dict)
    assert m["tri"].shape == (N_Q5, N_ROOT, N_Q5, N_ROOT, N_Q5)
    assert "uni" in m
    assert m.get("nonexistent_key") is None


def test_context_prior_model_score_candidates_matches_module_function():
    m = ContextPriorModel(_empty_model())
    prev, nxt = (2, MIN), (0, MAJ)
    qc, dp, qp, dn, qn = trigram_key(prev, (7, DOM), nxt)
    m["tri"][qc, dp, qp, dn, qn] = 20.0
    m["prevbi"][qc, dp, qp] = 20.0
    m["nextbi"][qc, dn, qn] = 20.0

    via_method = m.score_candidates(prev, nxt)
    via_function = score_candidates(prev, nxt, model=m)
    assert np.allclose(via_method, via_function)
    assert via_method[cand_index(7, DOM)] == via_method.max()


def test_build_and_load_return_context_prior_model_instances(tmp_path):
    built = build_context_prior(cache_path=tmp_path / "x.npz", corpus="jazz")
    assert isinstance(built, ContextPriorModel)
    loaded = load_context_prior(cache_path=tmp_path / "x.npz", corpus="jazz")
    assert isinstance(loaded, ContextPriorModel)
    # both expose the bound method, both usable interchangeably with the
    # module-level function via model=
    probs = loaded.score_candidates((2, MIN), (0, MAJ))
    assert probs.shape == (N_CAND,)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_old_cache_without_corpus_key_or_model_class_still_yields_working_scorer(tmp_path):
    """Regression guard: a cache file written by code that predates BOTH the
    corpus= arg and ContextPriorModel (i.e. exactly the Phase 1 artifact
    already on disk at data/cache/chord_context_prior.npz) must still load
    into an object with a working .score_candidates method."""
    cache = tmp_path / "legacy.npz"
    prev, cand, nxt = (2, MIN), (7, DOM), (0, MAJ)
    qc, dp, qp, dn, qn = trigram_key(prev, cand, nxt)
    tri = np.zeros((N_Q5, N_ROOT, N_Q5, N_ROOT, N_Q5), dtype=np.float32)
    prevbi = np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float32)
    nextbi = np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float32)
    tri[qc, dp, qp, dn, qn] = 5.0
    prevbi[qc, dp, qp] = 5.0
    nextbi[qc, dn, qn] = 5.0
    np.savez_compressed(
        cache, tri=tri, prevbi=prevbi, nextbi=nextbi,
        uni=np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        train_ids=np.array(["a:1"], dtype=object),
        heldout_ids=np.array(["a:2"], dtype=object),
        stats_json=np.array("{}"),
        # deliberately NO "corpus" key -- simulates the pre-genre-arm cache
    )
    loaded = load_context_prior(cache_path=cache)
    assert isinstance(loaded, ContextPriorModel)
    probs = loaded.score_candidates(prev, nxt)
    assert probs[cand_index(7, DOM)] == probs.max()
