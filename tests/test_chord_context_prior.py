"""Tests for harmonia/models/chord_context_prior.py (Phase 1, issue: chord
context prior for lock propagation, docs/design_chord_context_prior.md)."""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.chord_context_prior import (
    N_CAND,
    N_Q5,
    N_ROOT,
    cand_decode,
    cand_index,
    chord_name,
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
