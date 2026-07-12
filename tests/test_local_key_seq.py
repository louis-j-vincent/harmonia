"""Unit tests for the per-chord local-key SEQUENCE model (#20/#23).

Covers the distillation dataset (per-chord heuristic targets, index alignment,
song-level split), the transpose-equivariance operator, the churn/collection
helpers, and — when a trained checkpoint exists — that the MODEL (not just the
raw heuristic) resolves the user's canonical genuine-collection-change case
Gm7→F / Eb→Bb, and that it is no noisier than its teacher on the "A Beautiful
Friendship" secondary-dominant bridge.
"""
from __future__ import annotations

import pytest

from harmonia.models.local_key_seq_data import (
    DEFAULT_DB,
    build_seq_examples,
    collection_of,
    count_collection_changes,
    heuristic_track_for_tokens,
    rel_to_abs_key,
    split_seq_examples,
    tokens_to_rel_example,
    token_to_q5,
)
from harmonia.models.local_key_seq_model import (
    LocalKeySeqGRU,
    load_seq_model,
    predict_sequence,
)
from harmonia.theory.local_key import parse_token, transpose_token

CKPT = DEFAULT_DB.parent.parent / "cache" / "local_key_seq_gru.pt"


def _pred_abs(model, tokens, gt=0, gmode="major"):
    """Model's ABSOLUTE per-chord key idx for a token stream (via rel encoding)."""
    seq, _ = tokens_to_rel_example(tokens, gt, gmode)
    return [rel_to_abs_key(r, gt) for r in predict_sequence(model, seq)]


# ── helpers ────────────────────────────────────────────────────────────────────
def test_collection_of_relative_major_minor_share_collection():
    # C major (0) and A minor (12+9=21) are the same diatonic collection.
    assert collection_of(0) == collection_of(21) == 0
    assert collection_of(5) == collection_of(12 + 2) == 5   # F major == D minor


def test_count_collection_changes_ignores_relative_flip():
    # C major → A minor → C major is 0 collection changes (same 7 pcs)…
    assert count_collection_changes([0, 21, 0]) == 0
    # …but C major → F major → Bb major is 2.
    assert count_collection_changes([0, 5, 10]) == 2


def test_rel_to_abs_key_roundtrip():
    # F major (5) relative to global C (0) is delta 5; add G (7) back → C major?
    assert rel_to_abs_key(5, 0) == 5                 # C-home: F major stays F
    assert rel_to_abs_key(5, 7) == 0                 # G-home: +5 from G = C major
    assert rel_to_abs_key(12 + 9, 0) == 12 + 9       # minor mode offset preserved


def test_relative_encoding_is_transpose_invariant_by_construction():
    # The same harmonic motif seeded in two different keys yields BIT-IDENTICAL
    # relative (seq, target) — the equivariance the coordinator asked for, not
    # merely learned via augmentation.
    motif = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    seq_c, y_c = tokens_to_rel_example(motif, 0, "major")           # C major
    motif_e = [transpose_token(t, 4, flats=False) for t in motif]   # +4 → E major
    seq_e, y_e = tokens_to_rel_example(motif_e, 4, "major")
    assert seq_c == seq_e
    assert y_c == y_e


# ── dataset ──────────────────────────────────────────────────────────────────
def test_build_seq_examples_aligns_input_and_target():
    ex = build_seq_examples()
    assert ex, "dataset should be non-empty"
    for e in ex[:200]:
        assert len(e["seq"]) == len(e["y"])       # index-aligned
        assert all(0 <= r < 12 and 0 <= q < 5 for r, q in e["seq"])
        assert all(0 <= y < 24 for y in e["y"])


def test_split_is_by_song_and_disjoint():
    ex = build_seq_examples()
    tr, va = split_seq_examples(ex)
    tr_songs = {e["song_idx"] for e in tr}
    va_songs = {e["song_idx"] for e in va}
    assert tr_songs.isdisjoint(va_songs)
    assert len(va) > 0 and len(tr) > 0


# ── heuristic teacher sanity (the labels the model distils) ─────────────────────
def test_heuristic_target_gm7_eb_case():
    # the user's genuine-collection-change spec: in C, Gm7 → F, Eb^7 → Bb.
    names = heuristic_track_for_tokens(["C^7", "G-7", "Eb^7"], 0, "major")
    assert names == [0, 5, 10]   # C major, F major, Bb major


# ── model behaviour (requires trained checkpoint) ───────────────────────────────
def _load_or_skip():
    if CKPT is None or not CKPT.exists():
        pytest.skip("no trained checkpoint — run scripts/train_local_key_seq_model.py")
    return load_seq_model(CKPT)


def test_model_forward_shapes():
    # architecture is well-formed regardless of training
    m = LocalKeySeqGRU()
    seq, _ = tokens_to_rel_example(["C^7", "G-7", "Eb^7"], 0, "major")
    out = predict_sequence(m, seq)
    assert len(out) == 3 and all(0 <= k < 24 for k in out)


def test_model_resolves_genuine_collection_change_gm7_eb():
    # The MODEL (not just the raw heuristic) must still fire the two genuine
    # collection changes Gm7→F and Eb→Bb — smoothing must not blur real jumps.
    m = _load_or_skip()
    pred = _pred_abs(m, ["C^7", "G-7", "Eb^7"])
    assert [collection_of(k) for k in pred] == [0, 5, 10]  # C, F, Bb collections


def test_model_resolves_genuine_change_in_another_key():
    # same genuine-change motif transposed to A major (+9): must still fire the
    # two collection jumps, now A→D→G — checks the equivariance end-to-end.
    m = _load_or_skip()
    toks = [transpose_token(t, 9, flats=False) for t in ["C^7", "G-7", "Eb^7"]]
    pred = _pred_abs(m, toks, gt=9, gmode="major")
    assert [collection_of(k) for k in pred] == [9, 2, 7]  # A, D, G collections


def test_model_not_noisier_than_heuristic_on_abf_bridge():
    # On the secondary-dominant chain (Em7 A7 D7 G7#5) the model should churn
    # collections no MORE than the raw heuristic (goal: less; guard: not more).
    m = _load_or_skip()
    abf = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    heur = heuristic_track_for_tokens(abf, 0, "major")
    pred = _pred_abs(m, abf)
    assert count_collection_changes(pred) <= count_collection_changes(heur)
