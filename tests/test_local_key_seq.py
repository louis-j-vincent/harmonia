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
    DOM_Q5,
    NO_NEXT,
    build_rel_example,
    build_seq_examples,
    collection_of,
    count_collection_changes,
    heuristic_track_for_tokens,
    rel_features,
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
    # relative (seq, features, BOTH targets) — the equivariance the coordinator
    # asked for, not merely learned via augmentation.
    motif = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    ex_c = build_rel_example(motif, 0, "major")                     # C major
    motif_e = [transpose_token(t, 4, flats=False) for t in motif]   # +4 → E major
    ex_e = build_rel_example(motif_e, 4, "major")
    for k in ("seq", "intervals", "dom_prep", "y", "y_v2"):
        assert ex_c[k] == ex_e[k], k


# ── relational features (#23 follow-up) ─────────────────────────────────────────
def test_rel_features_flags_descending_fifth_dominant_prep():
    # A7 (dominant) → D7 is a descending fifth (interval 5) ⇒ dom_prep=1; the
    # last chord has no successor ⇒ interval NO_NEXT, dom_prep 0.
    motif = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    ex = build_rel_example(motif, 0, "major")
    intervals, dom_prep = ex["intervals"], ex["dom_prep"]
    a7 = motif.index("A7")
    assert intervals[a7] == 5 and dom_prep[a7] == 1           # A7 → D7
    assert intervals[motif.index("D7")] == 5 and dom_prep[motif.index("D7")] == 1
    # E-7 (minor) → A7 is a fifth but NOT a dominant-prep (wrong quality)
    e7 = motif.index("E-7")
    assert intervals[e7] == 5 and dom_prep[e7] == 0
    assert intervals[-1] == NO_NEXT and dom_prep[-1] == 0

    # rel_features on the raw seq must agree with build_rel_example
    seq, _ = tokens_to_rel_example(motif, 0, "major")
    assert rel_features(seq) == (intervals, dom_prep)


def test_dom_q5_is_the_dominant_family_index():
    assert token_to_q5("A7") == DOM_Q5
    assert token_to_q5("C^7") != DOM_Q5


def test_build_rel_example_v3_consolidates_dominant_chain():
    # the distillation TARGET (y = v3) reads the ABF tail as one key, unlike the
    # raw v2 target (y_v2) it is derived from.
    motif = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    ex = build_rel_example(motif, 0, "major")
    y3 = [rel_to_abs_key(r, 0) for r in ex["y"]]
    y2 = [rel_to_abs_key(r, 0) for r in ex["y_v2"]]
    assert count_collection_changes(y3) < count_collection_changes(y2)
    assert len({collection_of(k) for k in y3[4:]}) == 1      # tail = one collection


# ── dataset ──────────────────────────────────────────────────────────────────
def test_build_seq_examples_aligns_input_and_target():
    ex = build_seq_examples()
    assert ex, "dataset should be non-empty"
    for e in ex[:200]:
        n = len(e["seq"])
        assert len(e["y"]) == len(e["y_v2"]) == n          # all index-aligned
        assert len(e["intervals"]) == len(e["dom_prep"]) == n
        assert all(0 <= r < 12 and 0 <= q < 5 for r, q in e["seq"])
        assert all(0 <= y < 24 for y in e["y"])
        assert all(0 <= iv <= NO_NEXT for iv in e["intervals"])
        assert all(d in (0, 1) for d in e["dom_prep"])


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


# The genuine-borrowing spec Gm7→F / Eb→Bb, tested IN CONTEXT. A bare 3-chord
# toy is out of distribution for a whole-song model (a dangling final Eb^7 is
# ambiguous between its own key and IV-of-Bb); embedded in a real turnaround the
# model tracks the borrowing exactly — that is the capability that must survive
# the dominant-chain smoothing.
_BORROW_CTX = ["C^7", "A-7", "D-7", "G7", "C^7", "G-7", "C7", "F^7", "Eb^7", "Bb^7"]
_BORROW_COLLS = [0, 0, 0, 0, 0, 5, 5, 5, 10, 10]   # C…C, F(=Gm7 region), Bb(Eb→Bb)


def test_model_resolves_genuine_collection_change_gm7_eb():
    # The MODEL (not just the raw heuristic) must still fire the genuine
    # collection changes C→F (Gm7) and F→Bb (Eb→Bb) — smoothing must not blur
    # real jumps. Tested in a song-length context (see _BORROW_CTX note).
    m = _load_or_skip()
    pred = _pred_abs(m, _BORROW_CTX)
    assert [collection_of(k) for k in pred] == _BORROW_COLLS


def test_model_resolves_genuine_change_in_another_key():
    # same genuine-change context transposed to A major (+9): the collection
    # track shifts by +9 — checks the equivariance end-to-end.
    m = _load_or_skip()
    toks = [transpose_token(t, 9, flats=False) for t in _BORROW_CTX]
    pred = _pred_abs(m, toks, gt=9, gmode="major")
    assert [collection_of(k) for k in pred] == [(c + 9) % 12 for c in _BORROW_COLLS]


def test_model_still_leaves_home_on_borrowed_chord_toy():
    # Guard even on the bare toy: smoothing must not collapse a genuine borrowing
    # to a single home key. The model need not nail the exact tonic on a dangling
    # final maj7 (I-vs-IV ambiguous out of context), but it MUST fire ≥2 distinct
    # collections away from home — the jump is not blurred.
    m = _load_or_skip()
    pred = _pred_abs(m, ["C^7", "G-7", "Eb^7"])
    colls = [collection_of(k) for k in pred]
    assert colls[0] == 0 and len(set(colls)) == 3        # C, then two more, all distinct


def test_model_not_noisier_than_heuristic_on_abf_bridge():
    # On the secondary-dominant chain (Em7 A7 D7 G7#5) the model should churn
    # collections strictly LESS than the raw v2 heuristic (the #23 payoff).
    m = _load_or_skip()
    abf = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    heur = heuristic_track_for_tokens(abf, 0, "major")
    pred = _pred_abs(m, abf)
    assert count_collection_changes(pred) < count_collection_changes(heur)


def test_model_reads_dominant_chain_as_single_key():
    # The direct #23 goal: with the relational feature + consolidated target, the
    # trained MODEL reads the descending-fifths tail E-7 A7 D7 G7#5 as ONE key
    # (its resolution, C major = the home), not 3–4 collections flickering past.
    m = _load_or_skip()
    abf = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
    pred = _pred_abs(m, abf)
    assert len({collection_of(k) for k in pred[4:]}) == 1     # tail = one key
    assert collection_of(pred[-1]) == 0                        # resolves to C
