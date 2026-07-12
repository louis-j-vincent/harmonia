"""Transpose-invariance tests for the key-relative local-key features (#20/#23).

Both the Volet-1 reranker (chord_pipeline_v1.rerank_local_key_qualities) and the
Volet-2 ctx-model feature block (train_ctx_model_v2) are meant to be **key
agnostic**: they encode a chord root only as a *scale degree relative to the
local key*, never as an absolute tonic. So transposing a whole song by any number
of semitones must leave the features — and therefore the predictions — bit-for-
bit identical. These tests pin that invariant, the same property validated today
for LocalKeySeqGRU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TOKENS = ["C", "A-", "D-7", "G7", "C", "E-7", "A7", "D-7", "G7", "C"]  # ii-V's in C
FLAT_TO_SHARP = {  # iReal writes flats; transpose by ± semitones via pc math
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb", 7: "G",
    8: "Ab", 9: "A", 10: "Bb", 11: "B",
}


def _transpose_token(tok: str, semis: int) -> str:
    """Transpose an iReal token's root by ``semis`` (keeps quality tail)."""
    from harmonia.theory.local_key import parse_token
    root, qual, bass = parse_token(tok)
    new_root = FLAT_TO_SHARP[(root + semis) % 12]
    out = new_root + qual
    if bass is not None:
        out += "/" + FLAT_TO_SHARP[(bass + semis) % 12]
    return out


# ── Volet 2: training-script feature block ─────────────────────────────────────

def test_song_local_key_labels_degree_invariant():
    """(degree, mode) labels are identical for a song and its transpositions."""
    from train_ctx_model_v2 import _song_local_key_labels
    from harmonia.theory.local_key import parse_token

    roots0 = [parse_token(t)[0] for t in TOKENS]
    base = _song_local_key_labels(TOKENS, roots0, home_tonic=0, home_mode="major",
                                  mode="v2")
    for semis in (1, 3, 5, 7, 11):
        toks = [_transpose_token(t, semis) for t in TOKENS]
        roots = [(r + semis) % 12 for r in roots0]
        got = _song_local_key_labels(toks, roots, home_tonic=semis % 12,
                                     home_mode="major", mode="v2")
        assert got == base, f"transpose +{semis} changed local-key degree labels"


def test_localkey_ctx_onehots_transpose_invariant():
    """The 9×13 windowed feature block is identical under transposition."""
    from train_ctx_model_v2 import _localkey_ctx_onehots, LK_POS_DIM, LK_DEG_DIM
    from harmonia.theory.local_key import parse_token
    from train_ctx_model_v2 import _song_local_key_labels

    def build(semis):
        toks = [_transpose_token(t, semis) for t in TOKENS]
        roots = [(parse_token(t)[0] + semis) % 12 for t in TOKENS]
        labs = _song_local_key_labels(toks, roots, home_tonic=semis % 12,
                                      home_mode="major", mode="v2")
        recs = [{"lk_degree": d, "lk_mode": m} for d, m in labs]
        return _localkey_ctx_onehots(recs)

    base = build(0)
    assert base.shape[1] == (2 * 4 + 1) * LK_POS_DIM  # 9 positions × 13
    # each active position is a valid one-hot degree + a {0,1} mode bit
    for semis in (2, 5, 9):
        np.testing.assert_array_equal(build(semis), base)


def test_localkey_block_off_is_all_zero():
    """No lk_degree on the records → an all-zero block (baseline compatibility)."""
    from train_ctx_model_v2 import _localkey_ctx_onehots
    recs = [{"root_pc": 0}, {"root_pc": 7}, {"root_pc": 9}]
    assert not _localkey_ctx_onehots(recs).any()
