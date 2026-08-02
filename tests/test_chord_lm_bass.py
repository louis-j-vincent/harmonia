"""The bass fusion term."""
from __future__ import annotations

import numpy as np

from harmonia_min.chord_lm import bass, vocab


def _onehot(pc: int) -> np.ndarray:
    v = np.zeros(13, dtype=np.float32)
    v[1 + pc] = 1.0
    return v


def test_matrix_rows_are_the_documented_convention():
    """col 0 = N, col i>=1 = pitch class i-1 — determined empirically (93.8%
    agreement with the decoded triad root); it is documented nowhere else."""
    M = bass.chord_bass_matrix()
    assert M.shape == (13, vocab.VOCAB_SIZE)
    c_maj = vocab.chord_id(0, "maj")
    assert M[1 + 0, c_maj] == bass.BASS_WEIGHTS[0]     # root C
    assert M[1 + 4, c_maj] == bass.BASS_WEIGHTS[1]     # third E
    assert M[1 + 7, c_maj] == bass.BASS_WEIGHTS[2]     # fifth G
    assert M[0, vocab.NC] == 1.0
    assert M[:, vocab.REP].sum() == 0.0                # REP borrows, owns nothing


def test_each_chord_column_sums_to_one():
    M = bass.chord_bass_matrix()
    for tok in range(vocab.N_CHORD):
        assert abs(float(M[:, tok].sum()) - 1.0) < 1e-6


def test_a_heard_root_favours_chords_on_that_root():
    lp = bass.token_logp(_onehot(7))                   # a G in the bass
    best = int(np.argmax(lp[:vocab.N_CHORD]))
    assert vocab.split_chord(best)[0] == 7


def test_an_inversion_is_unlikely_but_not_impossible():
    """A C/E sounds an E. Scoring only roots would make every first inversion
    impossible, and pop plays them constantly."""
    lp = bass.token_logp(_onehot(4))                   # an E in the bass
    c_maj = vocab.chord_id(0, "maj")                   # C/E
    e_maj = vocab.chord_id(4, "maj")
    assert lp[c_maj] > np.log(1e-3)                    # allowed
    assert lp[e_maj] > lp[c_maj]                       # but root position wins


def test_bass_cannot_choose_a_quality():
    """The whole reason the LM is still needed: a bass note says nothing about
    maj vs min vs sus."""
    lp = bass.token_logp(_onehot(0))
    same_root = [lp[vocab.chord_id(0, f)] for f in vocab.FAMILIES]
    assert max(same_root) - min(same_root) < 1e-6


def test_fuse_is_a_noop_at_zero_weight():
    lm = np.log(np.full((4, vocab.VOCAB_SIZE), 1.0 / vocab.VOCAB_SIZE, np.float32))
    b = np.tile(_onehot(0), (4, 1))
    out = bass.fuse(lm, b, [vocab.chord_id(0, "maj")] * 4, [None] * 4, 0.0)
    assert np.allclose(out, lm)


def test_fuse_scores_rep_with_the_bass_of_the_chord_it_stands_for():
    """Otherwise REP is the one token the bass can never support and every hold
    gets fused away."""
    V = vocab.VOCAB_SIZE
    lm = np.log(np.full((2, V), 1.0 / V, dtype=np.float32))
    g = vocab.chord_id(7, "maj")
    b = np.stack([_onehot(7), _onehot(7)])             # G sounding throughout
    out = bass.fuse(lm, b, [g, vocab.REP], [g, g], 3.0)
    assert out[1, vocab.REP] > out[1, vocab.chord_id(0, "maj")]


def test_slot_bass_lines_up_with_the_bar_grid():
    from harmonia_min import musx
    plane = np.zeros((400, 13), dtype=np.float32)
    plane[:200, 1 + 0] = 1.0                           # C for the first bar
    plane[200:, 1 + 7] = 1.0                           # G for the second
    t = 200 * musx.FRAME_DT
    out = bass.slot_bass(plane, [0.0, t, 2 * t], 4, slots_per_bar=2)
    assert out.shape == (4, 13)
    assert int(np.argmax(out[0])) == 1 + 0 and int(np.argmax(out[1])) == 1 + 0
    assert int(np.argmax(out[2])) == 1 + 7 and int(np.argmax(out[3])) == 1 + 7
