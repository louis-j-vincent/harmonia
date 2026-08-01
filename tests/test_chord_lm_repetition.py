"""Tests for the within-song repetition cache."""
from __future__ import annotations

import numpy as np

from harmonia_min.chord_lm import vocab
from harmonia_min.chord_lm.repetition import RepetitionCache, fit_lambdas, mix

C = vocab.chord_id(0, "maj")
A = vocab.chord_id(9, "min")
F = vocab.chord_id(5, "maj")
G = vocab.chord_id(7, "dom")
R = vocab.REP


def _aaba() -> list[int]:
    """Two verbatim copies of an 8-slot phrase, then the phrase again."""
    phrase = [C, R, A, R, F, R, G, R]
    return phrase * 3


def test_cache_predicts_the_repeat():
    seq = _aaba()
    cp, mt = RepetitionCache().distributions(seq)
    # position 16 opens the third copy; its 8 predecessors matched once before
    assert mt[16] > 0
    assert int(cp[16].argmax()) == C


def test_cache_is_confident_not_just_correct():
    """Regression: the first implementation used alpha=0.05, spread over 90
    tokens = 4.5 pseudo-counts, so a chord seen once before got p=0.19. Top-1
    was right 94% of the time and mixing it in still moved almost nothing
    (+0.6pp instead of +8.8pp). Accuracy alone would not have caught it."""
    seq = _aaba()
    cp, mt = RepetitionCache().distributions(seq)
    assert cp[16, C] > 0.7, f"copy probability collapsed to {cp[16, C]:.3f}"


def test_cache_never_sees_the_position_it_predicts():
    seq = _aaba()
    cp, mt = RepetitionCache().distributions(seq)
    # first pass through the phrase has nothing earlier to copy from
    assert mt[:8].max() == 0
    assert cp[:8].sum() == 0


def test_longest_match_wins():
    """A 16-slot match needs TWO prior copies behind it, so it can only appear
    from the fourth copy on — with three copies the cache correctly reports 8."""
    three = [C, R, A, R, F, R, G, R] * 3
    _, mt3 = RepetitionCache(widths=(16, 8, 4, 2)).distributions(three)
    assert mt3[16:24].max() == 8

    four = [C, R, A, R, F, R, G, R] * 4
    _, mt4 = RepetitionCache(widths=(16, 8, 4, 2)).distributions(four)
    assert mt4[24] == 16


def test_mix_is_identity_where_nothing_matched():
    T, V = 5, vocab.VOCAB_SIZE
    lp = np.log(np.full((T, V), 1.0 / V, dtype=np.float32))
    cp = np.zeros((T, V), dtype=np.float32)
    mt = np.zeros(T, dtype=np.int32)
    assert np.allclose(mix(lp, cp, mt, {16: 0.9}), lp, atol=1e-5)


def test_mix_moves_toward_the_cache_where_it_matched():
    V = vocab.VOCAB_SIZE
    lp = np.log(np.full((1, V), 1.0 / V, dtype=np.float32))
    cp = np.zeros((1, V), dtype=np.float32)
    cp[0, C] = 1.0
    mt = np.array([8], dtype=np.int32)
    out = np.exp(mix(lp, cp, mt, {8: 0.9}))
    assert out[0, C] > 0.85


def test_fit_lambdas_picks_high_weight_for_a_perfect_cache():
    V = vocab.VOCAB_SIZE
    n = 50
    lp = np.log(np.full((n, V), 1.0 / V, dtype=np.float32))
    y = np.full(n, C, dtype=np.int64)
    cp = np.zeros((n, V), dtype=np.float32)
    cp[:, C] = 1.0
    mt = np.full(n, 8, dtype=np.int32)
    lam = fit_lambdas([lp], [cp], [mt], [y], widths=(8,))
    assert lam[8] >= 0.8


def test_fit_lambdas_ignores_a_useless_cache():
    V = vocab.VOCAB_SIZE
    n = 50
    lp = np.full((n, V), -20.0, dtype=np.float32)
    lp[:, C] = 0.0
    y = np.full(n, C, dtype=np.int64)
    cp = np.zeros((n, V), dtype=np.float32)
    cp[:, G] = 1.0          # cache is confidently WRONG
    mt = np.full(n, 8, dtype=np.int32)
    lam = fit_lambdas([lp], [cp], [mt], [y], widths=(8,))
    assert lam[8] == 0.0
