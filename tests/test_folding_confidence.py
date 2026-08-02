"""Folded chords must report an ACOUSTIC confidence, on the same scale as
unfolded ones.

Measured 2026-08-01 on GuitarSet (879 decoded spans, verified GT shipped with
the audio):

    branch                              n    root acc   AUC(root)
    acoustic (_segment_confidence)    712      0.875      0.819
    folding  (0.5 + 0.08*n_obs)       167      0.952      0.019
    pooled into one `c` field         879      0.890      0.800

Folded spans are the MOST accurate in the set and carried the LOWER confidence
(0.66 vs a 0.784 acoustic median), because `0.5 + 0.08*n_obs` is a repetition
count on an unaligned scale. Pooling the two cost 2 points of AUC, which is the
difference between a confidence worth putting in the intervention rule and one
that is not.

The fix is not to drop the repetition signal — it stays in `n_obs` — but to put
in `c` the acoustic posterior of the chord the fold actually decoded, measured
on the AVERAGED template it decoded from. Averaging n observations legitimately
raises that posterior, so the number rises for the right reason and stays
comparable to every unfolded chord.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min import folding, musx


FOLD_FORMULA = {round(min(0.97, 0.5 + 0.08 * k), 3) for k in range(1, 30)}


def _flat(bars):
    return [c for bar in bars for c in bar]


def _fake_probs(n_frames: int, root: int = 0, fam: int = 1, strength: float = 0.8):
    """A musx-shaped posterior stack whose triad plane favours one chord.

    Real shape (musx.frame_posteriors): [triad(73), bass(13), s7(4), s9(4),
    s11(3), s13(3)] — six planes, not five.
    """
    triad = np.full((n_frames, 73), (1.0 - strength) / 72.0, dtype=np.float32)
    triad[:, 1 + (fam - 1) * 12 + root] = strength
    rest = [np.full((n_frames, k), 1.0 / k, dtype=np.float32)
            for k in (13, 4, 4, 3, 3)]
    return [triad] + rest


def test_label_confidence_matches_the_pipeline_formula():
    """musx.label_confidence must be the same quantity pipeline used to compute
    inline — otherwise folded and unfolded chords land on different scales
    again, which is the whole bug."""
    triad = _fake_probs(400, root=0, fam=1, strength=0.9)[0]
    c = musx.label_confidence(triad, 0.0, 400 * musx.FRAME_DT, "C:maj")
    assert c == pytest.approx(0.9, abs=1e-3)
    # a chord that is NOT what the plane favours must score low
    c2 = musx.label_confidence(triad, 0.0, 400 * musx.FRAME_DT, "F:min")
    assert c2 < 0.05


def test_label_confidence_handles_no_chord_and_unknown_quality():
    triad = _fake_probs(100)[0]
    assert 0.0 <= musx.label_confidence(triad, 0.0, 1.0, "N") <= 1.0
    # unknown quality falls back to the neutral 0.5 rather than crashing
    assert musx.label_confidence(triad, 0.0, 1.0, "C:notaquality") == 0.5
    # empty window falls back too
    assert musx.label_confidence(triad, 5.0, 5.0, "C:maj") == 0.5


def test_write_position_takes_a_confidence_per_chord():
    """Regression: `_write_position` used to take ONE confidence for the whole
    bar (the folding formula). Two chords in one bar can differ in how well the
    averaged evidence supports them, and must be allowed to say so."""
    bars = [[{"root": 9, "q": "-", "bass": -1, "nc": False, "beat": 0}]]
    grid = [0.0, 2.0]
    chords_k = [
        {"root": 0, "q": "", "bass": -1, "nc": False, "beat": 0, "c": 0.91},
        {"root": 7, "q": "7", "bass": -1, "nc": False, "beat": 2, "c": 0.42},
    ]
    folding._write_position(bars, grid, 0, chords_k, 4, n_obs=6)
    got = [c["c"] for c in bars[0]]
    assert got == [0.91, 0.42]
    assert all(c["n_obs"] == 6 for c in bars[0])
    assert all(c["folded"] is True for c in bars[0])


def test_folded_confidence_is_not_the_repetition_count():
    """The load-bearing assertion. A folded chord's `c` must be an acoustic
    posterior, never a value the folding formula could have produced from
    n_obs alone."""
    bars = [[{"root": 9, "q": "-", "bass": -1, "nc": False, "beat": 0}]]
    grid = [0.0, 2.0]
    chords_k = [{"root": 0, "q": "", "bass": -1, "nc": False, "beat": 0,
                 "c": 0.913}]
    folding._write_position(bars, grid, 0, chords_k, 4, n_obs=6)
    c = bars[0][0]["c"]
    assert c == 0.913
    assert c not in FOLD_FORMULA, (
        f"c={c} is a value 0.5+0.08*n_obs can produce — the repetition count "
        "is still being laundered into the confidence field")


def test_template_chords_attaches_acoustic_confidence():
    """`_template_chords` must return chords carrying `c`, computed from the
    averaged posteriors it decoded from.

    The period alternates C and G so the template actually contains a chord
    CHANGE. A period that decodes to one chord throughout produces no event
    starting inside the middle tile and `_template_chords` returns None — a
    pre-existing edge case, unrelated to confidence, noted in known_issues.
    """
    Lf = 40                      # frames per bar in the template
    P = 2
    c_maj = _fake_probs(Lf, root=0, fam=1, strength=0.85)
    g_maj = _fake_probs(Lf, root=7, fam=1, strength=0.85)

    def bar_probs(b):
        return c_maj if b % 2 == 0 else g_maj

    pos_members = [[0, 2], [1, 3]]
    out = folding._template_chords(pos_members, bar_probs, len(c_maj), Lf, 4, P)
    assert out is not None
    n_with_conf = 0
    for pos in out:
        for ch in pos:
            assert "c" in ch, "template chords must carry an acoustic confidence"
            assert 0.0 <= ch["c"] <= 1.0
            assert ch["c"] not in FOLD_FORMULA
            n_with_conf += 1
    assert n_with_conf >= 2
    # the planes strongly favour their chords, so the decode should be confident
    assert max(ch["c"] for pos in out for ch in pos) > 0.5
