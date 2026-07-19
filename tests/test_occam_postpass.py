"""Occam post-pass: minimal-vocabulary loop compression (opt-in).

known_issues.md 2026-07-19 ★ CHORDS — Mayer Hawthorne 'Henny & Gingerale' A|Bm7
vamp polluted with spurious E7 / quality wobble. The pass snaps a 2-chord vamp
onto its dominant reciprocal-bigram vocabulary, keeping only margin-surviving
deviations. Uses ONLY the song's own structure (no corpus grammar prior).
"""
from __future__ import annotations
import numpy as np
from harmonia.models.chord_pipeline_v1 import occam_compress_bars, NOTE

A, B, E = 9, 11, 7


def _post(roots, strong=0.7):
    p = np.full((len(roots), 12), 0.02)
    for i, r in enumerate(roots):
        p[i, r] = strong
    return p


def test_two_chord_vamp_snaps_spurious_chord():
    roots = [A, B, A, B, A, E, A, B, A, B, A, B]   # bar 5 (pos 1) should be B
    qual = ["maj", "min7"] * 6
    post = _post(roots)
    post[5, E] = 0.30; post[5, B] = 0.25            # weak E, low calibrated conf
    conf = np.full(len(roots), 0.6); conf[5] = 0.20
    nr, nq, dec = occam_compress_bars(roots, qual, post, [0] * len(roots), bar_conf=conf)
    applied = [d for d in dec if d.get("applied")]
    assert applied and set(applied[0]["vocab"]) == {"A", "B"} and applied[0]["period"] == 2
    assert nr[5] == B                                # low-conf spurious E snapped to the loop


def test_high_evidence_deviation_survives_margin():
    roots = [A, B, A, B, A, B, E, B, A, B, A, B]     # bar6 = strong E turnaround
    qual = ["maj", "min7"] * 6
    post = _post(roots)
    post[6] = 0.02; post[6, E] = 0.9; post[6, A] = 0.03   # decisive E evidence
    nr, nq, dec = occam_compress_bars(roots, qual, post, [0] * len(roots))
    assert nr[6] == E                                 # kept as a real turnaround
    assert any(d.get("kept_deviation") for d in dec)


def test_through_composed_family_abstains():
    # no dominant 2-chord alternation -> leave untouched
    roots = [0, 2, 4, 5, 7, 9, 11, 0, 4, 7, 2, 5]
    post = _post(roots)
    nr, nq, dec = occam_compress_bars(roots, [None] * 12, post, [0] * 12)
    assert nr == roots                                # unchanged
    assert all(not d.get("applied") for d in dec)


def test_short_family_untouched():
    roots = [A, B, A]
    post = _post(roots)
    nr, _, dec = occam_compress_bars(roots, [None] * 3, post, [0, 0, 0])
    assert nr == roots


def test_bayes_arbitration_low_conf_snaps_high_conf_keeps():
    """Part C (2026-07-19): per-bar Bayes arbitration — an UNCERTAIN off-vamp bar
    defers to the coherent pattern; a CONFIDENT one keeps its own harmony."""
    import numpy as np
    from harmonia.models.chord_pipeline_v1 import occam_compress_bars, NOTE
    A, B, E = 9, 11, 7
    roots = [A, B, A, B, A, B, E, B, A, B, A, B, E, B]   # bars 6,12 off-vamp (E)
    qual = ["maj", "min7"] * 7
    post = np.full((len(roots), 12), 0.02)
    for i, r in enumerate(roots):
        post[i, r] = 0.7
    # bar 6: LOW calibrated conf -> should SNAP; bar 12: HIGH conf -> should KEEP
    conf = np.full(len(roots), 0.5)
    conf[6] = 0.20; post[6, E] = 0.55; post[6, A] = 0.20
    conf[12] = 0.85; post[12, E] = 0.85; post[12, A] = 0.03
    nr, nq, dec = occam_compress_bars(roots, qual, post, [0]*len(roots), bar_conf=conf)
    assert nr[6] in (A, B)      # low-confidence deviation snapped into the vamp
    assert nr[12] == E          # high-confidence deviation kept
