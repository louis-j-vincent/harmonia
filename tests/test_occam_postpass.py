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
    roots = [A, B, A, B, A, E, A, B, A, B, A, B]   # one spurious E (weak)
    qual = ["maj", "min7"] * 6
    post = _post(roots)
    post[5, E] = 0.30; post[5, A] = 0.28            # E barely beats A -> snap
    nr, nq, dec = occam_compress_bars(roots, qual, post, [0] * len(roots))
    applied = [d for d in dec if d.get("applied")]
    assert applied and set(applied[0]["vocab"]) == {"A", "B"}
    assert nr[5] in (A, B)                           # spurious E snapped into vamp


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
