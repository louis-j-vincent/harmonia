"""Unit tests for harmonia/models/section_structure.py (issue #22).

No audio — synthetic per-beat chord sequences with a planted AABA form.
"""
from __future__ import annotations

import numpy as np

from harmonia.models.section_structure import build_chord_ssm, detect_section_boundaries, label_sections

BPB = 4


def _section(root_qual_bars: list[tuple[int, int]], bpb: int = BPB) -> list[tuple[int, int]]:
    """Expand a list of per-bar (root, qual) into a per-beat sequence."""
    seq: list[tuple[int, int]] = []
    for root, qual in root_qual_bars:
        seq.extend([(root, qual)] * bpb)
    return seq


# An 8-bar A phrase (ii-V-I-ish) and a contrasting 8-bar B (bridge) phrase.
A8 = [(2, 1), (7, 2), (0, 3), (0, 3), (2, 1), (7, 2), (0, 3), (0, 3)]
B8 = [(9, 1), (2, 2), (7, 3), (7, 3), (11, 1), (4, 2), (9, 3), (9, 3)]


def test_build_chord_ssm_shape_and_diagonal():
    seq = _section(A8)
    ssm = build_chord_ssm(seq)
    assert ssm.shape == (len(seq), len(seq))
    # diagonal is self-similarity == 1 for non-empty beats
    assert np.allclose(np.diagonal(ssm), 1.0)
    # cosine similarity stays in [0, 1]
    assert ssm.min() >= 0.0 and ssm.max() <= 1.0 + 1e-6
    # symmetric
    assert np.allclose(ssm, ssm.T)


def test_empty_and_short_inputs():
    assert build_chord_ssm([]).shape == (0, 0)
    # song shorter than the smallest candidate section -> no boundaries
    assert detect_section_boundaries(build_chord_ssm(_section(A8)), BPB) == []


def test_detect_aaba_boundaries():
    # iReal "A16 B8 A8": A A B A, 32 bars. GT interior boundaries at the label
    # changes: bar 16 (A->B, beat 64) and bar 24 (B->A, beat 96).
    seq = _section(A8 + A8 + B8 + A8)
    ssm = build_chord_ssm(seq)
    bnds = detect_section_boundaries(ssm, beats_per_bar=BPB)
    assert bnds == [64, 96], bnds


def test_repeated_a_phrases_are_merged_not_split():
    # The two adjacent identical A8 phrases (bars 0-8, 8-16) must merge into one
    # A16 section, i.e. NO boundary at bar 8 (beat 32).
    seq = _section(A8 + A8 + B8 + A8)
    bnds = detect_section_boundaries(build_chord_ssm(seq), beats_per_bar=BPB)
    assert 32 not in bnds


def test_aba_32bar_form():
    # A8 B8 A8 C8 -> boundaries at bars 8, 16, 24 (beats 32, 64, 96); no A-merge
    # because the A phrases are non-adjacent.
    C8 = [(5, 1), (10, 2), (3, 3), (3, 3), (5, 1), (10, 2), (3, 3), (3, 3)]
    seq = _section(A8 + B8 + A8 + C8)
    bnds = detect_section_boundaries(build_chord_ssm(seq), beats_per_bar=BPB)
    assert bnds == [32, 64, 96], bnds


def test_no_chord_beats_yield_zero_rows():
    """Beats with root<0 must produce all-zero rows/cols in the SSM (documented behaviour)."""
    # Mix normal beats and no-chord beats (root=-1)
    seq: list[tuple[int | None, int]] = [(0, 0), (-1, 0), (0, 0), (-1, -1)]
    ssm = build_chord_ssm(seq)
    assert ssm.shape == (4, 4)
    # Rows and columns for no-chord beats must be all zero
    assert np.all(ssm[1] == 0.0), "no-chord row should be all-zero"
    assert np.all(ssm[:, 1] == 0.0), "no-chord col should be all-zero"
    assert np.all(ssm[3] == 0.0), "no-chord row should be all-zero"
    # Valid beats still have self-similarity 1.0
    assert np.isclose(ssm[0, 0], 1.0)
    assert np.isclose(ssm[2, 2], 1.0)


def test_all_same_chord_produces_no_boundaries():
    """All beats identical → every adjacent pair merges → one section, no interior boundaries."""
    # 32 bars of the same chord — all cross-similarities are 1.0, everything merges
    seq = [(0, 0)] * (32 * BPB)
    bnds = detect_section_boundaries(build_chord_ssm(seq), beats_per_bar=BPB)
    assert bnds == [], bnds


def test_label_sections_aaba():
    """label_sections: explicit 4-section AABA boundary → ['A', 'A', 'B', 'A'].

    Uses hand-specified boundary_beats (0, 32, 64, 96, 128) so the test isolates
    the labelling algorithm from detect_section_boundaries' merging behaviour.

    B8_sharp uses roots 9-11 and quality indices 5-7 (no overlap with A8's
    roots 0/2/7 and qualities 1-3), so the A-B cross-section cosine is ~0
    and S[A0,A1] >> 0.70 > S[B,A].
    """
    # Bridge phrase with zero harmonic overlap with A8
    B8_sharp = [(9, 5), (10, 6), (11, 7), (11, 7), (9, 5), (10, 6), (11, 7), (11, 7)]
    seq = _section(A8 + A8 + B8_sharp + A8)
    ssm = build_chord_ssm(seq)
    # Explicit 4-section boundary: A(0-32), A(32-64), B(64-96), A(96-128)
    cut_beats = [0, 32, 64, 96, len(seq)]
    labels = label_sections(ssm, cut_beats)
    assert labels == ["A", "A", "B", "A"], f"Expected ['A','A','B','A'], got {labels}"


# ── bar-locked repetition-first section pass (2026-07-19) ──────────────────────
from harmonia.models.section_structure import barlocked_sections  # noqa: E402


def _bars_to_times(nbars: int, barlen: float = 2.0):
    return [(b * barlen, (b + 1) * barlen) for b in range(nbars)]


# Two 2-bar LOOPS (the Mayer failure case, generalised): A = Emaj7|F#m7 (roots
# 0|7), B = G#m7|F#m7 (roots 4|7) — F#m7 shared, Emaj7 vs G#m7 discriminative.
_A_LOOP = [(0, 0), (7, 1)]           # E | F#
_B_LOOP = [(4, 1), (7, 1)]           # G# | F#
_INTRO = [(11, 2), (5, 0)]           # 2 bars of non-recurring junk
_SONG = (_INTRO + _A_LOOP * 4 + _B_LOOP * 4 + _A_LOOP * 4 + _B_LOOP * 2
         + _A_LOOP * 4)              # intro, A, B, A, B, A


def test_barlocked_finds_intro_and_loop_families():
    """Derived-grain: the 2-bar loop is detected; leading junk = Intro; the
    Emaj7|F#m7 loop = A (most bars); the G#m7|F#m7 loop = B."""
    secs = barlocked_sections(_SONG, _bars_to_times(len(_SONG)))
    assert secs, "expected non-empty sections"
    assert secs[0]["label"] == "Intro"
    labels = [s["label"] for s in secs]
    assert "A" in labels and "B" in labels
    from collections import Counter
    freq = Counter(s["label"] for s in secs if s["label"] != "Intro"
                   for _ in range(s["n_bars"]))
    assert freq.most_common(1)[0][0] == "A"  # A = the loop with the most bars


def test_barlocked_boundaries_are_loop_unit_aligned():
    """Every boundary lands on a 2-bar loop-unit multiple (never inside a loop)."""
    secs = barlocked_sections(_SONG, _bars_to_times(len(_SONG)))
    assert secs
    b = 0
    for s in secs:
        assert b % 2 == 0, f"boundary at bar {b} splits a loop unit"
        b += s["n_bars"]


def test_barlocked_no_intro_when_song_opens_on_a_loop():
    """A song that opens directly on a loop (no non-recurring junk) has no Intro."""
    bars = _A_LOOP * 4 + _B_LOOP * 4 + _A_LOOP * 4 + _B_LOOP * 2 + _A_LOOP * 4
    secs = barlocked_sections(bars, _bars_to_times(len(bars)))
    assert secs
    assert secs[0]["label"] != "Intro"
    assert secs[0]["label"] == "A"


def test_barlocked_single_chord_loop_defers():
    """A pure single-content loop collapses to one label -> [] (caller defers to
    the acoustic detector)."""
    bars = [(0, 0)] * 32
    assert barlocked_sections(bars, _bars_to_times(len(bars))) == []


def test_barlocked_accepts_posterior_matrix():
    """The production input is a per-bar (n_bars, 12) root-posterior matrix; a
    near-one-hot soft matrix reproduces the tuple-input Intro + loop result."""
    rng = np.random.default_rng(0)
    mat = np.zeros((len(_SONG), 12))
    for i, (r, _q) in enumerate(_SONG):
        mat[i, r % 12] = 1.0
    mat = mat + 0.02 * rng.random(mat.shape)  # soft perturbation
    secs = barlocked_sections(mat, _bars_to_times(len(_SONG)))
    assert secs
    assert secs[0]["label"] == "Intro"
    assert {"A", "B"} <= {s["label"] for s in secs}


def test_barlocked_short_song_defers():
    assert barlocked_sections([(0, 0)] * 4, _bars_to_times(4)) == []
