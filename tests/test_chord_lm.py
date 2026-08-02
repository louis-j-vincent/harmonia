"""Tests for the half-bar chord-LM tokenizer (harmonia_min/chord_lm)."""
from __future__ import annotations

import random

import pytest

from harmonia_min.chord_lm import grid, vocab
from harmonia_min.chord_lm.from_app import app_chart_to_grid


# ── vocabulary ──────────────────────────────────────────────────────────────
def test_chord_id_roundtrip():
    for pc in range(12):
        for fam in vocab.FAMILIES:
            tok = vocab.chord_id(pc, fam)
            assert vocab.is_chord(tok)
            assert vocab.split_chord(tok) == (pc, fam)


def test_specials_are_not_chords():
    for t in (vocab.NC, vocab.REP, vocab.PAD, vocab.BOS, vocab.EOS, vocab.MASK):
        assert not vocab.is_chord(t)
        with pytest.raises(ValueError):
            vocab.split_chord(t)


def test_transpose_is_cyclic_and_family_preserving():
    for tok in range(vocab.N_CHORD):
        assert vocab.transpose(tok, 12) == tok
        pc, fam = vocab.split_chord(tok)
        for k in range(1, 12):
            pc2, fam2 = vocab.split_chord(vocab.transpose(tok, k))
            assert fam2 == fam
            assert pc2 == (pc + k) % 12


def test_transpose_leaves_specials_alone():
    for t in (vocab.NC, vocab.REP, vocab.PAD, vocab.MASK):
        assert vocab.transpose(t, 5) == t


@pytest.mark.parametrize("token,root,family", [
    ("C", 0, "maj"), ("C^7", 0, "maj"), ("C6", 0, "maj"), ("C69", 0, "maj"),
    ("D-7", 2, "min"), ("D-", 2, "min"), ("D-9", 2, "min"), ("D-^7", 2, "min"),
    ("G7", 7, "dom"), ("G7b9", 7, "dom"), ("G13", 7, "dom"), ("G7#9#5", 7, "dom"),
    ("Bo7", 11, "dim"), ("Bo", 11, "dim"),
    ("Eh7", 4, "hdim"), ("E-7b5", 4, "hdim"),
    ("F7sus", 5, "sus"), ("F7sus4", 5, "sus"),
    ("Ab+", 8, "aug"),
    ("D-7/A", 2, "min"),          # slash bass dropped: functional root
    ("sC7", 0, "dom"),            # iReal 'small chord' size prefix
    ("F#-7", 6, "min"), ("Bb^7", 10, "maj"),
])
def test_parse_ireal_token(token, root, family):
    assert vocab.parse_ireal_token(token) == vocab.chord_id(root, family)


def test_parse_ireal_specials():
    assert vocab.parse_ireal_token("n") == vocab.NC
    assert vocab.parse_ireal_token("p") == vocab.REP
    assert vocab.parse_ireal_token("zzz") is None


def test_quality_to_family_never_silently_defaults():
    """Unknown qualities must be None, not maj — the bug labels.py warns about."""
    assert vocab.quality_to_family("totally-not-a-chord") is None


# ── within-bar placement ────────────────────────────────────────────────────
@pytest.mark.parametrize("n,beats,want", [
    (1, 4, [0]), (2, 4, [0, 2]), (3, 4, [0, 2, 3]), (4, 4, [0, 1, 2, 3]),
    (2, 3, [0, 2]), (3, 3, [0, 1, 2]), (5, 4, [0, 1, 2, 3]),
])
def test_place_in_bar(n, beats, want):
    assert grid.place_in_bar(n, beats) == want


def test_bar_to_slots_halfbar():
    C = vocab.chord_id(0, "maj")
    F = vocab.chord_id(5, "maj")
    G = vocab.chord_id(7, "dom")
    A = vocab.chord_id(9, "min")
    assert grid.bar_to_slots([C], 2, 4) == [C, C]
    assert grid.bar_to_slots([C, F], 2, 4) == [C, F]
    # 3 chords = 2+1+1, so the 2nd half-bar starts on the SECOND chord
    assert grid.bar_to_slots([C, F, G], 2, 4) == [C, F]
    assert grid.bar_to_slots([C, F, G, A], 2, 4) == [C, G]


def test_bar_to_slots_quarterbar_keeps_all_four():
    toks = [vocab.chord_id(p, "maj") for p in (0, 2, 4, 5)]
    assert grid.bar_to_slots(toks, 4, 4) == toks


def test_empty_bar_is_all_none():
    assert grid.bar_to_slots([], 2, 4) == [None, None]


# ── REP encoding ────────────────────────────────────────────────────────────
def test_repeat_roundtrip_simple():
    C, F, G = (vocab.chord_id(p, q) for p, q in ((0, "maj"), (5, "maj"), (7, "dom")))
    absolute = [C, C, F, F, F, vocab.NC, F, G]
    toks, n_rep = grid.apply_repeats(absolute)
    assert toks == [C, vocab.REP, F, vocab.REP, vocab.REP, vocab.NC, vocab.REP, G]
    assert n_rep == 4
    assert grid.expand_repeats(toks) == absolute


def test_first_slot_is_never_rep():
    C = vocab.chord_id(0, "maj")
    toks, _ = grid.apply_repeats([C, C, C])
    assert toks[0] == C


def test_repeat_roundtrip_random():
    rng = random.Random(0)
    for _ in range(200):
        absolute = [rng.choice([vocab.chord_id(rng.randrange(12),
                                               rng.choice(vocab.FAMILIES)),
                                vocab.NC]) for _ in range(rng.randrange(2, 40))]
        toks, _ = grid.apply_repeats(absolute)
        assert vocab.REP not in toks[:1]
        assert grid.expand_repeats(toks) == absolute


def test_unmapped_slot_inherits_previous_chord():
    C = vocab.chord_id(0, "maj")
    toks, _ = grid.apply_repeats([C, None, None])
    assert toks == [C, vocab.REP, vocab.REP]


def test_leading_unmapped_becomes_nc():
    toks, _ = grid.apply_repeats([None, None])
    assert toks == [vocab.NC, vocab.NC]


# ── chart -> grid ───────────────────────────────────────────────────────────
def test_chart_to_grid_autumn_leaves_head():
    measures = [("A", ["C-7"]), ("A", ["F7"]), ("A", ["Bb^7"]), ("A", ["Eb^7"]),
                ("A", ["Ah7"]), ("A", ["D7b13"]), ("A", ["G-6"]), ("A", ["G-6"])]
    g = grid.chart_to_grid(measures, title="Autumn Leaves")
    assert g.n_bars == 8
    assert len(g.tokens) == 16
    assert g.n_unmapped == 0
    expect_bar_starts = ["C:min", "F:dom", "Bb:maj", "Eb:maj",
                         "A:hdim", "D:dom", "G:min", "%"]
    got = [vocab.token_name(g.tokens[2 * i]) for i in range(8)]
    assert got == expect_bar_starts
    # every second slot holds
    assert all(g.tokens[2 * i + 1] == vocab.REP for i in range(8))


def test_grid_to_bars_inverts():
    measures = [("A", ["C"]), ("A", ["A-7", "D7"]), ("A", ["G^7"])]
    g = grid.chart_to_grid(measures)
    bars = grid.grid_to_bars(g)
    assert bars == [
        [vocab.chord_id(0, "maj"), vocab.chord_id(0, "maj")],
        [vocab.chord_id(9, "min"), vocab.chord_id(2, "dom")],
        [vocab.chord_id(7, "maj"), vocab.chord_id(7, "maj")],
    ]


# ── app charts ──────────────────────────────────────────────────────────────
def _bar(*chords):
    return [{"root": r, "q": q, "bass": -1, "nc": False, "beat": b}
            for r, q, b in chords]


def test_app_chart_uses_explicit_beats():
    chart = {"title": "t", "bpb": 4, "sections": [{
        "label": "A", "barRanges": [[0, 2]],
        "bars": [_bar((0, "", 0)), _bar((9, "-7", 0), (2, "7", 2)),
                 _bar((7, "^7", 0))]}]}
    g = app_chart_to_grid(chart)
    assert [vocab.token_name(t) for t in g.tokens] == [
        "C:maj", "%", "A:min", "D:dom", "G:maj", "%"]


def test_app_chart_beat4_chord_is_dropped_but_carries_out():
    """A half-bar grid cannot hold a beat-4 chord — same limit as the iReal path.

    The 2nd half-bar starts on beat 2, where C is still sounding, so the slot is
    C and the G7 is not representable. It must still CARRY into the next bar:
    dropping it from the grid is a resolution limit, not the chord ending.
    """
    chart = {"title": "t", "bpb": 4, "sections": [{
        "label": "A", "barRanges": [[0, 1]],
        "bars": [_bar((0, "", 0), (7, "7", 3)), []]}]}
    g = app_chart_to_grid(chart)
    assert [vocab.token_name(t) for t in g.tokens] == [
        "C:maj", "%", "G:dom", "%"]


def test_app_chart_leading_slot_takes_a_chord_starting_inside_it():
    """Nothing sounds at slot 0's start (no pickup carry) — take what starts inside."""
    chart = {"title": "t", "bpb": 4, "sections": [{
        "label": "A", "barRanges": [[0, 0]],
        "bars": [_bar((7, "7", 1))]}]}
    g = app_chart_to_grid(chart)
    assert [vocab.token_name(t) for t in g.tokens] == ["G:dom", "%"]


def test_app_chart_nc_and_carry():
    chart = {"title": "t", "bpb": 4, "sections": [{
        "label": "A", "barRanges": [[0, 1]],
        "bars": [[{"root": 0, "q": "", "bass": -1, "nc": True, "beat": 0}],
                 _bar((0, "", 0))]}]}
    g = app_chart_to_grid(chart)
    assert [vocab.token_name(t) for t in g.tokens] == ["N.C.", "N.C.", "C:maj", "%"]


def test_app_chart_slash_bass_ignored():
    """D-7/A and D-7 must be the same token: the LM's roots are functional."""
    a = app_chart_to_grid({"bpb": 4, "sections": [{"label": "A", "barRanges": [[0, 0]],
        "bars": [[{"root": 2, "q": "-7", "bass": 9, "nc": False, "beat": 0}]]}]})
    b = app_chart_to_grid({"bpb": 4, "sections": [{"label": "A", "barRanges": [[0, 0]],
        "bars": [[{"root": 2, "q": "-7", "bass": -1, "nc": False, "beat": 0}]]}]})
    assert a.tokens == b.tokens


# ── corpus-level (slow-ish but cheap: symbolic only) ────────────────────────
def test_ireal_corpus_vocabulary_is_essentially_exhaustive():
    from harmonia_min.chord_lm import corpus
    charts = corpus.load_charts(min_bars=8)
    assert len(charts) > 1500, "iReal corpus should yield >1500 4/4 charts"
    sym = sum(c.n_symbols for c in charts)
    unm = sum(c.n_unmapped for c in charts)
    assert unm / sym < 1e-3, f"{unm}/{sym} chord symbols unmapped"


def test_splits_are_song_disjoint_and_stable():
    from harmonia_min.chord_lm import corpus
    s = corpus.load_splits(min_bars=8)
    titles = [{c.title for c in x} for x in (s.train, s.val, s.test)]
    assert not (titles[0] & titles[1]) and not (titles[0] & titles[2])
    assert not (titles[1] & titles[2])
    assert corpus.split_of("Autumn Leaves") == corpus.split_of("Autumn Leaves")


# ── display folding: sections stored as a template + occurrence ranges ──────
def test_app_chart_expands_folded_repetitions():
    """Regression (2026-08-02): display folding stores a section ONCE as its
    repeating template plus the bar ranges where it recurs. Emitting the
    template once — correct before folding, when every bar was written out —
    turned Let It Be's 70 bars into 4 and cost the LM the repetition it exists
    to exploit. It degraded silently; nothing raised."""
    tmpl = [_bar((0, "", 0)), _bar((7, "", 0))]
    chart = {"title": "t", "bpb": 4, "nBars": 6, "sections": [{
        "label": "A", "reps": 3, "barRanges": [[0, 1], [2, 3], [4, 5]],
        "bars": tmpl}]}
    g = app_chart_to_grid(chart)
    assert g.n_bars == 6, "occurrences must be expanded onto the real bar grid"
    assert [vocab.token_name(t) for t in g.tokens] == [
        "C:maj", "%", "G:maj", "%", "C:maj", "%",
        "G:maj", "%", "C:maj", "%", "G:maj", "%"]


def test_app_chart_cycles_a_template_shorter_than_its_occurrence():
    tmpl = [_bar((0, "", 0)), _bar((7, "", 0))]
    chart = {"title": "t", "bpb": 4, "nBars": 4, "sections": [{
        "label": "A", "reps": 1, "barRanges": [[0, 3]], "bars": tmpl}]}
    g = app_chart_to_grid(chart)
    assert g.n_bars == 4
    assert [vocab.token_name(t) for t in g.tokens[::2]] == [
        "C:maj", "G:maj", "C:maj", "G:maj"]


def test_app_chart_grid_length_matches_nbars():
    """Bar indices must line up with chart['barGrid'], or every timestamp a
    suggestion reports is wrong."""
    chart = {"title": "t", "bpb": 4, "nBars": 5, "sections": [{
        "label": "A", "reps": 1, "barRanges": [[1, 3]],
        "bars": [_bar((0, "", 0))]}]}
    g = app_chart_to_grid(chart)
    assert g.n_bars == 5
    assert len(g.sections) == 5
