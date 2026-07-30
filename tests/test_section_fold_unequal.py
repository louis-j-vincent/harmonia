"""Under-fold, never over-fold: occurrences of unequal length are separate sections.

Louis, 2026-07-30: "Yes we shouldn't have folded this. Why was it folded? I'd
rather under fold than over fold every time."

The case that prompted it — Norah Jones, "Don't Know Why", section B, written
once as 8 bars (`G-7 | C7 | A# | A# | A# | A# | D#^7 | D`) but occurring as
**4** real bars (song bars 24-27) and **8** real bars (124-131). A real bar there
is 1.358 s, so the 8 written bars of the short pass got 0.680 s each and the
highlight ran through that section at double speed — correct at both ends, up to
two bars ahead in the middle.

Folding across unequal lengths was deliberate: the fold used to key on
(letter, merged tails) alone, so that a verse the decoder heard as 8, 12 and 4
bars still rendered as one verse instead of A, A¹, A². The cost is that the
playhead cannot line up inside the odd-length pass — there is no correct answer
to draw. Louis's rule settles the trade-off in favour of showing the difference.
"""
import pytest

from harmonia.output.chart_display import _fold_units


def _u(label, bar0, bar1, *, tails=(), d_bars=4):
    parts = [{"label": label}] + [{"label": t} for t in tails]
    return {"parts": parts, "bar0": bar0, "bar1": bar1,
            "d_bars": d_bars, "label": label}


def test_equal_length_occurrences_still_fold_into_one_section():
    """The fold must keep folding what it should — this is the regression guard."""
    units = [_u("A", 0, 8), _u("A", 16, 24), _u("A", 32, 40)]
    order, groups, suffix = _fold_units(units)
    assert len(order) == 1
    assert len(groups[order[0]]) == 3
    assert suffix[order[0]] == ""


def test_unequal_length_occurrences_become_separate_sections():
    """Don't Know Why's B: 4 bars and 8 bars are not the same section."""
    units = [_u("B", 24, 28), _u("B", 124, 132)]
    order, groups, suffix = _fold_units(units)
    assert len(order) == 2, "a 4-bar and an 8-bar occurrence must not fold together"
    assert sorted(len(groups[k]) for k in order) == [1, 1]
    assert sorted(suffix[k] for k in order) == ["", "′"]


def test_the_most_played_length_keeps_the_plain_letter():
    """This Love's verse plays 8 bars twice, 12 once, 4 once — the 8-bar one is
    the verse, the others are the variants."""
    units = [_u("A", 0, 8), _u("A", 8, 20), _u("A", 20, 28), _u("A", 28, 32)]
    order, groups, suffix = _fold_units(units)
    assert len(order) == 3
    plain = [k for k in order if suffix[k] == ""]
    assert len(plain) == 1
    assert len(groups[plain[0]]) == 2, "the twice-played 8-bar length is canonical"
    assert (plain[0][2] if len(plain[0]) > 2 else None) == 8


def test_merged_tails_still_separate_sections_with_the_same_letter():
    """`B→C` and `B→E` were already kept apart; length must not undo that."""
    units = [_u("B", 0, 8, tails=("C",)), _u("B", 16, 24, tails=("E",))]
    order, _groups, _suffix = _fold_units(units)
    assert len(order) == 2


def test_kill_switch_restores_the_old_over_folding(monkeypatch):
    monkeypatch.setenv("HARMONIA_FOLD_UNEQUAL", "1")
    units = [_u("B", 24, 28), _u("B", 124, 132)]
    order, groups, _suffix = _fold_units(units)
    assert len(order) == 1
    assert len(groups[order[0]]) == 2


def test_play_order_is_preserved():
    units = [_u("A", 0, 8), _u("B", 8, 12), _u("A", 12, 20)]
    order, _groups, _suffix = _fold_units(units)
    assert order[0][0] == "A" and order[1][0] == "B"


@pytest.mark.parametrize("n_variants", [2, 3, 4])
def test_each_variant_gets_a_distinct_label(n_variants):
    units = [_u("A", i * 40, i * 40 + 4 * (i + 1)) for i in range(n_variants)]
    order, _groups, suffix = _fold_units(units)
    assert len(order) == n_variants
    assert len({suffix[k] for k in order}) == n_variants
