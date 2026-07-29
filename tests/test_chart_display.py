"""Regrid display re-derivation (harmonia.output.chart_display) — the This Love
lead-sheet display rules: per-section chords-per-bar, loop fold ×N, 1st/2nd
endings, and the compact form string. Built and gated 2026-07-29; see
docs/this_love_target_spec.md and the module docstring.
"""

from __future__ import annotations

from harmonia.output import chart_display as cd
from harmonia.output.chart_display import regrid_display_sections


# ── bar builders ────────────────────────────────────────────────────────────
def _bar(*chords, base=0.0):
    """A bar from (root, q) pairs; ``base`` is the bar's start time (seconds).
    Two chords split the 2-second bar at beat 0 and beat 2."""
    out = []
    for i, (root, q) in enumerate(chords):
        beat = 0 if i == 0 else 2
        t0 = base + beat * 0.5
        out.append({"root": root, "q": q, "c": 0.9, "t0": t0, "t1": t0 + 1.0,
                    "bar": 0, "beat": beat})
    return out


# This Love's three phrase types (4 bars each).
def _A(base, head=7):               # verse, 1 chord/bar (head = bar-1 root; the
    # real decode reads G / C / Bm across passes — a bit of noise so the 2-bar
    # grain doesn't cluster cleanly, matching real audio; bars 2-4 are stable.
    return [_bar((head, ""), base=base + 0), _bar((0, "-"), base=base + 2),
            _bar((5, "-7"), base=base + 4), _bar((2, "h7"), base=base + 6)]


def _B(base, second=False):         # chorus, 2 chords/bar, 2-bar ending
    tail = ([_bar((0, ""), (5, ""), base=base + 4), _bar((8, ""), (7, ""), base=base + 6)]
            if second else
            [_bar((0, "-"), (5, "-"), base=base + 4), _bar((10, ""), (3, ""), base=base + 6)])
    return [_bar((0, "-"), (5, "-"), base=base + 0),
            _bar((10, ""), (3, ""), base=base + 2)] + tail


def _C(base, second=False):         # bridge, 1 chord/bar, 2-bar ending
    tail = ([_bar((7, ""), base=base + 4), _bar((7, "7"), base=base + 6)] if second else
            [_bar((7, "7"), base=base + 4), _bar((0, "-"), base=base + 6)])
    return [_bar((5, "-7"), base=base + 0), _bar((3, "^7"), base=base + 2)] + tail


def _lay(*phrases):
    """Concatenate phrase blocks (each a list of 4 bars) into (bars, n_bars),
    re-timing so bar b starts at 2·b seconds."""
    bars, t = [], 0.0
    for ph in phrases:
        for bar in ph:
            shifted = [{**c, "t0": t + (c["beat"] * 0.5), "t1": t + (c["beat"] * 0.5) + 1.0}
                       for c in bar]
            bars.append(shifted)
            t += 2.0
    return bars, len(bars)


# ── density ─────────────────────────────────────────────────────────────────
class TestDensity:
    def test_verse_collapses_to_one_per_bar(self):
        assert cd._density_one_per_bar(_A(0)) is True

    def test_chorus_keeps_two_per_bar(self):
        assert cd._density_one_per_bar(_B(0)) is False

    def test_apply_density_drops_passing_chord(self):
        block = [_bar((2, "o"), (5, "o"))]        # Ddim + passing Fdim in one bar
        assert [len(b) for b in cd._apply_density(block, one_per_bar=True)] == [1]
        assert cd._apply_density(block, one_per_bar=True)[0][0]["root"] == 2  # downbeat kept
        assert [len(b) for b in cd._apply_density(block, one_per_bar=False)] == [2]


# ── 1st/2nd ending detection ────────────────────────────────────────────────
class TestEndingTail:
    def test_two_bar_root_change_is_tail_2(self):
        assert cd._ending_tail(_B(0), _B(0, second=True)) == 2

    def test_single_bar_root_change_is_tail_1(self):
        e = [_bar((0, "")), _bar((5, "")), _bar((7, "")), _bar((0, ""))]
        o = [_bar((0, "")), _bar((5, "")), _bar((7, "")), _bar((9, "-"))]
        assert cd._ending_tail(e, o) == 1

    def test_pure_quality_wobble_same_root_is_not_an_ending(self):
        # A verse decode noise: G vs G7 (head), Ddim vs Dø (tail) — SAME roots
        e = [_bar((7, "")), _bar((0, "-")), _bar((5, "-7")), _bar((2, "o"))]
        o = [_bar((7, "7")), _bar((0, "-")), _bar((5, "-7")), _bar((2, "h7"))]
        assert cd._ending_tail(e, o) == 0

    def test_identical_passes_no_ending(self):
        assert cd._ending_tail(_A(0), _A(0)) == 0


# ── held-bar fill ───────────────────────────────────────────────────────────
class TestFillHeld:
    def test_forward_fills_empty_bar(self):
        block = [_bar((7, "")), [], _bar((5, ""))]
        filled = cd._fill_held(block)
        assert [b[0]["root"] for b in filled] == [7, 7, 5]   # held G fills bar 1

    def test_backfills_leading_empty(self):
        block = [[], _bar((0, "-")), _bar((5, "-"))]
        filled = cd._fill_held(block)
        assert filled[0][0]["root"] == 0                     # leading empty → next chord


# ── end-to-end: This Love shape ─────────────────────────────────────────────
class TestRegridDisplaySections:
    def _this_love(self):
        # A×3 B A×3 B A C B×3 — the target form. Verse heads carry the real
        # decode's G/C/Bm wobble (7/0/11) so the 2-bar grain isn't spuriously
        # clean; the 4-bar grain still clusters the verses (bars 2-4 stable).
        heads = iter([7, 0, 11, 7, 0, 11, 7])
        A = lambda b: _A(b, head=next(heads))
        phrases = ([A, A, A, lambda b: _B(b), lambda b: _B(b, True)]
                   + [A, A, A, lambda b: _B(b), lambda b: _B(b, True)]
                   + [A]
                   + [lambda b: _C(b), lambda b: _C(b, True)]
                   + [lambda b: _B(b), lambda b: _B(b, True)] * 3)
        built, base = [], 0.0
        for fn in phrases:
            built.append(fn(base))
            base += 8.0
        return _lay(*built)

    def test_form_string_matches_target(self):
        bars, n = self._this_love()
        out = regrid_display_sections(bars, n, tonic_pc=0)
        assert out is not None
        sections, form = out
        assert form == "A×3 B A×3 B A C B×3"

    def test_verse_folds_one_per_bar_no_ending(self):
        bars, n = self._this_love()
        sections, _ = regrid_display_sections(bars, n, tonic_pc=0)
        a = sections[0]
        assert a["label"] == "A" and a["reps"] == 3
        assert len(a["bars"]) == 4 and all(len(b) == 1 for b in a["bars"])   # 1 chord/bar
        assert "endings" not in a
        assert [b[0]["root"] for b in a["bars"]] == [7, 0, 5, 2]             # G Cm Fm Dø

    def test_chorus_has_second_ending_two_per_bar(self):
        bars, n = self._this_love()
        sections, _ = regrid_display_sections(bars, n, tonic_pc=0)
        b = next(s for s in sections if s["label"] == "B")
        assert len(b["bars"][0]) == 2                    # 2 chords/bar
        assert "endings" in b and b["endings"]["tail"] == 2
        assert len(b["endings"]["variants"]) == 2
        assert b["endings"]["variants"][0]["passes"][0] == 0   # timing anchor

    def test_bridge_is_found_with_ending(self):
        bars, n = self._this_love()
        sections, _ = regrid_display_sections(bars, n, tonic_pc=0)
        c = next(s for s in sections if s["label"] == "C")
        assert "endings" in c and c["endings"]["tail"] == 2
        assert [b[0]["root"] for b in c["bars"]] == [5, 3, 7, 0]   # Fm Eb G7 Cm

    def test_final_chorus_folds_times_three(self):
        bars, n = self._this_love()
        sections, _ = regrid_display_sections(bars, n, tonic_pc=0)
        assert sections[-1]["label"] == "B" and sections[-1]["reps"] == 3
        assert len(sections[-1]["spans"]) == 6           # 3 phrase-cycles × 2 passes


class TestGating:
    def test_single_loop_song_defers(self):
        # one 4-bar loop repeated → dominant cluster covers 100% → defer (None)
        bars, n = _lay(*[_A(0) for _ in range(10)])
        assert regrid_display_sections(bars, n, tonic_pc=0) is None

    def test_too_short_defers(self):
        bars, n = _lay(_A(0))
        assert regrid_display_sections(bars, n, tonic_pc=0) is None

    def test_through_composed_defers(self):
        # every 4-bar phrase distinct (no repetition) → coverage 0 → defer
        blocks = []
        for k in range(8):
            blocks.append([_bar((k % 12, "")), _bar(((k + 1) % 12, "")),
                           _bar(((k + 2) % 12, "")), _bar(((k + 3) % 12, ""))])
        bars, n = _lay(*blocks)
        assert regrid_display_sections(bars, n, tonic_pc=0) is None
