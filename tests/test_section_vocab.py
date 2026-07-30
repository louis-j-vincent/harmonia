"""Vocabulary section detection (harmonia.models.section_vocab).

The acceptance target is Louis's hand-written This Love lead sheet
(docs/this_love_target_spec.md), reproduced end-to-end in
`TestThisLoveEndToEnd`. The rest are red-first regression tests for bugs that
actually shipped during the build — each names the wrong behaviour it pins.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import section_vocab as sv


def _bar(*chords, bpb=4):
    """A bar from (root, q) pairs, placed at beat 0 and beat bpb/2."""
    return [{"root": r, "q": q, "c": 0.9, "beat": 0 if i == 0 else bpb // 2,
             "t0": 0.0, "t1": 1.0}
            for i, (r, q) in enumerate(chords)]


def _nc_bar():
    """A bar whose only event is a no-chord — the decoder declining to name it."""
    return [{"root": 0, "q": "N", "c": 0.0, "nc": True, "beat": 0,
             "t0": 0.0, "t1": 1.0}]


# ── the SSM must encode Louis's chord distance ───────────────────────────────

class TestChordSSM:
    def test_bb_is_closer_to_gm_than_to_f(self):
        """Louis's standing requirement, 2026-07-30, stated as an absolute."""
        S = sv.chord_ssm(["Bb", "G-", "F"])
        assert S[0, 1] > S[0, 2]
        assert S[0, 1] == pytest.approx(0.701, abs=0.01)
        assert S[0, 2] == pytest.approx(0.276, abs=0.01)

    def test_quality_wobble_still_reads_as_the_same_chord(self):
        """G vs G7 and Dø vs Dm7b5 are the decoder wobbling, not a chord change."""
        S = sv.chord_ssm(["G", "G7", "Do", "Dh7"])
        assert S[0, 1] > 0.95
        assert S[2, 3] > 0.95

    def test_different_sections_stay_far_apart(self):
        S = sv.chord_ssm(["G", "C-", "Bb"])
        assert S[0, 1] < 0.35 and S[1, 2] < 0.35

    def test_true_zero_to_one_scale(self):
        """No 0.5 floor. `build_chord_ssm` with a constant quality slot has one,
        which halved the dynamic range of every downstream curve (CLAUDE.md #1)."""
        S = sv.chord_ssm(["C", "Gb"])
        assert S[0, 0] == pytest.approx(1.0)
        assert S[0, 1] < 0.1


# ── the slot layer ───────────────────────────────────────────────────────────

class TestBuildSlots:
    def test_held_chord_forward_fills_and_stays_known(self):
        bars = [_bar((7, "")), []]
        tokens, _roots, known = sv.build_slots(bars, 2)
        assert tokens == ["G", "G", "G", "G"]
        assert all(known)

    def test_no_chord_is_unknown_and_inherits_nothing(self):
        """RED-FIRST for the phantom-section bug: bar 28 of This Love is a
        no-chord, and forward-filling it with bar 27's passing Bb△7 dropped the
        verse repeat below threshold, minting a bogus section."""
        bars = [_bar((10, "^7")), _nc_bar()]
        tokens, _roots, known = sv.build_slots(bars, 2)
        assert tokens[:2] == ["Bb^7", "Bb^7"]
        assert known[2] is False          # NOT filled with Bb^7
        assert tokens[2] == ""

    def test_two_chords_land_in_separate_half_bars(self):
        tokens, _roots, _known = sv.build_slots([_bar((0, "-"), (5, "-"))], 1)
        assert tokens == ["C-", "F-"]


class TestStripeDiag:
    def test_exact_repeat_scores_one(self):
        tokens = ["C-", "F-", "C-", "F-"]
        S = sv.chord_ssm(tokens)
        assert sv.stripe_diag(S, 0, 2)[2] == pytest.approx(1.0)

    def test_unknown_slots_abstain_rather_than_penalise(self):
        """An unnamed chord must neither help nor hurt a match."""
        tokens = ["C-", "F-", "", "F-"]
        S = sv.chord_ssm(tokens)
        known = [True, True, False, True]
        assert sv.stripe_diag(S, 0, 2, known)[2] == pytest.approx(1.0)
        # without the mask the unknown slot scores 0 and drags the mean down
        assert sv.stripe_diag(S, 0, 2)[2] < 0.6

    def test_too_little_evidence_is_nan(self):
        tokens = ["C-", "F-", "", ""]
        S = sv.chord_ssm(tokens)
        assert np.isnan(sv.stripe_diag(S, 0, 2, [True, True, False, False])[2])


class TestMinimalPeriod:
    def test_finds_the_loop_not_a_multiple_of_it(self):
        """A multiple of the true period always scores at least as well, so
        argmax picks a multiple; the smallest passing lag is the answer."""
        tokens = ["C-", "F-"] * 8
        S = sv.chord_ssm(tokens)
        roots = [0, 5] * 8
        assert sv.minimal_period(S, roots, 0, len(tokens)) == 2

    def test_rejects_a_held_chord_as_a_pattern(self):
        """A chord held h slots trivially matches itself at every lag <= h, which
        made a held `Cm Cm` a perfect 1-bar 'pattern' scoring 1.00."""
        tokens = ["C-"] * 8
        S = sv.chord_ssm(tokens)
        assert sv.minimal_period(S, [0] * 8, 0, 8) is None

    def test_returns_none_when_nothing_reproduces_the_span(self):
        tokens = ["C", "D", "E", "Gb", "Ab", "Bb", "C", "D"]
        S = sv.chord_ssm(tokens)
        assert sv.minimal_period(S, [0, 2, 4, 6, 8, 10, 0, 2], 0, 8) is None


# ── knowing when to stay quiet ───────────────────────────────────────────────

class TestDeferGates:
    def test_single_loop_song_defers(self):
        """One loop repeated has no section structure to report; "the whole song
        is A×10" is true and useless."""
        bars = [b for _ in range(10) for b in
                (_bar((7, "")), _bar((0, "-")), _bar((5, "-7")), _bar((2, "h7")))]
        assert sv.vocab_sections(bars, len(bars)) is None

    def test_through_composed_defers(self):
        """RED-FIRST: with no repetition anywhere, the hole-is-the-pattern
        fallback made the ENTIRE SONG one 'pattern' matching itself once.

        The root sequence has to be genuinely APERIODIC. A first attempt used
        ``k % 12`` over 32 bars, which has period 12 — and the detector correctly
        found that 12-bar cycle repeating twice, failing the test for the right
        reason. Alternating qualities keep any accidental root coincidence from
        reading as a repeat.
        """
        roots = [0, 4, 7, 2, 9, 5, 11, 3, 8, 1, 6, 10, 0, 5, 9, 4,
                 11, 2, 7, 1, 8, 3, 10, 6, 0, 7, 2, 9, 4, 11, 5, 1]
        bars = [_bar((r, "" if i % 3 else "-")) for i, r in enumerate(roots)]
        assert sv.vocab_sections(bars, len(bars)) is None

    def test_too_short_defers(self):
        assert sv.vocab_sections([_bar((0, ""))], 1) is None

    def test_no_named_chords_defers(self):
        assert sv.vocab_sections([_nc_bar() for _ in range(16)], 16) is None


# ── end-to-end against Louis's lead sheet ────────────────────────────────────

class TestThisLoveEndToEnd:
    TARGET = "A×4 B×3 C A×3 B×3 C A D B×3 E B×3 E B×3 E"

    def _model(self):
        import os
        from harmonia.serving.render import _chart_model_for
        prev = os.environ.get("HARMONIA_REGRID")
        os.environ["HARMONIA_REGRID"] = "1"
        try:
            return _chart_model_for("inferred_maroon_5_this_love.html")
        finally:
            if prev is None:
                os.environ.pop("HARMONIA_REGRID", None)
            else:
                os.environ["HARMONIA_REGRID"] = prev

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_form_matches_the_lead_sheet(self):
        assert self._model().get("form") == self.TARGET

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_every_verse_opens_on_g(self):
        """RED-FIRST: `_fill_held` could only look inside a block, so a verse
        whose first bar holds a chord over from the previous bar backfilled from
        its SECOND bar and printed `Cm | Cm | Fm | Dø`, losing the G. Louis: "I'm
        telling you the A bars always open on G"."""
        for s in self._model()["sections"]:
            if s["label"] == "A":
                assert s["bars"][0][0]["root"] == 7, f"A at {s['barRanges'][0]}"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_spans_partition_time_so_the_playhead_is_unambiguous(self):
        """RED-FIRST: `_span_of` measures a block by chord sustain, and a chord's
        t1 runs to the next chord's onset — so the chorus tail's G7, which really
        does ring through bar 24, made that section's span overrun the verse that
        owns bar 24. The audit measured a 3.78 s overlap; the highlight sat on the
        verse while the bridge sounded."""
        spans = sorted(sp for s in self._model()["sections"] for sp in s["spans"])
        worst = max((a[1] - b[0] for a, b in zip(spans, spans[1:])), default=0.0)
        assert worst <= 0.01, f"sections overlap by {worst:.2f}s"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_verse_is_one_chord_per_bar_and_chorus_is_two(self):
        secs = self._model()["sections"]
        a = next(s for s in secs if s["label"] == "A")
        b = next(s for s in secs if s["label"] == "B")
        assert all(len(bar) == 1 for bar in a["bars"])       # verse: 1 chord/bar
        assert len(b["bars"][0]) == 2                        # chorus: 2 chords/bar

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_every_written_section_meets_the_minimum(self):
        """Louis's fixed rule: a section is at least 8 bars. The one documented
        exception is a lone phrase whose only neighbour is a full-length section
        (This Love's 4-bar verse at bars 44-47) — forcing that one would merge a
        verse into a bridge, which is worse than being short."""
        for s in self._model()["sections"]:
            assert len(s["bars"]) >= 8, f"{s['label']} is {len(s['bars'])} bars"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_each_section_written_exactly_once(self):
        """The minimal chart: "if you've already written the A section, you don't
        write it again". Every occurrence still has to be reachable for playback."""
        secs = self._model()["sections"]
        assert len(secs) == len({s["label"] for s in secs})
        covered = sum(b - a + 1 for s in secs for a, b in s["barRanges"])
        assert covered == self._model()["nBars"]

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "docs/plots/inferred_maroon_5_this_love.html").exists(),
        reason="This Love chart payload not present")
    def test_two_choruses_collapse_into_one_with_two_endings(self):
        """The a-priori rule: two sections differing only in their last bars are
        ONE section with 1st/2nd endings, named after the base letter. This Love's
        two choruses differ only in the final bar (`Ab G7` vs `Ab`)."""
        b = next(s for s in self._model()["sections"] if s["label"] == "B")
        assert "endings" in b, "the two chorus variants did not collapse"
        v = b["endings"]["variants"]
        assert [x["label"] for x in v] == ["B1", "B2"]
        assert b["endings"]["tail"] <= 2
        # the two endings must actually differ, and partition the occurrences
        assert v[0]["bars"] != v[1]["bars"]
        passes = sorted(p for x in v for p in x["passes"])
        assert passes == list(range(len(b["barRanges"])))


class TestMinSectionBarsIsTweakable:
    """Louis: "sections defined as a minimum of 4 or 8 bars, should be an option
    we can tweak"."""

    def test_env_var_changes_the_minimum(self, monkeypatch):
        from harmonia.output import chart_display as cd
        monkeypatch.setenv("HARMONIA_MIN_SECTION_BARS", "4")
        assert cd._min_section_bars() == 4
        monkeypatch.setenv("HARMONIA_MIN_SECTION_BARS", "16")
        assert cd._min_section_bars() == 16

    def test_garbage_falls_back_to_the_default(self, monkeypatch):
        from harmonia.output import chart_display as cd
        monkeypatch.setenv("HARMONIA_MIN_SECTION_BARS", "not-a-number")
        assert cd._min_section_bars() == cd._MIN_SECTION_BARS
        monkeypatch.setenv("HARMONIA_MIN_SECTION_BARS", "0")
        assert cd._min_section_bars() == cd._MIN_SECTION_BARS

    def test_a_lower_minimum_keeps_shorter_sections(self):
        """At a 4-bar minimum the 4-bar verse loop is its own unit, so grouping
        must not glue two loops together."""
        from harmonia.output.chart_display import _group_to_min_bars
        vocab = [{"label": "A", "bar0": 0, "bar1": 16, "d_bars": 4, "reps": 4}]
        assert len(_group_to_min_bars(vocab, min_bars=4)) == 4
        assert len(_group_to_min_bars(vocab, min_bars=8)) == 2
