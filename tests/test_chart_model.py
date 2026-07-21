"""ChartModel adapter + the key-string parser it depends on."""

from __future__ import annotations

import pytest

from harmonia.output.chart_interactive import _parse_home_key
from harmonia.output.chart_model import to_chart_model


class TestParseHomeKey:
    """Two key-string dialects reach this: the iReal DB format ("G-", "Ab")
    and pipeline_v1's global_key ("G# major"). The word "major" contains an
    'm', which used to fall through to the minor branch — so every real-audio
    chart was baked with mode="minor" regardless of its actual mode.
    """

    @pytest.mark.parametrize("key,expected", [
        ("C major", (0, "major")),          # regression: was (0, "minor")
        ("G# major", (8, "major")),         # regression: was (8, "minor")
        ("Bb major", (10, "major")),
        ("F minor", (5, "minor")),
        ("G# minor", (8, "minor")),
        ("Ab", (8, "major")),               # iReal DB: bare letter = major
        ("G-", (7, "minor")),               # iReal DB: trailing '-' = minor
        ("Cm", (0, "minor")),
        ("Cmaj", (0, "major")),
        ("", (0, "major")),
    ])
    def test_mode_and_tonic(self, key, expected):
        assert _parse_home_key(key) == expected


def _chord(bar, beat, root, q, c, t0, t1):
    return {"root": root, "bass": -1, "bar": bar, "beat": beat, "t0": t0, "t1": t1,
            "lv": {"family": {"q": "", "c": c}, "seventh": {"q": q, "c": c},
                   "exact": {"q": q, "c": c}}}


class TestToChartModel:
    def test_sections_come_from_chips_not_per_bar_labels(self):
        """Real-audio charts put the KEY NAME in the per-bar `sections` array;
        the actual form is in `sectionChips`. Trusting the per-bar array yields
        one section named "G# major" spanning the whole tune."""
        payload = {
            "nBars": 4, "bpb": 4, "home": {"tonic": 8, "mode": "minor"},
            "sections": ["G# major"] * 4,
            "sectionChips": [{"label": "A", "start_s": 0.0}, {"label": "B", "start_s": 4.0}],
            "chords": [_chord(b, 0, b, "-7", 0.8, b * 2.0, b * 2.0 + 2.0) for b in range(4)],
            "keyName": "G# major",
        }
        m = to_chart_model(payload, filename="inferred_x.html")
        assert [s["label"] for s in m["sections"]] == ["A", "B"]
        assert m["key"] == {"tonic": 8, "mode": "major"}   # keyName overrides stale home.mode
        assert sum(len(s["bars"]) for s in m["sections"]) == 4

    def test_repeats_fold_with_reps_and_both_spans(self):
        bars = [_chord(0, 0, 0, "^7", 0.9, 0.0, 2.0), _chord(1, 0, 7, "7", 0.9, 2.0, 4.0),
                _chord(2, 0, 0, "^7", 0.9, 4.0, 6.0), _chord(3, 0, 7, "7", 0.9, 6.0, 8.0)]
        payload = {"nBars": 4, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A", "A", "A", "A"],
                   "sectionChips": [{"label": "A", "start_s": 0.0}, {"label": "A", "start_s": 4.0}],
                   "chords": bars}
        m = to_chart_model(payload, filename="f.html")
        assert len(m["sections"]) == 1                 # rendered once…
        assert m["sections"][0]["reps"] == 2           # …badged ×2
        assert len(m["sections"][0]["bars"]) == 2      # never printed twice
        assert m["sections"][0]["spans"] == [[0.0, 4.0], [4.0, 8.0]]   # both passes addressable

    def test_third_chord_in_a_bar_is_dropped_by_confidence(self):
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [],
                   "chords": [_chord(0, 0, 0, "", 0.9, 0.0, 0.6),
                              _chord(0, 1, 2, "-7", 0.2, 0.6, 1.2),   # weakest → dropped
                              _chord(0, 2, 7, "7", 0.7, 1.2, 2.0)]}
        m = to_chart_model(payload, filename="f.html")
        bar = m["sections"][0]["bars"][0]
        assert len(bar) == 2
        assert [c["root"] for c in bar] == [0, 7]      # kept in time order, not conf order

    def test_trusted_import_keeps_up_to_four_chords_per_bar(self):
        """Follow-up 2026-07-21: the ≤2/bar cap is right for noisy audio but
        drops a real 3-4-chord walking turnaround on a TRUSTED iReal import
        ("il manque les 4 accords de la dernière barre"). Trusted imports keep
        up to 4; audio decodes still cap at 2."""
        chords = [_chord(0, b, r, "7", 1.0, b * 0.5, b * 0.5 + 0.5)
                  for b, r in enumerate([0, 2, 4, 7])]           # 4 chords in bar 0
        base = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                "sections": ["A"], "sectionChips": [], "chords": chords}
        trusted = to_chart_model({**base, "sections_trusted": True}, filename="f.html")
        assert len(trusted["sections"][0]["bars"][0]) == 4       # all kept
        noisy = to_chart_model(base, filename="f.html")
        assert len(noisy["sections"][0]["bars"][0]) == 2         # audio still capped

    def test_slash_bass_flows_through(self):
        """Follow-up 2026-07-21: a slash-chord's sounding bass pc reaches the
        ChartModel so the app glyph can render the "/D" suffix."""
        ch = _chord(0, 0, 10, "6", 1.0, 0.0, 2.0)
        ch["bass"] = 2                                            # A#6/D
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [], "chords": [ch]}
        m = to_chart_model(payload, filename="f.html")
        assert m["sections"][0]["bars"][0][0]["bass"] == 2

    def test_legacy_label_only_sidecar_does_not_crash(self):
        """Regression, 2026-07-21: annotation files written before the root/q
        schema settled (docs/annotation_sidecar_schema.md) store a plain
        {"label": "A-"} instead of {"root": int, "q": str} — confirmed via 3
        real /api/library entries all crashing with KeyError('root') on a
        file dated 2026-07-13. _normalize_fix parses the label; this exercises
        the FULL to_chart_model path (not just the parser in isolation) to
        pin the fix at the actual crash site.
        """
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [],
                   "chords": [_chord(0, 0, 0, "", 0.5, 0.0, 2.0)]}
        ann = {"chords": [{"bar": 0, "beat": 0, "label": "A-"}]}
        m = to_chart_model(payload, filename="f.html", annotation=ann)
        c = m["sections"][0]["bars"][0][0]
        assert (c["root"], c["q"], c["confirmed"]) == (9, "-", True)

    def test_unparseable_legacy_sidecar_is_ignored_not_crashed(self):
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [],
                   "chords": [_chord(0, 0, 0, "", 0.5, 0.0, 2.0)]}
        ann = {"chords": [{"bar": 0, "beat": 0, "label": "???"}]}
        m = to_chart_model(payload, filename="f.html", annotation=ann)
        c = m["sections"][0]["bars"][0][0]
        assert c["root"] == 0 and c.get("confirmed") is None   # original chord, untouched

    def test_sidecar_correction_locks_the_chord(self):
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [],
                   "chords": [_chord(0, 0, 7, "-7b5", 0.31, 0.0, 2.0)]}
        ann = {"chords": [{"bar": 0, "beat": 0, "root": 7, "q": "7"}]}
        m = to_chart_model(payload, filename="f.html", annotation=ann)
        c = m["sections"][0]["bars"][0][0]
        assert (c["root"], c["q"], c["c"], c["confirmed"]) == (7, "7", 1.0, True)


class TestNoChordCells:
    """N (no-chord) cells: musx's 'N' must render as N.C., never a bogus C.

    Regression for known_issues.md 2026-07-19 ★ CHORDS / NO-CHORD: the intro of
    Mayer Hawthorne 'Henny & Gingerale' (musx N 0-18.3s) was rendered as a run of
    invented chords at nonzero confidence.
    """

    def test_nc_flag_yields_sentinel_q_and_zero_conf(self):
        c = _chord(0, 0, 0, "", 0.9, 0.0, 2.0)
        c["nc"] = True                      # marked no-chord by chart_interactive
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [], "chords": [c]}
        m = to_chart_model(payload, filename="f.html")
        cell = m["sections"][0]["bars"][0][0]
        assert cell.get("nc") is True
        assert cell["q"] == "N"             # sentinel, distinct from C major ""
        assert cell["c"] == 0.0             # confidence clamped

    def test_sidecar_correction_overrides_nc(self):
        c = _chord(0, 0, 0, "", 0.0, 0.0, 2.0)
        c["nc"] = True
        payload = {"nBars": 1, "bpb": 4, "home": {"tonic": 0, "mode": "major"},
                   "sections": ["A"], "sectionChips": [], "chords": [c]}
        ann = {"chords": [{"bar": 0, "beat": 0, "root": 9, "q": "-7"}]}
        m = to_chart_model(payload, filename="f.html", annotation=ann)
        cell = m["sections"][0]["bars"][0][0]
        assert cell.get("nc") is None and (cell["root"], cell["q"]) == (9, "-7")


def test_musx_no_chord_per_segment_flags_only_explicit_N():
    from harmonia.models.musx_bass import no_chord_per_segment
    labels = [(0.0, 5.0, "N"), (5.0, 10.0, "A:maj"), (10.0, 12.0, "X")]
    # seg midpoints: 2.5 (N), 7.5 (chord), 11.0 (X), 20.0 (no overlap)
    segs = [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0), (18.0, 22.0)]
    mask = no_chord_per_segment(labels, segs)
    assert list(mask) == [True, False, True, False]


class TestFoldAndRelabel:
    """Directive 2/3 (2026-07-19): iReal-style loop fold + rank relabel."""

    @staticmethod
    def _bar(root, q):
        return [{"root": root, "q": q, "c": 0.9, "t0": 0.0, "t1": 1.0, "bar": 0, "beat": 0}]

    def test_clean_two_bar_loop_folds(self):
        from harmonia.output.chart_model import _fold_bar_run
        A, Bm = self._bar(9, ""), self._bar(11, "-7")
        bars = [A, Bm] * 6                       # clean A|Bm7 vamp ×6
        folded = _fold_bar_run(bars, 0, "A")
        assert folded is not None
        assert len(folded) == 1 and folded[0]["reps"] == 6
        assert len(folded[0]["bars"]) == 2       # the loop shown once
        assert len(folded[0]["spans"]) == 6      # every pass addressable

    def test_noisy_run_abstains(self):
        from harmonia.output.chart_model import _fold_bar_run
        # no dominant loop block → None (never crush into a fake loop)
        roots = [0, 2, 4, 5, 7, 9, 11, 1, 3, 6, 8, 10]
        bars = [self._bar(r, "") for r in roots]
        assert _fold_bar_run(bars, 0, "A") is None

    def test_relabel_by_reps_ranks_chronological_first_appearance(self):
        # 2026-07-20 (user correction): letters follow FIRST-APPEARANCE order,
        # not repetition count -- "la première partie c'est A, pas l'inverse".
        # Repetition count no longer decides the LETTER, only whether two
        # blocks are the same cluster (content-based merging, unchanged).
        from harmonia.output.chart_model import _relabel_by_reps
        # distinct content types by chord ROOTS: {0,7} vs {2,9}
        verse = [[{"root": 0, "q": ""}], [{"root": 7, "q": "7"}]]
        chorus = [[{"root": 2, "q": "-"}], [{"root": 9, "q": ""}]]
        secs = [{"label": "C", "id": "C", "reps": 2, "bars": chorus},
                {"label": "D", "id": "D", "reps": 5, "bars": verse},
                {"label": "Intro", "id": "Intro", "reps": 1, "bars": [[]]}]
        _relabel_by_reps(secs)
        assert secs[0]["label"] == "A"           # chorus: appears first chronologically
        assert secs[1]["label"] == "B"           # verse: appears second, despite more reps
        assert secs[2]["label"] == "Intro"       # untouched

    def test_relabel_same_content_same_letter(self):
        from harmonia.output.chart_model import _relabel_by_reps
        loop = [[{"root": 9, "q": ""}], [{"root": 11, "q": "-7"}]]
        # two non-adjacent occurrences of the SAME content must share a letter
        secs = [{"label": "A", "id": "A", "reps": 1, "bars": loop},
                {"label": "B", "id": "B", "reps": 1, "bars": [[{"root": 2, "q": ""}]]},
                {"label": "C", "id": "C", "reps": 1, "bars": loop}]
        _relabel_by_reps(secs)
        assert secs[0]["label"] == secs[2]["label"]      # same content → same letter
        assert secs[1]["label"] != secs[0]["label"]      # different content → different


def _sec(label, bar0, bar1, reps=1, extra_ranges=None):
    """Build a minimal section dict for _detect_and_correct_form tests — bars
    content doesn't matter here, only barRanges/reps (what the detector reads)
    and enough bars/spans structure for _shift_boundary to rewrite."""
    ranges = [[bar0, bar1]] + (extra_ranges or [])
    return {"id": label, "label": label, "tag": "", "reps": reps,
            "bars": [[] for _ in range(bar1 - bar0 + 1)],
            "spans": [[float(bar0), float(bar1 + 1)]] * len(ranges),
            "barRanges": ranges}


class TestFoldRepeatingSectionGroups:
    """2026-07-20 user directive: a repeating GROUP of sections ("deux A et un
    B, répétés deux fois") should be written once with a ×2 on the whole
    group, not A A B A A B in full — mirrors iReal's own ‖: :‖ ×k notation."""

    def test_aab_group_repeated_twice_folds_to_ab(self):
        from harmonia.output.chart_model import _fold_repeating_section_groups
        a_bars = [[{"root": 0, "q": "", "c": 0.9, "t0": float(i), "t1": float(i) + 1.0}] for i in range(2)]
        b_bars = [[{"root": 7, "q": "7", "c": 0.9, "t0": 0.0, "t1": 1.0}]]
        secs = [
            _sec("A", 0, 1, extra_ranges=None), _sec("B", 2, 2),
            _sec("A", 3, 4), _sec("B", 5, 5),
        ]
        for s in secs:                     # give matching content per label
            s["bars"] = a_bars if s["label"] == "A" else b_bars
        out = _fold_repeating_section_groups(secs)
        assert [s["label"] for s in out] == ["A", "B"]
        assert out[0]["reps"] == 2 and out[1]["reps"] == 2
        assert out[0]["barRanges"] == [[0, 1], [3, 4]]
        assert out[1]["barRanges"] == [[2, 2], [5, 5]]

    def test_non_repeating_group_left_unchanged(self):
        from harmonia.output.chart_model import _fold_repeating_section_groups
        secs = [_sec("A", 0, 7), _sec("B", 8, 15), _sec("C", 16, 23)]
        out = _fold_repeating_section_groups(secs)
        assert [s["label"] for s in out] == ["A", "B", "C"]


class TestDetectEndings:
    """1st/2nd-ending detection (2026-07-21): a folded reps≥2 phrase whose
    passes share a leading region but diverge ONLY in the trailing 1-2 bars is
    the classic ``|: … 1.__ :| 2.__`` — capture the per-pass tails instead of
    crushing to one representative (which silently drops the alternate ending
    from BOTH display and playback). RED-FIRST: pre-fix, ``_fold_bar_run``
    folded these to one block with NO ``endings`` field and no way to recover
    the alt ending."""

    @staticmethod
    def _bar(root, q="", t0=0.0):
        return [{"root": root, "q": q, "c": 0.9, "t0": t0, "t1": t0 + 1.0,
                 "bar": 0, "beat": 0}]

    def test_last_bar_diverges_two_endings(self):
        from harmonia.output.chart_model import _detect_endings
        # 4-bar phrase ×2: shared [C,F,G] prefix, last bar C (1st) vs A- (2nd)
        p1 = [self._bar(0), self._bar(5), self._bar(7), self._bar(0)]
        p2 = [self._bar(0), self._bar(5), self._bar(7), self._bar(9, "-")]
        end = _detect_endings([p1, p2])
        assert end is not None
        assert end["tail"] == 1
        assert [v["passes"] for v in end["variants"]] == [[0], [1]]
        assert end["variants"][0]["bars"][0][0]["root"] == 0     # 1st ending: C
        assert end["variants"][1]["bars"][0][0]["root"] == 9     # 2nd ending: A-

    def test_identical_passes_no_endings(self):
        from harmonia.output.chart_model import _detect_endings
        p = [self._bar(0), self._bar(5), self._bar(7), self._bar(0)]
        assert _detect_endings([p, [b[:] for b in p]]) is None

    def test_mid_divergence_not_tail_abstains(self):
        from harmonia.output.chart_model import _detect_endings
        # differ in bar 2 (middle), agree on the last bar → NOT a 1st/2nd ending
        p1 = [self._bar(0), self._bar(5), self._bar(7), self._bar(0)]
        p2 = [self._bar(0), self._bar(2), self._bar(7), self._bar(0)]
        assert _detect_endings([p1, p2]) is None

    def test_ragged_lengths_abstain(self):
        from harmonia.output.chart_model import _detect_endings
        p1 = [self._bar(0), self._bar(5), self._bar(7)]
        p2 = [self._bar(0), self._bar(5), self._bar(7), self._bar(9)]
        assert _detect_endings([p1, p2]) is None

    def test_two_bar_tail(self):
        from harmonia.output.chart_model import _detect_endings
        # last TWO bars diverge; first two agree
        p1 = [self._bar(0), self._bar(5), self._bar(7), self._bar(0)]
        p2 = [self._bar(0), self._bar(5), self._bar(2), self._bar(9)]
        end = _detect_endings([p1, p2])
        assert end is not None and end["tail"] == 2
        assert len(end["variants"][0]["bars"]) == 2

    def test_fold_bar_run_attaches_endings(self):
        """Integration: a 4-bar loop ×2 whose last bar differs folds to ONE
        reps=2 block that now CARRIES the per-pass endings (red-first: pre-fix
        this produced a fold with no ``endings`` and the alt ending vanished)."""
        from harmonia.output.chart_model import _fold_bar_run
        p1 = [self._bar(0), self._bar(5), self._bar(7), self._bar(0)]
        p2 = [self._bar(0), self._bar(5), self._bar(7), self._bar(9, "-")]
        folded = _fold_bar_run(p1 + p2, 0, "A")
        assert folded is not None and len(folded) == 1
        sec = folded[0]
        assert sec["reps"] == 2
        assert "endings" in sec and sec["endings"]["tail"] == 1
        assert len(sec["endings"]["variants"]) == 2

    def test_section_group_fold_layer2_endings(self):
        """Layer-2: a repeating GROUP where the same slot's section differs only
        in its trailing bar between groups. No real corpus example trips this
        today, so it's covered synthetically."""
        from harmonia.output.chart_model import _fold_section_group_run
        # 4-bar A so the shared prefix (3/4 = 0.75) clears the group-match frac
        # while the last bar still diverges between the two groups.
        a_pre = [self._bar(0), self._bar(5), self._bar(7)]
        a1 = {"label": "A", "id": "A", "reps": 1,
              "bars": a_pre + [self._bar(0)],
              "spans": [[0.0, 4.0]], "barRanges": [[0, 3]]}
        b1 = {"label": "B", "id": "B", "reps": 1,
              "bars": [self._bar(5)], "spans": [[4.0, 5.0]], "barRanges": [[4, 4]]}
        a2 = {"label": "A", "id": "A", "reps": 1,
              "bars": a_pre + [self._bar(9, "-")],   # last bar differs
              "spans": [[5.0, 9.0]], "barRanges": [[5, 8]]}
        b2 = {"label": "B", "id": "B", "reps": 1,
              "bars": [self._bar(5)], "spans": [[9.0, 10.0]], "barRanges": [[9, 9]]}
        out = _fold_section_group_run([a1, b1, a2, b2], frac=0.7)
        assert [s["label"] for s in out] == ["A", "B"]
        assert "endings" in out[0] and out[0]["endings"]["tail"] == 1
        assert "endings" not in out[1]      # B is identical across groups


class TestDetectAndCorrectForm:
    def _bars(self, n):
        return [[] for _ in range(n)]

    def test_twelve_bar_blues(self):
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 11, reps=3)]
        assert _detect_and_correct_form(secs, self._bars(36)) == "12-bar blues"

    def test_n_bar_loop(self):
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7, reps=4)]
        assert _detect_and_correct_form(secs, self._bars(32)) == "8-bar loop"

    def test_single_pass_no_form_name(self):
        """A single, non-repeated (reps=1) section isn't a "loop" — nothing to
        name confidently from one pass alone."""
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7, reps=1)]
        assert _detect_and_correct_form(secs, self._bars(8)) is None

    def test_aaba_already_equal_lengths_classified_no_correction(self):
        # bars 0-7 and 8-15: the two folded A passes (reps=2); 16-23: B; 24-31: final A
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7, reps=2, extra_ranges=[[8, 15]]),    # 2×A, 8 bars each
                _sec("B", 16, 23, reps=1),                          # B, 8 bars
                _sec("A", 24, 31, reps=1)]                          # A, 8 bars
        form = _detect_and_correct_form(secs, self._bars(32))
        assert form == "AABA (32-bar song form)"
        assert secs[1]["barRanges"] == [[16, 23]]     # untouched: already correct
        assert secs[2]["barRanges"] == [[24, 31]]

    def test_aaba_boundary_off_by_one_gets_corrected(self):
        """B decoded 1 bar too long (9), final A 1 bar too short (7) — a
        classic off-by-one boundary error. Total (16) conserves 2×8, so the
        detector should reassign the misplaced bar rather than abstain."""
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7, reps=2, extra_ranges=[[8, 15]]),    # 2×A, 8 bars
                _sec("B", 16, 24, reps=1),                          # B, 9 bars (wrong)
                _sec("A", 25, 31, reps=1)]                          # A, 7 bars (wrong)
        form = _detect_and_correct_form(secs, self._bars(32))
        assert form == "AABA (32-bar song form)"
        assert secs[1]["barRanges"] == [[16, 23]]     # corrected to 8 bars
        assert secs[2]["barRanges"] == [[24, 31]]     # corrected to 8 bars
        assert len(secs[1]["bars"]) == 8 and len(secs[2]["bars"]) == 8

    def test_aaba_non_conserving_mismatch_abstains(self):
        """B and final-A disagree but DON'T sum to 2×target (e.g. a genuine
        extra tag bar) — must not force a correction it can't verify."""
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7, reps=2, extra_ranges=[[8, 15]]),
                _sec("B", 16, 25, reps=1),      # 10 bars
                _sec("A", 26, 33, reps=1)]      # 8 bars — 10+8=18 != 16
        before_b, before_a2 = list(secs[1]["barRanges"]), list(secs[2]["barRanges"])
        form = _detect_and_correct_form(secs, self._bars(34))
        assert form is None                      # not confidently AABA
        assert secs[1]["barRanges"] == before_b   # left untouched
        assert secs[2]["barRanges"] == before_a2

    def test_verse_chorus_not_misclassified_as_aaba(self):
        from harmonia.output.chart_model import _detect_and_correct_form
        secs = [_sec("A", 0, 7), _sec("B", 8, 15), _sec("A", 16, 23), _sec("B", 24, 31)]
        assert _detect_and_correct_form(secs, self._bars(32)) is None


class TestD9FamilyRepresentation:
    """2026-07-21: _sections_by_largest_unit's default bar representation is
    (root, is_dominant) (the user's D9 rule — only a dominant-7th changes the
    section-comparison family, maj/maj7/6 and min/min7 stay equivalent).
    HARMONIA_SECTION_REPR=root restores the old plain-root-only behaviour."""

    @staticmethod
    def _bars_for(roots, q):
        return [[{"root": r, "q": q, "t0": float(i), "t1": float(i) + 1.0}]
                for i, r in enumerate(roots)]

    def test_same_roots_different_dominant_flavor_stay_distinct_by_default(self, monkeypatch):
        from harmonia.output.chart_model import _sections_by_largest_unit
        roots = [0, 2, 4, 5, 7, 9, 11, 1]
        # 3 identical major passes + 1 dominant-quality pass over the SAME roots:
        # plain-root sees 4 identical blocks; D9 sees the 4th as a different family.
        bars = (self._bars_for(roots, "") * 3) + self._bars_for(roots, "7")
        out = _sections_by_largest_unit(bars, len(bars))
        assert out is not None
        labels = [s["label"] for s in out]
        assert labels == ["A", "B"]
        assert out[0]["reps"] == 3
        assert out[1]["reps"] == 1

    def test_kill_switch_restores_plain_root_merge(self, monkeypatch):
        from harmonia.output.chart_model import _sections_by_largest_unit
        monkeypatch.setenv("HARMONIA_SECTION_REPR", "root")
        roots = [0, 2, 4, 5, 7, 9, 11, 1]
        bars = (self._bars_for(roots, "") * 3) + self._bars_for(roots, "7")
        out = _sections_by_largest_unit(bars, len(bars))
        assert out is not None
        assert [s["label"] for s in out] == ["A"]
        assert out[0]["reps"] == 4


class TestDistinctiveChordVeto:
    """2026-07-21: section_arbiter.veto blocks a merge when one block has a
    chord recurring >=2 bars that's wholly absent from the other, even when
    the overall sequence similarity clears the merge threshold — catches an
    over-merge a coincidental partial-sequence match would miss."""

    @staticmethod
    def _bars_for(roots):
        return [[{"root": r, "q": "", "t0": float(i), "t1": float(i) + 1.0}]
                for i, r in enumerate(roots)]

    def test_veto_blocks_merge_despite_high_sequence_similarity(self):
        from harmonia.output.chart_model import _sections_by_largest_unit
        a = [0, 0, 2, 4, 5, 7, 9, 11]      # root 0 recurs twice
        b = [3, 3, 2, 4, 5, 7, 9, 11]      # root 3 recurs twice, root 0 absent
        bars = self._bars_for(a) + self._bars_for(a) + self._bars_for(b)
        out = _sections_by_largest_unit(bars, len(bars))
        assert out is not None
        labels = [s["label"] for s in out]
        assert labels == ["A", "B"]
        assert out[0]["reps"] == 2
        assert out[1]["reps"] == 1

    def test_kill_switch_restores_over_merge(self, monkeypatch):
        from harmonia.output.chart_model import _sections_by_largest_unit
        monkeypatch.setenv("HARMONIA_SECTION_VETO", "0")
        a = [0, 0, 2, 4, 5, 7, 9, 11]
        b = [3, 3, 2, 4, 5, 7, 9, 11]
        bars = self._bars_for(a) + self._bars_for(a) + self._bars_for(b)
        out = _sections_by_largest_unit(bars, len(bars))
        assert out is not None
        assert [s["label"] for s in out] == ["A"]
        assert out[0]["reps"] == 3


def test_leading_one_off_block_becomes_intro():
    """User convention 2026-07-20: a LEADING one-off phrase (appears once, before
    the first repeated phrase) is an Intro (label-only), not a letter."""
    from harmonia.output.chart_model import _sections_by_largest_unit
    # n=24 so only L=8 applies (L=16 needs 2·16≤n): intro(8) + verse ×2
    seq = ([0, 2, 4, 5, 7, 9, 11, 1]       # intro (one-off leading)
           + [0, 7, 9, 5, 0, 7, 9, 5] * 2)  # main phrase repeated ×2
    bars = [[{"root": r, "q": "", "t0": float(i), "t1": float(i) + 1.0}]
            for i, r in enumerate(seq)]
    out = _sections_by_largest_unit(bars, len(bars))
    assert out is not None
    assert out[0]["label"] == "Intro"
    assert out[1]["label"] == "A" and out[1]["reps"] == 2
