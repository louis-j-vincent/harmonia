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
