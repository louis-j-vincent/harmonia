"""iReal<->YouTube discrete downbeat-anchored alignment (2026-07-21) — the
pure warp-construction math, which doesn't need real audio/models. The
real-audio yield (~1-2/10 pairs accepted at a 250ms median gate) is
measured in scratchpad/mission1_phase2_pop.py, not a unit test — it needs
real matched audio + iReal charts + the beat_this model.
"""
from __future__ import annotations

import numpy as np

from harmonia.data.ireal_youtube_align import (
    _downbeat_anchored_warp, _section_runs, _mma_chart_to_chords_for_bars,
    _model_root_shape_agreement,
)


class _FakeHeads:
    """Stubs heads.root_proba: returns a one-hot argmax at the NEXT root from
    a fixed sequence, in call order (_model_root_shape_agreement's roll+L2
    preprocessing zeroes out a constant-valued test feature, so the stub
    can't key off feature content — it doesn't need to, since chords are
    processed in a fixed, known order)."""
    def __init__(self, roots_in_call_order: list):
        self._roots = list(roots_in_call_order)
        self._i = 0

    def root_proba(self, feat):
        import numpy as np
        root = self._roots[self._i]
        self._i += 1
        out = np.zeros((1, 12))
        out[0, root] = 1.0
        return out


class _FakeChart:
    def __init__(self, timeline, beats_per_bar=4, tempo=120.0):
        self.timeline = timeline
        self.beats_per_bar = beats_per_bar
        self.tempo = tempo


def _bar(barno, label, mma, tok=None):
    return (barno, label, [(0, tok or mma, mma)])


def _chord(bar, start_s):
    return {"bar": bar, "start_s": start_s}


class TestDownbeatAnchoredWarp:
    def test_maps_bar_starts_onto_real_downbeats(self):
        # chart at 120 BPM, 4/4 -> 2s/bar; downbeats land at 10, 12, 14, 16s
        downbeats = [10.0, 12.0, 14.0, 16.0, 18.0]
        chords = [_chord(1, 0.0), _chord(2, 2.0), _chord(3, 4.0)]
        warp = _downbeat_anchored_warp(chords, __import__("numpy").asarray(downbeats),
                                       d0=0, bpb=4, spb_nominal=0.5)
        assert warp(0.0) == 10.0     # bar 1 start
        assert warp(2.0) == 12.0     # bar 2 start
        assert warp(4.0) == 14.0     # bar 3 start

    def test_interpolates_within_a_bar(self):
        downbeats = [10.0, 14.0]   # one real bar spans 4s
        chords = [_chord(1, 0.0)]
        warp = _downbeat_anchored_warp(chords, __import__("numpy").asarray(downbeats),
                                       d0=0, bpb=4, spb_nominal=0.5)
        # halfway through the chart's bar (2 of 4 beats = 1.0s at spb=0.5)
        # should land halfway through the REAL bar (10 + 0.5*4 = 12.0)
        assert warp(1.0) == 12.0

    def test_follows_real_tempo_not_chart_nominal_tempo(self):
        # chart nominal tempo implies 2s/bar, but the REAL recording's bars
        # (from real downbeats) are 3s apart — within-bar interpolation must
        # follow the REAL spacing, not silently re-impose the chart's own.
        downbeats = [0.0, 3.0, 6.0]
        chords = [_chord(1, 0.0), _chord(2, 2.0)]
        warp = _downbeat_anchored_warp(chords, __import__("numpy").asarray(downbeats),
                                       d0=0, bpb=4, spb_nominal=0.5)
        assert warp(0.0) == 0.0
        assert warp(2.0) == 3.0    # bar 2 lands on the REAL second downbeat

    def test_none_when_too_few_anchors_available(self):
        downbeats = [10.0]  # only one downbeat — can't anchor even 1 bar span
        chords = [_chord(1, 0.0), _chord(5, 8.0)]
        warp = _downbeat_anchored_warp(chords, __import__("numpy").asarray(downbeats),
                                       d0=0, bpb=4, spb_nominal=0.5)
        assert warp is None


class TestSectionRuns:
    """Per-section alignment (2026-07-21, user's own idea after seeing the
    whole-song version's ~10-20% yield): split the chart into iReal's own
    contiguous section runs so one divergent bridge/extra-chorus can't sink
    every other section's alignment."""

    def test_contiguous_same_label_bars_form_one_run(self):
        timeline = [_bar(1, "i", "C"), _bar(2, "i", "C"),
                   _bar(3, "A", "F"), _bar(4, "A", "G"), _bar(5, "A", "C")]
        chart = _FakeChart(timeline)
        assert _section_runs(chart) == [("i", 1, 2), ("A", 3, 5)]

    def test_non_contiguous_recurrence_yields_separate_runs(self):
        # "A" appears twice, separated by "B" — a real second-verse pattern.
        # Each physical occurrence must get its OWN run (own independent
        # alignment target), not be merged as if it were one span.
        timeline = [_bar(1, "A", "C"), _bar(2, "A", "F"),
                   _bar(3, "B", "G"),
                   _bar(4, "A", "C"), _bar(5, "A", "F")]
        chart = _FakeChart(timeline)
        assert _section_runs(chart) == [("A", 1, 2), ("B", 3, 3), ("A", 4, 5)]

    def test_single_run_covering_whole_chart(self):
        timeline = [_bar(1, "A", "C"), _bar(2, "A", "F")]
        chart = _FakeChart(timeline)
        assert _section_runs(chart) == [("A", 1, 2)]


class TestMmaChartToChordsForBars:
    def test_rebases_bar_numbers_to_start_at_one(self):
        timeline = [_bar(1, "i", "C"), _bar(2, "i", "F"),
                   _bar(3, "A", "G"), _bar(4, "A", "C")]
        chart = _FakeChart(timeline)
        chords = _mma_chart_to_chords_for_bars(chart, 3, 4)
        assert [c["bar"] for c in chords] == [1, 2]
        assert [c["mma"] for c in chords] == ["G", "C"]
        assert chords[0]["start_s"] == 0.0   # section-local time starts at 0

    def test_repeat_previous_chord_resolves_across_the_slice_boundary(self):
        # bar 2 ('p' = repeat previous) is BEFORE the slice [3,4], but its
        # resolved chord must still be tracked so a 'p' INSIDE the slice
        # resolves correctly.
        timeline = [_bar(1, "i", "C"),
                   (2, "i", [(0, "p", "z")]),
                   (3, "A", [(0, "p", "z")]),
                   _bar(4, "A", "F")]
        chart = _FakeChart(timeline)
        chords = _mma_chart_to_chords_for_bars(chart, 3, 4)
        assert [c["mma"] for c in chords] == ["C", "F"]


class TestModelRootShapeAgreement:
    """User's own safeguard (2026-07-21): use the chord model's OWN
    independent decode to cross-check a candidate alignment — but only via
    transposition-invariant root-INTERVAL shape, never an exact label, so it
    can't reintroduce the circularity the whole method exists to avoid."""

    @staticmethod
    def _chord(root, start_s, end_s):
        return {"root": root, "start_s": start_s, "end_s": end_s}

    def test_perfect_shape_match_scores_one(self):
        # chart root sequence 0 -> 5 -> 9 (up a 4th, up a maj3rd); model
        # "hears" completely different absolute roots (7,0,4) but the SAME
        # relative intervals (+5, +4 mod 12) -> shape agreement is still 1.0
        chords = [self._chord(0, 0.0, 1.0), self._chord(5, 1.0, 2.0), self._chord(9, 2.0, 3.0)]
        heads = _FakeHeads([7, 0, 4])
        arr = np.array([[0.0] * 24, [1.0] * 24, [2.0] * 24])
        times = np.array([0.5, 1.5, 2.5])
        warp = lambda t: t
        agree = _model_root_shape_agreement(chords, warp, arr, times, heads)
        assert agree == 1.0

    def test_disagreeing_shape_scores_low(self):
        chords = [self._chord(0, 0.0, 1.0), self._chord(5, 1.0, 2.0), self._chord(9, 2.0, 3.0)]
        heads = _FakeHeads([0, 3, 4])   # random-ish roots, wrong intervals
        arr = np.array([[0.0] * 24, [1.0] * 24, [2.0] * 24])
        times = np.array([0.5, 1.5, 2.5])
        warp = lambda t: t
        agree = _model_root_shape_agreement(chords, warp, arr, times, heads)
        assert agree == 0.0

    def test_none_with_too_few_chord_spans(self):
        chords = [self._chord(0, 0.0, 1.0)]
        heads = _FakeHeads([0])
        arr = np.array([[0.0] * 24])
        times = np.array([0.5])
        warp = lambda t: t
        assert _model_root_shape_agreement(chords, warp, arr, times, heads) is None
