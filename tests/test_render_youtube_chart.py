"""chart_to_interactive_inputs's beat-snapping display layer.

Regression for the "Hot n Cold" freeze (2026-07-21): a beat tracker that
loses lock before the song's audio actually ends (real case — a whispered-
bridge section with almost no rhythmic onset content) leaves
``pipeline_chart.beat_times`` covering only PART of the duration. Snapping
every later chord to the single nearest (last) real beat collapsed the
entire back third of the song onto one frozen timestamp, breaking playback
and the displayed loop for that whole stretch.
"""
from __future__ import annotations

from harmonia.pipeline import ChordChart
from scripts.render_youtube_chart import chart_to_interactive_inputs


def _chart(chords, beat_times, duration_s=300.0, tempo_bpm=120.0):
    return ChordChart(
        source_path="x", duration_s=duration_s, tempo_bpm=tempo_bpm,
        time_signature="4/4", global_key="C major", global_key_confidence=0.9,
        style="v1", modulations=[], chords=chords, segments=[],
        beat_times=beat_times,
    )


def _ch(label, start_s, end_s):
    return {"label": label, "start_s": start_s, "end_s": end_s,
            "duration_beats": 1, "confidence": 0.8}


class TestBeatSnappingPastTrackerCoverage:
    def test_chords_after_lost_beat_lock_do_not_freeze(self):
        # Real beats only cover 0..100s; chords continue to 200s (the tracker
        # lost lock, e.g. a quiet bridge) — every later chord must NOT collapse
        # onto the single last real beat at 99.5s.
        beat_times = [i * 0.5 for i in range(200)]     # 0.0 .. 99.5, period 0.5
        chords = [_ch("C:maj", 90.0, 92.0), _ch("D:maj", 150.0, 152.0),
                  _ch("G:maj", 199.0, 201.0)]
        chart_obj, chord_dicts = chart_to_interactive_inputs(_chart(chords, beat_times), "t", "s")
        t0s = [c["start_s"] for c in chord_dicts]
        assert len(set(t0s)) == len(t0s), f"chords froze onto a shared timestamp: {t0s}"
        # still monotonically increasing — no collapse, no reordering
        assert t0s == sorted(t0s)

    def test_chords_within_tracker_coverage_still_snap(self):
        beat_times = [i * 0.5 for i in range(200)]      # 0.0 .. 99.5
        chords = [_ch("C:maj", 10.02, 12.0)]             # 0.02s off a real beat at 10.0
        _obj, chord_dicts = chart_to_interactive_inputs(_chart(chords, beat_times), "t", "s")
        assert abs(chord_dicts[0]["start_s"] - 10.0) < 1e-9

    def test_no_beat_times_is_a_no_op(self):
        chords = [_ch("C:maj", 5.0, 7.0)]
        _obj, chord_dicts = chart_to_interactive_inputs(_chart(chords, []), "t", "s")
        assert chord_dicts[0]["start_s"] == 5.0
