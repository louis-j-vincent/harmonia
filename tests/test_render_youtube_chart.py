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


class TestHarmonicBarPhaseReanchor:
    """Regression for the This Love bar-phase misrender (2026-07-30).

    Measured on the baked inferred_maroon_5_this_love.html payload: 75/120
    chords sat on beat 3 (and only 1/120 on beat 0) — every chord that
    musically opens a bar rendered as the TAIL of the previous displayed bar,
    and the following bar showed a held "%". Cause: the bar grid's phase comes
    from the beat tracker's downbeat estimate (grid_anchor_beats → bar1_offset
    _beats = 2), which on this song is one beat off from where the harmony
    actually changes (tracker beats k ≡ 1 mod 4). When nearly all chords agree
    on one non-zero beat-in-bar, the phase must follow the chords.

    The mimic below is This Love's exact shape at tempo 120: the tracker's
    grid starts at 0.35 s (phase 0.7 beat — a chord's start_s/beat_dur has
    fractional part ≈ 0.7, exactly the measured 0.72–0.86 band), chords
    change every 4 beats starting at tracker beat 1, and the claimed downbeat
    anchor is 2.
    """

    BEAT = 0.5                      # 120 bpm
    PHI = 0.35                      # tracker grid origin: 0.7 beat past t=0

    def _this_love_shaped(self, n=8):
        beat_times = [self.PHI + self.BEAT * k for k in range(60)]
        chords = []
        for j in range(n):          # chord onsets at tracker beats 1, 5, 9, …
            t = self.PHI + self.BEAT * (1 + 4 * j)
            chords.append(_ch("C:maj" if j % 2 else "F:maj", t, t + 4 * self.BEAT))
        return _chart(chords, beat_times, duration_s=40.0, tempo_bpm=120.0)

    def test_downbeat_chords_land_on_beat_0_not_beat_3_of_previous_bar(self):
        chart = self._this_love_shaped()
        _obj, dicts = chart_to_interactive_inputs(chart, "t", "s", bar1_offset_beats=2)
        assert [c["beat"] for c in dicts] == [0] * len(dicts), \
            f"bar-opening chords must sit on beat 0, got beats {[c['beat'] for c in dicts]}"
        assert [c["bar"] for c in dicts] == list(range(len(dicts)))

    def test_correctly_anchored_chart_is_untouched(self):
        # Chords already on the anchor's downbeats (tracker beats 2, 6, 10, …):
        # beat-0 is modal, the re-anchor must not fire, bars stay put.
        beat_times = [self.PHI + self.BEAT * k for k in range(60)]
        chords = []
        for j in range(8):
            t = self.PHI + self.BEAT * (2 + 4 * j)
            chords.append(_ch("C:maj" if j % 2 else "F:maj", t, t + 4 * self.BEAT))
        chart = _chart(chords, beat_times, duration_s=40.0, tempo_bpm=120.0)
        _obj, dicts = chart_to_interactive_inputs(chart, "t", "s", bar1_offset_beats=2)
        assert [c["beat"] for c in dicts] == [0] * len(dicts)
        assert [c["bar"] for c in dicts] == list(range(len(dicts)))

    def test_no_supermajority_no_rotation(self):
        # Residues split 3/3/2 across beats 1, 2, 0 — no consensus, layout
        # must equal the naive floor layout (nothing silently rotates).
        beat_times = [self.PHI + self.BEAT * k for k in range(80)]
        # (k-2)%4 → 1,2,0,1,2,0,1,2,0: counts {1:3, 2:3, 0:3} — modal share 1/3.
        # Trailing k=22 keeps the last two-onset bar followed by a NON-empty
        # bar so rebalance_near_boundary_onsets stays inert too.
        ks = [3, 4, 6, 11, 12, 14, 19, 20, 22]
        chords = []
        for i, k in enumerate(ks):
            t = self.PHI + self.BEAT * k
            chords.append(_ch("C:maj" if i % 2 else "F:maj", t, t + 2 * self.BEAT))
        chart = _chart(chords, beat_times, duration_s=40.0, tempo_bpm=120.0)
        _obj, dicts = chart_to_interactive_inputs(chart, "t", "s", bar1_offset_beats=2)
        assert [(c["bar"], c["beat"]) for c in dicts] == \
            [((k - 2) // 4, (k - 2) % 4) for k in ks]

    def test_kill_switch_restores_old_layout(self, monkeypatch):
        monkeypatch.setenv("HARMONIA_HARMONIC_REANCHOR", "0")
        chart = self._this_love_shaped()
        _obj, dicts = chart_to_interactive_inputs(chart, "t", "s", bar1_offset_beats=2)
        # old behavior: every chord one beat before its bar line → beat 3
        assert [c["beat"] for c in dicts] == [3] * len(dicts)
