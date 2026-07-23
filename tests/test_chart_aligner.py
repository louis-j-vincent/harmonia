"""Unit tests for the productionized chart-to-audio aligner seam
(harmonia.align.chart_aligner).

These drive the PURE, audio-free adapter (``_to_chart_alignment``) with a
hand-built ``FusionAlignment`` — the same style as ``test_fusion.py`` (the audio
front-end is covered by the scratchpad losslessness harness against the frozen
goldens). The contract under test: the adapter is LOSSLESS vs the fusion output
(gt_chords<->markers bijection, section<->placement mapping, downbeat +
confidence passthrough) so a serving route can build either the
``_load_ireal_alignment`` per-chord payload or the section-align ``markers`` /
``sections`` payload from a ``ChartAlignment`` with no information loss."""
import numpy as np
import pytest

from harmonia.align.fusion import FusionAlignment, Placement, DownbeatPhase
from harmonia.align.chart_aligner import (
    SectionMarker, ChartAlignment, ChartAligner, FusionChartAligner,
    _to_chart_alignment, _assign_section, _marker_bar,
)


# ── a hand-built FusionAlignment: 2 sections x 2 chords, 16-beat lattice ───────
def _fake_alignment(*, db_phase=1, db_conf=0.3, db_flag=False, antic=True):
    period, phase = 0.5, 0.0
    beats = np.arange(16) * period                     # 0..7.5s, bars of 4 beats
    plA = Placement(0, 0, "A", 0, 8, emission=0.5); plA.confidence = 0.80
    plB = Placement(1, 1, "B", 8, 8, emission=0.4); plB.confidence = 0.40
    # A spans beats [0,8) = [0,4)s ; B spans [8,16) = [4,8)s
    gt_chords = [
        dict(t0=0.0, t1=2.0, root_pc=0, quality="maj", bass_pc=0, label="C:maj"),
        dict(t0=2.0, t1=4.0, root_pc=7, quality="maj", bass_pc=7, label="G:maj"),
        dict(t0=4.0, t1=6.0, root_pc=9, quality="min", bass_pc=9, label="A:min"),
        dict(t0=6.0, t1=8.0, root_pc=5, quality="maj", bass_pc=5, label="F:maj"),
    ]
    db = DownbeatPhase(db_phase, db_phase, 2, db_conf, antic, db_flag, 4, ["note"])
    low = [dict(label="B2", t0=4.0, t1=8.0, confidence=0.2,
                reliability=0.05, reason="flat posterior")]
    return FusionAlignment(
        song_id="fake", transpose=0, period=period, phase=phase, beats=beats,
        placements=[plA, plB], downbeat=db, gt_chords=gt_chords,
        whole_song_confidence=0.66, low_confidence_regions=low,
        diag={})


# ═══════════════════════════ adapter: shape + losslessness ═════════════════════

def test_sections_map_to_placements():
    ca = _to_chart_alignment(_fake_alignment())
    assert isinstance(ca, ChartAlignment)
    assert [s["label"] for s in ca.sections] == ["A", "B"]
    assert [s["confidence"] for s in ca.sections] == [0.80, 0.40]
    # section span == first/last contained marker time
    assert ca.sections[0]["t0"] == 0.0 and ca.sections[0]["t1"] == 4.0
    assert ca.sections[1]["t0"] == 4.0 and ca.sections[1]["t1"] == 8.0


def test_gt_chords_markers_bijection():
    fa = _fake_alignment()
    ca = _to_chart_alignment(fa)
    all_markers = [m for s in ca.sections for m in s["markers"]]
    assert len(all_markers) == len(fa.gt_chords)          # bijection: no drop/dup
    # each marker carries the union of both consumers' fields, mma == label
    for m, c in zip(all_markers, fa.gt_chords):
        assert isinstance(m, SectionMarker)
        assert (m.t0, m.t1, m.label) == (c["t0"], c["t1"], c["label"])
        assert m.mma == m.label
    assert [m.section_idx for m in all_markers] == [0, 0, 1, 1]


def test_marker_conf_inherits_section_confidence():
    ca = _to_chart_alignment(_fake_alignment())
    assert all(m.conf == 0.80 for m in ca.sections[0]["markers"])
    assert all(m.conf == 0.40 for m in ca.sections[1]["markers"])


def test_downbeat_and_confidence_passthrough():
    fa = _fake_alignment(db_phase=1, db_conf=0.3, db_flag=False)
    ca = _to_chart_alignment(fa)
    assert ca.downbeat_phase == 1
    assert ca.downbeat_confidence == 0.3
    assert ca.downbeat_flagged is False
    assert ca.whole_song_confidence == 0.66
    assert ca.low_confidence_regions == fa.low_confidence_regions
    assert ca.tempo_bpm == 120.0                          # 60 / 0.5
    assert np.array_equal(ca.beats, fa.beats)
    # diag carries the acoustic/form split for the anticipation self-detection
    assert ca.diag["downbeat_form_phase"] == 1
    assert ca.diag["downbeat_acoustic_phase"] == 2
    assert ca.diag["downbeat_anticipation"] is True


def test_flagged_downbeat_passthrough():
    ca = _to_chart_alignment(_fake_alignment(db_flag=True, db_conf=0.05))
    assert ca.downbeat_flagged is True
    assert ca.downbeat_confidence == 0.05


# ═══════════════════════════ partition robustness ═════════════════════════════

def test_assign_section_max_overlap():
    windows = [(0.0, 4.0), (4.0, 8.0)]
    # a chord straddling the boundary goes to the side it overlaps MORE
    assert _assign_section(3.5, 4.2, windows) == 0        # 0.5 in A vs 0.2 in B
    assert _assign_section(3.8, 5.0, windows) == 1        # 0.2 in A vs 1.0 in B


def test_assign_section_gap_chord_nearest_center():
    windows = [(0.0, 4.0), (10.0, 14.0)]
    # a chord in the gap (overlaps neither) attaches to the nearest window centre
    assert _assign_section(5.0, 5.5, windows) == 0
    assert _assign_section(9.0, 9.5, windows) == 1


def test_boundary_chord_not_double_counted():
    """A chord whose midpoint sits exactly on a section boundary must appear in
    EXACTLY one section (the +1 duplicate the naive midpoint rule produced)."""
    fa = _fake_alignment()
    # insert a chord straddling the A/B boundary at 4.0s
    fa.gt_chords.insert(2, dict(t0=3.9, t1=4.1, root_pc=2, quality="min",
                                bass_pc=2, label="D:min"))
    ca = _to_chart_alignment(fa)
    all_markers = [m for s in ca.sections for m in s["markers"]]
    assert len(all_markers) == len(fa.gt_chords)          # still a bijection
    dmin = [m for m in all_markers if m.label == "D:min"]
    assert len(dmin) == 1


def test_marker_bar_monotone():
    # bars of 4 beats at period 0.5s from phase 0: beat idx = round(t/0.5)
    assert _marker_bar(0.0, 0.0, 0.5) == 0                # beat 0 -> bar 0
    assert _marker_bar(2.0, 0.0, 0.5) == 1                # beat 4 -> bar 1
    assert _marker_bar(4.0, 0.0, 0.5) == 2                # beat 8 -> bar 2


# ═══════════════════════════ ABC + thin wrapper ═══════════════════════════════

def test_chart_aligner_is_abstract():
    with pytest.raises(TypeError):
        ChartAligner()                                    # abstract, no align()


def test_fusion_chart_aligner_is_a_chart_aligner():
    assert issubclass(FusionChartAligner, ChartAligner)
    assert isinstance(FusionChartAligner(), ChartAligner)


def test_align_delegates_to_align_fusion(monkeypatch):
    """The wrapper is thin: align() calls align_fusion then the pure adapter, and
    returns exactly ``_to_chart_alignment(fusion_output)`` — no extra numerics."""
    fa = _fake_alignment()
    captured = {}

    def _fake_align_fusion(audio, chart, **kw):
        captured["audio"] = audio
        captured["chart"] = chart
        captured["kw"] = kw
        return fa

    monkeypatch.setattr("harmonia.align.chart_aligner.align_fusion",
                        _fake_align_fusion)
    ca = FusionChartAligner(w_bass=0.05, w_hr=0.05).align(
        "audio.m4a", {"song_id": "fake"}, features={"x": 1})
    assert captured["audio"] == "audio.m4a"
    assert captured["chart"] == {"song_id": "fake"}
    assert captured["kw"]["features"] == {"x": 1}
    assert captured["kw"]["w_bass"] == 0.05
    # identical to calling the pure adapter directly
    ref = _to_chart_alignment(fa)
    assert [s["label"] for s in ca.sections] == [s["label"] for s in ref.sections]
    assert ca.downbeat_phase == ref.downbeat_phase
    assert ca.whole_song_confidence == ref.whole_song_confidence
