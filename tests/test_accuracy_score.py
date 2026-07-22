"""Tests for the Brick-0 chord-accuracy scorer (harmonia/eval/accuracy_score.py).

Two layers, per the Brick-0 spec:

1. SCORING-MATH correctness — hand-computed synthetic GT+prediction pairs whose
   MIREX / partial / strict / bass scores are worked out by hand below. This is
   the correctness proof; it needs no real ground truth and no pipeline run.
2. WIRING smoke — an opt-in end-to-end run on one real audio file with a
   PLACEHOLDER GT, purely to prove the pipeline-run + schema-consumption path.
   Its number is NOT an accuracy result (GT is placeholder/unverified). Guarded
   behind HARMONIA_RUN_SMOKE=1 so the normal suite stays fast.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harmonia.eval.accuracy_score import (
    Chord,
    DatasetScore,
    UnverifiedGTError,
    chord_family,
    chord_from_label,
    load_frozen_gt,
    mireval_crosscheck,
    parse_label,
    score_song,
    score_timeline,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "golden" / "brick0" / "_fixture_4chord_synthetic.gt.json"


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_harte_basic(self):
        assert parse_label("C:maj") == (0, "maj", 0)
        assert parse_label("A:min7") == (9, "min7", 9)
        assert parse_label("Bb:maj7") == (10, "maj7", 10)

    def test_slash_bass_sounding(self):
        # G7 with the 3rd (B) in the bass -> sounding bass pc 11.
        assert parse_label("G:7/B") == (7, "7", 11)
        # C major over E -> sounding bass 4.
        assert parse_label("C:maj/E") == (0, "maj", 4)

    def test_no_chord(self):
        assert parse_label("N") == (None, "N", None)
        assert parse_label("") == (None, "N", None)

    def test_colonless_harmonia_label(self):
        assert parse_label("Cmaj7") == (0, "maj7", 0)

    def test_families(self):
        assert chord_family("maj7") == "maj"
        assert chord_family("min7") == "min"
        assert chord_family("7") == "dom"
        assert chord_family("dim7") == "dim"
        assert chord_family("hdim7") == "hdim"
        assert chord_family("aug") == "aug"
        assert chord_family("sus4") == "sus"
        assert chord_family("N") == "N"


# ---------------------------------------------------------------------------
# Scoring math — the hand-computed correctness proof
# ---------------------------------------------------------------------------
#
# GT timeline (4 chords x 2s = 8s):
#   [0,2] C:maj    (root 0, fam maj,  bass 0)
#   [2,4] A:min7   (root 9, fam min,  bass 9)
#   [4,6] D:maj7   (root 2, fam maj,  bass 2)
#   [6,8] G:7/B    (root 7, fam dom,  bass 11)   <- sounding bass = B = 11
#
# Prediction (same boundaries, one wrong root, one maj7-for-maj, one wrong-bass):
#   [0,2] C:maj    correct everywhere
#   [2,4] F:min7   WRONG ROOT (5 != 9)     -> all fail; bass 5 != 9
#   [4,6] D:maj    root ok, maj7-for-maj   -> partial YES, strict NO, 7ths NO,
#                                             majmin YES, bass 2==2 YES
#   [6,8] G:7      root ok, dom==dom       -> partial/strict/7ths YES;
#                                             bass 7 (root, no slash) != 11 NO
#
# Per chord (each weight 2s = 1/4):
#   root:     1,3,4 ok               -> 3/4 = 0.75
#   majmin:   1,3,4 ok               -> 3/4 = 0.75
#   sevenths: 1,4 ok (3 = maj7!=maj) -> 2/4 = 0.50
#   partial:  1,3,4 ok               -> 3/4 = 0.75
#   strict:   1,4 ok                 -> 2/4 = 0.50
#   bass:     1,3 ok                 -> 2/4 = 0.50

GT_4 = [
    Chord(0, 2, 0, "maj", 0, "C:maj"),
    Chord(2, 4, 9, "min7", 9, "A:min7"),
    Chord(4, 6, 2, "maj7", 2, "D:maj7"),
    Chord(6, 8, 7, "7", 11, "G:7/B"),
]
PRED_4 = [
    chord_from_label(0, 2, "C:maj"),
    chord_from_label(2, 4, "F:min7"),
    chord_from_label(4, 6, "D:maj"),
    chord_from_label(6, 8, "G:7"),
]

EXPECT_4 = {
    "mirex_root": 0.75, "mirex_majmin": 0.75, "mirex_sevenths": 0.50,
    "partial_credit": 0.75, "strict": 0.50, "bass_root": 0.50,
}


class TestScoreTimelineMath:
    def test_four_chord_hand_computed(self):
        r = score_timeline(GT_4, PRED_4)
        for k, v in EXPECT_4.items():
            assert r[k] == pytest.approx(v), f"{k}: {r[k]} != {v}"
        assert r["duration_s"] == pytest.approx(8.0)

    def test_perfect_match_is_one(self):
        r = score_timeline(GT_4, GT_4)
        for k in EXPECT_4:
            assert r[k] == pytest.approx(1.0)

    def test_duration_weighting_not_chord_count(self):
        # GT one 4s chord; pred right for 1s, wrong for 3s. Duration-weighted
        # root = 1/4 = 0.25 (chord-count weighting would wrongly give 0.5).
        gt = [Chord(0, 4, 0, "maj", 0, "C:maj")]
        pred = [chord_from_label(0, 1, "C:maj"),
                chord_from_label(1, 4, "A:min")]
        r = score_timeline(gt, pred)
        assert r["mirex_root"] == pytest.approx(0.25)

    def test_no_chord_regions(self):
        # GT: [0,2] N, [2,4] C:maj ; pred all C:maj.
        # root: N-vs-C mismatch for 2s, C-vs-C for 2s -> 0.5
        gt = [Chord(0, 2, None, "N", None, "N"),
              Chord(2, 4, 0, "maj", 0, "C:maj")]
        pred = [chord_from_label(0, 4, "C:maj")]
        r = score_timeline(gt, pred)
        assert r["mirex_root"] == pytest.approx(0.5)

    def test_both_no_chord_matches(self):
        gt = [Chord(0, 2, None, "N", None, "N")]
        pred = [chord_from_label(0, 2, "N")]
        r = score_timeline(gt, pred)
        assert r["mirex_root"] == pytest.approx(1.0)
        assert r["bass_root"] == pytest.approx(1.0)

    def test_partial_vs_strict_divergence(self):
        # A single maj7-for-maj covering the whole span: partial 1.0, strict 0.
        gt = [Chord(0, 4, 0, "maj", 0, "C:maj")]
        pred = [chord_from_label(0, 4, "C:maj7")]
        r = score_timeline(gt, pred)
        assert r["partial_credit"] == pytest.approx(1.0)
        assert r["strict"] == pytest.approx(0.0)
        assert r["mirex_majmin"] == pytest.approx(1.0)
        assert r["mirex_sevenths"] == pytest.approx(0.0)

    def test_sounding_bass_target(self):
        # Right root+quality, wrong bass -> bass metric fails, others pass.
        gt = [Chord(0, 4, 0, "maj", 4, "C:maj/E")]      # sounding bass E
        pred = [chord_from_label(0, 4, "C:maj")]         # bass = root C
        r = score_timeline(gt, pred)
        assert r["mirex_root"] == pytest.approx(1.0)
        assert r["strict"] == pytest.approx(1.0)
        assert r["bass_root"] == pytest.approx(0.0)


class TestMirEvalCrossCheck:
    def test_own_engine_agrees_with_mireval(self):
        pytest.importorskip("mir_eval")
        x = mireval_crosscheck(GT_4, PRED_4)
        assert x is not None
        assert x["root"] == pytest.approx(0.75, abs=1e-6)
        assert x["majmin"] == pytest.approx(0.75, abs=1e-6)
        assert x["sevenths"] == pytest.approx(0.50, abs=1e-6)


# ---------------------------------------------------------------------------
# Schema loading + honesty gate
# ---------------------------------------------------------------------------

class TestFrozenGTAndGate:
    def test_load_fixture(self):
        gt = load_frozen_gt(FIXTURE)
        assert gt.song_id == "_fixture_4chord_synthetic"
        assert gt.verified is False
        assert len(gt.gt_chords) == 4
        assert gt.gt_chords[3].bass_pc == 11  # sounding bass of G:7/B

    def test_score_song_with_supplied_pred_matches_math(self):
        # Score the fixture GT against PRED_4 without running the pipeline.
        s = score_song(FIXTURE, allow_unverified=True, pred_chords=PRED_4)
        assert s.mirex_root == pytest.approx(0.75)
        assert s.mirex_sevenths == pytest.approx(0.50)
        assert s.partial_credit == pytest.approx(0.75)
        assert s.strict == pytest.approx(0.50)
        assert s.bass_root == pytest.approx(0.50)
        assert s.verified is False

    def test_refuses_unverified_by_default(self):
        with pytest.raises(UnverifiedGTError):
            score_song(FIXTURE, pred_chords=PRED_4)

    def test_missing_required_field_raises(self, tmp_path):
        bad = tmp_path / "bad.gt.json"
        bad.write_text(json.dumps({"song_id": "x", "audio_path": "y"}))
        with pytest.raises(ValueError):
            load_frozen_gt(bad)


class TestPooling:
    def _write_gt(self, tmp_path, name, chords):
        p = tmp_path / f"{name}.gt.json"
        p.write_text(json.dumps({
            "song_id": name, "title": name, "audio_path": "x.wav",
            "verified": False,
            "gt_chords": [
                {"t0": c.t0, "t1": c.t1, "root_pc": c.root_pc,
                 "quality": c.quality, "bass_pc": c.bass_pc, "label": c.label}
                for c in chords],
        }))
        return p

    def test_pooled_is_duration_weighted_micro(self, tmp_path):
        # Song1: 4-chord fixture (root num=6s over 8s).
        # Song2: single 4s chord, perfect (root num=4s over 4s).
        # Pooled root = (6+4)/(8+4) = 10/12 = 0.8333...
        g1 = self._write_gt(tmp_path, "s1", GT_4)
        s1 = score_song(g1, allow_unverified=True, pred_chords=PRED_4)
        gt2 = [Chord(0, 4, 0, "maj", 0, "C:maj")]
        g2 = self._write_gt(tmp_path, "s2", gt2)
        s2 = score_song(g2, allow_unverified=True,
                        pred_chords=[chord_from_label(0, 4, "C:maj")])
        ds = DatasetScore(per_song=[s1, s2])
        pooled = ds.pooled
        assert pooled["n_songs"] == 2
        assert pooled["mirex_root"] == pytest.approx(10 / 12, abs=1e-4)
        assert pooled["duration_s"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# WIRING smoke (opt-in; runs the real pipeline)
# ---------------------------------------------------------------------------

SMOKE_AUDIO = REPO / "docs" / "audio" / "rwc_rwc_p001.m4a"


@pytest.mark.skipif(
    os.environ.get("HARMONIA_RUN_SMOKE") != "1" or not SMOKE_AUDIO.exists(),
    reason="set HARMONIA_RUN_SMOKE=1 (and provide rwc_rwc_p001.m4a) to run the "
           "heavy end-to-end wiring smoke",
)
def test_pipeline_wiring_smoke(tmp_path):
    """WIRING ONLY — proves the pipeline runs on real audio and the scorer
    consumes the schema. The returned number is NOT an accuracy result: the GT
    is a placeholder and verified=False."""
    placeholder = tmp_path / "placeholder.gt.json"
    placeholder.write_text(json.dumps({
        "song_id": "rwc_p001_PLACEHOLDER",
        "title": "RWC P001 (PLACEHOLDER GT — not real)",
        "audio_path": str(SMOKE_AUDIO),
        "verified": False,
        "downbeat_times": [],
        "gt_chords": [
            {"t0": 0.0, "t1": 10.0, "root_pc": 0, "quality": "maj",
             "bass_pc": 0, "label": "C:maj"},
            {"t0": 10.0, "t1": 20.0, "root_pc": 7, "quality": "7",
             "bass_pc": 7, "label": "G:7"},
        ],
    }))
    s = score_song(placeholder, allow_unverified=True, work_dir=tmp_path)
    # Pipeline produced a scored timeline over the GT span; schema consumed.
    assert s.verified is False
    assert s.duration_s == pytest.approx(20.0, abs=1e-3)
    assert 0.0 <= s.mirex_root <= 1.0
    assert 0.0 <= s.bass_root <= 1.0
