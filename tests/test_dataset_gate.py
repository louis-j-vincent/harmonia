"""Audio-free unit tests for the dataset harvest gate + a small integration test.

The gate (``harmonia.dataset.gate``) is a pure function on pre-normalised
signals, so it is fully testable with no audio / models / I/O. The integration
test drives ``build_segment_signals`` + ``harvest_song`` + ``write_manifests``
with a synthetic aligner-proposal dict and a synthetic (audio-free) BeatLock, so
it exercises the join + row-grouping + manifest emission without touching Beat
This!, librosa or ffmpeg.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from harmonia.dataset.gate import (Bucket, GateConfig, SegmentSignals,
                                    gate_segment)
from harmonia.dataset.harvest import (BeatLock, build_segment_signals,
                                      harvest_song, write_manifests)


# ── helpers ──────────────────────────────────────────────────────────────────

def _sig(**kw) -> SegmentSignals:
    base = dict(
        t0=0.0, t1=2.0, label="C:maj",
        agreement=0.50, agreement_norm=0.90,
        beat_lock=0.80, octave_locked=True,
        coverage_full=True, near_gap=False,
        transpose_margin=0.10, xrep_consistent=True, xrep_divergence=False,
    )
    base.update(kw)
    return SegmentSignals(**base)


CFG = GateConfig()


# ═════════════════════════════════════════════════════════════════════════════
# Gate: bucket 1 (CLEAN)
# ═════════════════════════════════════════════════════════════════════════════

def test_clean_when_all_locks_and_high_agreement():
    d = gate_segment(_sig(), CFG)
    assert d.bucket is Bucket.CLEAN
    assert 0.0 < d.confidence <= 1.0


def test_agreement_threshold_is_the_precision_knob():
    just_below = gate_segment(_sig(agreement=CFG.agr_keep - 0.01), CFG)
    at = gate_segment(_sig(agreement=CFG.agr_keep + 0.01), CFG)
    assert just_below.bucket is Bucket.DROP
    assert at.bucket is Bucket.CLEAN


def test_norm_floor_can_veto_clean():
    # high raw agr but its per-song-normalised value is under the floor
    d = gate_segment(_sig(agreement=0.5, agreement_norm=CFG.agr_norm_keep - 0.1), CFG)
    assert d.bucket is Bucket.DROP


# ═════════════════════════════════════════════════════════════════════════════
# Gate: bucket 3 (DROP) — precision-first
# ═════════════════════════════════════════════════════════════════════════════

def test_low_agreement_but_time_aligned_drops_not_reviews():
    # NOT a clean substitution (no divergence signal) -> drop, never emit/review
    d = gate_segment(_sig(agreement=0.15, agreement_norm=0.30), CFG)
    assert d.bucket is Bucket.DROP


def test_weak_beat_lock_drops():
    d = gate_segment(_sig(beat_lock=CFG.beat_lock_min - 0.05), CFG)
    assert d.bucket is Bucket.DROP


def test_weak_transpose_margin_drops():
    d = gate_segment(_sig(transpose_margin=CFG.transpose_margin_min - 0.01), CFG)
    assert d.bucket is Bucket.DROP


def test_near_gap_drops():
    d = gate_segment(_sig(near_gap=True), CFG)
    assert d.bucket is Bucket.DROP


def test_uncovered_drops():
    d = gate_segment(_sig(coverage_full=False), CFG)
    assert d.bucket is Bucket.DROP


def test_octave_unlocked_drops():
    d = gate_segment(_sig(octave_locked=False), CFG)
    assert d.bucket is Bucket.DROP


def test_xrep_inconsistent_drops():
    d = gate_segment(_sig(xrep_consistent=False), CFG)
    assert d.bucket is Bucket.DROP


def test_song_level_broad_misalignment_veto():
    d = gate_segment(_sig(frac_low_song=CFG.frac_low_veto + 0.05), CFG)
    assert d.bucket is Bucket.DROP
    assert "veto" in d.reasons[0]


# ═════════════════════════════════════════════════════════════════════════════
# Gate: bucket 2 (REVIEW / substitution) — valuable, not noise
# ═════════════════════════════════════════════════════════════════════════════

def test_xrep_divergence_time_aligned_routes_to_review():
    d = gate_segment(_sig(agreement=0.10, agreement_norm=0.2, xrep_divergence=True), CFG)
    assert d.bucket is Bucket.REVIEW


def test_strong_split_time_aligned_routes_to_review():
    d = gate_segment(_sig(agreement=0.20, agreement_norm=0.4,
                          split_fire=True, split_margin=CFG.split_review_margin + 0.05), CFG)
    assert d.bucket is Bucket.REVIEW


def test_weak_split_does_not_review():
    # a lone weak split-fire is treated as spurious (melody flux) -> drop
    d = gate_segment(_sig(agreement=0.20, agreement_norm=0.4,
                          split_fire=True, split_margin=CFG.split_review_margin - 0.05), CFG)
    assert d.bucket is Bucket.DROP


def test_divergence_but_misaligned_does_not_review():
    # the key safety property: a merely-misaligned span must NOT masquerade as a
    # substitution. Divergence + weak time-lock -> DROP, not REVIEW.
    d = gate_segment(_sig(agreement=0.10, beat_lock=0.1, xrep_divergence=True), CFG)
    assert d.bucket is Bucket.DROP


def test_high_agreement_never_review():
    # a divergence flag requires low agreement; a confidently-fitting chord is CLEAN
    d = gate_segment(_sig(agreement=0.55, xrep_divergence=False,
                          split_fire=True, split_margin=0.3), CFG)
    assert d.bucket is Bucket.CLEAN


# ═════════════════════════════════════════════════════════════════════════════
# Gate: confidence
# ═════════════════════════════════════════════════════════════════════════════

def test_confidence_monotonic_in_agreement():
    lo = gate_segment(_sig(agreement=0.35), CFG).confidence
    hi = gate_segment(_sig(agreement=0.60), CFG).confidence
    assert hi > lo


def test_confidence_penalised_by_weak_beat_lock():
    strong = gate_segment(_sig(beat_lock=1.0), CFG).confidence
    weak = gate_segment(_sig(beat_lock=0.5), CFG).confidence
    assert strong > weak


# ═════════════════════════════════════════════════════════════════════════════
# Integration: proposal-dict -> signals -> rows -> manifest (no audio)
# ═════════════════════════════════════════════════════════════════════════════

def _fake_proposal() -> dict:
    """A synthetic aligner proposal with one clean chord, one low-agr (drop) chord,
    one cross-rep divergence (review) chord, and a gap boundary."""
    gt_chords = [
        dict(t0=1.0, t1=3.0, root_pc=0, quality="maj", bass_pc=0, label="C:maj"),
        dict(t0=3.0, t1=5.0, root_pc=7, quality="maj", bass_pc=7, label="G:maj"),
        dict(t0=5.0, t1=7.0, root_pc=9, quality="min", bass_pc=9, label="A:min"),
        dict(t0=8.0, t1=10.0, root_pc=5, quality="maj", bass_pc=5, label="F:maj"),
    ]
    per_chord = [
        dict(t0=1.0, t1=3.0, label="C:maj", agr=0.55),   # clean
        dict(t0=3.0, t1=5.0, label="G:maj", agr=0.50),   # clean
        dict(t0=5.0, t1=7.0, label="A:min", agr=0.10),   # divergence -> review
        dict(t0=8.0, t1=10.0, label="F:maj", agr=0.18),  # low + near gap -> drop
    ]
    return dict(
        audio_path="docs/audio/does_not_matter.m4a",
        gt_chords=gt_chords,
        proposal=dict(
            beat_this=dict(beat_period=0.5),
            harmonic_agreement=dict(overall=0.4, worst=0.1, frac_low=0.05, ceiling=0.6),
            agreement_detail=dict(per_chord=per_chord,
                                  per_transpose=[[0, 0.40], [5, 0.20]]),
            section_alignment=dict(
                coverage=0.9, avg_agreement=0.4, n_placed=2, start_margin=0.1,
                # trailing outro gap: makes F:maj (t1=10) near_gap without touching
                # the interior divergence chord A:min (t1=7).
                gaps_unlabeled=[dict(after="B1", before="", t0=10.0, t1=11.0, dur=1.0)],
                sections=[
                    dict(label="A", t0=1.0, t1=7.0, agreement=0.4, occ=0, sec_idx=0),
                    dict(label="B", t0=8.0, t1=10.0, agreement=0.3, occ=0, sec_idx=1),
                ]),
            refinement=dict(cross_repetition=[
                dict(label="A", n_occ=2, var_after=0.05, occ_overall=[0.4, 0.4],
                     positions=["C:maj", "G:maj", "A:min"],
                     divergences=[dict(pos=2, label="A:min", mean=0.10, std=0.03,
                                       t_first=5.0, note="chart!=recording")]),
            ]),
            georgia_bundle=dict(split_detector_fires=[]),
        ),
    )


def _flat_beatlock(level: float = 0.30) -> BeatLock:
    times = np.linspace(0, 12, 120)
    rel = np.full_like(times, level)
    return BeatLock(octave_locked=True, ref_level=level, _rel_times=times, _rel=rel)


def test_build_segment_signals_join_and_flags():
    sigs = build_segment_signals(_fake_proposal(), _flat_beatlock())
    assert len(sigs) == 4
    by_label = {s.label: s for _, s in sigs}
    assert by_label["C:maj"].agreement == 0.55
    assert by_label["A:min"].xrep_divergence is True         # flagged uniform-low
    assert by_label["F:maj"].near_gap is True                # abuts the 7-8s gap
    assert abs(by_label["C:maj"].transpose_margin - 0.20) < 1e-9
    assert by_label["C:maj"].beat_lock == pytest.approx(1.0)  # at this song's ref


def test_harvest_song_three_way_split_and_rows():
    song = dict(song_id="unit_song", title="Unit", audio="docs/audio/x.m4a",
                ireal_file="none", tune_title="Unit")
    res = harvest_song(song, proposal=_fake_proposal(), beat_lock=_flat_beatlock())
    # C:maj + G:maj are contiguous cleans -> one grouped row of 2 chords
    assert res.stats["n_clean_rows"] == 1
    assert res.stats["n_clean_chords"] == 2
    assert res.clean_rows[0]["chords"][0]["label"] == "C:maj"
    assert res.clean_rows[0]["chords"][1]["label"] == "G:maj"
    # A:min -> review; F:maj -> drop
    assert res.stats["n_review"] == 1
    assert res.review_rows[0]["chart_suggestion"] == "A:min"
    assert res.stats["n_dropped"] == 1
    # row references audio + offsets, does not embed audio
    row = res.clean_rows[0]
    assert set(["audio_path", "t0", "t1", "chords", "confidence", "song_id", "source"]) <= set(row)


def test_write_manifests_jsonl(tmp_path):
    song = dict(song_id="unit_song", title="Unit", audio="docs/audio/x.m4a",
                ireal_file="none", tune_title="Unit")
    res = harvest_song(song, proposal=_fake_proposal(), beat_lock=_flat_beatlock())
    clean = tmp_path / "manifest.jsonl"
    review = tmp_path / "review.jsonl"
    stats = write_manifests([res], clean_path=clean, review_path=review)
    assert clean.exists() and review.exists()
    clean_lines = [json.loads(l) for l in clean.read_text().splitlines()]
    review_lines = [json.loads(l) for l in review.read_text().splitlines()]
    assert len(clean_lines) == 1 and len(review_lines) == 1
    assert stats["n_clean_pairs"] == 2
    assert review_lines[0]["status"] == "unlabeled"


def test_frac_low_veto_emits_nothing():
    prop = _fake_proposal()
    prop["proposal"]["harmonic_agreement"]["frac_low"] = 0.9   # broadly misaligned
    song = dict(song_id="bad", title="Bad", audio="x", ireal_file="none", tune_title="Bad")
    res = harvest_song(song, proposal=prop, beat_lock=_flat_beatlock())
    assert res.stats["n_clean_rows"] == 0
    assert res.stats["n_review"] == 0
