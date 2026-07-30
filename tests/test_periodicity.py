"""
Unit tests for harmonia/models/periodicity.py (docs/known_issues.md #1,
candidate C: periodicity/structure folding).

No audio — synthetic beat_probs with a planted period.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.periodicity import (
    build_period_folds,
    find_loop_phase,
    fold_beat_probs,
    score_periods,
    slice_folded_views,
)


def _tiled_beat_probs(period: int, n_repeats: int, n_keys: int = 88, seed: int = 0) -> np.ndarray:
    """Tile `period` distinct random vectors `n_repeats` times, with a
    little per-repeat noise so it's not a perfectly trivial signal."""
    rng = np.random.RandomState(seed)
    slots = rng.rand(period, n_keys).astype(np.float32)
    beat_probs = np.tile(slots, (n_repeats, 1))
    beat_probs += rng.normal(scale=0.02, size=beat_probs.shape).astype(np.float32)
    return np.clip(beat_probs, 0, None)


class TestScorePeriods:

    def test_recovers_planted_period(self):
        beat_probs = _tiled_beat_probs(period=32, n_repeats=6)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=3)

        assert 32 in scores
        assert scores[32] == max(scores.values())

    def test_recovers_planted_period_at_different_bar_length(self):
        beat_probs = _tiled_beat_probs(period=16, n_repeats=8)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=3)

        assert 16 in scores
        assert scores[16] == max(scores.values())

    def test_drops_harmonics_of_a_kept_period(self):
        """If L=32 is the real period, L=64 (its harmonic) shouldn't also
        be reported as independent evidence."""
        beat_probs = _tiled_beat_probs(period=32, n_repeats=8)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=3)

        assert 64 not in scores

    def test_pure_noise_does_not_falsely_dominate(self):
        """With no real structure, no candidate period should score
        dramatically higher than the others — there's nothing to find."""
        rng = np.random.RandomState(1)
        beat_probs = rng.rand(200, 88).astype(np.float32)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=3)

        values = list(scores.values())
        assert max(values) - min(values) < 0.2

    def test_returns_at_most_top_k(self):
        beat_probs = _tiled_beat_probs(period=32, n_repeats=8)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=2)
        assert len(scores) <= 2

    def test_short_input_returns_empty_rather_than_crashing(self):
        beat_probs = np.random.RandomState(2).rand(3, 88).astype(np.float32)
        scores = score_periods(beat_probs, beats_per_bar=4, top_k=3)
        assert scores == {}


class TestFindLoopPhase:

    def test_anchors_to_first_downbeat(self):
        is_downbeat = np.zeros(40, dtype=bool)
        is_downbeat[5] = True  # first downbeat at beat 5, not beat 0
        is_downbeat[5 + 16] = True
        phase = find_loop_phase(period=16, is_downbeat=is_downbeat)
        assert phase == 5

    def test_zero_offset_when_song_starts_on_a_downbeat(self):
        is_downbeat = np.zeros(40, dtype=bool)
        is_downbeat[0] = True
        is_downbeat[16] = True
        phase = find_loop_phase(period=16, is_downbeat=is_downbeat)
        assert phase == 0

    def test_phase_is_reduced_mod_period(self):
        is_downbeat = np.zeros(40, dtype=bool)
        is_downbeat[20] = True  # first downbeat past one full period
        phase = find_loop_phase(period=16, is_downbeat=is_downbeat)
        assert phase == 4

    def test_no_downbeats_returns_zero(self):
        is_downbeat = np.zeros(40, dtype=bool)
        assert find_loop_phase(period=16, is_downbeat=is_downbeat) == 0

    def test_non_positive_period_returns_zero(self):
        is_downbeat = np.zeros(40, dtype=bool)
        is_downbeat[3] = True
        assert find_loop_phase(period=0, is_downbeat=is_downbeat) == 0


class TestFoldBeatProbs:

    def test_shape_preserved(self):
        beat_probs = _tiled_beat_probs(period=8, n_repeats=4)
        folded = fold_beat_probs(beat_probs, period=8)
        assert folded.shape == beat_probs.shape

    def test_averages_out_noise_across_repeats(self):
        """Folding a noisy repeating signal should land close to the clean
        underlying pattern — averaging cancels the per-repeat noise."""
        rng = np.random.RandomState(3)
        period = 8
        n_repeats = 20
        clean_slots = rng.rand(period, 88).astype(np.float32)
        beat_probs = np.tile(clean_slots, (n_repeats, 1))
        beat_probs += rng.normal(scale=0.3, size=beat_probs.shape).astype(np.float32)

        folded = fold_beat_probs(beat_probs, period=period)

        # every occurrence of a given slot should fold to the same vector
        slot0_rows = folded[0::period]
        for row in slot0_rows[1:]:
            np.testing.assert_allclose(row, slot0_rows[0])

        # and that vector should be much closer to the clean pattern than
        # any single noisy observation was
        clean = clean_slots[0]
        raw_beat0_error = np.abs(beat_probs[0] - clean).mean()
        folded_error = np.abs(folded[0] - clean).mean()
        assert folded_error < raw_beat0_error

    def test_same_slot_positions_get_identical_values(self):
        beat_probs = np.random.RandomState(4).rand(24, 88).astype(np.float32)
        folded = fold_beat_probs(beat_probs, period=6)
        for slot in range(6):
            rows = folded[slot::6]
            for row in rows[1:]:
                np.testing.assert_allclose(row, rows[0])


class TestBuildPeriodFoldsAndSlice:
    """The fold-then-slice pair extracted 2026-07-30 from the deleted
    `pipeline.py::HarmoniaPipeline.run`, which was its only caller and had no
    test coverage at all. These pin the one property that made the original
    code correct."""

    def test_weights_normalise_to_one(self):
        beat_probs = _tiled_beat_probs(period=16, n_repeats=8)
        _, weights = build_period_folds(beat_probs, beats_per_bar=4, top_k=3)
        assert weights
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w > 0 for w in weights.values())

    def test_folds_cover_full_track_and_match_direct_fold(self):
        beat_probs = _tiled_beat_probs(period=16, n_repeats=8)
        folded_full, weights = build_period_folds(beat_probs, beats_per_bar=4, top_k=3)
        for L, folded in folded_full.items():
            assert folded.shape == beat_probs.shape
            np.testing.assert_allclose(folded, fold_beat_probs(beat_probs, L))
        assert set(folded_full) == set(weights)

    def test_empty_track_yields_no_views(self):
        """Too short to score any period -> empty dicts -> slice returns None,
        which the decoder treats as 'no extra evidence'."""
        folded_full, weights = build_period_folds(
            np.random.RandomState(1).rand(2, 88).astype(np.float32), beats_per_bar=4,
        )
        assert folded_full == {} and weights == {}
        assert slice_folded_views(folded_full, weights, 0, 2) is None

    def test_slice_uses_absolute_beat_position(self):
        """THE correctness property. A slice starting at an absolute beat that
        is NOT a multiple of the period must not begin at the loop's slot 0 —
        that is exactly the off-by-phase bug that folding per segment would
        introduce, silently averaging each segment against wrong positions."""
        period = 8
        beat_probs = _tiled_beat_probs(period=period, n_repeats=10)
        folded = fold_beat_probs(beat_probs, period)
        folded_full, weights = {period: folded}, {period: 1.0}

        start = 3  # deliberately not a multiple of the period
        views = slice_folded_views(folded_full, weights, start, start + period)
        assert views is not None and len(views) == 1
        sl, w = views[0]
        assert w == 1.0
        # the slice is the absolute window, i.e. it starts on slot 3
        np.testing.assert_allclose(sl, folded[start:start + period])
        # and that is NOT what folding the segment on its own would give
        naive = fold_beat_probs(beat_probs[start:start + period], period)
        assert not np.allclose(sl, naive)

    def test_slice_end_beat_is_clamped(self):
        """An over-long final segment must not raise or silently pad."""
        beat_probs = _tiled_beat_probs(period=8, n_repeats=4)  # 32 beats
        folded_full, weights = build_period_folds(beat_probs, beats_per_bar=4, top_k=1)
        views = slice_folded_views(folded_full, weights, 24, 999)
        assert views is not None
        assert views[0][0].shape[0] == 8

    def test_slices_align_1to1_with_the_raw_segment(self):
        """Each returned view must be usable as extra evidence beat-for-beat
        against the segment's own beat_probs slice — same length, same order."""
        beat_probs = _tiled_beat_probs(period=16, n_repeats=6)
        folded_full, weights = build_period_folds(beat_probs, beats_per_bar=4, top_k=3)
        for start, end in ((0, 13), (13, 40), (40, beat_probs.shape[0])):
            raw = beat_probs[start:end]
            for sl, _ in slice_folded_views(folded_full, weights, start, end):
                assert sl.shape == raw.shape
