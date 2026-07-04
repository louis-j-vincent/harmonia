"""
Unit tests for harmonia/models/periodicity.py (docs/known_issues.md #1,
candidate C: periodicity/structure folding).

No audio — synthetic beat_probs with a planted period.
"""

from __future__ import annotations

import numpy as np

from harmonia.models.periodicity import find_loop_phase, fold_beat_probs, score_periods


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
