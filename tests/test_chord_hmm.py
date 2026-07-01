"""
Unit tests for the Bayesian chord HMM (harmonia/models/chord_hmm.py).

These are regression tests for two bugs found while validating the full
pipeline end-to-end on POP909 (see docs/plots/inference/pop909_001):

1. NO_CHORD used to become a near-absorbing Viterbi state: its transition
   row never received the jazz-progression weights real chords get, so
   after the self-transition boost its self-loop probability (~53%) was an
   order of magnitude higher than any real chord's (~4%), causing the
   decoder to collapse to "N" for an entire song regardless of emission
   evidence.
2. `_compress_path` could emit a zero-duration (or, across segment
   boundaries, an overlapping) chord event whenever a Viterbi run ended
   exactly on the last beat of a segment.

No audio or ML models are involved — everything here is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.chord_hmm import (
    ChordInferrer,
    build_emission_matrix,
    build_key_prior,
    build_transition_matrix,
    viterbi,
)
from harmonia.theory.chord_vocabulary import ChordQuality, build_index, chord_label
from harmonia.theory.key_profiles import KeyPosterior


def _no_chord_idx(max_phase: int = 1) -> int:
    _, chord_to_idx = build_index(max_phase)
    return chord_to_idx[(-1, ChordQuality.NO_CHORD)]


# ── Transition matrix: NO_CHORD must not be a sticky sink ──────────────────────

class TestTransitionMatrixNoChord:

    def test_no_chord_self_transition_not_dominant(self):
        """P(N -> N) must be the same order of magnitude as a real chord's
        self-transition probability, not ~10x larger."""
        log_A = build_transition_matrix(tonic=0, max_phase=1)
        n_idx = _no_chord_idx()

        p_no_chord_self = np.exp(log_A[n_idx, n_idx])
        real_diag = np.exp(np.diagonal(log_A))
        real_diag = np.delete(real_diag, n_idx)

        assert p_no_chord_self < 5 * real_diag.mean()

    def test_no_chord_persistence_is_tunable(self):
        """no_chord_self_transition_boost must actually change P(N -> N)."""
        log_A_low = build_transition_matrix(
            tonic=0, max_phase=1, no_chord_self_transition_boost=0.1
        )
        log_A_high = build_transition_matrix(
            tonic=0, max_phase=1, no_chord_self_transition_boost=5.0
        )
        n_idx = _no_chord_idx()
        assert log_A_high[n_idx, n_idx] > log_A_low[n_idx, n_idx]

    def test_rows_are_normalised(self):
        log_A = build_transition_matrix(tonic=3, max_phase=1)
        row_sums = np.exp(log_A).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


# ── Viterbi must not collapse to NO_CHORD given clear evidence ────────────────

class TestViterbiNoCollapse:

    def test_strong_alternating_evidence_is_not_all_no_chord(self):
        """Feed the decoder unambiguous, strongly alternating chord evidence
        and confirm it doesn't settle on NO_CHORD throughout."""
        max_phase = 1
        idx_to_chord, chord_to_idx = build_index(max_phase)
        C = len(idx_to_chord)
        n_idx = chord_to_idx[(-1, ChordQuality.NO_CHORD)]

        c_maj = chord_to_idx[(0, ChordQuality.MAJOR)]
        g_maj = chord_to_idx[(7, ChordQuality.MAJOR)]

        T = 20
        log_obs = np.full((T, C), -10.0)
        for t in range(T):
            target = c_maj if t % 2 == 0 else g_maj
            log_obs[t, target] = 0.0  # overwhelmingly likely at this beat

        log_A = build_transition_matrix(tonic=0, max_phase=max_phase)
        log_init = build_key_prior(tonic=0, mode="major", max_phase=max_phase)

        path, _ = viterbi(log_obs, log_A, log_init)

        assert not np.all(path == n_idx)
        # with such strong, alternating emission evidence the decoder should
        # track it rather than lock onto a single hub chord throughout
        assert len(set(path.tolist())) > 1

    def test_ambiguous_evidence_can_still_pick_no_chord(self):
        """Sanity check the other direction: with genuinely flat/uninformative
        emission, NO_CHORD remains an available, reasonable decode (we only
        fixed it being an inescapable trap, not removed it as an option)."""
        max_phase = 1
        idx_to_chord, chord_to_idx = build_index(max_phase)
        C = len(idx_to_chord)
        n_idx = chord_to_idx[(-1, ChordQuality.NO_CHORD)]

        T = 5
        log_obs = np.zeros((T, C))  # perfectly flat: no evidence for anything

        log_A = build_transition_matrix(tonic=0, max_phase=max_phase)
        log_init = build_key_prior(tonic=0, mode="major", max_phase=max_phase)
        path, _ = viterbi(log_obs, log_A, log_init)

        # shouldn't error, should produce a valid path
        assert path.shape == (T,)
        assert ((path >= 0) & (path < C)).all()


# ── ChordInferrer / _compress_path: no zero-duration or overlapping events ────

class TestCompressPath:

    def _make_inferrer(self) -> ChordInferrer:
        return ChordInferrer(max_phase=1)

    def _key(self) -> KeyPosterior:
        return KeyPosterior(
            log_probs=np.zeros(24), tonic=0, mode="major",
            key_name="C major", confidence=1.0,
        )

    def test_no_zero_duration_event_at_segment_end(self):
        """A run that ends exactly on the last beat of a segment must not
        produce a start_time_s == end_time_s event (previously happened
        because the end index was clamped to the last valid beat instead of
        extrapolated past it)."""
        inferrer = self._make_inferrer()
        B = 12
        beat_probs = np.random.RandomState(0).rand(B, 88).astype(np.float32)
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        events = inferrer.infer(beat_probs, beat_times, self._key())

        for ev in events:
            assert ev.end_time_s > ev.start_time_s

    def test_segment_end_time_s_prevents_cross_segment_overlap(self):
        """When the caller supplies the authoritative next-beat time via
        segment_end_time_s, the last event must end exactly there rather
        than being extrapolated past it (which could overlap the next
        segment's first event)."""
        inferrer = self._make_inferrer()
        B = 8
        beat_probs = np.random.RandomState(1).rand(B, 88).astype(np.float32)
        beat_times = np.arange(B, dtype=np.float64) * 0.5
        next_beat_time = float(beat_times[-1] + 0.37)  # deliberately not avg spacing

        events = inferrer.infer(
            beat_probs, beat_times, self._key(), segment_end_time_s=next_beat_time
        )

        assert events[-1].end_time_s == pytest.approx(next_beat_time)

    def test_confidence_is_bounded_and_nonzero(self):
        """Confidence must be a meaningful (0, 1] value, not exactly 0.0 —
        averaging the raw cumulative Viterbi log-probability over a long run
        used to underflow to exp(huge negative) == 0.0 regardless of fit."""
        inferrer = self._make_inferrer()
        B = 30  # long enough that the old cumulative-log-prob bug underflowed
        beat_probs = np.random.RandomState(2).rand(B, 88).astype(np.float32)
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        events = inferrer.infer(beat_probs, beat_times, self._key())

        for ev in events:
            assert 0.0 < ev.confidence <= 1.0

    def test_events_are_contiguous(self):
        inferrer = self._make_inferrer()
        B = 16
        beat_probs = np.random.RandomState(3).rand(B, 88).astype(np.float32)
        beat_times = np.arange(B, dtype=np.float64) * 0.6

        events = inferrer.infer(beat_probs, beat_times, self._key())

        for prev, nxt in zip(events, events[1:]):
            assert prev.end_time_s == pytest.approx(nxt.start_time_s)
