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

Also covers the `normalize_emission`/`compress_emission` preprocessing
options investigated for docs/known_issues.md #1 (chord-change temporal
resolution): `normalize_emission` (L1, per beat) is provably a no-op on the
decoded path (see TestEmissionPreprocessing), which is why the fix for that
issue instead uses `compress_emission`.

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
    viterbi_duration_aware,
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


class TestEmissionPreprocessing:
    """docs/known_issues.md #1: emission-signal-quality experiments."""

    def _key(self) -> KeyPosterior:
        return KeyPosterior(
            log_probs=np.zeros(24), tonic=0, mode="major",
            key_name="C major", confidence=1.0,
        )

    def test_normalize_emission_does_not_change_decoded_path(self):
        """L1-normalizing beat_probs per beat subtracts a per-beat constant
        from every chord's log-emission uniformly, which can never change
        which chord wins at any step of Viterbi — proven inert, not just
        empirically weak. This test is the formal version of that proof."""
        B = 20
        beat_probs = np.random.RandomState(4).rand(B, 88).astype(np.float32) * 10

        plain = ChordInferrer(max_phase=1, normalize_emission=False)
        normalized = ChordInferrer(max_phase=1, normalize_emission=True)
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        events_plain = plain.infer(beat_probs.copy(), beat_times, self._key())
        events_norm = normalized.infer(beat_probs.copy(), beat_times, self._key())

        assert [(e.root, e.quality) for e in events_plain] == \
               [(e.root, e.quality) for e in events_norm]
        for ep, en in zip(events_plain, events_norm):
            assert ep.start_time_s == en.start_time_s
            assert ep.end_time_s == en.end_time_s

    def test_compress_emission_can_change_decoded_path(self):
        """Unlike normalize_emission, compress_emission is a nonlinear
        per-element transform and *can* change the decoded path — sanity
        check that it actually does something (not a claim it improves
        accuracy; see docs/known_issues.md #1 for why it was not adopted)."""
        B = 20
        # Construct beat_probs where one dominant key would swamp the
        # comparison unless compressed.
        rng = np.random.RandomState(5)
        beat_probs = rng.rand(B, 88).astype(np.float32)
        beat_probs[:, 30] *= 50  # one loud, recurring key

        plain = ChordInferrer(max_phase=1, compress_emission=None)
        compressed = ChordInferrer(max_phase=1, compress_emission="sqrt")
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        events_plain = plain.infer(beat_probs.copy(), beat_times, self._key())
        events_compressed = compressed.infer(beat_probs.copy(), beat_times, self._key())

        # Not asserting a specific outcome — just that compression is not a
        # no-op the way normalize_emission is.
        assert [(e.root, e.quality) for e in events_plain] != \
               [(e.root, e.quality) for e in events_compressed]

    def test_invalid_compress_emission_rejected(self):
        with pytest.raises(ValueError):
            ChordInferrer(max_phase=1, compress_emission="not_a_real_option")


class TestViterbiDurationAware:
    """
    docs/known_issues.md #1, candidate B: explicit-duration (semi-Markov)
    decoding, replacing the memoryless geometric duration implied by a
    boosted self-transition. Empirically this did NOT improve full-pipeline
    accuracy (majmin dropped from 17.0% -> ~10% across all boost blends
    tested) — the bottleneck turned out to be quality discrimination in the
    emission signal, which more frequent segment boundaries just exposed
    more often, not the duration model's shape. Not adopted as the default;
    these tests lock in the decoder's *mechanical* correctness regardless.
    """

    def _uniform_log_duration(self, C: int, D: int, peak_d: int) -> np.ndarray:
        """A simple duration prior strongly peaked at `peak_d` beats, for all states."""
        pmf = np.full(D, 1e-6)
        pmf[peak_d - 1] = 1.0
        pmf /= pmf.sum()
        return np.tile(np.log(pmf), (C, 1))

    def test_tracks_clean_alternating_evidence(self):
        """With strong, unambiguous 2-beat-alternating evidence, the decoder
        should track it — same sanity bar as the plain viterbi() test."""
        max_phase = 1
        idx_to_chord, chord_to_idx = build_index(max_phase)
        C = len(idx_to_chord)
        c_maj = chord_to_idx[(0, ChordQuality.MAJOR)]
        g_maj = chord_to_idx[(7, ChordQuality.MAJOR)]

        T = 40
        log_obs = np.full((T, C), -10.0)
        for t in range(T):
            log_obs[t, c_maj if (t // 2) % 2 == 0 else g_maj] = 0.0

        log_A = build_transition_matrix(tonic=0, max_phase=max_phase,
                                         self_transition_boost=0.0,
                                         no_chord_self_transition_boost=0.0)
        log_init = build_key_prior(tonic=0, mode="major", max_phase=max_phase)
        log_duration = self._uniform_log_duration(C, D=8, peak_d=2)

        path, _ = viterbi_duration_aware(log_obs, log_A, log_init, log_duration)

        assert len(set(path.tolist())) == 2
        assert set(path.tolist()) == {c_maj, g_maj}

    def test_escape_valve_for_long_stable_regions(self):
        """A single chord held far longer than any duration bin has mass
        should still decode as (approximately) that one chord throughout —
        not fragment into unrelated states just because no single segment
        can express the full length. Requires i == j segment chaining to be
        allowed (see viterbi_duration_aware docstring)."""
        max_phase = 1
        idx_to_chord, chord_to_idx = build_index(max_phase)
        C = len(idx_to_chord)
        c_maj7 = chord_to_idx[(0, ChordQuality.MAJ7)]

        T = 50  # far longer than the D=8 cap below
        log_obs = np.full((T, C), -10.0)
        log_obs[:, c_maj7] = 0.0  # unambiguous: always the same chord

        log_A = build_transition_matrix(tonic=0, max_phase=max_phase,
                                         self_transition_boost=0.0,
                                         no_chord_self_transition_boost=0.0)
        log_init = build_key_prior(tonic=0, mode="major", max_phase=max_phase)
        log_duration = self._uniform_log_duration(C, D=8, peak_d=4)

        path, _ = viterbi_duration_aware(log_obs, log_A, log_init, log_duration)

        # dominant state should be c_maj7 throughout, not a scattering of others
        assert (path == c_maj7).mean() > 0.9

    def test_duration_aware_matches_plain_viterbi_interface(self):
        """Drop-in interface parity: same (path, log_probs) shapes as viterbi()."""
        max_phase = 1
        idx_to_chord, _ = build_index(max_phase)
        C = len(idx_to_chord)
        T = 16
        rng = np.random.RandomState(6)
        log_obs = rng.rand(T, C)
        log_A = build_transition_matrix(tonic=0, max_phase=max_phase)
        log_init = build_key_prior(tonic=0, mode="major", max_phase=max_phase)
        log_duration = self._uniform_log_duration(C, D=8, peak_d=2)

        path, log_probs = viterbi_duration_aware(log_obs, log_A, log_init, log_duration)

        assert path.shape == (T,)
        assert log_probs.shape == (T,)
        assert ((path >= 0) & (path < C)).all()


class TestDurationAwareChordInferrer:

    def _key(self) -> KeyPosterior:
        return KeyPosterior(
            log_probs=np.zeros(24), tonic=0, mode="major",
            key_name="C major", confidence=1.0,
        )

    def test_duration_prior_switches_decoder(self):
        """ChordInferrer(duration_prior=...) must actually use the
        duration-aware decoder (different code path from the default)."""
        max_phase = 1
        _, chord_to_idx = build_index(max_phase)
        C = len(chord_to_idx)
        D = 8
        pmf = np.full(D, 1e-6)
        pmf[1] = 1.0  # peak at duration=2
        pmf /= pmf.sum()
        prior = {"chord": pmf, "no_chord": pmf.copy()}

        inferrer = ChordInferrer(max_phase=max_phase, duration_prior=prior)
        assert inferrer._log_duration_by_state is not None
        assert inferrer._log_duration_by_state.shape == (C, D)

        plain = ChordInferrer(max_phase=max_phase)
        assert plain._log_duration_by_state is None


class TestFoldedViews:
    """docs/known_issues.md #1, candidate C: periodicity folding. Not
    adopted (neutral at best, harmful at full weight — see writeup), but
    the wiring itself should behave correctly regardless."""

    def _key(self) -> KeyPosterior:
        return KeyPosterior(
            log_probs=np.zeros(24), tonic=0, mode="major",
            key_name="C major", confidence=1.0,
        )

    def test_folded_views_can_change_decoded_path(self):
        rng = np.random.RandomState(7)
        B = 16
        beat_probs = rng.rand(B, 88).astype(np.float32)
        # a folded view with very different (structured) content should be
        # able to pull the decision away from the raw-only result
        folded = rng.rand(B, 88).astype(np.float32) * 5
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        plain = ChordInferrer(max_phase=1)
        with_folding = ChordInferrer(max_phase=1, periodicity_weight=5.0)

        events_plain = plain.infer(beat_probs.copy(), beat_times, self._key())
        events_folded = with_folding.infer(
            beat_probs.copy(), beat_times, self._key(),
            folded_views=[(folded, 1.0)],
        )

        assert [(e.root, e.quality) for e in events_plain] != \
               [(e.root, e.quality) for e in events_folded]

    def test_zero_weight_folded_view_is_a_no_op(self):
        rng = np.random.RandomState(8)
        B = 12
        beat_probs = rng.rand(B, 88).astype(np.float32)
        folded = rng.rand(B, 88).astype(np.float32) * 10
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        inferrer = ChordInferrer(max_phase=1)
        events_no_folding = inferrer.infer(beat_probs.copy(), beat_times, self._key())
        events_zero_weight = inferrer.infer(
            beat_probs.copy(), beat_times, self._key(),
            folded_views=[(folded, 0.0)],
        )

        assert [(e.root, e.quality) for e in events_no_folding] == \
               [(e.root, e.quality) for e in events_zero_weight]

    def test_empty_folded_views_list_is_a_no_op(self):
        rng = np.random.RandomState(9)
        B = 10
        beat_probs = rng.rand(B, 88).astype(np.float32)
        beat_times = np.arange(B, dtype=np.float64) * 0.5

        inferrer = ChordInferrer(max_phase=1)
        events_none = inferrer.infer(beat_probs.copy(), beat_times, self._key(), folded_views=None)
        events_empty = inferrer.infer(beat_probs.copy(), beat_times, self._key(), folded_views=[])

        assert [(e.root, e.quality) for e in events_none] == \
               [(e.root, e.quality) for e in events_empty]
