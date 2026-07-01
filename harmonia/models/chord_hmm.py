"""
Bayesian chord inference via Hidden Markov Model.

The HMM operates per structural segment, conditioned on the inferred key.
This is the core of what makes Harmonia better than frame-level classifiers.

Model:
    Hidden states:   chord vocabulary (root × quality + N)
    Observations:    beat-level note probability vectors (B, 88)

    P(chord_t | obs_t, key, chord_{t-1}, style) ∝
        P(obs_t | chord_t)              [emission — chord template match]
      × P(chord_t | key)               [key prior — diatonic chords preferred]
      × P(chord_t | chord_{t-1}, style) [transition — jazz priors]

Inference:
    Viterbi algorithm → MAP chord sequence per segment.
    Also computes forward-backward for full posterior if needed.

Key design choices:
    - Emission model built from ChordTemplate soft weights (music-theory-informed)
      with a small noise floor to handle notes outside the chord (passing tones).
    - Key prior up-weights the 7 diatonic chords, down-weights chromatic ones.
    - Transition matrix encodes jazz progressions in scale-agnostic form,
      then instantiated for the inferred key at inference time.
    - Harmonic rhythm prior: minimum chord duration enforced via self-transition
      boost, scaled by style × tempo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from harmonia.theory.chord_vocabulary import (
    CHORD_TEMPLATES,
    ChordQuality,
    build_index,
    chord_label,
    get_vocabulary,
    n_chords,
)
from harmonia.theory.jazz_priors import (
    PROGRESSIONS,
    STYLE_PRIORS,
    build_relative_transition_matrix,
)
from harmonia.theory.key_profiles import KeyPosterior

logger = logging.getLogger(__name__)

MIDI_START = 21   # A0
N_KEYS_PER_PIANO = 88


# ---------------------------------------------------------------------------
# Chord result
# ---------------------------------------------------------------------------

@dataclass
class ChordEvent:
    """A chord spanning a range of beats."""
    root: int               # 0–11, or -1 for N
    quality: ChordQuality
    label: str              # e.g. "Cmaj7", "G7", "N"
    start_beat: int
    end_beat: int
    start_time_s: float
    end_time_s: float
    confidence: float       # posterior probability at Viterbi path

    @property
    def duration_beats(self) -> int:
        return self.end_beat - self.start_beat


# ---------------------------------------------------------------------------
# Emission model
# ---------------------------------------------------------------------------

def build_emission_matrix(
    max_phase: int = 1,
    noise_floor: float = 0.05,
) -> np.ndarray:
    """
    Build the emission matrix E: (n_chords, 88).

    E[c, k] = probability that piano key k is active given chord c.

    Construction:
      - For each (root, quality) chord, look up ChordTemplate.
      - Map chord intervals to piano key indices across all octaves.
      - Set emission weight = template weight at that interval.
      - All other keys: noise_floor (handles passing tones, mistakes).

    Args:
        max_phase:    chord vocabulary phase.
        noise_floor:  base emission probability for notes not in chord.
                      Prevents zero-probability observations on passing tones.

    Returns:
        E: (n_chords, 88) float32, rows normalised to sum to 1.
    """
    idx_to_chord, _ = build_index(max_phase)
    C = len(idx_to_chord)
    E = np.full((C, N_KEYS_PER_PIANO), noise_floor, dtype=np.float32)

    for chord_idx, (root, quality) in enumerate(idx_to_chord):
        if quality == ChordQuality.NO_CHORD:
            # No chord: flat emission, slightly lower than noise
            E[chord_idx] = noise_floor * 0.5
            continue

        template = CHORD_TEMPLATES[quality]
        for key_idx in range(N_KEYS_PER_PIANO):
            midi = MIDI_START + key_idx
            pc = midi % 12
            # Distance from chord root in semitones (mod 12)
            interval = (pc - root) % 12
            if interval in template.weights:
                weight = template.weights[interval]
                # Scale by octave relevance: midrange octaves carry most harmonic info
                octave = midi // 12
                octave_weight = np.exp(-0.1 * (octave - 4.5) ** 2)
                E[chord_idx, key_idx] = float(weight * octave_weight)

    # Normalise rows
    row_sums = E.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    E /= row_sums

    return E


# ---------------------------------------------------------------------------
# Key-conditioned chord prior
# ---------------------------------------------------------------------------

def build_key_prior(
    tonic: int,
    mode: str,
    max_phase: int = 1,
    diatonic_boost: float = 3.0,
) -> np.ndarray:
    """
    Build log-prior over chords given the inferred key.

    Diatonic chords get a `diatonic_boost` factor; chromatic chords get 1.0.

    Diatonic chords for major key (scale degrees I–VII):
        I   → intervals [0]    → maj7
        ii  → intervals [2]    → min7
        iii → intervals [4]    → min7
        IV  → intervals [5]    → maj7
        V   → intervals [7]    → dom7
        vi  → intervals [9]    → min7
        vii → intervals [11]   → hdim7 (half-dim)

    Diatonic chords for natural minor (Aeolian):
        i   → min7, iv → min7, v → min7, bIII → maj7, bVI → maj7, bVII → dom7, ii° → hdim7

    Returns:
        log_prior: (n_chords,) float — unnormalised log-prior.
    """
    idx_to_chord, _ = build_index(max_phase)
    C = len(idx_to_chord)
    prior = np.ones(C, dtype=np.float32)

    if mode == "major":
        diatonic = {
            (tonic + 0) % 12:  [ChordQuality.MAJOR, ChordQuality.MAJ7],
            (tonic + 2) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7],
            (tonic + 4) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7],
            (tonic + 5) % 12:  [ChordQuality.MAJOR, ChordQuality.MAJ7],
            (tonic + 7) % 12:  [ChordQuality.MAJOR, ChordQuality.DOM7],
            (tonic + 9) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7],
            (tonic + 11) % 12: [ChordQuality.HALF_DIM7, ChordQuality.DIMINISHED],
        }
    else:  # minor / Aeolian
        diatonic = {
            (tonic + 0) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7, ChordQuality.MIN_MAJ7],
            (tonic + 2) % 12:  [ChordQuality.HALF_DIM7, ChordQuality.DIMINISHED],
            (tonic + 3) % 12:  [ChordQuality.MAJOR, ChordQuality.MAJ7],
            (tonic + 5) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7],
            (tonic + 7) % 12:  [ChordQuality.MINOR, ChordQuality.MIN7, ChordQuality.DOM7],
            (tonic + 8) % 12:  [ChordQuality.MAJOR, ChordQuality.MAJ7],
            (tonic + 10) % 12: [ChordQuality.MAJOR, ChordQuality.DOM7],
        }

    for chord_idx, (root, quality) in enumerate(idx_to_chord):
        if quality == ChordQuality.NO_CHORD:
            prior[chord_idx] = 0.5
            continue
        if root in diatonic and quality in diatonic[root]:
            prior[chord_idx] = diatonic_boost

    return np.log(prior + 1e-9)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def build_transition_matrix(
    tonic: int,
    max_phase: int = 1,
    style: str = "jazz_medium_swing",
    min_chord_beats: float = 1.0,
    self_transition_boost: float = 2.0,
    no_chord_self_transition_boost: float = 0.5,
) -> np.ndarray:
    """
    Build log-transition matrix A: (n_chords, n_chords).

    A[i, j] = log P(chord_t = j | chord_{t-1} = i, key, style)

    Construction:
      1. Start from root-movement weights (cycle-of-fifths, semitone, etc.)
      2. Add jazz progression priors (II-V-I, tritone sub, etc.) instantiated
         for the given tonic.
      3. Boost self-transitions to enforce minimum harmonic rhythm.
      4. Normalise rows.

    Args:
        tonic:                  pitch class of the tonic (0–11).
        max_phase:              chord vocabulary phase.
        style:                  style name from STYLE_PRIORS (sets progression weights).
        min_chord_beats:        minimum expected chord duration in beats.
                                Controls self-transition probability.
        self_transition_boost:  additional log-weight for staying on the same chord.
        no_chord_self_transition_boost: separate, smaller self-transition weight for
                                the NO_CHORD state. NO_CHORD gets the same flat 0.1
                                base rate to/from every chord as the rest of the matrix
                                (unlike real chords it has no root, so it cannot use
                                root-movement or jazz-progression weights) — using the
                                real-chord `self_transition_boost` here would make its
                                row sum, and hence its self-transition share, wildly
                                out of scale with real chords' progression-inflated
                                rows, turning NO_CHORD into a near-absorbing Viterbi
                                sink regardless of emission evidence.

    Returns:
        log_A: (C, C) float — log transition matrix, rows sum to 0 in log space
               (i.e., exp(log_A).sum(axis=1) ≈ 1.0 up to normalisation).
    """
    idx_to_chord, chord_to_idx = build_index(max_phase)
    C = len(idx_to_chord)

    # Base transition weights from root movement
    rel_transitions = build_relative_transition_matrix()
    A = np.ones((C, C), dtype=np.float32) * 0.01  # small base probability

    for i, (root_i, qual_i) in enumerate(idx_to_chord):
        if qual_i == ChordQuality.NO_CHORD:
            A[i, :] = 0.1  # flat: no root to anchor root-movement weights on
            continue
        for j, (root_j, qual_j) in enumerate(idx_to_chord):
            if qual_j == ChordQuality.NO_CHORD:
                A[i, j] = 0.1
                continue
            from_interval = (root_i - tonic) % 12
            to_interval = (root_j - tonic) % 12
            A[i, j] = rel_transitions.get((from_interval, to_interval), 0.1)

    # Add jazz progression priors
    style_prior = STYLE_PRIORS.get(style, STYLE_PRIORS["jazz_medium_swing"])
    for prog_name, weight in style_prior.progression_weights.items():
        prog = PROGRESSIONS.get(prog_name)
        if prog is None:
            continue
        chords = prog.instantiate(tonic)  # [(root, quality), ...]
        for k in range(len(chords) - 1):
            from_chord = chords[k]
            to_chord = chords[k + 1]
            if from_chord in chord_to_idx and to_chord in chord_to_idx:
                i = chord_to_idx[from_chord]
                j = chord_to_idx[to_chord]
                A[i, j] += weight * prog.weight

    # Self-transition boost (harmonic rhythm prior)
    no_chord_idx = chord_to_idx[(-1, ChordQuality.NO_CHORD)]
    for c in range(C):
        boost = no_chord_self_transition_boost if c == no_chord_idx else self_transition_boost
        A[c, c] += boost * min_chord_beats

    # Normalise rows → log
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    A /= row_sums
    return np.log(A + 1e-30).astype(np.float32)


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def viterbi(
    log_emission: np.ndarray,   # (T, C) — log P(obs_t | chord_c)
    log_transition: np.ndarray, # (C, C) — log P(chord_t | chord_{t-1})
    log_init: np.ndarray,       # (C,)   — log P(chord_0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Viterbi algorithm in log space.

    Returns:
        path:        (T,) — MAP chord sequence (integer indices).
        log_probs:   (T,) — log-probability of the MAP state at each step.
    """
    T, C = log_emission.shape
    viterbi_mat = np.full((T, C), -np.inf, dtype=np.float64)
    backpointer = np.zeros((T, C), dtype=np.int32)

    # Initialise
    viterbi_mat[0] = log_init + log_emission[0]

    # Recursion
    for t in range(1, T):
        # (C, C): scores[i, j] = viterbi[t-1, i] + log_transition[i, j]
        scores = viterbi_mat[t - 1, :, np.newaxis] + log_transition
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi_mat[t] = scores[backpointer[t], np.arange(C)] + log_emission[t]

    # Traceback
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(viterbi_mat[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    log_probs = viterbi_mat[np.arange(T), path]
    return path, log_probs


# ---------------------------------------------------------------------------
# ChordInferrer — public interface
# ---------------------------------------------------------------------------

class ChordInferrer:
    """
    Full Bayesian chord inference for a structural segment.

    Usage:
        inferrer = ChordInferrer(max_phase=1)
        events = inferrer.infer(
            beat_probs=segment.beat_probs,
            beat_times=beat_times[segment.start_beat:segment.end_beat],
            key=key_posterior,
            style="jazz_medium_swing",
        )
        for ev in events:
            print(f"  {ev.label:10s}  {ev.start_time_s:.2f}s – {ev.end_time_s:.2f}s")
    """

    def __init__(
        self,
        max_phase: int = 1,
        noise_floor: float = 0.05,
        diatonic_boost: float = 3.0,
        self_transition_boost: float = 2.0,
        no_chord_self_transition_boost: float = 0.5,
    ):
        self.max_phase = max_phase
        self.noise_floor = noise_floor
        self.diatonic_boost = diatonic_boost
        self.self_transition_boost = self_transition_boost
        self.no_chord_self_transition_boost = no_chord_self_transition_boost

        # Build emission matrix once (independent of key)
        self._emission = build_emission_matrix(max_phase, noise_floor)
        self._idx_to_chord, _ = build_index(max_phase)

        logger.info(
            f"ChordInferrer ready: {len(self._idx_to_chord)} chords, "
            f"phase {max_phase}"
        )

    def infer(
        self,
        beat_probs: np.ndarray,
        beat_times: np.ndarray,
        key: KeyPosterior,
        style: str = "jazz_medium_swing",
        min_chord_beats: float = 1.0,
        segment_end_time_s: float | None = None,
    ) -> list[ChordEvent]:
        """
        Infer chord sequence for a segment.

        Args:
            beat_probs:  (B, 88) — beat-level note probabilities.
            beat_times:  (B,) — times in seconds of each beat.
            key:         KeyPosterior from key inference.
            style:       style name for transition prior.
            min_chord_beats: minimum expected chord duration.
            segment_end_time_s: authoritative end time for the final chord
                event in this segment (typically the start of the next beat
                after the segment, read from the full-track beat grid). If
                None, it is extrapolated from the average beat spacing —
                only correct for the last segment of a track.

        Returns:
            List of ChordEvent objects (consecutive chords, no gaps).
        """
        B = beat_probs.shape[0]
        if B == 0:
            return []

        # 1. Log-emission: P(beat_obs | chord)
        # dot each beat's note_prob vector with each chord's emission vector
        # log_obs: (B, C)
        log_obs = np.log(
            beat_probs @ self._emission.T + 1e-30
        ).astype(np.float64)

        # 2. Key-conditioned prior (initial distribution)
        log_init = build_key_prior(
            key.tonic, key.mode, self.max_phase, self.diatonic_boost
        )

        # 3. Transition matrix
        log_A = build_transition_matrix(
            tonic=key.tonic,
            max_phase=self.max_phase,
            style=style,
            min_chord_beats=min_chord_beats,
            self_transition_boost=self.self_transition_boost,
            no_chord_self_transition_boost=self.no_chord_self_transition_boost,
        ).astype(np.float64)

        # 4. Viterbi
        path, log_probs = viterbi(log_obs, log_A, log_init)

        # 5. Compress runs → ChordEvent list
        return self._compress_path(path, log_obs, beat_times, segment_end_time_s)

    def _compress_path(
        self,
        path: np.ndarray,
        log_obs: np.ndarray,
        beat_times: np.ndarray,
        segment_end_time_s: float | None = None,
    ) -> list[ChordEvent]:
        """Compress consecutive identical chords into ChordEvent objects."""
        events: list[ChordEvent] = []
        if len(path) == 0:
            return events

        # Per-beat emission posterior (softmax over chords), used for a bounded
        # confidence score. The raw Viterbi score is a cumulative log-probability
        # over the whole segment — averaging and exponentiating that underflows
        # to exactly 0.0 for any run of more than a few beats, so it can't be
        # used as a per-event confidence directly.
        obs_posterior = np.exp(log_obs - log_obs.max(axis=1, keepdims=True))
        obs_posterior /= obs_posterior.sum(axis=1, keepdims=True)

        # Beat duration used to extrapolate the end time of a run that reaches
        # the last beat of the segment, where there is no "next beat" to use
        # as an end boundary.
        avg_beat_s = float(np.mean(np.diff(beat_times))) if len(beat_times) > 1 else 0.5

        start = 0
        for t in range(1, len(path) + 1):
            if t == len(path) or path[t] != path[start]:
                chord_idx = int(path[start])
                root, quality = self._idx_to_chord[chord_idx]
                # end_time_s is the start of beat `t` (the beat after this run);
                # if the run reaches the end of the segment, there is no beat
                # `t` to read, so extrapolate one average beat past the last one.
                if t < len(beat_times):
                    end_time_s = float(beat_times[t])
                elif segment_end_time_s is not None:
                    end_time_s = segment_end_time_s
                else:
                    end_time_s = float(beat_times[-1] + avg_beat_s)
                avg_conf = float(obs_posterior[start:t, chord_idx].mean())

                events.append(ChordEvent(
                    root=root,
                    quality=quality,
                    label=chord_label(root, quality),
                    start_beat=start,
                    end_beat=t,
                    start_time_s=float(beat_times[start]),
                    end_time_s=end_time_s,
                    confidence=min(avg_conf, 1.0),
                ))
                start = t

        return events
