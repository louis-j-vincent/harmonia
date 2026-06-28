"""
Key inference via Bayesian chroma matching.

Uses the Krumhansl-Schmuckler (1990) key profiles as the prior over
pitch class distributions given a key. Posterior key is inferred per
structural segment via:

    P(key | chroma) ∝ P(chroma | key) × P(key)

where P(chroma | key) is modelled as a Dirichlet-Multinomial likelihood
and P(key) is a uniform prior over 24 keys (12 major + 12 minor).

For modulation detection, we apply Bayesian changepoint detection over
the sequence of per-segment key posteriors.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler profiles
# (from Krumhansl, C.L. (1990). Cognitive Foundations of Musical Pitch.)
# ---------------------------------------------------------------------------

# Major key profile — correlation with perceived stability of each pitch class
# relative to the tonic. Index 0 = tonic, index 1 = minor 2nd, etc.
KS_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
])

# Minor key profile (natural minor / Aeolian)
KS_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
])

# Normalise to sum to 1 (treat as probability distributions)
KS_MAJOR = KS_MAJOR / KS_MAJOR.sum()
KS_MINOR = KS_MINOR / KS_MINOR.sum()

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 24 keys: 0–11 = C major through B major, 12–23 = C minor through B minor
KEY_NAMES = (
    [f"{pc} major" for pc in PITCH_CLASSES] +
    [f"{pc} minor" for pc in PITCH_CLASSES]
)

N_KEYS = 24


# ---------------------------------------------------------------------------
# Build rotated profile matrix  (24 × 12)
# ---------------------------------------------------------------------------

def _build_profile_matrix() -> np.ndarray:
    """
    Returns a (24, 12) matrix where row k is the KS profile for key k,
    rotated to start at the tonic of key k.
    """
    profiles = np.zeros((N_KEYS, 12))
    for root in range(12):
        profiles[root] = np.roll(KS_MAJOR, root)           # major keys
        profiles[root + 12] = np.roll(KS_MINOR, root)      # minor keys
    return profiles


KEY_PROFILES: np.ndarray = _build_profile_matrix()  # shape (24, 12)


# ---------------------------------------------------------------------------
# Bayesian key inference
# ---------------------------------------------------------------------------

@dataclass
class KeyPosterior:
    """Result of key inference for a single segment."""
    log_probs: np.ndarray   # shape (24,) — unnormalised log-posterior
    tonic: int              # 0–11, most probable tonic pitch class
    mode: str               # "major" or "minor"
    key_name: str           # e.g. "G major"
    confidence: float       # posterior probability of the MAP key

    @property
    def probs(self) -> np.ndarray:
        p = np.exp(self.log_probs - self.log_probs.max())
        return p / p.sum()

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (key_name, probability) pairs."""
        probs = self.probs
        idx = np.argsort(probs)[::-1][:k]
        return [(KEY_NAMES[i], float(probs[i])) for i in idx]


def infer_key(
    chroma: np.ndarray,
    key_prior: np.ndarray | None = None,
    alpha: float = 1.0,
) -> KeyPosterior:
    """
    Infer key from a chroma vector (or summed chroma over a segment).

    Args:
        chroma:     shape (12,) — pitch class energy / activation sum.
                    Can be raw counts, probabilities, or energy values.
        key_prior:  shape (24,) — log-prior over keys. Uniform if None.
        alpha:      Dirichlet smoothing concentration. Higher = more influence
                    of the KS profile prior, less of the data.

    Returns:
        KeyPosterior with full posterior distribution over 24 keys.
    """
    chroma = np.asarray(chroma, dtype=float)
    assert chroma.shape == (12,), f"Expected shape (12,), got {chroma.shape}"

    # Normalise chroma to avoid scale sensitivity
    total = chroma.sum()
    if total > 0:
        chroma_norm = chroma / total
    else:
        chroma_norm = np.ones(12) / 12  # flat chroma = no information

    # Log-likelihood: KL-divergence inspired, using dot product correlation
    # P(chroma | key k) ∝ exp(profile_k · chroma)  [cosine similarity in log space]
    # More precisely: use log of the dot product as a proxy for the
    # Dirichlet-Multinomial log-likelihood (computationally efficient approximation)
    log_likelihood = KEY_PROFILES @ chroma_norm  # shape (24,)

    # Add Dirichlet smoothing via alpha-weighted KS profile
    log_likelihood = log_likelihood * (1.0 + alpha * total / 12.0)

    # Add prior
    if key_prior is None:
        log_prior = np.zeros(N_KEYS)  # uniform
    else:
        log_prior = np.asarray(key_prior, dtype=float)
        assert log_prior.shape == (N_KEYS,)

    log_posterior = log_likelihood + log_prior

    best_idx = int(np.argmax(log_posterior))
    tonic = best_idx % 12
    mode = "major" if best_idx < 12 else "minor"

    probs = np.exp(log_posterior - log_posterior.max())
    probs /= probs.sum()

    return KeyPosterior(
        log_probs=log_posterior,
        tonic=tonic,
        mode=mode,
        key_name=KEY_NAMES[best_idx],
        confidence=float(probs[best_idx]),
    )


# ---------------------------------------------------------------------------
# Chroma extraction from probabilistic note activations
# ---------------------------------------------------------------------------

def activations_to_chroma(
    note_probs: np.ndarray,
    weight_by_octave: bool = True,
) -> np.ndarray:
    """
    Fold a (frames, 88) piano-key probability matrix into a (12,) chroma vector.

    Args:
        note_probs:      shape (frames, 88) — P(note active | audio frame).
                         Piano keys: MIDI 21 (A0) to MIDI 108 (C8).
        weight_by_octave: down-weight very low and very high octaves
                         (less harmonically informative).

    Returns:
        chroma: shape (12,) — summed and normalised pitch class activations.
    """
    n_frames, n_keys = note_probs.shape
    assert n_keys == 88, f"Expected 88 piano keys, got {n_keys}"

    midi_start = 21  # A0
    chroma = np.zeros(12)

    for key_idx in range(n_keys):
        midi = midi_start + key_idx
        pc = midi % 12
        octave = midi // 12

        # Octave weight: midrange (octaves 3–6) weighted highest
        if weight_by_octave:
            ow = np.exp(-0.15 * (octave - 4.5) ** 2)
        else:
            ow = 1.0

        chroma[pc] += note_probs[:, key_idx].sum() * ow

    # Normalise
    total = chroma.sum()
    if total > 0:
        chroma /= total

    return chroma


# ---------------------------------------------------------------------------
# Bayesian changepoint detection for modulation
# ---------------------------------------------------------------------------

def detect_modulations(
    segment_posteriors: list[KeyPosterior],
    threshold: float = 0.6,
) -> list[int]:
    """
    Detect modulation points in a sequence of per-segment key posteriors.

    A modulation is flagged when the MAP key of a segment differs from
    the previous segment AND the confidence exceeds `threshold`.

    Returns:
        List of segment indices where a modulation occurs.
    """
    modulations: list[int] = []
    if not segment_posteriors:
        return modulations

    prev_key = segment_posteriors[0].key_name

    for i, kp in enumerate(segment_posteriors[1:], start=1):
        if kp.key_name != prev_key and kp.confidence >= threshold:
            modulations.append(i)
            prev_key = kp.key_name

    return modulations
