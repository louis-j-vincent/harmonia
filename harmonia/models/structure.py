"""
Structural segmentation: identify section boundaries in audio.

Approach:
    1. Compute per-beat chroma (from beat-quantised note activations)
    2. Build self-similarity matrix (SSM) — how similar is beat i to beat j?
    3. Compute novelty curve: how much does the music change at each beat?
       (diagonal Gaussian kernel applied to SSM)
    4. Detect peaks in novelty curve → segment boundaries
    5. Optionally: Bayesian changepoint detection for finer-grained inference

Output: list of Segment objects, each with start/end beat indices and
aggregated chroma for Bayesian key inference.

Why segment before inferring chords?
    Key inference is most accurate over longer stretches (8–32 bars).
    Chord inference within a segment is conditioned on the inferred key,
    so getting segment boundaries right dramatically improves chord accuracy
    at modulation points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """
    A structural section of the piece.

    start_beat:   inclusive beat index.
    end_beat:     exclusive beat index.
    start_time_s: start time in seconds.
    end_time_s:   end time in seconds.
    chroma:       (12,) summed RAW (unnormalised) chroma over the segment —
                  input to key inference.
    beat_probs:   (n_beats_in_segment, 88) note probabilities — input to chord HMM.
    """
    start_beat: int
    end_beat: int
    start_time_s: float
    end_time_s: float
    chroma: np.ndarray          # (12,)
    beat_probs: np.ndarray      # (n_beats, 88)

    @property
    def n_beats(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


# ---------------------------------------------------------------------------
# Self-similarity matrix
# ---------------------------------------------------------------------------

def _beat_chroma(beat_probs: np.ndarray, norm: str = "l2") -> np.ndarray:
    """
    Fold (B, 88) beat note probabilities into (B, 12) beat chroma, each beat
    normalised independently so no single beat's magnitude dominates.

    norm="l2": unit-norm per beat -- for cosine-similarity use (SSM).
    norm="l1": each beat's chroma sums to 1 -- for key-inference chroma
        aggregation (see _make_segment), where summing L1-normalised rows
        makes every beat count as exactly one unit of evidence. Raw
        activation-probability magnitude is NOT a meaningful independent-
        trial count (many pitch classes co-sound within one beat, inflating
        it arbitrarily) -- see docs/known_issues.md #0.
    """
    n_beats = beat_probs.shape[0]
    chroma = np.zeros((n_beats, 12), dtype=np.float32)
    midi_start = 21  # A0
    for key_idx in range(88):
        pc = (midi_start + key_idx) % 12
        chroma[:, pc] += beat_probs[:, key_idx]
    if norm == "l2":
        denom = np.linalg.norm(chroma, axis=1, keepdims=True)
    elif norm == "l1":
        denom = chroma.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown norm: {norm!r}")
    denom = np.where(denom > 0, denom, 1.0)
    return chroma / denom


def build_ssm(beat_probs: np.ndarray) -> np.ndarray:
    """
    Build cosine self-similarity matrix from beat-level note probabilities.

    Returns:
        ssm: (B, B) float32 — entry (i, j) ∈ [0, 1] = cosine similarity
             between beat i and beat j chroma vectors.
    """
    chroma = _beat_chroma(beat_probs)  # (B, 12)
    # Cosine similarity: chroma is already L2-normalised
    ssm = chroma @ chroma.T           # (B, B)
    ssm = np.clip(ssm, 0.0, 1.0)
    return ssm.astype(np.float32)


# ---------------------------------------------------------------------------
# Novelty curve
# ---------------------------------------------------------------------------

def compute_novelty(ssm: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    """
    Compute a novelty curve from the SSM using a checkerboard kernel.

    The checkerboard kernel detects transitions: high novelty at beat t means
    the music before t is dissimilar from the music after t — i.e., a boundary.

    Args:
        ssm:         (B, B) self-similarity matrix.
        kernel_size: half-width of the checkerboard kernel in beats.
                     Larger = detects coarser structure (sections).
                     Smaller = detects finer structure (phrases).

    Returns:
        novelty: (B,) float array. Peaks indicate likely segment boundaries.
    """
    B = ssm.shape[0]
    k = kernel_size

    # Build checkerboard kernel: +1 in top-left and bottom-right, -1 elsewhere
    kernel = np.ones((2 * k, 2 * k), dtype=np.float32)
    kernel[:k, k:] = -1
    kernel[k:, :k] = -1

    # Apply Gaussian taper to reduce edge effects
    gy, gx = np.ogrid[-k:k, -k:k]
    gauss = np.exp(-(gx**2 + gy**2) / (2 * (k / 2) ** 2))
    kernel *= gauss

    novelty = np.zeros(B, dtype=np.float32)
    for t in range(k, B - k):
        block = ssm[t - k:t + k, t - k:t + k]
        if block.shape == kernel.shape:
            novelty[t] = float(np.sum(kernel * block))

    # Smooth and normalise
    novelty = gaussian_filter1d(novelty, sigma=2.0)
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())

    return novelty


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def detect_boundaries(
    novelty: np.ndarray,
    min_segment_beats: int = 8,
    prominence: float = 0.2,
) -> np.ndarray:
    """
    Detect segment boundary positions from the novelty curve.

    Args:
        novelty:            (B,) novelty curve.
        min_segment_beats:  minimum number of beats per segment.
                            Prevents over-segmentation.
        prominence:         minimum peak prominence (0–1).
                            Higher = fewer, more confident boundaries.

    Returns:
        boundaries: sorted array of beat indices where boundaries occur.
                    Always includes 0 and B (start and end).
    """
    peaks, properties = find_peaks(
        novelty,
        distance=min_segment_beats,
        prominence=prominence,
    )
    # Always include start and end
    n_beats = len(novelty)
    boundaries = np.unique(np.concatenate([[0], peaks, [n_beats]]))
    return boundaries.astype(int)


# ---------------------------------------------------------------------------
# Bayesian changepoint refinement (optional, post-peak-picking)
# ---------------------------------------------------------------------------

def _bayesian_changepoint_score(
    chroma_seq: np.ndarray,
    t: int,
    window: int = 4,
) -> float:
    """
    Local Bayesian evidence for a changepoint at beat t.

    Compares the log-likelihood of the data under two models:
        H1: same distribution before and after t
        H2: different distributions before and after t

    Uses Gaussian model with conjugate Normal-Wishart prior (simplified).

    Returns:
        Score > 0 means evidence for a changepoint at t.
    """
    lo = max(0, t - window)
    hi = min(len(chroma_seq), t + window)

    before = chroma_seq[lo:t]
    after = chroma_seq[t:hi]

    if len(before) < 2 or len(after) < 2:
        return 0.0

    def log_evidence(X: np.ndarray) -> float:
        # Simplified: negative mean squared deviation from mean (proxy for log-likelihood)
        mu = X.mean(axis=0)
        return -float(((X - mu) ** 2).sum())

    joint = log_evidence(chroma_seq[lo:hi])
    separate = log_evidence(before) + log_evidence(after)
    return separate - joint   # positive = changepoint improves fit


def refine_boundaries_bayesian(
    boundaries: np.ndarray,
    beat_probs: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Optionally merge or split boundaries using local Bayesian evidence.

    Currently: prunes boundaries with low changepoint score (merges segments
    that don't show enough harmonic contrast).

    Args:
        boundaries:  initial boundary beat indices from peak-picking.
        beat_probs:  (B, 88) beat-level note probabilities.
        threshold:   minimum changepoint score to keep a boundary.

    Returns:
        Refined boundary array.
    """
    chroma_seq = _beat_chroma(beat_probs)  # (B, 12)
    refined = [boundaries[0]]

    for b in boundaries[1:-1]:
        score = _bayesian_changepoint_score(chroma_seq, b)
        if score >= threshold:
            refined.append(b)
        else:
            logger.debug(f"Pruned weak boundary at beat {b} (score={score:.3f})")

    refined.append(boundaries[-1])
    return np.array(refined, dtype=int)


# ---------------------------------------------------------------------------
# Segmenter — public interface
# ---------------------------------------------------------------------------

class Segmenter:
    """
    Detect structural boundaries and build Segment objects.

    Usage:
        seg = Segmenter()
        segments = seg.segment(beat_probs, beat_times)
        for s in segments:
            print(f"  [{s.start_time_s:.1f}s – {s.end_time_s:.1f}s] "
                  f"{s.n_beats} beats")
    """

    def __init__(
        self,
        kernel_size: int = 8,
        min_segment_beats: int = 8,
        peak_prominence: float = 0.2,
        use_bayesian_refinement: bool = True,
        bayesian_threshold: float = 0.05,
    ):
        self.kernel_size = kernel_size
        self.min_segment_beats = min_segment_beats
        self.peak_prominence = peak_prominence
        self.use_bayesian_refinement = use_bayesian_refinement
        self.bayesian_threshold = bayesian_threshold

    def segment(
        self,
        beat_probs: np.ndarray,
        beat_times: np.ndarray,
    ) -> list[Segment]:
        """
        Segment a piece into structural sections.

        Args:
            beat_probs:  (B, 88) — beat-level note probabilities from BeatGrid.quantise_frames
            beat_times:  (B,) — time in seconds of each beat

        Returns:
            List of Segment objects in chronological order.
        """
        B = len(beat_times)
        if B < self.min_segment_beats * 2:
            # Too short to segment — return as single segment
            logger.warning(f"Only {B} beats — returning as single segment")
            return [self._make_segment(beat_probs, beat_times, 0, B)]

        logger.info(f"Segmenting {B} beats...")

        # 1. Self-similarity matrix
        ssm = build_ssm(beat_probs)

        # 2. Novelty curve
        novelty = compute_novelty(ssm, kernel_size=self.kernel_size)

        # 3. Peak-picking
        boundaries = detect_boundaries(
            novelty,
            min_segment_beats=self.min_segment_beats,
            prominence=self.peak_prominence,
        )

        # 4. Bayesian refinement
        if self.use_bayesian_refinement and len(boundaries) > 2:
            boundaries = refine_boundaries_bayesian(
                boundaries, beat_probs, threshold=self.bayesian_threshold
            )

        # 5. Build Segment objects
        segments = []
        for i in range(len(boundaries) - 1):
            start_b = int(boundaries[i])
            end_b = int(boundaries[i + 1])
            seg = self._make_segment(beat_probs, beat_times, start_b, end_b)
            segments.append(seg)

        logger.info(f"Found {len(segments)} segments: "
                    f"{[s.n_beats for s in segments]} beats each")
        return segments

    def _make_segment(
        self,
        beat_probs: np.ndarray,
        beat_times: np.ndarray,
        start_b: int,
        end_b: int,
    ) -> Segment:
        seg_probs = beat_probs[start_b:end_b]   # (n, 88)

        # Aggregate chroma for key inference: each beat is L1-normalised
        # (one unit of evidence) then summed, so the segment's raw total
        # scales with n_beats -- a meaningful, uninflated evidence count --
        # rather than with the raw activation-probability magnitude (which
        # double-counts co-sounding notes and isn't a real trial count).
        # Deliberately NOT normalised further after summing: infer_key()
        # needs this magnitude to calibrate posterior confidence (see
        # docs/known_issues.md #0).
        chroma = _beat_chroma(seg_probs, norm="l1").sum(axis=0)

        end_b_clamped = min(end_b, len(beat_times) - 1)
        return Segment(
            start_beat=start_b,
            end_beat=end_b,
            start_time_s=float(beat_times[start_b]),
            end_time_s=float(beat_times[end_b_clamped]),
            chroma=chroma,
            beat_probs=seg_probs,
        )
