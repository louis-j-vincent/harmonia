"""
Song-structure periodicity: find repeated harmonic loops and use them to
reinforce the beat-level emission evidence.

Candidates A (emission normalization) and B (explicit-duration decoding)
both improved *when* the decoder places chord boundaries without improving
*what* it decides at them — the bottleneck is how discriminable the raw
per-beat evidence is, not decoder structure (see docs/known_issues.md #1).
This module is the one candidate that targets evidence quality directly:
if a song's accompaniment loops every L beats, averaging beat t with
beat t+L, t+2L, ... across all repeats should raise the signal-to-noise
ratio of "what's actually being played at this position in the loop",
since noise/passing-tones differ between repeats while the true harmony
at each slot doesn't.

Reuses `build_ssm()` from structure.py (already computed for segmentation,
so this is nearly free) rather than re-deriving similarity from scratch.
"""

from __future__ import annotations

import numpy as np

from harmonia.models.structure import build_ssm


def score_periods(
    beat_probs: np.ndarray,
    beats_per_bar: int = 4,
    max_period_bars: int = 8,
    top_k: int = 3,
) -> dict[int, float]:
    """
    Score candidate loop lengths (in beats) by how self-similar the song is
    at that lag, and return the top-k non-redundant candidates.

    Candidates are constrained to musically plausible multiples of the bar
    length (`beats_per_bar x {1, 2, 4, 8}`) rather than an unconstrained lag
    sweep — genuine harmonic loops in 4/4 pop/jazz are overwhelmingly bar
    multiples, and an unconstrained sweep would be dominated by the
    trivial/misleading lag=1 peak (adjacent beats are usually still the same
    chord — that's the over-smoothing problem this whole investigation is
    about, not song structure).

    score(L) = mean_i SSM[i, i+L] — the L-th off-diagonal of the
    self-similarity matrix, averaged. This is exactly an autocorrelation of
    beat-to-beat similarity: if beat i and beat i+L sound alike for many i
    simultaneously, L is a real periodicity, not noise.

    A period is dropped if it's an exact multiple of an already-kept,
    higher-scoring period — e.g. if L=32 wins, L=64 (its first harmonic)
    is redundant evidence of the same underlying loop, not new information.

    Returns:
        {period_in_beats: score}, at most `top_k` entries, highest score first.
    """
    ssm = build_ssm(beat_probs)
    B = ssm.shape[0]

    candidates = sorted({
        beats_per_bar * k
        for k in (1, 2, 4, 8)
        if 0 < beats_per_bar * k < B
    })
    if not candidates:
        return {}

    scores = {L: float(np.diagonal(ssm, offset=L).mean()) for L in candidates}

    kept: list[int] = []
    for L in sorted(scores, key=lambda x: -scores[x]):
        if any(L % k == 0 for k in kept):
            continue
        kept.append(L)
        if len(kept) >= top_k:
            break

    return {L: scores[L] for L in kept}


def find_loop_phase(period: int, is_downbeat: np.ndarray) -> int:
    """
    Given a period already chosen by `score_periods`, find which residue
    class `0..period-1` is the true start of a repeat.

    `score_periods` only answers "does a repeat of length L exist" — its
    score, `mean_i SSM[i, i+L]`, averages over every starting position `i`
    simultaneously, which is exactly invariant to which absolute beat gets
    called position 0 within the loop. Self-similarity alone can't break
    that symmetry either: a cleanly repeating signal is, by construction,
    just as internally coherent under any phase choice, so there's no
    signal in the SSM to prefer one residue class over another. Anything
    that needs "distance into the loop" (e.g. `beat_idx % period`) has been
    silently assuming beat 0 of the *song* is also beat 0 of the loop,
    which need not be true (a pickup beat, an intro, or any offset before
    the first full repeat breaks that assumption) — see
    docs/known_issues.md #1.

    The only thing that can actually break the symmetry is external
    information about where a real metrical unit starts: the annotated
    downbeat grid. This anchors phase 0 to the first annotated downbeat,
    so `(beat_idx - phase) % period == 0` lines up with a downbeat whenever
    `period` is a multiple of the bar length and downbeats recur regularly
    from that point on.

    Args:
        period: loop length in beats, as returned by `score_periods`.
        is_downbeat: boolean array, same length as the beat grid, True at
            annotated downbeats.

    Returns:
        The phase in `[0, period)` such that residue class 0 aligns with
        the first downbeat. Returns 0 if `period <= 0` or no downbeat is
        annotated (nothing to anchor to).
    """
    if period <= 0:
        return 0
    downbeat_idxs = np.flatnonzero(is_downbeat)
    if len(downbeat_idxs) == 0:
        return 0
    return int(downbeat_idxs[0] % period)


def fold_beat_probs(beat_probs: np.ndarray, period: int) -> np.ndarray:
    """
    Circular-average beat_probs at the given period: every beat is replaced
    by the mean of itself and all beats an exact multiple of `period` away
    (same "slot" in the loop). Same shape as the input.

    Beat index here is absolute (position in the full song's beat grid),
    not relative to any structural segment — folding must use absolute
    position so slot alignment is consistent across segment boundaries.
    """
    B = beat_probs.shape[0]
    folded = np.empty_like(beat_probs, dtype=np.float64)
    for slot in range(period):
        idx = np.arange(slot, B, period)
        folded[idx] = beat_probs[idx].mean(axis=0)
    return folded
