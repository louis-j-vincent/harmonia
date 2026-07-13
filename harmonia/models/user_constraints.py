"""User inputs as inference factors (Mission 3, handoff §8 "User inputs are
factors in the same graph").

Two constraint types, both translated into factors the existing joint decode
already understands (``harmonia/models/joint_decode.py``) — no new recursion:

  * **chord-confirm** ``{span [t0,t1), root, q5}`` — the user confirms/edits a
    chord. Becomes (a) a dominant additive emission log-bonus (~+20 nats,
    ``joint_decode.CLAMP_NATS``) on the confirmed ``(root, q5)`` cells over the
    overlapping segments — a soft-but-dominant delta prior that still lets the
    transition factor re-decode the NEIGHBOURS (propagation), and (b) a
    duration-boundary HINT at ``t0``/``t1`` so the segmentation actually carves a
    span there (so the clamp lands on a slot the user meant, and freeing a
    neighbour of the confirmed beats' evidence is itself a propagation channel).

  * **section-merge** ``{spans A, B, ...}`` of equal musical length — the user
    asserts "these parts are the same." Corresponding segments across the spans
    are TIED and their emission log-scores POOLED (summed) — P3
    parallelism-as-denoising, gated by the user's assertion (never blind: blind
    averaging was Gen-1 Candidate C and it hurt). Unequal segment counts are
    rejected with a clear error (v1).

This module is pure bookkeeping over beat/segment indices — it holds no model
state — so it is cheap to unit-test in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from harmonia.models.joint_decode import CLAMP_NATS


@dataclass
class ChordConfirm:
    """A user confirmation/edit of one chord over a time span (seconds)."""
    t0: float
    t1: float
    root: int                       # 0..11 pitch class
    q5: int | None = None           # 0..4 (maj/min/dom/hdim/dim); None = any quality
    bonus: float = CLAMP_NATS


@dataclass
class SectionMerge:
    """A user assertion that ≥2 time spans are the same section (P3 pooling)."""
    spans: list[tuple[float, float]] = field(default_factory=list)  # [(t0,t1), ...] seconds


def _time_to_beat(t: float, bt: np.ndarray) -> int:
    """Nearest beat index to time ``t`` (seconds) on the beat grid ``bt``."""
    return int(np.clip(np.abs(bt - t).argmin(), 0, len(bt) - 1))


def confirm_cut_beats(confirms: list[ChordConfirm], bt: np.ndarray) -> list[int]:
    """Beat indices where the segmentation should be FORCED to carve a boundary."""
    cuts: set[int] = set()
    for c in confirms:
        cuts.add(_time_to_beat(c.t0, bt))
        cuts.add(_time_to_beat(c.t1, bt))
    return sorted(cuts)


def force_boundaries(segs: list[tuple[int, int]], cut_beats: list[int]) -> list[tuple[int, int]]:
    """Split any segment that straddles a requested cut beat.

    Idempotent; preserves coverage/order. A cut at a beat already on a boundary
    is a no-op. This is the duration-boundary HINT half of a chord-confirm.
    """
    cuts = set(int(b) for b in cut_beats)
    out: list[tuple[int, int]] = []
    for (s, e) in segs:
        interior = sorted(b for b in cuts if s < b < e)
        prev = s
        for b in interior:
            out.append((prev, b))
            prev = b
        out.append((prev, e))
    return out


def _span_to_segments(t0: float, t1: float, segs: list[tuple[int, int]],
                      bt: np.ndarray) -> list[int]:
    """Segment indices whose beat range overlaps [t0, t1) (via the beat grid)."""
    b0 = _time_to_beat(t0, bt)
    b1 = max(b0 + 1, _time_to_beat(t1, bt))
    return [i for i, (s, e) in enumerate(segs) if s < b1 and e > b0]


def build_segment_constraints(
    confirms: list[ChordConfirm],
    segs: list[tuple[int, int]],
    bt: np.ndarray,
) -> list[dict | None]:
    """Per-segment clamp dicts (length ``len(segs)``) for ``joint_decode``.

    Every segment overlapping a confirm's span gets the confirm's (root, q5)
    clamp. When a segment is claimed by two confirms (rare after boundary
    hints), the last one wins — deterministic and adequate for v1.
    """
    out: list[dict | None] = [None] * len(segs)
    for c in confirms:
        for i in _span_to_segments(c.t0, c.t1, segs, bt):
            out[i] = {"root": int(c.root) % 12,
                      "q5": None if c.q5 is None else int(c.q5),
                      "bonus": float(c.bonus)}
    return out


def _span_to_beats(t0: float, t1: float, bt: np.ndarray) -> tuple[int, int]:
    b0 = _time_to_beat(t0, bt)
    b1 = max(b0 + 1, _time_to_beat(t1, bt))
    return b0, b1


def pool_beat_evidence(
    merges: list[SectionMerge],
    bt: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Pool per-beat evidence across the beat ranges of each merge's spans (P3).

    For every merge, its spans are mapped to beat ranges (equal MUSICAL length ⇒
    equal beat COUNT, so corresponding beats align by offset — this is why merge
    pooling lives at the beat level, robust to the fact that two acoustically-
    similar spans rarely segment into the same NUMBER of chord segments). For
    each ``arr`` in ``arrays`` (e.g. the (T,12) root posterior and the (T,88)
    onset/note pooled features), the corresponding beats are replaced by their
    SUM across the spans — the superimposed-observation likelihood ("N snippets
    of the same chord, variance ↓ ~1/N"). Posteriors are re-normalised per beat
    so downstream log() stays well-scaled; raw feature arrays are left summed.

    Returns copies (originals untouched). Raises ``ValueError`` on unequal beat
    counts (v1 rejects unequal musical lengths).
    """
    outs = [a.copy() for a in arrays]
    for m in merges:
        if len(m.spans) < 2:
            continue
        beat_ranges = [_span_to_beats(t0, t1, bt) for (t0, t1) in m.spans]
        lens = [b1 - b0 for (b0, b1) in beat_ranges]
        if len(set(lens)) != 1:
            raise ValueError(
                f"section-merge spans cover {lens} beats — must be equal "
                f"musical length (equal beat count) in v1")
        n = lens[0]
        for arr in outs:
            for off in range(n):
                idxs = [b0 + off for (b0, _b1) in beat_ranges if b0 + off < arr.shape[0]]
                if len(idxs) < 2:
                    continue
                pooled = np.sum([arr[i] for i in idxs], axis=0)
                # renormalise probability rows (sum≈1 originally); leave raw
                # feature rows summed (their scale is not a simplex).
                s = float(pooled.sum())
                orig = float(arr[idxs[0]].sum())
                if 0.5 < orig < 1.5 and s > 1e-9:      # looks like a posterior row
                    pooled = pooled / s
                for i in idxs:
                    arr[i] = pooled
    return tuple(outs)


def build_pool_groups(
    merges: list[SectionMerge],
    segs: list[tuple[int, int]],
    bt: np.ndarray,
) -> list[list[int]]:
    """Tied segment-index groups for ``joint_decode`` pooling.

    Each merge maps its spans to segment-index lists; the k-th segment of every
    span is tied into one group. Raises ``ValueError`` if the spans do not cover
    an equal number of segments (v1: unequal musical lengths are rejected).
    """
    groups: list[list[int]] = []
    for m in merges:
        if len(m.spans) < 2:
            continue
        per_span = [_span_to_segments(t0, t1, segs, bt) for (t0, t1) in m.spans]
        n = len(per_span[0])
        for j, ss in enumerate(per_span):
            if len(ss) != n:
                raise ValueError(
                    f"section-merge span {j} covers {len(ss)} segments, "
                    f"expected {n} (spans must be equal musical length in v1)")
        if n == 0:
            continue
        for k in range(n):
            group = sorted({per_span[j][k] for j in range(len(per_span))})
            if len(group) >= 2:
                groups.append(group)
    return groups
