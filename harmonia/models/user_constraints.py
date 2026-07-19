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

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Minimum spans that must survive intra-group exclusion for a merge to still
# pool. Pooling is a "superimposed observations, variance ↓ ~1/N" operation —
# with N=1 there is nothing to superimpose, so a group that collapses to a
# single surviving span is treated as unpoolable (reported, not a silent
# no-op success). See pool_beat_evidence's docstring.
MIN_POOL_SPANS = 2

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
    """Map a time span ``[t0, t1)`` to a ``(start_beat, end_beat)`` index pair.

    ``b0`` is the beat index NEAREST ``t0`` — the true START anchor, so a span
    still begins on the beat the user meant and never over/under-runs its
    neighbour at the start.

    The beat COUNT ``b1 - b0`` is the SINGLE-quantized span DURATION,
    ``round((t1 - t0) / period)`` — NOT the double-quantized
    ``argmin(t1) - argmin(t0)`` the original used.

    Root-cause (2026-07-19, ★ CHORD-ROBUSTNESS / BAR-MERGE, "Part A"):
    differencing two INDEPENDENT nearest-index lookups sums each endpoint's
    ±0.5-beat rounding error into a ±1-beat scatter (``round(a) - round(b)``
    noise ≈ 1.0), whereas quantizing the DURATION once has noise ≈ 0.5. On the
    3 real songs' failing merge clusters this recovers equal beat counts on
    7/10 groups vs the old method's 1/10 (e.g. aretha cluster B, spans of one
    static 16-bar vamp: old ``[32, 33, 32, 32, 32]`` → new
    ``[32, 32, 32, 32, 32]``), turning routine ±1 quantization noise from a
    ``pool_beat_evidence`` exclusion trigger back into a clean FULL pool. This
    leaves ``pool_beat_evidence``'s weak-link exclusion ("Part B") as the
    safety-net for GENUINE length-outliers only (e.g. aretha cluster A's real
    43-beat block, still ``[31, 32, 32, 43]`` under this fix), not routine
    drift — the two fixes are complementary, closing the loop the split
    Part A / Part B brief opened.

    ``period`` is the MEDIAN grid spacing. ``bt`` is the production constant-
    tempo uniform grid, so its interior spacings are all ~period; the median is
    robust to the two irregular endpoint gaps chord_pipeline_v1 introduces by
    prepending ``0.0`` and appending the duration.

    Consistency (the subtlety Part A flagged for whoever implemented this):
    ``pool_beat_evidence`` consumes the returned ``(b0, b1)`` in TWO places —
    the equal-length PRECONDITION (``b1 - b0``) AND the offset-alignment loop
    (``b0 + off for off in range(b1 - b0)``). Both must see the SAME count for
    the pooled beats to stay aligned. Because ``b0`` is unchanged (still the
    true start anchor) and ``b1 = b0 + count``, the loop pools exactly ``count``
    beats starting at the real span start — it does NOT shift a span into its
    neighbour's beats. ``b1`` is used ONLY for the count here; it is
    deliberately NOT clipped to ``len(bt) - 1`` (the offset loop bounds-checks
    each ``b0 + off`` against the array length independently), so a near-end
    span reports its honest duration count rather than a truncated one that
    would spuriously mismatch its equal-length siblings.
    """
    b0 = _time_to_beat(t0, bt)
    period = float(np.median(np.diff(bt))) if len(bt) >= 2 else 0.0
    if period > 1e-9:
        count = max(1, int(round((t1 - t0) / period)))
    else:
        # Degenerate grid (single point / zero spacing): fall back to the old
        # endpoint-difference count so we never divide by ~0.
        count = max(1, _time_to_beat(t1, bt) - b0)
    return b0, b0 + count


def _choose_mode_beat_count(lens: list[int]) -> int:
    """Pick the beat count to pool a mismatched merge group around.

    The MODE (most-frequent beat count) is the majority reading; the spans that
    match it are pooled and the rest excluded. Ties are broken deterministically
    toward the LARGER beat count — when two candidate lengths are equally
    supported, keeping the longer one retains more musical evidence per pooled
    beat, and (importantly) BOTH candidate subgroups are internally
    equal-length, so either choice is safe; the tiebreak only needs to be
    stable, not "correct". Returns the chosen beat count.
    """
    counts = Counter(lens)
    # max by (support, beat_value): most spans first, larger beat count as tiebreak.
    return max(counts, key=lambda L: (counts[L], L))


def pool_beat_evidence(
    merges: list[SectionMerge],
    bt: np.ndarray,
    *arrays: np.ndarray,
    rejected: list | None = None,
    pooled_report: list | None = None,
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

    Returns copies (originals untouched).

    Order-independence (2026-07-18, ★ CHORD-ROBUSTNESS / BAR-MERGE)
    --------------------------------------------------------------
    Multiple merges are INDEPENDENT (each pools its own spans' ORIGINAL
    evidence) — not a sequential pipeline where a later merge sees an earlier
    merge's already-pooled output. The original implementation read from and
    wrote into the SAME mutable array across the ``for m in merges`` loop, so
    when two merge groups shared a beat index (e.g. bar 69 in both a [53,69]
    and a [69,73] pair), the second merge silently pooled the FIRST merge's
    already-pooled value — order-dependent (reproduced on 2 real songs /
    production endpoint). Fixed by computing every merge's pooled value from an
    immutable snapshot (``orig``) of the ORIGINAL arrays; writes still land in
    ``outs``, so if two merges genuinely disagree about a shared beat the LAST
    merge in the list wins it (deterministic, matches the last-write-wins
    convention for overlapping chord-confirms — see ``build_segment_constraints``)
    but WITHOUT compounding through an intermediate pooled value.

    Graceful degradation on unequal beat counts (2026-07-19, ★ CHORD-ROBUSTNESS
    / BAR-MERGE — "one weak link should not break the whole chain")
    --------------------------------------------------------------------------
    Whether a merge group's spans all quantise to the SAME beat count is a
    fragile precondition on real audio: the beat grid ``bt`` is estimated
    independently of whatever generated the span boundaries, and drift
    accumulates over a span's length, so one span in an N-span group routinely
    lands +/-1 beat off the others. The history of this precondition:

      * v1: unequal beat counts inside ANY merge → hard ``ValueError`` for the
        WHOLE request.
      * 2026-07-18: a malformed merge is skipped and recorded (``rejected``
        out-list) while OTHER, valid merges in the same batch still pool —
        per-BATCH degradation. But a mismatch INSIDE one group still killed
        that ENTIRE group. For N-way section-cluster pooling (5–10 blocks per
        group) this meant almost every real group died to a single off-by-one
        member (aretha 0/2, abba 0/3, autumn_leaves 1/5 groups applied), and a
        caller-side per-bar-offset workaround was built instead of fixing this.
      * 2026-07-19 (this docstring): per-GROUP degradation. When a group's
        spans disagree on beat count we no longer discard the group. Instead we
        pool the MODE (majority) beat count's spans and EXCLUDE the mismatched
        span(s) — explicitly, never silently and never by force-aligning
        (truncating/padding) a mismatched span onto the mode, which would
        silently corrupt data (a span off by 8 beats is not the same music as
        one off by 1; excluding is always safe, forcing is not). The survivors
        are all exactly the mode length, so they remain genuinely equal musical
        length AMONG THEMSELVES — the property the whole beat-offset pooling
        relies on holds for what actually gets pooled.

    Subtleties, and how each is resolved:
      * No clear majority (e.g. a 2-2 split): ``_choose_mode_beat_count`` breaks
        the tie deterministically toward the larger beat count. Both tied
        subgroups are internally equal-length, so either is safe to pool; only
        stability matters.
      * Minimum viable group size: pooling one span is not pooling. If fewer
        than ``MIN_POOL_SPANS`` (=2) spans match the chosen mode, the whole
        group is UNPOOLABLE — reported via ``rejected``, not returned as a
        silent no-op success.
      * Near-miss vs far-miss: exclusion is distance-agnostic in ACTION (we
        only ever exclude, never truncate/pad), but the miss magnitude is
        reported (``expected_beats`` vs ``got_beats``) so a caller/UI can tell
        a benign +/-1 drift from a suspicious grouping.

    Reporting contract (so callers can be honest about what happened):
      * ``rejected`` (backward-compatible): appended one dict per merge that
        pooled NOTHING — ``{"spans", "beat_lens", "excluded", "reason"}``. A
        partially-pooled merge does NOT appear here (it DID apply).
      * ``pooled_report`` (new, optional): appended one dict per considered
        merge (>=2 spans) describing exactly what happened —
        ``{"status": "pooled" | "partial" | "unpoolable",
           "mode_beats": int, "pooled_spans": [{"span", "got_beats"}, ...],
           "excluded": [{"span", "expected_beats", "got_beats", "delta"}, ...]}``.

    If EVERY considered merge pooled nothing (n_applied == 0), a ``ValueError``
    is still raised — a request in which not one correction could be applied
    surfaces as a hard failure, not a silent no-op (api_reinfer relies on this).
    """
    orig = [a.copy() for a in arrays]
    outs = [a.copy() for a in arrays]
    n_considered = 0     # merges with >=2 spans (single-span merges are silently no-ops, unchanged)
    n_applied = 0
    local_rejected: list[dict] = []
    local_report: list[dict] = []

    def _pool_span_group(beat_ranges: list[tuple[int, int]], n: int) -> None:
        """Pool the given (already equal-length) beat ranges, offset by offset,
        reading originals from ``orig`` and writing into ``outs``."""
        for src, arr in zip(orig, outs):
            for off in range(n):
                idxs = [b0 + off for (b0, _b1) in beat_ranges if b0 + off < src.shape[0]]
                if len(idxs) < 2:
                    continue
                pooled = np.sum([src[i] for i in idxs], axis=0)
                # renormalise probability rows (sum≈1 originally); leave raw
                # feature rows summed (their scale is not a simplex).
                s = float(pooled.sum())
                o = float(src[idxs[0]].sum())
                if 0.5 < o < 1.5 and s > 1e-9:      # looks like a posterior row
                    pooled = pooled / s
                for i in idxs:
                    arr[i] = pooled

    for m in merges:
        if len(m.spans) < 2:
            continue
        n_considered += 1
        beat_ranges = [_span_to_beats(t0, t1, bt) for (t0, t1) in m.spans]
        lens = [b1 - b0 for (b0, b1) in beat_ranges]

        if len(set(lens)) == 1:
            # Happy path: all spans equal length — pool them all.
            n = lens[0]
            _pool_span_group(beat_ranges, n)
            n_applied += 1
            if pooled_report is not None:
                local_report.append({
                    "status": "pooled",
                    "mode_beats": n,
                    "pooled_spans": [{"span": list(sp), "got_beats": n}
                                     for sp in m.spans],
                    "excluded": [],
                })
            continue

        # Unequal beat counts — graceful per-group degradation (2026-07-19).
        # Pool the MODE beat count's spans; exclude the rest EXPLICITLY.
        mode = _choose_mode_beat_count(lens)
        keep_idx = [i for i, L in enumerate(lens) if L == mode]
        drop_idx = [i for i, L in enumerate(lens) if L != mode]
        excluded = [{"span": list(m.spans[i]), "expected_beats": mode,
                     "got_beats": lens[i], "delta": lens[i] - mode}
                    for i in drop_idx]

        if len(keep_idx) < MIN_POOL_SPANS:
            # Not enough spans agree on any single length — unpoolable.
            reason = (f"no beat count shared by >={MIN_POOL_SPANS} spans "
                      f"(beat_lens={lens})")
            local_rejected.append({"spans": list(m.spans), "beat_lens": lens,
                                   "excluded": excluded, "reason": reason})
            if pooled_report is not None:
                local_report.append({
                    "status": "unpoolable", "mode_beats": mode,
                    "pooled_spans": [], "excluded": excluded, "reason": reason})
            continue

        _pool_span_group([beat_ranges[i] for i in keep_idx], mode)
        n_applied += 1
        if pooled_report is not None:
            local_report.append({
                "status": "partial",
                "mode_beats": mode,
                "pooled_spans": [{"span": list(m.spans[i]), "got_beats": mode}
                                 for i in keep_idx],
                "excluded": excluded,
            })

    if n_considered > 0 and n_applied == 0:
        # EVERY merge in the batch pooled nothing -- preserve the original
        # hard-reject contract (test_pool_beat_evidence_rejects_unequal_beats /
        # _all_bad_still_raises) so a request where not one correction could be
        # applied surfaces as a clear failure, not a silent no-op.
        lens = local_rejected[0]["beat_lens"] if local_rejected else []
        raise ValueError(
            f"section-merge spans cover {lens} beats — must be equal "
            f"musical length (equal beat count) in v1")
    if rejected is not None:
        rejected.extend(local_rejected)
    if pooled_report is not None:
        pooled_report.extend(local_report)
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
