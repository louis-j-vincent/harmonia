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

Two folders live here, and the second supersedes the first:

  * `fold_beat_probs(beat_probs, period)` — one global period for the whole
    song. Cheap, needs no structure, and WRONG whenever a song has more than
    one loop: it averages verse beats into chorus beats, and it averages a
    1st ending into a 2nd ending, destroying both.
  * `fold_by_vocabulary(...)` — groups occurrences by the learned section
    vocabulary (`harmonia.models.section_vocab`), so only genuine repeats of
    the same material pool. Louis's 2026-07-30 direction.
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


def _bar_beat_index(
    beat_times: np.ndarray, bar_bounds_sec: "list[float] | np.ndarray"
) -> tuple[list[np.ndarray], int]:
    """``(beats_of_bar, n_bars)`` — which absolute beat indices each bar owns.

    Purely by TIME: bar ``b`` owns every beat whose time is in
    ``[bounds[b], bounds[b+1])``. Deliberately reads no ``beat`` field from
    anywhere. ``rigid_grid.py`` backs every bar edge off 0.15 bar and never
    compensates it in the ``beat`` label it writes, so on This Love no downbeat
    lands on beat 0 (histogram ``{1:75, 2:1, 3:43}``, docs/known_issues.md OPEN
    #3) — any bar→beat map that trusted that label would be off by one on 118 of
    119 bars. A time-based map inherits only the grid's constant phase, which
    shifts *which* beats a bar owns identically for every bar and therefore
    leaves slot alignment between occurrences intact.

    Beats before the first edge (a pickup) and after the last belong to no bar.
    """
    bounds = np.asarray(sorted(float(x) for x in bar_bounds_sec), dtype=np.float64)
    n_bars = max(len(bounds) - 1, 0)
    if n_bars == 0:
        return [], 0
    bt = np.asarray(beat_times, dtype=np.float64)
    edge = np.searchsorted(bt, bounds, side="left")
    return [np.arange(edge[b], edge[b + 1]) for b in range(n_bars)], n_bars


def fold_by_vocabulary(
    beat_probs: np.ndarray,
    beat_times: np.ndarray,
    sections: "list[dict] | None",
    bar_bounds_sec: "list[float] | np.ndarray",
) -> np.ndarray:
    """Average the OBSERVATIONS across every occurrence of each vocabulary item.

    Same shape/dtype contract as :func:`fold_beat_probs`, and the same idea —
    raise the signal-to-noise of "what is actually being played at this position
    in the loop" by averaging repeats — but grouped by the song's *learned
    section vocabulary* (:func:`harmonia.models.section_vocab.vocab_sections`)
    instead of by one global period.

    Why this replaces the fixed-period folder (Louis, 2026-07-30): a single
    period assumes the whole song is one loop, so it averages verse beats into
    chorus beats, and it averages a 1st ending into a 2nd ending — destroying
    both. The vocabulary knows This Love's chorus tail ``C`` (``Cm F△7 | Ab G7``)
    and its outro tail ``E`` (``Cm7 F | Ab Ab``) are different items, so their
    observations never mix, while the verse ``A``'s eight occurrences all pool.

    The decision this is FOR: with N observations of the same slot, a 7th that is
    weakly present in every pass survives averaging (√N SNR) where a vote on
    N decoded *labels* can only ever report the majority label and has already
    thrown the 7th away.

    Args:
        beat_probs: ``(B, K)`` per-beat note activations. Not mutated.
        beat_times: ``(B,)`` beat times in seconds, ascending. ABSOLUTE
            position in the full track's beat grid (same discipline as
            ``fold_beat_probs``: fold on absolute position, then slice per
            segment, never the other way round).
        sections: contiguous sections as returned by ``vocab_sections`` —
            ``{"label", "bar0", "bar1" (EXCLUSIVE), "d_bars", ...}``. Every
            section sharing a ``label`` is an occurrence of one item, and
            ``d_bars`` is that item's own loop length. ``None``/empty → no-op.
        bar_bounds_sec: bar edges in seconds, length ``n_bars + 1``, as returned
            by ``harmonia.models.rigid_grid.rigid_grid_for``.

    Returns:
        ``(B, K)`` float64. Beats belonging to no complete occurrence of any
        item — a pickup before the grid, a section shorter than its own item, a
        one-off bridge — are returned untouched.

    Alignment, and the ragged-count decision
    ---------------------------------------
    A slot is keyed by ``(item label, bar index WITHIN the item, ordinal beat
    within that bar)``; every beat carrying the same key is averaged. Bars come
    from ``bar_bounds_sec`` by time (see :func:`_bar_beat_index`).

    Occurrences of one item routinely disagree on beat count — the bar grid is
    rigid but beat *detection* is not, so one bar in one pass picks up a 5th
    beat. The three candidate policies and why this one:

    * **truncate to the minimum / zip** — REJECTED. One extra beat early in an
      occurrence shifts every later beat of that occurrence by one slot, so the
      whole pass gets averaged against the wrong musical positions while the
      output still looks plausibly smoothed. This is exactly CLAUDE.md error
      pattern #1, and it is why the keying is per-BAR: a wobble is contained
      inside the one bar where it happened.
    * **resample onto a common slot count** — REJECTED. It interpolates across
      chord changes, inventing evidence at the boundary the decoder most needs
      to be sharp at.
    * **skip the whole occurrence** — REJECTED as too lossy: on real audio a
      ±1-beat disagreement somewhere is the common case, not the exception
      (``user_constraints.pool_beat_evidence`` measured almost every real merge
      group losing to a single off-by-one before it was made per-group), so
      skipping would routinely discard the majority of the evidence.
    * **ADOPTED: per-bar ordinal keying, ragged tail folded only across the
      occurrences that actually supply it.** A slot supplied by one occurrence
      folds to itself (mean of one = identity) — never zipped against a
      different musical position. Exact when the passes agree, locally
      degrading when they don't, and it never averages two beats that the bar
      grid says sit at different positions in the loop.

    Every value is read from an immutable snapshot of ``beat_probs``, so
    overlapping/duplicated sections can never compound through an already-folded
    value — the same order-independence rule ``pool_beat_evidence`` had to learn.

    What this does NOT solve
    ------------------------
    * It inherits the bar grid and the vocabulary wholesale. A wrong metrical
      octave (docs/known_issues.md OPEN #1, ~⅓ of the corpus) produces a
      confidently wrong grouping, and folding then averages genuinely different
      music. There is no internal consistency check here — the caller must gate
      on grid/vocabulary confidence.
    * It does not weight occurrences. A badly-mixed or masked pass counts as
      much as a clean one; a reliability weight per occurrence is the obvious
      next lever and is not implemented.
    * It is a MEAN, not a sum. ``pool_beat_evidence`` sums (and renormalises
      posteriors); a mean keeps the folded array on the same scale as the raw
      one, which is what every ``folded_views`` consumer assumes when it scores
      both with the same emission model.
    """
    folded = np.array(beat_probs, dtype=np.float64, copy=True)
    if folded.size == 0 or not sections:
        return folded

    beats_of_bar, n_bars = _bar_beat_index(beat_times, bar_bounds_sec)
    if n_bars == 0:
        return folded

    # (label, bar-within-item, ordinal-beat-within-bar) → absolute beat indices
    slots: dict[tuple[str, int, int], list[int]] = {}
    for sec in sections:
        try:
            b0, b1 = int(sec["bar0"]), int(sec["bar1"])
            d = int(sec.get("d_bars") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if d < 1:
            continue
        b0, b1 = max(b0, 0), min(b1, n_bars)
        label = str(sec.get("label", ""))
        # Only COMPLETE occurrences fold. A section shorter than its own item
        # (vocab_sections reports reps = max(1, ...), so a 1-bar remainder of a
        # 4-bar item still says reps=1) has none, and its bars pass through
        # rather than being folded against a truncated pattern.
        for r in range((b1 - b0) // d):
            for k in range(d):
                for j, i in enumerate(beats_of_bar[b0 + r * d + k]):
                    slots.setdefault((label, k, j), []).append(int(i))

    for idxs in slots.values():
        if len(idxs) > 1:
            folded[idxs] = beat_probs[idxs].mean(axis=0)
    return folded


# ---------------------------------------------------------------------------
# folded_views assembly: fold once on ABSOLUTE beats, then slice per segment
# ---------------------------------------------------------------------------
#
# Extracted 2026-07-30 from the deleted `pipeline.py::HarmoniaPipeline.run`,
# which was the only place this two-phase discipline existed. It is the
# load-bearing part of feeding `folded_views` to a per-segment decoder, so it
# outlived the pipeline it was written for.

def build_period_folds(
    beat_probs: np.ndarray,
    beats_per_bar: int = 4,
    max_period_bars: int = 8,
    top_k: int = 3,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Fold the WHOLE track at each of its top-k candidate periods.

    Phase 1 of 2. Run once per track, then call :func:`slice_folded_views` per
    structural segment — never the other way round (see below).

    Why the whole track and not each segment: a structural segment on its own
    is usually too short to see an 8-bar loop at all (an 8-bar loop in 4/4 is
    32 beats; segments routinely come in at 8-16), so scoring periodicity
    inside a segment mostly measures noise.

    Args:
        beat_probs: ``(B, K)`` per-beat activations for the FULL track. Not
            mutated.
        beats_per_bar: metre, passed through to :func:`score_periods`.
        max_period_bars: longest candidate loop, in bars.
        top_k: how many non-redundant periods to keep.

    Returns:
        ``(folded_full, period_weights)``.
        ``folded_full[L]`` is ``beat_probs`` folded at period ``L``, same shape
        as the input, indexed by ABSOLUTE beat.
        ``period_weights[L]`` is ``L``'s score normalised to sum to 1 across
        the kept periods, so a track with one clean loop concentrates its
        weight and an ambiguous one spreads it.
        Both are empty dicts when no period is scoreable (track shorter than
        one bar), which :func:`slice_folded_views` turns into ``None``.

    What this does NOT solve
    ------------------------
    * Period LENGTH only, never PHASE. Nothing here knows which residue class
      starts a repeat; :func:`find_loop_phase` answers that separately and this
      function does not call it. Folding is phase-invariant (every slot averages
      with every other beat an exact multiple of ``L`` away regardless of where
      the loop "begins"), so phase does not corrupt the fold — but any consumer
      that wants to *name* a slot needs the phase and will not get it here.
    * One global period per track. When a song has more than one loop this
      averages verse beats into chorus beats and a 1st ending into a 2nd
      ending. :func:`fold_by_vocabulary` is the fix for that and supersedes
      this pair; keep these for the no-vocabulary/no-bar-grid case.
    """
    periods = score_periods(
        beat_probs,
        beats_per_bar=beats_per_bar,
        max_period_bars=max_period_bars,
        top_k=top_k,
    )
    if not periods:
        return {}, {}
    total = sum(periods.values()) or 1.0
    period_weights = {L: s / total for L, s in periods.items()}
    folded_full = {L: fold_beat_probs(beat_probs, L) for L in periods}
    return folded_full, period_weights


def slice_folded_views(
    folded_full: dict[int, np.ndarray],
    period_weights: dict[int, float],
    start_beat: int,
    end_beat: int,
) -> "list[tuple[np.ndarray, float]] | None":
    """Cut one structural segment's view out of each whole-track fold.

    Phase 2 of 2. Returns exactly the ``folded_views`` argument that
    :meth:`harmonia.models.chord_hmm.ChordInferrer.infer` takes: a list of
    ``(folded_slice, weight)`` pairs, or ``None`` when there is nothing to add
    (so the caller can pass the result straight through).

    THE CORRECTNESS POINT — fold, THEN slice, using the segment's ABSOLUTE
    beat range. A fold assigns each beat to slot ``position mod period``. If a
    segment were folded on its own, its first beat would become slot 0, so the
    same musical position would land in a different slot in every segment and
    the extra evidence would be averaged against the wrong beats — while still
    looking like a plausibly smoothed array (CLAUDE.md error pattern #1).
    Slicing an already-folded whole-track array with the same absolute
    ``[start_beat, end_beat)`` used for the segment's raw ``beat_probs`` keeps
    slot alignment identical across every segment boundary.

    Args:
        folded_full: ``{period: folded (B, K)}`` from :func:`build_period_folds`.
        period_weights: ``{period: weight}`` from the same call.
        start_beat: segment's first beat, ABSOLUTE index into the full grid.
        end_beat: segment's end beat, ABSOLUTE, EXCLUSIVE. Clamped to the
            folded length, so an over-long final segment is safe.

    Returns:
        ``[(folded_slice, weight), ...]`` aligned 1:1 with the segment's own
        ``beat_probs`` slice, or ``None`` if ``folded_full`` is empty.

    What this does NOT solve
    ------------------------
    It cannot verify the caller sliced ``beat_probs`` with the same range — the
    alignment guarantee is only as good as the caller passing the same
    ``start_beat``/``end_beat`` it used for the raw evidence. There is no
    cross-check here.
    """
    if not folded_full:
        return None
    views = []
    for L, folded in folded_full.items():
        hi = min(int(end_beat), folded.shape[0])
        views.append((folded[int(start_beat):hi], period_weights[L]))
    return views or None
