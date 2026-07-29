"""Rigid-grid regridding — the receiving end for the per-section
repetition-period grid corrector (2026-07-29).

Problem (docs/known_issues.md ★ STRUCTURE 2026-07-29 "This Love"): the pipeline's
beat/bar grid is sometimes misaligned — on This Love it glued G7+Cm into one bar
(a ~5 s bar, then irregular bars), thrown off by a spurious short pickup, so the
verse loop G7|Cm|Fm7|Ddim (one chord per 2.52 s bar, anchored on the G7 at
t=1.08) came out phase-shifted. The chords are right; the BAR GRID is wrong, and
everything per-bar (the SSM, the section detector, the displayed chart) inherits
it.

Fix (Louis's plan): we can't stop the beat tracker producing a bad grid, but the
SSM/chord-onsets reveal the true repetition period + phase PER SECTION — a RIGID
grid we can re-quantise onto. This module is the RECEIVING end:

  * ``rigid_grid_for(chords, ...)`` — STUB. Returns the corrected bar-boundary
    TIMES (seconds), or ``None`` to keep the pipeline's own grid. The per-section
    period/phase algorithm (in development, scratchpad prototype) drops in here.
  * ``apply_rigid_grid(chords, bar_bounds_sec)`` — mechanical re-binning: assign
    each chord to the corrected bar its onset ``t0`` falls in, recompute ``beat``
    from the sub-bar position. Implemented + validated here.

INTEGRATION HOOK (one edit, to be added when the finder lands) — in
``harmonia.output.chart_model.to_chart_model``, right AFTER ``n_bars`` is read
and BEFORE the ``chords → bars`` loop (~L145):

    from harmonia.models.rigid_grid import rigid_grid_for, apply_rigid_grid
    if os.environ.get("HARMONIA_REGRID") == "1":
        _grid = rigid_grid_for(payload.get("chords", []),
                               tonic_pc=(payload.get("home") or {}).get("tonic", 0))
        if _grid is not None:
            _chords, n_bars = apply_rigid_grid(payload.get("chords", []), _grid)
            payload = {**payload, "chords": _chords, "nBars": n_bars}

Everything downstream (the ``bars`` array, ``_section_runs``, ssm_block_sections)
then runs on the corrected grid unchanged. Default OFF (opt-in flag) until the
finder is validated corpus-wide — CLAUDE.md rule #5.
"""
from __future__ import annotations

import bisect

__all__ = ["apply_rigid_grid", "rigid_grid_for"]


def apply_rigid_grid(
    chords: list[dict],
    bar_bounds_sec: list[float],
    beats_per_bar: int = 4,
) -> tuple[list[dict], int]:
    """Re-bin chords onto a rigid grid given by bar-boundary TIMES.

    ``bar_bounds_sec`` is a sorted list of bar edges in seconds, length
    ``n_bars + 1`` (bar ``k`` spans ``[bounds[k], bounds[k+1])``). Each chord is
    assigned to the bar whose span contains its onset ``t0``; ``beat`` is
    recomputed from the sub-bar fraction, so two chords in one grid-bar land on
    distinct beats (a split bar) and a single chord lands on beat 0. ``t0``/``t1``
    are preserved (they are the ground-truth onsets; only the bar/beat *labels*
    change). Chords before the first / after the last edge clamp to bar 0 / last.

    Returns ``(new_chords, n_bars)``. Does not mutate the input.
    """
    bounds = sorted(float(x) for x in bar_bounds_sec)
    n_bars = max(len(bounds) - 1, 0)
    if n_bars == 0:
        return [dict(c) for c in chords], 0
    out: list[dict] = []
    for c in chords:
        t0 = float(c.get("t0", 0.0))
        k = bisect.bisect_right(bounds, t0) - 1
        k = max(0, min(k, n_bars - 1))
        span = bounds[k + 1] - bounds[k]
        frac = (t0 - bounds[k]) / span if span > 1e-9 else 0.0
        beat = max(0, min(beats_per_bar - 1, int(round(frac * beats_per_bar))))
        nc = dict(c)
        nc["bar"] = k
        nc["beat"] = beat
        out.append(nc)
    return out, n_bars


def rigid_grid_for(
    chords: list[dict],
    *,
    tonic_pc: int = 0,
    sections: "list[dict] | None" = None,
    beats_per_bar: int = 4,
) -> "list[float] | None":
    """Corrected bar-boundary times (seconds), or ``None`` to keep the pipeline's
    grid. STUB — the per-section repetition-period + phase algorithm plugs in
    here (SSM/chord-onset periodicity → loop length + downbeat anchor → a rigid
    grid). See this module's header for the contract and the integration hook.

    Contract the finder must satisfy:
      * input: ``chords`` (each with ``t0``, ``t1``, ``root``, quality) already
        time-ordered; ``tonic_pc`` for the key; optional detected ``sections``.
      * output: a sorted list of bar edges in seconds of length ``n_bars + 1``
        that puts each chord ONSET on (or very near) a bar edge for a periodic
        section — e.g. This Love: ~2.52 s bars anchored so G7 (t≈1.08) is a
        downbeat, giving one chord per bar (G7|Cm|Fm7|Ddim looping). Return
        ``None`` when no confident periodic grid is found (defer, no regression).
    """
    return None
