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

import numpy as np

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


# ── Bar-grid finder (period + phase from raw chord onsets) ─────────────────────
# Ported from the validated prototype scratchpad/bar_grid_v2.py (2026-07-29,
# subagent). The live beat tracker MIS-LABELS bars but never moves onsets, so the
# grid is recovered from chord onset SECONDS, grid-free. See
# docs/research_sessions/bar_grid_period_phase_2026-07-29.md for the derivation.


def _fourier_comb(t, w, Ls):
    W = max(float(np.sum(w)), 1e-9)
    mags, phases = {}, {}
    for L in Ls:
        z = np.sum(w * np.exp(2j * np.pi * t / L))
        mags[L] = float(np.abs(z) / W)
        phases[L] = float((np.angle(z) / (2 * np.pi) * L) % L)
    return mags, phases


def _fine_grid(t, w, lo=0.5, hi=4.0, step=0.01, frac=0.9):
    """Finest steady step the chord changes tap out = LARGEST comb wavelength
    within ``frac`` of the max magnitude (subharmonics sit below it)."""
    Ls = np.arange(lo, hi + step, step)
    mags, phases = _fourier_comb(t, w, Ls)
    mx = max(mags.values())
    g = max(L for L in Ls if mags[L] >= frac * mx)
    phi = phases[min(phases, key=lambda L: abs(L - g))]
    return float(g), float(phi)


def _refine_grid(t, g0, phi0, w, iters=12):
    g, phi = g0, phi0
    for _ in range(iters):
        k = np.round((t - phi) / g)
        A = np.vstack([np.ones_like(k), k]).T
        WA = A * w[:, None]
        coef = np.linalg.lstsq(A.T @ WA, A.T @ (w * t), rcond=None)[0]
        phi, g = float(coef[0]), float(coef[1])
    k = np.round((t - phi) / g).astype(int)
    return g, phi, k


def _loop_period_seconds(rows, key_pc, dt=0.05, lo=3.0, hi=30.0):
    """Loop length in seconds via a TIME-domain content-SSM autocorrelation
    (grid-free, NOT the pipeline's broken per-bar SSM): smallest lag whose
    self-similarity is a near-max local peak."""
    end = max(r[1] for r in rows)
    n = int(np.ceil(end / dt))
    quals = sorted({r[3] for r in rows})
    qi = {q: i for i, q in enumerate(quals)}
    F = np.zeros((n, 12 + max(len(quals), 1)), np.float32)
    for (t0, t1, root, q) in rows:
        a, b = int(t0 / dt), int(t1 / dt)
        F[a:b, (root - key_pc) % 12] = 1.0
        F[a:b, 12 + qi[q]] = 1.0
    F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)
    max_lag = int(hi / dt)
    A = np.full(max_lag, np.nan)
    for lag in range(int(lo / dt), max_lag):
        if lag >= n:
            break
        A[lag] = float(np.mean(np.sum(F[:-lag] * F[lag:], axis=1)))
    valid = ~np.isnan(A)
    if valid.sum() < 3:
        return None
    amax = np.nanmax(A)
    peaks = [i for i in range(1, max_lag - 1)
             if valid[i] and valid[i - 1] and valid[i + 1]
             and A[i] >= A[i - 1] and A[i] >= A[i + 1] and A[i] >= 0.88 * amax]
    if not peaks:
        peaks = [int(np.nanargmax(A))]
    return float((np.arange(max_lag) * dt)[min(peaks)])


def _largest_gap_cut(vals):
    v = sorted(vals, reverse=True)
    if len(v) < 2:
        return v[0] * 0.5 if v else 0.0
    _, ci = max((v[i] - v[i + 1], i) for i in range(len(v) - 1))
    return (v[ci] + v[ci + 1]) / 2.0


def _structural_spacing_bar(k, P, g, span, min_loops=4):
    """Bar = ``m*g`` where ``m`` is the spacing of the STRUCTURAL fine-slots: fold
    onsets into the loop, keep the slots that fire in ~every loop (a true downbeat)
    vs the occasional passing chord, and take their modal circular spacing. Fixes
    the bar-vs-half-bar octave ambiguity the modal onset interval falls into."""
    k = np.asarray(k, int)
    n_fine = int(round(P / g))
    n_loops_est = int(round((span[1] - span[0]) / P)) if P > 0 else 0
    if n_fine < 2 or n_loops_est < min_loops:
        return 1, 0
    wpos = k % n_fine
    loopix = k // n_fine
    loopix = loopix - loopix.min()
    n_loops = int(loopix.max() + 1)
    fired = {s: set() for s in range(n_fine)}
    for wp, li in zip(wpos, loopix):
        fired[int(wp)].add(int(li))
    frac = np.array([len(fired[s]) / max(n_loops, 1) for s in range(n_fine)])
    populated = [frac[s] for s in range(n_fine) if frac[s] > 0]
    cut = max(_largest_gap_cut(populated), 0.55 * float(frac.max()))
    struct = [s for s in range(n_fine) if frac[s] >= cut]
    if not struct:
        return 1, 0
    ss = sorted(struct)
    gaps = [ss[i + 1] - ss[i] for i in range(len(ss) - 1)] + [ss[0] + n_fine - ss[-1]]
    gaps = [gp for gp in gaps if gp >= 1]
    vals, counts = np.unique(gaps, return_counts=True)
    m_bar = int(vals[int(np.argmax(counts))])
    if m_bar < 1 or m_bar > n_fine // 2 + 1:
        m_bar = 1
    p_bar = int(np.bincount([s % m_bar for s in struct]).argmax()) if m_bar > 1 else 0
    return m_bar, p_bar


def _build_bounds(t, k, g, m_bar, p_bar, span, backoff=0.15):
    """Rigid bar edges anchored on the ACTUAL downbeat-class onsets (circular
    mean mod the bar), backed off a fraction of a bar so a slightly-early decoded
    downbeat still lands inside its own bar."""
    bar = m_bar * g
    lo, hi = span
    t = np.asarray(t, float)
    k = np.asarray(k, int)
    db_mask = (k % m_bar) == p_bar
    db_t = t[db_mask] if db_mask.any() else t
    ang = np.angle(np.mean(np.exp(2j * np.pi * db_t / bar)))
    anchor = float((ang / (2 * np.pi) * bar) % bar)
    first = float(t.min())
    while anchor > first + 0.5 * bar:
        anchor -= bar
    while anchor < first - 0.5 * bar:
        anchor += bar
    eps = backoff * bar
    klo = int(np.floor((lo - anchor) / bar)) - 1
    khi = int(np.ceil((hi - anchor) / bar)) + 1
    edges = [anchor + k2 * bar - eps for k2 in range(klo, khi + 1)]
    return sorted(e for e in edges if lo - bar <= e <= hi + bar)


def rigid_grid_for(
    chords: list[dict],
    *,
    tonic_pc: int = 0,
    sections: "list[dict] | None" = None,
    beats_per_bar: int = 4,
) -> "list[float] | None":
    """Corrected bar-boundary times (seconds), or ``None`` to keep the pipeline's
    grid. Recovers period + phase from the raw chord onsets (the bad beat grid
    never moved them): fine grid → loop period (time-domain content SSM) →
    bar = structural-slot spacing (cross-loop recurrence, beats the half-bar
    octave trap) → phase on the downbeat-class onsets. This Love: ~2.52 s bars,
    G7 at bar 0, one chord per bar (G7|Cm|Fm7|Ddim looping).

    KNOWN LIMIT (docs/research_sessions/bar_grid_period_phase_2026-07-29.md): the
    bar is octave-ambiguous from onsets alone when harmonic rhythm ≠ 1 chord/bar
    (held chords double it; 2 chords/bar halve it). Opt-in (HARMONIA_REGRID=1)
    until an accent/tempo cue breaks the octave corpus-wide. Returns ``None`` on
    any failure or too-few chords (defer, no regression).
    """
    rows: list[tuple[float, float, int, str]] = []
    for c in chords:
        r = c.get("root")
        if r is None or int(r) < 0 or c.get("nc"):
            continue
        q = ((c.get("lv") or {}).get("exact") or {}).get("q", c.get("q", ""))
        rows.append((float(c.get("t0", 0.0)), float(c.get("t1", 0.0)), int(r), q))
    if len(rows) < 8:
        return None
    try:
        t = np.array([r[0] for r in rows])
        w = np.array([r[1] - r[0] for r in rows])
        lo, hi = float(t.min()), float(max(r[1] for r in rows))
        g0, phi0 = _fine_grid(t, w)
        g, _phi, k = _refine_grid(t, g0, phi0, w)
        period = _loop_period_seconds(rows, int(tonic_pc) % 12)
        if period is None:
            period = g * 8
        m_bar, p_bar = _structural_spacing_bar(k, period, g, (lo, hi))
        bounds = _build_bounds(t, k, g, m_bar, p_bar, (lo, hi))
    except Exception:
        return None
    return bounds if bounds and len(bounds) >= 3 else None
