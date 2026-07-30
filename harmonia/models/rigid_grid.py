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
    drop_before_grid: bool = False,
) -> tuple[list[dict], int]:
    """Re-bin chords onto a rigid grid given by bar-boundary TIMES.

    ``bar_bounds_sec`` is a sorted list of bar edges in seconds, length
    ``n_bars + 1`` (bar ``k`` spans ``[bounds[k], bounds[k+1])``). Each chord is
    assigned to the bar whose span contains its onset ``t0``; ``beat`` is
    recomputed from the sub-bar fraction, so two chords in one grid-bar land on
    distinct beats (a split bar) and a single chord lands on beat 0. ``t0``/``t1``
    are preserved (they are the ground-truth onsets; only the bar/beat *labels*
    change). Chords after the last edge clamp to the last bar.

    ``drop_before_grid``: drop chords whose onset precedes the first bar edge —
    the pre-grid intro pickup/anacrusis (This Love's spurious ``C`` at t=0.44 that
    otherwise clamps into bar 0 and steals the G7 downbeat). Off by default so the
    plain re-bin keeps every chord; the pipeline hook turns it on.

    Returns ``(new_chords, n_bars)``. Does not mutate the input.
    """
    bounds = sorted(float(x) for x in bar_bounds_sec)
    n_bars = max(len(bounds) - 1, 0)
    if n_bars == 0:
        return [dict(c) for c in chords], 0
    out: list[dict] = []
    for c in chords:
        t0 = float(c.get("t0", 0.0))
        if drop_before_grid and t0 < bounds[0] - 1e-6:
            continue                       # pre-grid intro pickup — drop it
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


def _duration_mass_phase(k, w, m):
    """Downbeat offset ``p`` in ``[0, m)``: the residue class of fine-slots that
    carries the most chord DURATION.  The E3 "metrical lift" of the 2026-07-29
    session, but with the metrical level ``m`` supplied rather than guessed."""
    if m <= 1:
        return 0
    mass = np.zeros(m)
    np.add.at(mass, np.asarray(k, int) % m, np.asarray(w, float))
    return int(np.argmax(mass))


def bar_len_from_downbeats(
    downbeat_times,
    beat_times=None,
    *,
    meters=(4, 3),
    meter_tol=0.12,
    max_spread=0.25,
    min_downbeats=4,
):
    """Bar length in seconds measured from a beat tracker's NATIVE downbeats —
    the external accent cue that breaks the metrical octave.  ``None`` when the
    tracker is not self-consistent enough to be believed.

    This is the load-bearing assumption of the octave fix, so it is gated rather
    than trusted (CLAUDE.md rule #1).  Two checks:

    * **steadiness** — the inter-downbeat spread (IQR / median) must be under
      ``max_spread``; a tracker that scatters its downbeats is not measuring bars.
    * **self-consistency** — ``downbeat spacing / beat spacing`` must land within
      ``meter_tol`` of a real meter (4 or 3 beats per bar).  This is the check
      that matters: on POP909 song 002 (the song CLAUDE.md flags as the known
      hard case) beat_this gets the BEATS right (63.8 vs GT 64.0 BPM) but places
      downbeats every ~2.2 beats, and its downbeat spacing is then 0.55x the true
      bar.  The ratio test catches exactly that and abstains.

    Measured 2026-07-30: with this gate, beat_this downbeat spacing equals the
    POP909 ``beat_midi.txt`` col-3 ground-truth bar length on 4/4 songs it
    accepts (5th abstained), and matches the externally-known bar length on every
    one of 25 real-audio charts cross-checked against published tempos.
    """
    d = np.diff(np.asarray(downbeat_times, dtype=float))
    if len(d) + 1 < min_downbeats or len(d) == 0:
        return None
    bar = float(np.median(d))
    if not np.isfinite(bar) or bar <= 0:
        return None
    spread = float(np.percentile(d, 75) - np.percentile(d, 25)) / bar
    if spread > max_spread:
        return None
    if beat_times is not None:
        db = np.diff(np.asarray(beat_times, dtype=float))
        if len(db) == 0:
            return None
        beat = float(np.median(db))
        if beat <= 0:
            return None
        bpb = bar / beat
        if not any(abs(bpb / m - 1.0) <= meter_tol for m in meters):
            return None
    return bar


def _snap_octave(m_bar, p_bar, g, k, w, bar_ref, tol=0.15):
    """Move the recovered bar to the metrical octave nearest the external cue.

    Returns ``(m_bar, p_bar, g_eff)`` for ``_build_bounds`` (bar = ``m_bar*g_eff``).
    Two branches:

    * **snap** (the normal one) — the bar stays an integer number of the finder's
      own least-squares fine slots (``m*g``), so the cue never imports the beat
      tracker's period error: it only says WHICH multiple of the chord-change
      grid is the musical bar.  Phase is re-derived at the new level from chord
      duration mass.
    * **adopt** — no integer multiple of ``g`` lands within ``tol`` of the cue,
      i.e. the fine grid and the cue disagree about more than the octave (the
      chord decode found a period the audio does not have).  The cue has already
      passed ``bar_len_from_downbeats``'s self-consistency gate, so it is the
      better of the two: take it wholesale as a one-slot grid.  3/25 charts.

    With ``bar_ref`` absent or zero, both branches are skipped and the finder's
    own answer is returned untouched — the kill switch is simply not passing a cue.
    """
    if not bar_ref or bar_ref <= 0 or g <= 0:
        return m_bar, p_bar, g
    m_ref = int(round(bar_ref / g))
    if m_ref >= 1 and abs(m_ref * g / bar_ref - 1.0) <= tol:
        if m_ref == m_bar:
            return m_bar, p_bar, g
        return m_ref, _duration_mass_phase(k, w, m_ref), g
    return 1, 0, float(bar_ref)


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
    bar_ref_sec: "float | None" = None,
) -> "list[float] | None":
    """Corrected bar-boundary times (seconds), or ``None`` to keep the pipeline's
    grid. Recovers period + phase from the raw chord onsets (the bad beat grid
    never moved them): fine grid → loop period (time-domain content SSM) →
    bar = structural-slot spacing (cross-loop recurrence) → octave snapped to the
    external downbeat cue → phase on the downbeat-class onsets. This Love:
    ~2.52 s bars, G7 at bar 0, one chord per bar (G7|Cm|Fm7|Ddim looping).

    ``bar_ref_sec`` — bar length in seconds measured from a beat tracker's native
    downbeats (``bar_len_from_downbeats``); ``None`` to run onsets-only.

    WHAT THE CUE FIXES (2026-07-30, Louis: "fix the 2x octave issue"). Chord
    onsets give the CHORD-CHANGE period robustly, but that equals the musical bar
    only at ~1 chord/bar: a held chord leaves no onset (the finder doubles the
    bar), two chords per bar add one (it halves it). Norah Jones' "Don't Know
    Why" changes chord twice a bar, so the chart came out 134 bars of 1.36 s
    instead of 67 of 2.72 s — every displayed bar was half a bar. The cue says
    which multiple of the chord grid is the bar; the period itself still comes
    from the finder's own least-squares fit, so the tracker's tempo error is
    never imported. Measured on 25 real-audio charts: wrong octave 15 → 3.

    KNOWN LIMITS (docs/research_sessions/bar_grid_period_phase_2026-07-29.md).
    Without a cue the octave ambiguity is unchanged — onsets alone cannot break
    it, and this function stays opt-in (HARMONIA_REGRID=1). With a cue it is
    still not fixed when (a) the beat tracker's own downbeats are not
    self-consistent, in which case ``bar_len_from_downbeats`` returns ``None``
    and nothing changes, or (b) the finder's fine grid ``g`` is not a whole
    fraction of the true bar, in which case the snap declines (3/25 charts:
    Alessi Brothers, Chain of Fools, Jackson 5 ABC — those are decode-quality
    failures upstream of the grid, not octave errors). It also does NOT correct
    the residual period DRIFT of a rigid constant-length grid over a
    tempo-varying take. Returns ``None`` on any failure or too-few chords
    (defer, no regression).
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
        m_bar, p_bar, g = _snap_octave(m_bar, p_bar, g, k, w, bar_ref_sec)
        bounds = _build_bounds(t, k, g, m_bar, p_bar, (lo, hi))
    except Exception:
        return None
    return bounds if bounds and len(bounds) >= 3 else None
