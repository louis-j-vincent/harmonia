"""repetition_prior.py — DEFAULT-OFF brick: a per-song HARMONIC-RHYTHM prior built
from the song's own repeated form.

Louis, 2026-07-27: *"unitairement c'est difficile, mais c'est la logique de la
repetition qui nous aide beaucoup et qu'il faut qu'on exploite au maximum."*
Do for the chord-CHANGE RHYTHM what the project already does for sections: learn
where changes fall in the first bars, then expect them at the same place in the
bars that follow — the pattern repeats within a section and across choruses **even
when the chords themselves differ**, so it is evidence exactly where per-instant
acoustics are weak.

WHAT WAS MEASURED (`docs/research_sessions/repetition_prior_2026-07-27.md`)
--------------------------------------------------------------------------
Premise screen (symbolic, no audio):

* 7 frozen songs — conditioned on bar position (so the metrical prior cannot claim
  the credit), the change indicator at the song's own form lag carries **0.205
  bits/slot**, a **62 % cut in conditional entropy**; odds lift 70x at lag 16 bars.
  Base rate to compare against: **87.1 % of GT changes fall on a downbeat**.
* POP909, N=880 (non-circular: independent per-song annotations, real downbeat GT)
  — lift ~5x, **6–10 % of the entropy**, best pooled lag 8 bars.  Per-song best lag
  is spread over 1–32 bars, so the period must be detected **per song**.

Three failure modes were diagnosed and fixed while building this:

1. **Self-confirmation.**  A profile estimated from the decode it then modifies
   cannot move that decode (measured: exactly 0.000 change on every metric, 2 of 7
   songs affected at all).  The profile must come from a *more sensitive* change
   detector than the one being corrected — a deliberately over-segmented decode
   (penalty 2) is the best source measured (AUC 0.714 vs 0.672 for the
   shipped-penalty decode, on GT change positions).
2. **Phase slip.**  Indexing the phase on a uniform-in-time grid loses the form
   over long takes (blue_bossa, 493 s: AUC 0.48 → **0.73** when the phase index is
   the REAL detected beat number instead).
3. **Leave-one-cycle-out.**  Beat *i*'s prior must be estimated from the OTHER
   cycles, or it just echoes *i*.

HONEST RESULT — READ BEFORE WIRING THIS
---------------------------------------
As a **detector of the chord changes we currently miss**, this works: at a matched
budget of 100 inserted cuts it recovers **32** missed GT changes vs **17** for the
same profile computed over the bar (metre only) and **15** for "split the longest
slices at their downbeats" — roughly **2x**.

As a **purity maximiser it does not beat the trivial metrical heuristic.**  Purity
(the fraction of playing time inside a >=80 %-pure slice) is monotone in cut count —
cutting blindly at every beat scores 0.870, *above* the GT-change oracle 0.863 — so
everything must be compared at a matched cut budget, and there:

* post-hoc insertion into the shipped chart: form-period profile beats a metre-only
  profile from the same source by **+1.0 … +4.5 pp**, but "downbeat x slice length"
  beats both once the budget exceeds ~150 cuts;
* inside the `musx_redecode` Viterbi as a graded transition cost: repetition
  0.6268 nameable @ 675 segments vs metre 0.6451 @ 701 — **identical efficiency
  (0.130 vs 0.124 pp per added segment) and no gain when composed** (metre+rep
  0.6450 @ 700 vs metre 0.6451 @ 701, i.e. **Δ = −0.0001 at matched budget**).

So: use it as a change-candidate scorer, not as a segmentation objective.  It is
default-OFF and unwired.

Also measured (Louis's "quand on n'est pas sur, ca change vers quelque chose de tres
proche"): changes to a chord sharing **3** tones are missed **64 %** of the time vs
**47 %** for 1 shared tone (odds ratio 1.28 near-vs-distant), and **63.5 % of merge
time is between NEAR chords** (>=2 shared tones).  This prior recovers a slightly
*more* near-biased slice of the misses than the metrical control (70.5 % vs 68.1 %
of a pool that is 68.0 % near) — i.e. it does fire on the ambiguous cases, weakly.

WHAT THIS DOES **NOT** SOLVE (CLAUDE.md #4)
-------------------------------------------
* It does not help songs without a repeating form: `close_to_you` (through-composed)
  and `georgia_on_my_mind` (rubato, 2 changes per bar) fail the gate outright, and
  `blue_bossa`'s own detected period is worthless (gain −0.001) even though its GT
  has strong 16-bar structure — its beat grid drifts.
* It does not place boundaries in *continuous time*; everything is beat-quantised,
  and only 18 % of GT changes lie within 30 ms of one of our beats.
* It says nothing about chord IDENTITY — segmentation only.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

#: Minimum cross-validated bits/beat that the form-period model must gain over the
#: metrical (P=4) model before the prior is used at all.  Below this the song has
#: no usable repeating harmonic rhythm and the prior degrades to a no-op.
DEFAULT_GAIN_GATE = 0.02

#: Candidate form periods, in beats.  Multiples of 4 from 2 bars to 32 bars.
DEFAULT_PERIODS = tuple(range(8, 132, 4))


def enabled() -> bool:
    """Default OFF.  Set ``HARMONIA_REPETITION_PRIOR=1`` to opt in (nothing reads
    this yet — the brick is unwired; the flag exists so a future wiring is a
    one-liner with a kill switch, matching `musx_redecode` / `seventh_upgrade`)."""
    return os.environ.get("HARMONIA_REPETITION_PRIOR", "0") == "1"


# ---------------------------------------------------------------------------
# change vector
# ---------------------------------------------------------------------------

def change_vector(change_times, beat_times, tol_beats: float = 0.5) -> np.ndarray:
    """Binary change indicator on the beat grid.

    A change further than ``tol_beats`` from every beat is DROPPED, not snapped —
    snapping a change that belongs between beats would invent structure.
    """
    bt = np.asarray(beat_times, float)
    v = np.zeros(len(bt), np.int8)
    ct = np.asarray(list(change_times), float)
    if len(bt) < 2 or len(ct) == 0:
        return v
    dt = float(np.median(np.diff(bt)))
    idx = np.abs(bt[None, :] - ct[:, None]).argmin(axis=1)
    err = np.abs(bt[idx] - ct) / max(dt, 1e-9)
    v[idx[err <= tol_beats]] = 1
    return v


# ---------------------------------------------------------------------------
# period detection (GT-free, cross-validated against the metrical model)
# ---------------------------------------------------------------------------

def _bern_xent(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).sum())


def _profile(c, idx, K, alpha: float = 1.0) -> np.ndarray:
    base = float(np.mean(c)) if len(c) else 0.0
    f = np.full(K, base)
    for k in range(K):
        m = idx == k
        if m.any():
            f[k] = (float(c[m].sum()) + alpha * base) / (int(m.sum()) + alpha)
    return f


def cv_bits(c, K: int, folds: int = 2) -> float:
    """Cross-validated bits/beat of the "position mod K" model of ``c``.

    Cycles are held out whole, so a long period cannot win by memorising.
    """
    c = np.asarray(c, float)
    n = len(c)
    idx = np.arange(n) % K
    cyc = np.arange(n) // K
    tot, cnt = 0.0, 0
    for f in range(folds):
        te = (cyc % folds) == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        p = _profile(c[tr], idx[tr], K)[idx[te]]
        tot += _bern_xent(p, c[te])
        cnt += int(te.sum())
    return tot / max(cnt, 1)


def detect_period(c, periods=DEFAULT_PERIODS, beats_per_bar: int = 4,
                  folds: int = 2) -> tuple[int, float]:
    """-> (best form period in beats, bits/beat gained over the metrical model).

    The comparison point is deliberately the METRICAL model (position within the
    bar), not the unconditional rate: "changes happen on downbeats" is the base
    rate (87.1 % of GT changes on the 7 frozen songs) and must not be sold as
    repetition.  A non-positive gain means no usable form structure.
    """
    c = np.asarray(c, float)
    n = len(c)
    if n < 3 * beats_per_bar or c.sum() < 2:
        return beats_per_bar, 0.0
    h_metre = cv_bits(c, beats_per_bar, folds)
    cands = [int(P) for P in periods if beats_per_bar < P <= max(n // 3, 0)]
    if not cands:
        return beats_per_bar, 0.0
    hs = {P: cv_bits(c, P, folds) for P in cands}
    P = min(hs, key=hs.get)
    return P, float(h_metre - hs[P])


# ---------------------------------------------------------------------------
# the prior itself
# ---------------------------------------------------------------------------

def loco_profile(c, period: int, alpha: float = 1.0) -> np.ndarray:
    """Leave-one-cycle-out P(change | phase), evaluated at every beat.

    Beat *i* is excluded from its own estimate — without this the prior echoes the
    very decode it is meant to correct (measured: a self-sourced prior moves the
    metric by exactly 0.000).
    """
    c = np.asarray(c, float)
    n = len(c)
    idx = np.arange(n) % int(period)
    tot = np.array([c[idx == q].sum() for q in range(int(period))])
    cnt = np.array([float((idx == q).sum()) for q in range(int(period))])
    base = float(c.mean()) if n else 0.0
    out = np.empty(n)
    for i in range(n):
        q = idx[i]
        t, k = tot[q] - c[i], cnt[q] - 1.0
        out[i] = (t + alpha * base) / (k + alpha) if k > 0 else base
    return out


def beat_change_prior(change_times, beat_times, *,
                      phase_beat_times=None,
                      gain_gate: float = DEFAULT_GAIN_GATE,
                      periods=DEFAULT_PERIODS,
                      beats_per_bar: int = 4) -> dict:
    """The brick's entry point.  GT-free.

    ``change_times``       — boundaries from a SENSITIVE change detector (a
                             deliberately over-segmented decode works best; the
                             decode you intend to correct does NOT — see module
                             docstring, failure mode 1).
    ``beat_times``         — the grid the returned prior is indexed on.
    ``phase_beat_times``   — REAL (non-uniform) detected beats used for the phase
                             index; defaults to ``beat_times``.  Supply the real
                             beats when ``beat_times`` is a uniform decode grid:
                             a uniform grid slips phase over long takes
                             (measured 0.48 → 0.73 AUC on a 493 s track).

    Returns ``{"prior": (n_beat,) float, "period": int, "gain": float,
    "gated_on": bool}``.  When the gate rejects the song the prior is the flat
    base rate and ``gated_on`` is False — callers must treat that as a no-op.
    """
    bt = np.asarray(beat_times, float)
    pbt = np.asarray(phase_beat_times if phase_beat_times is not None
                     else beat_times, float)
    c = change_vector(change_times, pbt)
    P, gain = detect_period(c, periods, beats_per_bar)
    base = float(c.mean()) if len(c) else 0.0
    if gain <= gain_gate:
        logger.info("repetition_prior: gate REJECTED (period %d, gain %+.3f "
                    "bits/beat <= %.3f) — prior is a no-op", P, gain, gain_gate)
        return dict(prior=np.full(len(bt), base), period=int(P),
                    gain=float(gain), gated_on=False)
    f = loco_profile(c, P)
    if pbt is not bt and len(pbt) and not np.array_equal(pbt, bt):
        j = np.abs(pbt[None, :] - bt[:, None]).argmin(axis=1)
        f = f[j]
    logger.info("repetition_prior: period %d beats (%.1f bars), gain %+.3f "
                "bits/beat over metre", P, P / beats_per_bar, gain)
    return dict(prior=np.asarray(f, float), period=int(P), gain=float(gain),
                gated_on=True)


def rank_candidates(prior, beat_times, existing_cuts, slice_len=None):
    """Score every beat that is NOT already a boundary, best first.

    ``slice_len`` (duration of the predicted slice each beat falls in) is used as a
    multiplier when supplied: it is what makes the ranking target MERGES, which is
    where the playing time is lost (18 % swallowed vs 23 % straddled on the shipped
    chart).  Returns indices into ``beat_times``, highest score first.
    """
    s = np.asarray(prior, float).copy()
    if slice_len is not None:
        s = s * np.asarray(slice_len, float)
    s[np.asarray(existing_cuts, bool)] = -np.inf
    order = np.argsort(-s, kind="mergesort")
    return order[np.isfinite(s[order]) & (s[order] > 0)]
