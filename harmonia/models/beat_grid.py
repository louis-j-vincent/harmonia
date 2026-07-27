"""Time-grid anchor logic: beat period refinement + downbeat-phase detection.

Consolidates the "where is beat 1, and how long is a beat" functions that used
to live scattered through chord_pipeline_v1.py (2026-07-19 through 2026-07-21
sessions). Each function here is a pure, independently-testable estimator over
already-extracted beat/chroma data — none of them touch audio I/O or the main
decode loop, so they can be unit-tested without the rest of the pipeline.

Composition, not a new algorithm: chord_pipeline_v1.py still owns the actual
anchor-selection CHAIN (SOTA downbeat_anchor -> flux -> structure, with the
tie-break/fallback order and kill-switches) since that logic is entangled with
the main inference function's local state (audio array, tempo, beat_backend
choice). This module holds the reusable estimators that chain calls into.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# TCS projection matrix (Harte & Sandler 2006) — same geometry as
# chord_pipeline_v1._tcs12 but applied here to (n, 12) chroma vectors for the
# optional hcdf novelty source in flux_downbeat_phase.
_r12 = np.arange(12, dtype=np.float32)
_TCS12 = np.stack([
    np.sin(_r12 * 7 * np.pi / 6), np.cos(_r12 * 7 * np.pi / 6),
    np.sin(_r12 * 3 * np.pi / 2), np.cos(_r12 * 3 * np.pi / 2),
    np.sin(_r12 * 2 * np.pi / 3), np.cos(_r12 * 2 * np.pi / 3),
], axis=0).astype(np.float32)  # (6, 12)
del _r12


def bestfit_beat_period(beat_times: np.ndarray, period_init: float) -> float:
    """Whole-song least-squares constant beat period.

    librosa's global tempo scalar is the *local median* beat spacing; it is
    locally accurate but carries a 0.5-2.3% systematic error vs the whole-song
    average, which accumulates to multi-bar grid drift (known_issues "BAR-GRID
    vs REAL-MUSIC DRIFT", 2026-07-19). This refits the constant period as the
    LSQ slope of detected-beat time vs beat index.

    Robust to occasional missed/doubled beats: each beat's integer index is
    assigned by cumulatively rounding the gap to the previous beat in units of
    ``period_init`` (a missed beat advances the index by 2, not 1).

    Does NOT solve: rubato (slow tempo drift within the song, <1 bar residual
    on the 2026-07-19 6-song sample), the 2x tempo octave-lock (a wrong-octave
    ``period_init`` yields a wrong-octave fit), or bar *phase*.
    """
    t = np.asarray(beat_times, dtype=float)
    if len(t) < 8 or period_init <= 0:
        return period_init
    steps = np.maximum(1, np.round(np.diff(t) / period_init).astype(int))
    idx = np.concatenate([[0], np.cumsum(steps)]).astype(float)
    slope = float(np.polyfit(idx, t, 1)[0])
    # Guard: this corrects small scalar errors, it is not a tempo re-estimator.
    # A fit outside ±10% of the tracker's period means something is wrong
    # (octave confusion, chaotic tracking) — keep the tracker's value.
    if not (0.9 * period_init <= slope <= 1.1 * period_init):
        return period_init
    return slope


def chroma_flux(arr: np.ndarray, times: np.ndarray) -> "tuple[np.ndarray, float]":
    """1-D harmonic-change novelty d(t) = ||delta treble-chroma||_2 (user's
    chroma-flux method, 2026-07-19). Returns ``(d, fps)``. Peaks at chord
    changes; because chords change on the bar grid, d(t) carries a comb
    periodicity locked to the TRUE grid — and it is derived from harmonic
    CONTENT, so it is reproducible across re-downloads (verified: two fresh
    yt-dlp pulls -> corr 1.000, phase 0ms), unlike ``librosa.beat_track``'s
    beat times (the round-2 grid-instability cause).

    Novelty source is selectable via ``HARMONIA_FLUX_NOVELTY`` (default
    ``raw`` = this exact treble-L2, byte-identical to before). ``hcdf``
    projects the treble chroma to Harte & Sandler 2006 tonal-centroid space
    (Gaussian sigma=4) before differencing — +52% rel change-F1@150ms vs raw
    on the matched set, but it changes the folded downbeat phase on ~half the
    songs so it is NOT the default. ``deepchroma`` (needs an audio path,
    handled in ``flux_downbeat_phase``) is the Korzeniowski & Widmer 2016 deep
    chroma (3.6x change-F1) — see the change-timing session log. See rule #6:
    swapping the novelty changes the folded phase, so the default stays
    ``raw``; the others are opt-in for a future change-time consumer.
    """
    fps = 1.0 / float(times[1] - times[0]) if len(times) > 1 else 1.0
    if os.environ.get("HARMONIA_FLUX_NOVELTY", "raw") == "hcdf":
        treb = arr[:, 12:24]
        s = treb.sum(1, keepdims=True)
        p = treb / np.where(s > 1e-9, s, 1.0)
        tcs = p @ _TCS12.T
        try:
            from scipy.ndimage import gaussian_filter1d
            tcs = gaussian_filter1d(tcs, 4.0, axis=0)
        except Exception:  # noqa: BLE001 — smoothing is optional
            pass
        d = np.sqrt((np.diff(tcs, axis=0) ** 2).sum(1))
        return np.concatenate([[0.0], d]).astype(np.float64), fps
    treb = arr[:, 12:24]
    d = np.sqrt((np.diff(treb, axis=0) ** 2).sum(1))
    d = np.concatenate([[0.0], d]).astype(np.float64)
    return d, fps


def flux_downbeat_phase(
    arr: np.ndarray, times: np.ndarray, bar_period: float, beats_per_bar: int = 4,
    audio_path: "Path | None" = None,
) -> "tuple[int, float]":
    """Structure-anchored downbeat phase (beats) from the chroma-flux comb.

    Fold d(t) modulo the bar period and take the phase where chord changes
    CLUSTER (the folded-curve peak) — that is the bar boundary / downbeat.
    Rounded to the nearest whole beat (the renderer's grid is beat-quantised)
    -> ``phi in [0, beats_per_bar)``. Returns ``(phi,
    folded_peak_over_mean_ratio)`` (ratio is the comb strength; ~1 = no comb).
    Reproducible across downloads because d(t) is content-derived.

    ``HARMONIA_FLUX_NOVELTY=deepchroma`` (opt-in, needs ``audio_path``): use
    the madmom DeepChroma novelty (3.6x change-F1) — sharper comb, and it
    agreed with the raw phase on 5/6 matched songs (measured 2026-07-20), so
    it is a more confident SAME grid, not a different one. Falls back to the
    raw/hcdf ``arr`` novelty on any failure (missing audio, broken madmom) so
    it can never break the analyse path.
    """
    novelty = os.environ.get("HARMONIA_FLUX_NOVELTY", "raw")
    d = fps = None
    if novelty == "deepchroma" and audio_path is not None:
        try:
            from harmonia.models._madmom_compat import deepchroma_novelty
            d, fps = deepchroma_novelty(Path(audio_path))
        except Exception as exc:  # noqa: BLE001 — never break analyse over an opt-in
            logger.warning("flux novelty=deepchroma failed (%s) — raw fallback", exc)
            d = fps = None
    if d is None:
        d, fps = chroma_flux(arr, times)
    beat = bar_period / beats_per_bar
    P = int(round(bar_period * fps))
    if P < 2 or len(d) < 2 * P or beat <= 0:
        return 0, 1.0
    n = (len(d) // P) * P
    folded = d[:n].reshape(-1, P).mean(0)
    phase_s = int(np.argmax(folded)) / fps
    phi = int(round(phase_s / beat)) % beats_per_bar
    ratio = float(folded.max() / (folded.mean() + 1e-9))
    return phi, ratio


def structure_anchor_phase(
    beat_proba: np.ndarray, beats_per_bar: int = 4, max_bars: int = 24,
    tonic_pc: int | None = None, loop_bars: int = 2,
) -> "tuple[int, list[float]]":
    """Structure-crispness-maximising downbeat phase (user's method, 2026-07-19).

    "Bien commencer la grille au bon debut": the DECIDER of where bar 1 starts
    is harmonic, not rhythmic — pick the downbeat phase ``phi in [0, bpb)``
    (which beat begins a bar) under which the SONG'S BEGINNING has the
    crispest loop structure. For each candidate phi we pool the first
    ``max_bars`` bars (``bpb`` beats each, starting at beat phi) and score
    crispness = mean top-1 posterior mass (bars aligned to single chords are
    PEAKED, bars that straddle two chords are smeared) PLUS the lag-recurrence
    contrast (lag-p minus lag-1, a clean loop repeats every p bars but changes
    every bar). Returns ``(best_phi, per_phi_scores)``. Grid-robust: the phase
    is an intrinsic property of the chord content, reproducible across
    downloads, unlike the raw beat-tracker sub-beat phase.
    """
    nb = len(beat_proba)
    scores: list[float] = []
    for phi in range(beats_per_bar):
        rows = []
        for b in range(max_bars):
            j0 = phi + b * beats_per_bar
            if j0 + beats_per_bar > nb:
                break
            v = beat_proba[j0:j0 + beats_per_bar].mean(0)
            s = v.sum()
            rows.append(v / s if s > 1e-9 else v)
        if len(rows) < 8:
            scores.append(-1.0)
            continue
        bars = np.array(rows)
        peak = float(bars.max(1).mean())
        fn = bars / np.clip(np.linalg.norm(bars, axis=1, keepdims=True), 1e-9, None)
        ssm = fn @ fn.T
        # loop-period contrast at the smallest strong even lag (2 = a 2-bar loop)
        lag2 = float(np.diagonal(ssm, offset=2).mean()) if bars.shape[0] > 2 else 0.0
        lag1 = float(np.diagonal(ssm, offset=1).mean()) if bars.shape[0] > 1 else 0.0
        score = peak + (lag2 - lag1)
        # Tonic-at-loop-start term: two phases can be equally crisp but differ
        # by a within-loop rotation (E|F# vs F#|E); prefer the one that places
        # the TONIC chord on the loop's FIRST bar (bar index == 0 mod
        # loop_bars) — the musically standard downbeat and the alignment that
        # keeps the tonic loop's discriminative chord bar-aligned (fixes the
        # A/B split seen when a merely-crisp-but-rotated phase is chosen on
        # the live grid).
        if tonic_pc is not None and bars.shape[0] >= 2 * loop_bars:
            t = tonic_pc % 12
            starts = bars[0::loop_bars, t].mean()
            others = bars[[i for i in range(bars.shape[0]) if i % loop_bars != 0], t].mean()
            score += 0.5 * float(starts - others)
        scores.append(score)
    best_phi = int(np.argmax(scores)) if scores else 0
    return best_phi, scores


# --------------------------------------------------------------------------- #
# Native per-bar-downbeat bar grid (Phase 2 sub-steps 1+2, 2026-07-21)
# --------------------------------------------------------------------------- #
# WHY (docs/known_issues.md PHASE 2 STEP 3/5): the live grid collapses Beat
# This!'s native per-bar downbeats to ONE circular-mean bar phase over a rigid
# uniform grid. That collapse is lossy — native downbeats score ~0.751 vs the
# shipped single-phase grid ~0.292 (best-single-phase ceiling only 0.534). Two
# songs (002, 008) whose native downbeats are near-perfect collapse to dbF=0.000
# because the circular-mean-mod-bar_period drifts to the WRONG beat. These two
# functions build the bar grid FROM the native downbeats instead, behind the
# HARMONIA_NATIVE_BARGRID kill-switch (default OFF — the live default is
# byte-unchanged until the switch is flipped).
#
# Does NOT solve (CLAUDE.md #4): the 345/341-class "native-can't-help" songs
# (native downbeats themselves wrong — needs the deferred multi-source sub-step
# 3); variable METER (bar-to-bar N changes — deferred sub-step 4; this is fixed
# 4/4); the bidirectional tempo octave-lock (orthogonal, unfixable at inference,
# STEP 4). It also does NOT change the RENDER's visual bar layout, which still
# redraws a uniform grid at a single phase (grid_anchor_beats) — the variable
# bar WIDTHS feed chroma pooling / sections here, and the corrected bar-1 anchor
# ships to the renderer; full variable-width DISPLAY is a follow-up render tweak.

NATIVE_BARGRID_ENV = "HARMONIA_NATIVE_BARGRID"
# Inter-downbeat REGULARITY gate for trusting native downbeats as the grid,
# matching downbeat_anchor.sota_downbeat_phase's min_confidence (0.85). Above it
# = PRIMARY (native downbeats ARE the bars); below = FALLBACK (best-supported
# single phase). "Regular" is the fraction of inter-downbeat gaps within 15% of
# the median (downbeat_anchor._regularity), NOT absolute correctness — a
# consistently-WRONG native track (341/345) still reads regular and is the
# documented native-can't-help remainder above.
NATIVE_BARGRID_MIN_CONF = 0.85


def native_bargrid_enabled() -> bool:
    """True iff the HARMONIA_NATIVE_BARGRID kill-switch is ON (default OFF)."""
    return os.environ.get(NATIVE_BARGRID_ENV, "off").strip().lower() in (
        "1", "on", "true", "yes")


def best_supported_phase(
    downbeats: np.ndarray, bts: np.ndarray, beats_per_bar: int = 4,
    tol_s: float = 0.07,
) -> int:
    """Real-beat-grid phase ``p in [0, beats_per_bar)`` whose subsample
    ``bts[p::bpb]`` best matches the (Beat This!) native downbeats.

    A GT-free, drift-free replacement for the lossy circular-mean phase: instead
    of averaging ``downbeats % bar_period`` (which smears when the native
    spacing != detected bar_period), pick the phase where the MOST native
    downbeats land within ``tol_s`` of a subsampled beat. Recovers the
    0.292->0.534 intra-phase headroom the circular mean throws away. Score ties
    keep the lower phase index (deterministic)."""
    downbeats = np.asarray(downbeats, dtype=float)
    bts = np.asarray(bts, dtype=float)
    if len(downbeats) == 0 or len(bts) < beats_per_bar:
        return 0
    best_p, best_score = 0, -1.0
    for p in range(beats_per_bar):
        grid = bts[p::beats_per_bar]
        if len(grid) == 0:
            continue
        score = float(np.mean(
            [np.min(np.abs(grid - t)) <= tol_s for t in downbeats]))
        if score > best_score:
            best_score, best_p = score, p
    return best_p


def native_bar_grid(
    downbeats: np.ndarray, conf: float, bts: np.ndarray, period: float,
    beats_per_bar: int = 4, min_conf: float = NATIVE_BARGRID_MIN_CONF,
    flux_phi: "int | None" = None,
) -> "tuple[np.ndarray | None, int | None, str]":
    """Resolve the bar-grid boundaries from native downbeats. Shared by the
    inference path (chroma pooling / sections) and the eval harness (scoring),
    so the number the gate measures is the grid the pipeline actually emits.

    Returns ``(bar_downbeats, anchor_beats, mode)``:
      * mode ``"native"`` (native regular, conf>=min_conf, >=5 downbeats):
        ``bar_downbeats`` = the native downbeats themselves (variable width at
        fixed 4/4); ``anchor_beats`` = beat-grid index of the first downbeat mod
        bpb (for the renderer's bar1_offset).
      * mode ``"phase"`` (native present but irregular): ``bar_downbeats`` =
        ``bts[p::bpb]`` at ``p = best_supported_phase`` (best-supported single
        phase); ``anchor_beats`` = p.
      * mode ``"flux"`` (native absent / too few, rare on POP909):
        ``(None, flux_phi, "flux")`` — the caller keeps its EXISTING
        circular-mean/flux/structure chain unchanged (no override).
    """
    downbeats = np.asarray(downbeats, dtype=float)
    bts = np.asarray(bts, dtype=float)
    if len(downbeats) < 5 or len(bts) < beats_per_bar + 1 or period <= 0:
        return None, flux_phi, "flux"

    def _anchor_of(t0: float) -> int:
        return int(np.argmin(np.abs(bts - t0))) % beats_per_bar

    if conf >= min_conf:
        # PRIMARY: the native per-bar downbeats ARE the bar boundaries.
        return downbeats, _anchor_of(float(downbeats[0])), "native"
    # FALLBACK: best-supported single phase over the real beat grid.
    p = best_supported_phase(downbeats, bts, beats_per_bar)
    return bts[p::beats_per_bar], p, "phase"


# --------------------------------------------------------------------------- #
# REAL-BEAT GRID brick (default-OFF, 2026-07-27) — "BOTH CLOCKS ARE SYNTHETIC"
# --------------------------------------------------------------------------- #
# WHY (docs/known_issues.md top entry, 2026-07-27).  The shipped pipeline
# detects real beats with Beat This! and then THROWS THEM AWAY: it collapses
# them to a single constant ``period`` (:func:`bestfit_beat_period`) plus a
# single circular-mean ``phase`` and rebuilds a metronomic lattice
# ``bt = np.arange(phase, duration_s + period, period)``
# (chord_pipeline_v1.py:3907-3921).  Every chord boundary the model emits is a
# time taken from that lattice — ``seg_bounds = [(bt[s], bt[e]) ...]``
# (harmonia/stages/chord_head.py:459) — so predicted boundaries sit **0.2 ms**
# from the lattice and **60-104 ms** from the beats actually detected.
#
# The lattice is measurably wrong about the audio.  Mean spectral-flux onset
# strength sampled at each grid, normalised by the envelope mean (= 1.0 is
# chance), 7 frozen benchmark songs, reproduced 2026-07-27 by
# ``scratchpad/gridunify_premise.py``:
#
#     song                    detected   lattice   null p95
#     bein_green                  1.70      1.16       1.16
#     blue_bossa                  1.33      1.01       1.15
#     blue_bossa_backing          3.09      1.12       1.30
#     close_to_you                2.93      0.83       1.30
#     every_breath_you_take       3.44      3.32       3.48
#     georgia_on_my_mind          1.51      0.94       1.05
#     stand_by_me                 2.15      1.02       1.23
#
# The detected beats beat a 200-draw random-phase null at p<0.005 on 6/7 songs;
# the lattice is inside the null on 6/7 (p = 0.03-0.96).  ``every_breath_you_
# take`` is the exception in BOTH columns: that recording is machine-steady, so
# the lattice IS the real grid there and neither grid separates from the null.
#
# WHY THE LATTICE EXISTS (do not undo it blindly — CLAUDE.md #6).  The constant
# period is not an accident: ``beat_period_mode="bestfit"`` was made the default
# on 2026-07-19 to kill multi-bar BAR-GRID DRIFT (librosa's tempo scalar is a
# LOCAL median and carries a 0.5-2.3% systematic error that accumulates to ~4
# bars over a song).  A uniform grid also lets every downstream consumer assume
# equal-width bars.  The trade-off this brick makes explicit:
#
#     uniform lattice  -> bars are equal-width and never drift apart from each
#                         other, but the whole grid can be up to half a beat off
#                         the audio at any given instant (rubato, fills, pushes).
#     detected beats   -> every boundary lands on a real onset, but bar widths
#                         vary and a missed/spurious beat locally warps the bar.
#
# So this brick does NOT delete the lattice.  It makes the grid selectable, and
# it REFUSES (falling back to the lattice) whenever the detected grid fails a
# coverage / tempo-octave guard.
#
# MEASURED (7 frozen songs, shipped config, `scratchpad/gridunify_score.py`,
# family-level partial_credit, pooled duration-weighted).  BOTH sides of the
# benchmark had to be moved, because the reference is metronomic too — moving
# only one breaks an agreement between two wrong clocks:
#
#                             reference as-is   reference retimed
#     model on the lattice        0.6679             0.6619
#     model snapped (snap)        0.6644             0.6719
#     model decoded (grid)        0.6647             0.6722
#
# Matched clocks, both synthetic -> both real: **+0.43 pp** partial / +0.64 pp
# root / +0.22 pp strict.  Small.  The sharper, harder-to-fool test is the
# +-0.25 s BOUNDARY-COLLAR diagnostic (blank a collar around every reference
# boundary; whatever that buys is error concentrated AT boundaries):
#
#     collar gain, pooled 7 songs   lattice/as-is +4.79 pp -> real/real +3.75 pp
#     collar gain, the 5 songs it
#       actually reached            lattice/as-is +4.04 pp -> real/real +2.13 pp
#       (root  +3.95 -> +1.72 pp;  strict  +1.62 -> +0.31 pp)
#
# and on the two songs it did NOT reach (see below) the collar gain is unchanged
# to the second decimal.  That dose-response — it shrinks exactly where the brick
# applies and nowhere else — is the load-bearing evidence.  Reading: **about half
# of the boundary-localised error was the synthetic clock; the other half is not.**
#
# WHAT THIS DOES NOT SOLVE (CLAUDE.md rule #4)
# --------------------------------------------
# * **Songs the beat tracker fails on.**  Blue Bossa is missing 25.2 % of its
#   beats; the coverage guard REFUSES it and its score is unchanged to four
#   decimals.  Nothing here rescues a bad beat track — it only declines to make
#   it worse.  (An earlier agent that interpolated into those holes accumulated
#   a silent 37 s error on this exact song.)
# * **The SECOND lattice, downstream.**  The Occam post-pass re-lays the
#   coalesced spans onto its OWN synthetic bar grid — ``bnds = np.arange(phi *
#   beat, times[-1] + bar_period, bar_period)`` (chord_pipeline_v1.py:2602) and
#   ``bar_times = anchor_t + b * bar_len`` (chord_head.py:604).  On any song
#   where Occam fires (stand_by_me on this benchmark) the real-beat grid is
#   silently undone after the fact: measured, ``grid`` mode there is bit-identical
#   to ``off``, while the same run with ``HARMONIA_OCCAM_POSTPASS=0`` lands its
#   boundaries 0.0 ms from the detected beats.  Fixing that means giving the
#   section/Occam pass a real bar grid (the ``HARMONIA_NATIVE_BARGRID`` brick is
#   the natural vehicle) — deliberately NOT done here.
# * **Bar-grid drift, which the lattice existed to prevent.**  ``grid`` mode
#   reintroduces variable bar widths; a locally missed beat locally warps a bar.
#   The 2026-07-19 drift finding is not refuted, it is traded against.
# * **The metric.**  +0.43 pp pooled is ~7 s of a 1654 s benchmark on 7 songs.
#   This is a correctness fix with a small measured payoff, not a lever.
# * **Songs where the lattice was already right.**  every_breath_you_take is
#   machine-steady (median re-lay move 7 ms); it LOSES 1.13 pp root, because the
#   reference moved under a model that did not need to move.

REAL_BEAT_GRID_ENV = "HARMONIA_REAL_BEAT_GRID"

#: Guard thresholds, deliberately the SAME numbers the 2026-07-27 reference
#: re-lay used (golden/frozen_parity/gt_repair_2026-07-27/README.md) so the
#: model side and the reference side accept/refuse exactly the same songs.
REAL_GRID_MAX_GAPFRAC = 0.10      # >10% of expected beats missing -> refuse
REAL_GRID_TEMPO_RATIO = (0.85, 1.18)   # detected/lattice period -> octave guard


def real_beat_grid_mode() -> str:
    """Which beat grid the chord stage uses.  ``HARMONIA_REAL_BEAT_GRID``:

      * ``off`` (DEFAULT) — the shipped synthetic lattice, byte-identical.
        :func:`apply_real_beat_grid` and :func:`snap_chord_times_to_beats` are
        exact no-ops and return the inputs unchanged.
      * ``snap`` — decode on the lattice exactly as today, then re-lay the
        FINAL chord boundary times onto their nearest detected beat (capped at
        half a beat, monotonicity enforced).  Timing-only; the bar layout,
        section grid, chroma pooling and every label are untouched.
      * ``grid`` — replace the pooling/decode grid itself with the detected
        beats, so chroma pooling, the musx re-decode's allowed-transition set,
        the segmentation indices AND the emitted times all live on real beats.

    Unknown values fall back to ``off`` with a warning (never break analyze).
    """
    v = os.environ.get(REAL_BEAT_GRID_ENV, "off").strip().lower()
    if v in ("", "0", "off", "false", "no", "none"):
        return "off"
    if v in ("snap", "grid"):
        return v
    logger.warning("%s=%r not understood (want off|snap|grid) — using off",
                   REAL_BEAT_GRID_ENV, v)
    return "off"


def real_grid_guard(beat_times_real, duration_s: float,
                    period: float) -> "tuple[bool, dict]":
    """Is the detected beat grid trustworthy enough to lay chords on?

    Two guards, both of which must pass (identical rules to the reference
    re-lay, so the two sides accept the same songs):

    ``coverage``  — ``gapfrac = 1 - n_beats / (duration_s / median_ibi)``.  A
    tracker that dropped beats leaves holes the size of several beats; snapping
    a boundary into such a hole moves it to the WRONG beat rather than to no
    beat at all.  ``blue_bossa`` fails here at **25.2 % missing** — its grid is
    refused, not interpolated (interpolating is how an earlier agent silently
    accumulated a 37 s error on this very song).

    ``tempo octave`` — ``median_ibi / period`` must sit in [0.85, 1.18].  A 2x
    or 0.5x octave lock (CLAUDE.md, song 002) makes the two grids describe
    different music; refuse rather than half-fix.

    Returns ``(ok, stats)``; ``stats`` is always populated for logging.
    """
    bts = np.asarray(beat_times_real, dtype=float) if beat_times_real is not None \
        else np.zeros(0)
    bts = bts[np.isfinite(bts)]
    stats = {"n_beats": int(len(bts)), "duration_s": float(duration_s),
             "gapfrac": None, "tempo_ratio": None, "reason": ""}
    if len(bts) < 8 or duration_s <= 0 or period <= 0:
        stats["reason"] = "too few detected beats"
        return False, stats
    ibi = float(np.median(np.diff(bts)))
    if not np.isfinite(ibi) or ibi <= 0:
        stats["reason"] = "degenerate inter-beat interval"
        return False, stats
    gapfrac = 1.0 - len(bts) / (duration_s / ibi)
    ratio = ibi / period
    stats["gapfrac"] = round(float(gapfrac), 4)
    stats["tempo_ratio"] = round(float(ratio), 4)
    if gapfrac > REAL_GRID_MAX_GAPFRAC:
        stats["reason"] = (f"beat-grid coverage: {gapfrac:.1%} of expected beats "
                           f"missing (>{REAL_GRID_MAX_GAPFRAC:.0%})")
        return False, stats
    if not (REAL_GRID_TEMPO_RATIO[0] <= ratio <= REAL_GRID_TEMPO_RATIO[1]):
        stats["reason"] = f"tempo octave: detected/lattice period {ratio:.3f}"
        return False, stats
    return True, stats


def apply_real_beat_grid(bt: np.ndarray, beat_times_real, duration_s: float,
                         period: float) -> "tuple[np.ndarray, dict]":
    """``grid`` mode: return the DETECTED-beat grid in place of the lattice.

    Same endpoint convention as the shipped lattice (``0.0`` and ``duration_s``
    are always present, duplicates removed), so every consumer that indexes
    ``bt`` keeps working; only the interior times change and the beat COUNT may
    differ (e.g. bein_green 225 lattice cells -> 208 real ones).

    Returns ``(bt_new, info)``.  On ``off``/``snap`` mode or a failed guard the
    input ``bt`` is returned **unchanged** (identity, not a copy-equal array),
    which is what makes the OFF path provably a no-op.
    """
    info = {"mode": real_beat_grid_mode(), "applied": False}
    if info["mode"] != "grid":
        return bt, info
    ok, stats = real_grid_guard(beat_times_real, duration_s, period)
    info.update(stats)
    if not ok:
        logger.warning("real-beat grid REFUSED (%s) — keeping the synthetic "
                       "lattice", stats["reason"])
        return bt, info
    bts = np.asarray(beat_times_real, dtype=float)
    bts = bts[(bts > 0.0) & (bts < duration_s)]
    bt_new = np.unique(np.concatenate([[0.0], bts, [float(duration_s)]]))
    info.update({"applied": True, "n_lattice": int(len(bt)),
                 "n_real": int(len(bt_new))})
    logger.warning("real-beat grid APPLIED: %d lattice cells -> %d detected "
                   "beats (gapfrac %.3f, tempo ratio %.3f)",
                   len(bt), len(bt_new), stats["gapfrac"], stats["tempo_ratio"])
    return bt_new, info


def snap_chord_times_to_beats(chords_out: list[dict], segments_out: list[dict],
                              beat_times_real, duration_s: float,
                              period: float) -> dict:
    """``snap`` mode: re-lay the FINAL chord boundary times onto detected beats.

    Operates on the *boundary set* (not per chord independently) so the emitted
    timeline stays contiguous and strictly increasing: for the sorted list of
    distinct boundaries ``b_0 < b_1 < ... < b_n`` each ``b_i`` moves to its
    nearest detected beat, the move is rejected if it exceeds half a beat, and
    a move that would cross or touch the previous accepted boundary is rejected
    (monotonicity beats accuracy — a zero/negative-length chord is worse than a
    boundary 100 ms late).  The first and last boundaries are pinned so the
    scored span never changes.

    Mutates ``chords_out`` / ``segments_out`` in place; returns a stats dict.
    Does nothing (and returns ``applied=False``) unless mode is ``snap`` and
    the guard passes.
    """
    info = {"mode": real_beat_grid_mode(), "applied": False}
    if info["mode"] != "snap" or not chords_out:
        return info
    ok, stats = real_grid_guard(beat_times_real, duration_s, period)
    info.update(stats)
    if not ok:
        logger.warning("chord-time snap REFUSED (%s) — timings unchanged",
                       stats["reason"])
        return info

    bts = np.asarray(beat_times_real, dtype=float)
    bnds = [float(chords_out[0]["start_s"])]
    for c in chords_out:
        bnds.append(float(c["end_s"]))
    cap = 0.5 * period

    new = list(bnds)
    moves = []
    for i in range(1, len(bnds) - 1):          # endpoints pinned
        j = int(np.argmin(np.abs(bts - bnds[i])))
        cand = float(bts[j])
        if abs(cand - bnds[i]) > cap:
            continue
        if cand <= new[i - 1] + 1e-6 or cand >= bnds[i + 1] - 1e-6:
            continue
        new[i] = cand
        moves.append(abs(cand - bnds[i]))

    for i, c in enumerate(chords_out):
        c["start_s"] = round(new[i], 3)
        c["end_s"] = round(new[i + 1], 3)
        c["duration_beats"] = max(1, round((new[i + 1] - new[i]) / period))
        if "onset_s" in c:      # keep the DISPLAY playhead consistent
            c["onset_s"] = c["start_s"]
        if "offset_s" in c:
            c["offset_s"] = c["end_s"]
    for i, s in enumerate(segments_out or []):
        if i + 1 < len(new):
            s["start_s"] = round(new[i], 3)
            s["end_s"] = round(new[i + 1], 3)

    info.update({
        "applied": True, "n_boundaries": len(bnds), "n_moved": len(moves),
        "frac_moved": round(len(moves) / max(len(bnds) - 2, 1), 4),
        "median_move_ms": round(float(np.median(moves)) * 1000, 1) if moves else 0.0,
        "max_move_ms": round(float(np.max(moves)) * 1000, 1) if moves else 0.0,
    })
    logger.warning("chord-time snap APPLIED: %d/%d boundaries moved to detected "
                   "beats (median %.0f ms, max %.0f ms)", info["n_moved"],
                   len(bnds) - 2, info["median_move_ms"], info["max_move_ms"])
    return info


def attach_musx_onset_hints(
    chords_out: list[dict],
    mx_labels: list[tuple[float, float, str]],
    period: float,
    no_chord_label: str = "N",
) -> int:
    """Attach a trusted DISPLAY onset/offset (``onset_s``/``offset_s``) per chord.

    Fixes a chord-START TIMING bug (user report 2026-07-20, This Love): the
    displayed bar-1 playhead highlighted a full beat late ("plutot que celui
    d'apres"). Root cause: the chord's onset that drives the display is the
    UNIFORM bar-grid time (This Love's opening G = 1.42s), and the display-snap
    (``_snap``, render_youtube_chart) then rounds it to the NEAREST real beat —
    1.42s tips just past the midpoint to the SECOND real beat (1.74s) instead
    of the first (1.09s). The uniform grid has no beat near the true onset (its
    phase put the nearest beat at 1.42s), so a grid-level fix can't reach it.

    music-x-lab's own change-times ARE accurate (opening G at 1.18s -> snaps to
    the correct first real beat 1.09s) and are already the label source when
    quality_frontend="musx". This carries that trusted change-time to the
    renderer as a DISPLAY hint only: the (bar, beat) LAYOUT still comes from
    the uniform grid (sections/folds byte-identical), but the playhead t0/t1
    snap to the music-x-lab onset/offset instead of the drifted uniform time.
    Same display-layer philosophy as the 2026-07-20 real-beat snap, just fed a
    better onset estimate. Mutates ``chords_out`` in place; returns the count
    changed.

    Match rule: the music-x-lab change-time (label t0) NEAREST the chord's
    uniform START, accepted only within +/-1 beat (``period``) — a bounded
    correction that can't wander onto a distant chord. If no change-time is
    within tolerance the uniform onset is kept (no hint). ``offset_s`` is set
    to the next chord's hint (or the covering label's t1) so the last-held
    chord's tail is also accurate. N.C. cells are skipped (no trusted onset).
    """
    if not chords_out or len(mx_labels) < 2:
        return 0
    changes = np.array([t0 for (t0, _t1, lab) in mx_labels if lab not in ("N", "X")],
                       dtype=float)
    if changes.size == 0:
        return 0
    tol = max(period, 1e-3)
    n = 0
    hinted: list[dict] = []
    for c in chords_out:
        if c.get("label") == no_chord_label:
            hinted.append(c)
            continue
        j = int(np.argmin(np.abs(changes - c["start_s"])))
        if abs(changes[j] - c["start_s"]) <= tol:
            c["onset_s"] = round(float(changes[j]), 3)
            n += 1
        hinted.append(c)
    # offset_s = next chord's display onset (or its own end) so tails stay tight.
    for i, c in enumerate(chords_out):
        if "onset_s" not in c:
            continue
        nxt = next((d for d in chords_out[i + 1:]
                    if d.get("label") != no_chord_label), None)
        if nxt is not None:
            c["offset_s"] = round(float(nxt.get("onset_s", nxt["start_s"])), 3)
    return n
