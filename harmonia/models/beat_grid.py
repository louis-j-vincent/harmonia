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
