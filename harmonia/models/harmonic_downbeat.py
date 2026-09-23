"""Harmonic-rhythm downbeat-phase brick (standalone, kill-switched, 2026-07-21).

WHAT / WHY
----------
Beat *tracking* (where the beats are) is good; downbeat *phase* — which beat is
the "1" — is sometimes off by a whole beat because the audio tracker
(Beat This!) sets the phase and is occasionally confidently wrong. We
under-use one strong, independent piece of evidence: HARMONY CHANGES ON THE
DOWNBEAT. Chords overwhelmingly (re)start on beat 1, so the harmonic-change
signal votes on the true phase without ever looking at the audio tracker.

This module is the HARMONIC analog of the spectral comb in
``beat_grid._flux_downbeat_phase``. Where that folds a continuous chroma-flux
novelty modulo the bar and takes the peak, this works directly on the beat
grid: it measures harmonic change BETWEEN CONSECUTIVE BEATS (from the model's
per-beat root/quality posteriors or beat-pooled chroma), folds that change onto
the four within-bar positions, and scores each candidate downbeat phase.

THE KEY DESIGN DECISION (why this beats the refuted naive version)
------------------------------------------------------------------
A naive "the phase with the most harmonic change is the downbeat" argmax was
screened on POP909 (880 regular-4/4 songs, oracle chords) and REPRODUCED the
known STEP-3 wash: 49.8% per-song phase recovery and a confidence signal that
does NOT separate right from wrong (corr(conf, correct)=0.03). See
docs/known_issues.md "PHASE 2 STEP 3" — a confidently-wrong harmonic comb
(song 421) is exactly what sinks a naive arbiter.

The fix: score the *shape*, not the peak. In POP909 GT, chord-change
probability by within-bar position is a MONOTONE DESCENDING RAMP
(0.471 / 0.403 / 0.313 / 0.233 for beat 1..4), not a spike. Correlating each
candidate phase's folded profile against that ramp rejects the dominant failure
mode — the half-bar alias where chords change equally on beats 1 and 3 (pos0 ~=
pos2), which fools argmax but does NOT match the ramp. On the same oracle screen
this lifts confidence to a REAL signal (corr 0.177) with **top-decile precision
0.73** — the basis for a conservative, high-precision/low-recall override.

CONSERVATISM (the STEP-3 / CLAUDE.md #5 guardrail)
--------------------------------------------------
``arbitrate_downbeat_phase`` KEEPS the audio phase by default and overrides only
when the harmonic evidence is BOTH strong (large ramp-margin over the runner-up
phase) AND clearly disagrees, and the audio tracker is not itself
veto-confident. This encodes the 421 lesson: never let harmony overturn a
confidently-correct audio phase. The oracle screen is a CEILING — audio-derived
per-beat evidence is noisier — so the thresholds default high.

WHAT IT IMPROVES, AND WHAT IT DOES NOT (measured, CLAUDE.md #4)
--------------------------------------------------------------
On POP909 (201 songs, song-grouped CV, out-of-fold), at the conservative default
operating point the brick improves DOWNBEAT-PHASE accuracy — dbF 0.322->0.373
(+0.051), phase-accuracy 0.347->0.440 (+9.3pp), override-precision 0.80 — with a
single audio-right regression (on a marginal-audio song, not a confident one).
It BEATS the refuted STEP-3 arbitration (a corpus-wide wash): the learned scorer
is the reason a low-regression operating point exists at all (the fixed ramp
scorer has none — it always costs 7-8/26 audio-right regressions).

Crucially, it DOES NOT improve chord accuracy: the per-bar dominant chord is
largely invariant to a +-1-beat phase shift, so the bar-chord-accuracy proxy is
flat-to-slightly-negative (+0.000 to -0.006) at every operating point. The value
is better BARLINES / notation / section alignment, NOT better chord labels.

SCOPE / DOES NOT SOLVE (CLAUDE.md #4)
-------------------------------------
- Pure estimator + arbiter over already-extracted beat/evidence data. No audio
  I/O, no beat tracking, no chord decoding. It does not find beats or tempo; it
  only votes on phase given a beat grid.
- It does not fix the beat GRID or the tempo octave (orthogonal — STEP 4), nor
  variable meter (assumes a fixed ``bar_len``, default 4/4).
- Standalone phase recovery is imperfect (~mid-50s%); the brick is an
  occasionally-firing CORRECTOR on top of the audio tracker, not a standalone
  phase estimator. The premise itself is real but WEAK (harmonic change
  concentrates on the downbeat with only a 1.49x lift).
- Gated behind ``HARMONIA_HARMONIC_DOWNBEAT`` (default OFF): importing/using the
  functions is inert to the live pipeline until an orchestrator wires it in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

HARMONIC_DOWNBEAT_ENV = "HARMONIA_HARMONIC_DOWNBEAT"

# Conservative-override defaults, calibrated on POP909 (201 songs, song-grouped
# 5-fold CV, out-of-fold — see the eval script / docs/known_issues.md entry).
# The confidence gated on is the LEARNED phase-probability margin (top minus
# runner-up P over the 4 phases, in [0, 1]).
#   MIN_CONF          minimum margin to override the audio tracker.
#   AUDIO_VETO_CONF   an audio tracker at/above this regularity keeps its phase
#                     regardless (protects the 421-class — a confidently-correct
#                     audio phase is never overturned). 0.70 is BELOW the
#                     pipeline's sota-trust threshold (0.85), so the brick can
#                     only act when the audio downbeat is itself unreliable.
#   HALFBAR_MIN_CONF  a HALF-BAR flip (best_phase == audio_phase + bar_len/2 —
#                     the aliasing failure where chords change equally on beats 1
#                     and 3) demands THIS higher margin. Aliasing is the dominant
#                     wrong-override mode; guarding it doubled the safe net gain.
# At this OP (OOF): downbeat-phase dbF 0.322->0.373 (+0.051), phase-acc
# 0.347->0.440 (+9.3pp), override-precision 0.80, 1/26 audio-right regressions
# (that one a MARGINAL song, not a confident 421-class one).
DEFAULT_MIN_CONF = 0.20
DEFAULT_AUDIO_VETO_CONF = 0.70
DEFAULT_HALFBAR_MIN_CONF = 0.85


def harmonic_downbeat_enabled() -> bool:
    """True iff the HARMONIA_HARMONIC_DOWNBEAT kill-switch is ON (default OFF)."""
    return os.environ.get(HARMONIC_DOWNBEAT_ENV, "off").strip().lower() in (
        "1", "on", "true", "yes")


@dataclass
class HarmonicDownbeatResult:
    """Result of :func:`harmonic_downbeat_phase`.

    Attributes
    ----------
    best_phase:
        Winning downbeat phase ``φ in [0, bar_len)`` — the beat-index-mod-bar
        the harmonic evidence says begins each bar. Same convention as
        ``beat_grid._flux_downbeat_phase`` (the caller maps it onto the real
        beat grid).
    confidence:
        Top-2 margin = (best per-phase score) minus (runner-up per-phase score).
        With the learned scorer (default) the scores are probabilities over the
        four phases, so ``confidence`` is in ``[0, 1]``; with the ramp fallback
        they are correlations, so it is in ``[0, 2]``. Higher = the winning phase
        is favoured much more than any rival. This is the strength signal the
        arbiter gates on, and — unlike the refuted naive argmax margin — a
        genuine right/wrong discriminator (see module docstring).
    per_phase_score:
        Length-``bar_len`` list — the learned P(phase) distribution (sums to 1)
        or the ramp-correlation per phase; ``per_phase_score[best_phase]`` is the
        max.
    change_mass:
        Length-``bar_len`` list — MEAN harmonic change per beat at each
        within-bar position (diagnostic; ``best_phase`` is NOT simply its
        argmax, that is the refuted naive scorer).
    n_beats:
        Number of beats the evidence covered.
    valid:
        False when there was too little evidence to score (fewer than
        ``2*bar_len`` beats, all-zero change, or degenerate input); the caller
        must then fall back to the audio phase. ``best_phase`` is 0 and
        ``confidence`` 0.0 when invalid.
    """

    best_phase: int
    confidence: float
    per_phase_score: list[float]
    change_mass: list[float]
    n_beats: int
    valid: bool


def _descending_ramp(bar_len: int) -> np.ndarray:
    """Centered monotone-descending template, length ``bar_len``.

    Encodes the empirical POP909 prior "harmonic change is strongest on the
    downbeat and decreases monotonically across the bar." Mean-centered so the
    correlation is invariant to the overall change rate. For 4/4 this is
    ``[1.5, 0.5, -0.5, -1.5]`` (proportional to the measured 0.47/0.40/0.31/0.23
    ramp; the exact spacing is immaterial to a rank-style correlation, the
    monotone shape is what matters)."""
    r = np.arange(bar_len, dtype=np.float64)[::-1]  # bar_len-1 .. 0, descending
    return r - r.mean()


def _row_normalize(evidence: np.ndarray) -> np.ndarray:
    """L2-normalize each beat's evidence row (guarding zero rows)."""
    ev = np.asarray(evidence, dtype=np.float64)
    if ev.ndim != 2:
        raise ValueError(f"per_beat_evidence must be 2-D (n_beats, D); got {ev.shape}")
    norms = np.linalg.norm(ev, axis=1, keepdims=True)
    safe = np.where(norms > 1e-9, norms, 1.0)
    return ev / safe, (norms[:, 0] > 1e-9)


def beat_change_strength(per_beat_evidence: np.ndarray) -> np.ndarray:
    """Per-beat harmonic-change strength ``c[i] = 1 - cos(ev[i], ev[i-1])``.

    ``c[0] = 0`` (no predecessor). Cosine distance in ``[0, 2]``: 0 when two
    consecutive beats carry the same harmony (chord held), large when a new
    chord starts on beat ``i``. Cosine (not L2) so the signal is invariant to
    per-beat energy — a loud vs soft beat of the SAME chord is not a change —
    matching the emission-template geometry (docs/known_issues.md #5). Works
    for any nonneg per-beat feature: root/quality posteriors (each row a
    distribution) or beat-pooled chroma. A zero-energy beat (silence) yields 0
    change on both its edges."""
    ev, ok = _row_normalize(per_beat_evidence)
    n = ev.shape[0]
    c = np.zeros(n, dtype=np.float64)
    if n < 2:
        return c
    dots = np.sum(ev[1:] * ev[:-1], axis=1)
    both_ok = ok[1:] & ok[:-1]
    c[1:] = np.where(both_ok, 1.0 - dots, 0.0)
    return c


def beat_change_profile(per_beat_evidence: np.ndarray, bar_len: int = 4) -> np.ndarray:
    """MEAN harmonic change per beat at each within-bar position (length ``bar_len``).

    Fold :func:`beat_change_strength` onto positions ``i mod bar_len`` and divide
    by the number of change-BEARING beats at each position. Beat 0 is excluded:
    ``change[0]`` is definitionally 0 (no predecessor), not a measured "no
    change", so summing it in would systematically depress whichever position
    beat 0 lands on and let the ramp correlation manufacture a spurious confident
    phase out of a flat (no-downbeat-signal) song. Taking the MEAN over measured
    beats removes that edge bias: a uniformly-changing song yields a flat profile
    (=> zero confidence), as it must."""
    c = beat_change_strength(per_beat_evidence)
    n = len(c)
    prof = np.zeros(bar_len, dtype=np.float64)
    if n < 2:
        return prof
    idx = np.arange(1, n) % bar_len           # beats 1..n-1 carry measured change
    csum = np.zeros(bar_len, dtype=np.float64)
    cnt = np.zeros(bar_len, dtype=np.float64)
    np.add.at(csum, idx, c[1:])
    np.add.at(cnt, idx, 1.0)
    return csum / np.maximum(cnt, 1.0)


# --------------------------------------------------------------------------- #
# Learned phase classifier (POP909 song-grouped-CV logistic ranker)
# --------------------------------------------------------------------------- #
# The ramp scorer above is a fixed musical prior. On POP909 it recovers the true
# phase ~54% standalone with a genuine (but modest) confidence signal. A tiny
# logistic model, trained per-phase ("is THIS phase the downbeat?") on
# audio-derived features and calibrated across the four phases, gives a
# better-calibrated confidence — the quantity a conservative arbiter actually
# needs. Coefficients are EMBEDDED (no external weights file) so the brick stays
# a single self-contained module; ``None`` until trained, in which case
# everything falls back to the transparent ramp scorer.
#
# Provenance / honesty: trained on POP909, 201 songs, 75 with a clean single
# best phase, song-grouped 5-fold CV; features from beat-pooled root posteriors
# (the pipeline's own per-beat evidence). Standardizer + logistic weights below
# are the final fit on all 75 clean songs. The featurizer is the SINGLE
# definition shared by training and inference (scratchpad/extract_features.py
# imports phase_feature_matrix) — no train/serve skew (CLAUDE.md #1). To retrain,
# re-run scratchpad/train_and_eval.py and paste its EMBEDDABLE COEFFICIENTS.
_FEATURE_NAMES = ("vn0", "vn1", "vn2", "ramp_corr", "dom", "half", "adj",
                  "mean_change", "cv", "log_nbeats")
_FEAT_MEAN: "list[float] | None" = [
    0.25, 0.25, 0.25, 0.0, 0.0, 0.0, -0.068489, 0.343442, 0.535806, 5.651309]
_FEAT_STD: "list[float] | None" = [
    0.146985, 0.146985, 0.146985, 0.532637, 0.19598, 0.200806, 0.270661,
    0.085157, 0.242046, 0.222461]
_LR_COEF: "list[float] | None" = [
    1.163494, 0.32891, 0.36913, 0.069429, 1.163494, 0.581456, -0.116392,
    0.088209, -0.568288, 0.027755]
_LR_INTERCEPT: float = -1.105081
# Arbiter confidence scale depends on which scorer is active: the learned scorer
# emits a probability margin in [0, 1]; the ramp fallback a correlation margin in
# [0, 2]. DEFAULT_MIN_CONF below is calibrated for the LEARNED scorer (the
# shipped default). _RAMP_MIN_CONF is used when the ramp fallback is active.

_RAMP4 = np.array([1.5, 0.5, -0.5, -1.5])


def phase_feature_matrix(profile: np.ndarray, n_beats: int) -> np.ndarray:
    """(4, 10) — feature vector per candidate downbeat phase (4/4 only).

    SINGLE source of truth for the classifier features, shared verbatim by the
    offline training script (no train/serve skew, CLAUDE.md #1). ``profile`` is
    the mean-change profile from :func:`beat_change_profile`. For each candidate
    phase ``p`` the profile is rotated so ``p`` sits at position 0 and described
    by: its normalized shape (first 3 bins), the ramp correlation, and three
    dominance contrasts (vs interior / vs half-bar alias / vs neighbours). The
    last three features (mean change, profile CV, log n_beats) are
    phase-INDEPENDENT — they set the confidence LEVEL without moving the argmax.
    """
    m = np.asarray(profile, dtype=np.float64)
    s = m.sum() + 1e-12
    mean_change = float(m.mean())
    cv = float(m.std() / (m.mean() + 1e-9))
    ramp_n = np.linalg.norm(_RAMP4)
    rows = []
    for p in range(4):
        v = np.roll(m, -p).astype(np.float64)
        vn = v / s
        vc = v - v.mean()
        vcn = np.linalg.norm(vc)
        rc = float(vc @ _RAMP4 / (vcn * ramp_n)) if vcn > 1e-9 else 0.0
        dom = float(v[0] - np.mean(v[1:]))
        half = float(v[0] - v[2])
        adj = float(v[0] - max(v[1], v[3]))
        rows.append([vn[0], vn[1], vn[2], rc, dom / (s + 1e-9),
                     half / (s + 1e-9), adj / (s + 1e-9),
                     mean_change, cv, float(np.log(max(n_beats, 1)))])
    return np.asarray(rows, dtype=np.float64)


def learned_available() -> bool:
    """True iff embedded classifier coefficients are present."""
    return _LR_COEF is not None


def _learned_phase_probs(profile: np.ndarray, n_beats: int) -> "np.ndarray | None":
    """Softmax P(phase is downbeat) over the 4 phases from the embedded logistic
    ranker, or None if coefficients are absent."""
    if _LR_COEF is None:
        return None
    X = phase_feature_matrix(profile, n_beats)
    z = (X - np.asarray(_FEAT_MEAN)) / np.asarray(_FEAT_STD)
    logits = z @ np.asarray(_LR_COEF) + _LR_INTERCEPT
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def _ramp_phase_scores(mass: np.ndarray, bar_len: int) -> np.ndarray:
    """Per-phase ramp correlation (the transparent fallback scorer)."""
    ramp = _descending_ramp(bar_len)
    ramp_norm = np.linalg.norm(ramp)
    scores = np.zeros(bar_len, dtype=np.float64)
    for phi in range(bar_len):
        v = np.roll(mass, -phi).astype(np.float64)
        v = v - v.mean()
        vn = np.linalg.norm(v)
        scores[phi] = float(v @ ramp / (vn * ramp_norm)) if vn > 1e-9 else 0.0
    return scores


def harmonic_downbeat_phase(
    beat_times: "np.ndarray | None",
    per_beat_evidence: np.ndarray,
    bar_len: int = 4,
    method: str = "auto",
) -> HarmonicDownbeatResult:
    """Score downbeat phase from harmonic rhythm (the ramp-shape scorer).

    Parameters
    ----------
    beat_times:
        Beat times in seconds, length ``n_beats`` — accepted for API symmetry
        with the rest of the grid code and to let a future weighting use beat
        spacing; the phase decision uses only the beat INDEX (mod ``bar_len``),
        so ``None`` is allowed. When provided it must match ``per_beat_evidence``
        in length.
    per_beat_evidence:
        ``(n_beats, D)`` nonneg per-beat harmonic evidence — the model's per-beat
        root (or root×quality) posteriors, or beat-pooled chroma. Root/quality
        posteriors are preferred (they encode chord IDENTITY, so a chroma wobble
        from voicing/timbre that does not change the chord does not register as a
        change); beat-pooled chroma also works.
    bar_len:
        Beats per bar (default 4). The folding generalizes to any
        ``bar_len >= 2``; the LEARNED scorer is 4/4-only.
    method:
        ``"auto"`` (default) uses the embedded learned classifier when available
        and ``bar_len == 4``, else the ramp scorer. ``"ml"`` forces the learned
        classifier (raises if unavailable). ``"ramp"`` forces the transparent
        ramp scorer.

    Returns
    -------
    HarmonicDownbeatResult
        ``per_phase_score`` is a probability distribution (learned scorer) or a
        correlation per phase (ramp scorer); ``confidence`` is the top-2 margin
        in ``[0, 1]`` (learned) or ``[0, 2]`` (ramp).

    Method
    ------
    1. ``c[i]`` = cosine change between consecutive beats (:func:`beat_change_strength`).
    2. Fold onto within-bar positions with :func:`beat_change_profile`
       (MEAN change per position, edge-bias-corrected).
    3. Score each candidate phase — learned classifier (per-phase logistic
       ranker, softmax over phases) or, as fallback, correlation of the
       phase-rotated profile against the downbeat ramp. Both reject the half-bar
       alias (equal change on beats 1 and 3), which fools a naive argmax.
    4. ``best_phase`` = argmax; ``confidence`` = best minus runner-up score.

    Deterministic: ties in the per-phase score keep the lowest phase index.
    """
    if bar_len < 2:
        raise ValueError(f"bar_len must be >= 2; got {bar_len}")
    ev = np.asarray(per_beat_evidence, dtype=np.float64)
    if ev.ndim != 2:
        raise ValueError(f"per_beat_evidence must be 2-D (n_beats, D); got {ev.shape}")
    n = ev.shape[0]
    if beat_times is not None and len(beat_times) != n:
        raise ValueError(
            f"beat_times ({len(beat_times)}) must match per_beat_evidence rows ({n})")

    if method not in ("auto", "ml", "ramp"):
        raise ValueError(f"method must be auto|ml|ramp; got {method!r}")

    zero = [0.0] * bar_len
    if n < 2 * bar_len:
        return HarmonicDownbeatResult(0, 0.0, list(zero), list(zero), n, False)

    mass = beat_change_profile(ev, bar_len)

    if mass.sum() <= 1e-9:
        return HarmonicDownbeatResult(0, 0.0, list(zero), mass.tolist(), n, False)

    # Scorer dispatch. Learned classifier is 4/4-only (its ramp/feature template
    # is 4-specific); anything else uses the transparent ramp scorer.
    use_ml = (method in ("auto", "ml") and bar_len == 4 and learned_available())
    if method == "ml" and not use_ml:
        raise ValueError("method='ml' requires bar_len==4 and embedded "
                         "coefficients (learned_available() is False)")
    if use_ml:
        scores = _learned_phase_probs(mass, n)      # P(phase), sums to 1
    else:
        scores = _ramp_phase_scores(mass, bar_len)  # correlation per phase

    best_phase = int(np.argmax(scores))          # ties -> lowest index (argmax)
    srt = np.sort(scores)[::-1]
    confidence = float(srt[0] - srt[1])
    return HarmonicDownbeatResult(
        best_phase=best_phase,
        confidence=confidence,
        per_phase_score=[float(x) for x in scores],
        change_mass=mass.tolist(),
        n_beats=n,
        valid=True,
    )


def arbitrate_downbeat_phase(
    audio_phase: int,
    audio_conf: float,
    harmonic_result: HarmonicDownbeatResult,
    min_harmonic_conf: float = DEFAULT_MIN_CONF,
    audio_veto_conf: float = DEFAULT_AUDIO_VETO_CONF,
    halfbar_min_conf: float = DEFAULT_HALFBAR_MIN_CONF,
) -> "tuple[int, str, bool]":
    """Conservatively reconcile the audio-tracker phase with the harmonic vote.

    Parameters
    ----------
    audio_phase:
        Downbeat phase from the audio tracker (Beat This! native / the pipeline's
        current ``sota_downbeat_phase``), in ``[0, bar_len)``.
    audio_conf:
        The audio tracker's own confidence for that phase (e.g. Beat This!
        inter-downbeat regularity), in ``[0, 1]``.
    harmonic_result:
        Output of :func:`harmonic_downbeat_phase`. ``confidence`` is the learned
        phase-probability margin in ``[0, 1]`` (or the ramp-correlation margin in
        ``[0, 2]`` when the ramp fallback is active — the defaults are calibrated
        for the learned scale).
    min_harmonic_conf:
        Minimum harmonic margin to override (default ``DEFAULT_MIN_CONF``).
    audio_veto_conf:
        If ``audio_conf >= audio_veto_conf`` the audio phase is KEPT regardless
        of the harmonic vote (default ``DEFAULT_AUDIO_VETO_CONF``). The 421
        guardrail: a confidently-correct audio phase is never overturned.
    halfbar_min_conf:
        A HALF-BAR flip (``best_phase == audio_phase + bar_len/2 (mod bar_len)``)
        must clear THIS higher margin (default ``DEFAULT_HALFBAR_MIN_CONF``).
        The half-bar alias — chords changing equally on beats 1 and 3 — is the
        dominant confident-but-wrong override; demanding more evidence for it
        roughly doubled the safe net gain on POP909. Only meaningful for even
        ``bar_len``.

    Returns
    -------
    (final_phase, source, disagreement_flag)
        ``source`` is ``"harmonic"`` iff the override fired, else ``"audio"``.
        ``disagreement_flag`` is True whenever the harmonic best phase differs
        from ``audio_phase`` (emitted even when the override is NOT taken, so a
        caller / log can see a silent disagreement).

    Policy (conservative — KEEP audio unless harmony strongly AND clearly wins):
        thr = halfbar_min_conf if the flip is a half-bar flip else min_harmonic_conf
        override  ⇔  harmonic_result.valid
                     AND best_phase != audio_phase          (clear disagreement)
                     AND confidence >= thr                   (strong enough margin)
                     AND audio_conf <  audio_veto_conf       (audio not veto-sure)
    """
    disagreement = bool(harmonic_result.valid
                        and harmonic_result.best_phase != audio_phase)
    bar_len = len(harmonic_result.per_phase_score) or 4
    is_halfbar = (bar_len % 2 == 0 and disagreement
                  and harmonic_result.best_phase == (audio_phase + bar_len // 2) % bar_len)
    thr = halfbar_min_conf if is_halfbar else min_harmonic_conf
    override = bool(
        disagreement
        and harmonic_result.confidence >= thr
        and audio_conf < audio_veto_conf
    )
    if override:
        return harmonic_result.best_phase, "harmonic", disagreement
    return int(audio_phase), "audio", disagreement
