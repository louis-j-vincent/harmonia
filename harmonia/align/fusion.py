"""Stage 2 of the Bayesian aligner — the FUSION DBN (first validated version).

This replaces the hand-tuned threshold stack (``scripts/brick0_propose.py``) with
a single **bar-pointer state-space** that FUSES the four instruments, each weighted
by its LOCAL reliability, and — for free from the same model — emits a per-region
and whole-song posterior **CONFIDENCE** (the self-detection / dataset-gate signal).
See ``docs/fusion_aligner_design.md`` for the full spec and the Stage 0/0b/bass
premise results this build stands on.

WHAT IS FUSED (the observation at bar-pointer state (section-position s, beat p)):

    logL(s, p) = Σ_i  w_i(p) · logL_i(s, p)

    | i | stream                         | strong where     | weight w_i(p)              |
    |---|--------------------------------|------------------|----------------------------|
    | 1 | harmonic agreement             | comping present  | comping salience (chart-   |
    |   | chroma vs the section's chart  |                  | fill; LOW in solos)        |
    |   | chord template                 |                  |                            |
    | 2 | drum beat/tempo grid           | everywhere incl. | drum reliability           |
    |   | (``drum_pattern``)             | solos            | (HIGH in solos)            |
    | 3 | bass root/PC match             | pop/soul         | bass concentration         |
    |   | (``bass_salience``)            |                  | (~0 on walking bass)       |
    | 4 | harmonic rhythm — chord CHANGE | clear changes    | flux salience              |
    |   | on strong beats (chroma flux)  |                  |                            |

The reliability weighting is the Bayesian win: in a solo w_harm and w_bass fall to
~0 automatically, and the section placement is carried by the FORM PRIOR (constant
tempo + chart section lengths in the transition model) plus the drum grid — exactly
where the harmony-only threshold stack drifted ("cacahuète"). No fixed thresholds.

DOWNBEAT = ONE GLOBAL PHASE, refined by the CHART FORM (anticipation-aware). The
acoustic global-phase resolver (``harmonia.align.downbeat``) gives a phase from
harmonic-rhythm + bass on the lattice; the aligned chart's bar-1 positions give a
FORM phase. When the two differ by exactly one beat we read it as the pop "push"
(the chord change anticipates the downbeat — it fooled the acoustic harmonic-rhythm
evidence on Every Breath) and PREFER the form phase; where they agree we are
confident; a half-bar disagreement is FLAGGED (precision-first).

INFERENCE = the *same* model with chord identity LATENT (design doc). This module
solves timing/sections with the chord identity OBSERVED (from the chart); the
downstream chord-inference variant reuses the identical streams + state-space.

STAGE 2b (2026-07-24) — the DRIFTING-τ state + vamp/gap propagation now CLOSE the
v1 remainder, so the DBN SUBSUMES brick0's drift/vamp within the principled model:
  * WITHIN-song tempo DRIFT is a STATE: a BOUNDED, strongly-regularised warp of the
    bar-pointer LATTICE (per-window τ, from brick0's windowed/whole-song drift
    DETECTION), on which the fusion DP re-places; accepted only if it RAISES the
    coverage-weighted harmonic agreement (constant-tempo prior + a tracked drift,
    never a free per-section warp). Reproduces blue_bossa's whole-song drift and
    every_breath's windowed drift; a clean/rubato song is never warped (self-check).
  * VAMPS (form-periodic turnarounds, e.g. Autumn) are discrete large-gap
    transitions (brick0's ``propagate_form_vamps``, re-tiled onto the fusion
    emission) — LARGE pause-gaps only, never sub-second catch-up gaps.
  See ``_drift_stage`` / ``_form_vamp_stage`` and the design-doc 2026-07-24 note.

WHAT THIS VERSION STILL DOES **NOT** SOLVE (rule #4 — state the remainder):
  * The bass/harmonic-rhythm streams enter the section-placement emission with
    small weights (harmony dominates) so the ear-approved frozen alignments
    reproduce; their larger role today is in the CONFIDENCE and the DOWNBEAT phase.
  * Per-song ear-overrides (Georgia's relabels/splits, the Close-To-You mirror
    fix, rubato-tail truncation) are NOT re-derived here — they are downstream GT
    edits. This module produces the section->time ALIGNMENT + confidence; the
    overrides are applied by the existing builder on top.

API
---
* ``align_fusion(audio, song_cfg) -> FusionAlignment`` — the audio-facing entry.
  Reuses ``scripts/brick0_propose.py`` (import only) for the proven, non-circular
  chart parse / chroma / transpose / constant-tempo grid, then runs the fusion
  observation + bar-pointer Viterbi + forward-backward confidence on top.
* Pure, audio-free core (what the unit tests drive), numpy-only:
  ``fused_agreement_curve`` (reliability-weighted harmony), ``bass_match_curve``,
  ``hr_match_curve``, ``fuse_emissions``, ``viterbi_bar_pointer`` (MAP alignment),
  ``forward_backward_confidence`` (posterior), ``refine_downbeat_phase``.

NON-CIRCULARITY. Every stream is raw audio (CQT chroma, percussive onset envelope,
bass CQT) or the chart — never the model's own chord decode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("harmonia.align.fusion")

# ── fusion weights + gaps (tunable; defaults keep harmony dominant so the frozen
#    ear-approved alignments reproduce — the hard gate) ─────────────────────────
_W_BASS = 0.05         # bass root/PC match weight into the section emission (small:
_W_HR = 0.05           # harmonic-rhythm weight — harmony dominates so the ear-
                       #   approved frozen alignments reproduce; 0.15 regresses
                       #   blue_bossa's chroma-flat jam. See the design-doc sweep.
_SALIENCE_FLOOR = 0.05  # min comping-salience weight per beat (den never zero)
_MIN_GAP_S = 4.0       # a legitimate inter-section gap spans >= this (a real vamp)
_GAP_OPEN = 0.60       # fixed agreement-mass a large gap must EARN to open
_GAP_COST = 0.010      # per-beat cost of a large gap (mild; matches brick0)
_MIN_FIT = 0.12        # drop a trailing placed section whose emission is ~noise
_FB_WINDOW = 3         # +/- beats each section may shift in the posterior support
_FB_TEMP = 0.08        # softmax temperature for the posterior (agreement-scale)
_CONF_FLAG = 0.35      # per-region confidence below this is FLAGGED
_METER = 4


# ═════════════════════════════════════════════════════════════════════════════
# small numpy helpers (kept local so the pure core is numpy-only)
# ═════════════════════════════════════════════════════════════════════════════

def _centre_norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise mean-centre over the 12 pcs then L2-normalise (for Pearson) —
    identical to the brick0 helper so the harmony term is byte-compatible."""
    mat = np.asarray(mat, float)
    c = mat - mat.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return c / nrm


# ═════════════════════════════════════════════════════════════════════════════
# Stream 1 — reliability-weighted harmonic agreement
# ═════════════════════════════════════════════════════════════════════════════

def fused_agreement_curve(template: np.ndarray, cn: np.ndarray,
                          salience: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-start-beat harmonic agreement of a chord-tone ``template`` (L, 12)
    against the centre-normed beat-chroma ``cn`` (N, 12), each beat WEIGHTED by
    its comping ``salience``.

    With uniform salience this is exactly brick0's ``_agreement_curve`` (mean
    per-beat Pearson) — so the frozen alignments reproduce. With a real comping
    salience (LOW in solos) a spuriously-high solo agreement no longer inflates a
    section's fit, so the DP stops being pulled off the grid in solos (the Autumn
    "cacahuète" fix) — the reliability weighting is the whole point. ``agr[p]`` =
    ``Σ_j salience[p+j]·<tcn[j], cn[p+j]>  /  Σ_j salience[p+j]``. -inf past the end.
    """
    template = np.asarray(template, float)
    cn = np.asarray(cn, float)
    L, N = len(template), len(cn)
    m = N - L + 1
    out = np.full(N, float("-inf"))
    if m <= 0:
        return out
    tcn = _centre_norm(template)
    D = tcn @ cn.T                                   # (L, N) per-beat inner products
    if salience is None:
        s = np.ones(N)
    else:
        s = np.clip(np.asarray(salience, float), _SALIENCE_FLOOR, None)
    num = np.zeros(m)
    den = np.zeros(m)
    for j in range(L):
        w = s[j:j + m]
        num += w * D[j, j:j + m]
        den += w
    out[:m] = num / np.maximum(den, 1e-9)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Stream 3 — bass root/PC match
# ═════════════════════════════════════════════════════════════════════════════

def bass_match_curve(section_bass_pc: Sequence[Optional[int]], N: int,
                     audio_bass_pc: np.ndarray, bass_rel: np.ndarray) -> np.ndarray:
    """Per-start-beat bass-root agreement: fraction of the section's beats whose
    CHART bass pitch-class matches the audio SOUNDING bass pc (``bass_salience``),
    weighted per beat by the bass reliability (walking/absent bass -> ~0 weight,
    so it self-downweights on jazz and reinforces on pop). Returns [0, 1]; -inf
    past the end. ``section_bass_pc[j]`` is the transposed chart bass pc at
    section-beat j, or ``None`` for a no-chord beat (skipped)."""
    L = len(section_bass_pc)
    m = N - L + 1
    out = np.full(N, float("-inf"))
    if m <= 0:
        return out
    apc = np.asarray(audio_bass_pc, int)
    rel = np.asarray(bass_rel, float)
    num = np.zeros(m)
    den = np.zeros(m)
    for j in range(L):
        bpc = section_bass_pc[j]
        if bpc is None:
            continue
        w = rel[j:j + m]
        match = (apc[j:j + m] == int(bpc)).astype(float)
        num += w * match
        den += w
    out[:m] = np.divide(num, den, out=np.zeros(m), where=den > 1e-9)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Stream 4 — harmonic rhythm (chart chord-changes on audio flux peaks)
# ═════════════════════════════════════════════════════════════════════════════

def hr_match_curve(change_offsets: Sequence[int], L: int, N: int,
                   flux: np.ndarray) -> np.ndarray:
    """Per-start-beat harmonic-rhythm agreement: mean audio chroma-change ``flux``
    at the beats where the section's CHART chords CHANGE. High when the chart's
    chord changes land on real audio changes (a correctly-placed section); this is
    what pins the metrical phase where comping is sparse. Returns the mean flux at
    those beats for each start p; -inf past the end. ``change_offsets`` are the
    within-section beat indices at which a new chord begins."""
    m = N - L + 1
    out = np.full(N, float("-inf"))
    if m <= 0:
        return out
    fx = np.asarray(flux, float)
    co = [c for c in change_offsets if 0 <= c < L]
    if not co:
        out[:m] = 0.0
        return out
    acc = np.zeros(m)
    for c in co:
        acc += fx[c:c + m]
    out[:m] = acc / len(co)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Fuse the streams into one per-section emission curve
# ═════════════════════════════════════════════════════════════════════════════

def fuse_emissions(harm: np.ndarray, bass: Optional[np.ndarray] = None,
                   hr: Optional[np.ndarray] = None, *, w_bass: float = _W_BASS,
                   w_hr: float = _W_HR) -> np.ndarray:
    """Fuse the per-start-beat stream curves into ONE section-emission curve:
    ``harm + w_bass·bass + w_hr·hr`` (streams absent past the grid stay -inf).
    Harmony is the base; bass/hr are corroborators with small default weights so
    a strong harmonic fit is never overturned (frozen-reproduction gate), but they
    break ties and add signal where harmony is weak."""
    out = np.array(harm, float)
    finite = np.isfinite(out)
    if bass is not None:
        b = np.asarray(bass, float)
        out[finite] += w_bass * np.where(np.isfinite(b[finite]), b[finite], 0.0)
    if hr is not None:
        h = np.asarray(hr, float)
        out[finite] += w_hr * np.where(np.isfinite(h[finite]), h[finite], 0.0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# The bar-pointer state-space — Viterbi MAP over (section-position, gap)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Placement:
    """A placed chart section on the constant-tempo lattice."""
    chorus: int
    sec_idx: int
    label: str
    start_beat: int
    n_beats: int
    emission: float           # fused emission score at the chosen start
    harm: float = 0.0         # harmony-only agreement (for the reproduction diff)
    offset_s: float = 0.0     # sub-beat refinement (filled downstream)
    occ: int = 0
    confidence: float = 0.0   # blended self-detection confidence (fit x posterior)
    posterior: float = 0.0    # raw forward-backward posterior mass at this placement
    fit: float = 0.0          # per-song-normalised harmonic agreement (confirmability)
    reliability: float = 0.0  # mean fused reliability over the span


def _min_gap_beats(period: float) -> int:
    return max(2, int(round(_MIN_GAP_S / period))) if period > 0 else 2


def viterbi_bar_pointer(order: list, emissions: list, N: int, p0: int,
                        min_gap_beats: int, gap_open: float = _GAP_OPEN,
                        gap_cost: float = _GAP_COST) -> list[Placement]:
    """Bar-pointer Viterbi MAP alignment (the state-space, not a threshold stack).

    ``order`` is the tiled chart form: a list of ``(chorus, sec_idx, section)``;
    ``emissions[i]`` is the fused per-start-beat emission curve for ``order[i]``.
    The latent chain places the sections IN FORM ORDER on the beat lattice; between
    two consecutive sections the transition is EITHER contiguous (gap == 0, the
    constant-tempo default) OR a single large VAMP gap (>= ``min_gap_beats``, which
    must EARN ``gap_open`` of agreement to open AND pay ``gap_cost`` per skipped
    beat — small catch-up gaps are structurally impossible). The first section is
    pinned to the intro-skip anchor ``p0``. This transition discipline + the
    per-beat gap cost are byte-identical to the brick0 min-gap DP, so with a
    harmony-only emission a reliable-harmony song reproduces exactly; the fusion
    changes the EMISSION, not the transition topology. Returns the MAP placements
    (tail dropped)."""
    M = len(order)
    if M == 0 or N <= 0:
        return []
    U = np.zeros(N + 1)                              # V(M, *) = 0 (drop tail)
    choice: list = [None] * M
    qidx = np.arange(N)
    # cumulative per-beat gap cost: C[p]-C[q] = cost to skip beats [q, p)
    C = np.concatenate([[0.0], np.cumsum(np.full(N, gap_cost))])
    for i in range(M - 1, -1, -1):
        _c, _si, s = order[i]
        L = s.n_beats
        agr = emissions[i]
        m = N - L + 1
        Bp = np.full(N, -1e9)                        # value of placing i at p (+ rest)
        if m > 0:
            f = np.where(np.isfinite(agr[:m]), agr[:m], -1e9)
            Bp[:m] = np.where(f > -8, f + U[L:L + m], -1e9)
        # contiguous branch: start == q
        cont = np.where(qidx < m, Bp, -1e9)
        # large-gap branch: start >= q + min_gap_beats; pays gap_cost/beat + gap_open
        Sp = Bp - C[:N]
        suffS, sargS = _suffix_max_arg(Sp)
        kk = np.minimum(qidx + min_gap_beats, N)
        large = C[qidx] + suffS[kk] - gap_open
        large_arg = sargS[kk]
        stack = np.vstack([np.zeros(N), cont, large])   # drop, contiguous, large-gap
        sel = np.argmax(stack, axis=0)
        Vi = stack[sel, qidx]
        pstar = np.where(sel == 1, qidx, np.where(sel == 2, large_arg, -1))
        choice[i] = pstar
        U = np.concatenate([Vi, [0.0]])
    placements: list[Placement] = []
    q = p0
    for i in range(M):
        c, si, s = order[i]
        pstar = p0 if i == 0 else (int(choice[i][q]) if q < N else -1)
        if pstar < 0 or pstar + s.n_beats > N:
            break
        placements.append(Placement(c, si, s.label, pstar, s.n_beats,
                                    float(emissions[i][pstar])))
        q = pstar + s.n_beats
        if q >= N - 1:
            break
    while placements and placements[-1].emission < _MIN_FIT:
        placements.pop()
    return placements


def _suffix_max_arg(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(suffmax, sargmax) length N+1: suffmax[p]=max_{p'>=p} a[p'], sargmax[p] the
    SMALLEST such p' (left tie-break); index N is the empty-suffix sentinel."""
    N = len(a)
    suff = np.empty(N + 1)
    suff[N] = -1e9
    if N:
        suff[:N] = np.maximum.accumulate(a[::-1])[::-1]
    sarg = np.full(N + 1, -1, dtype=int)
    if N:
        records = np.nonzero(a >= suff[:N])[0]
        idx = np.searchsorted(records, np.arange(N), side="left")
        ok = idx < len(records)
        sarg[:N][ok] = records[idx[ok]]
    return suff, sarg


# ═════════════════════════════════════════════════════════════════════════════
# Forward-backward posterior confidence (self-detection, for free)
# ═════════════════════════════════════════════════════════════════════════════

def forward_backward_confidence(placements: list[Placement], emissions_by_step: dict,
                                N: int, *, window: int = _FB_WINDOW,
                                temp: float = _FB_TEMP) -> list[float]:
    """Forward-backward posterior over the PLACED chain -> a per-section confidence.

    Each placed section may shift within ``+/- window`` beats of its MAP start; the
    emission gives the local potential (softmax at temperature ``temp``), and a
    soft contiguity transition couples neighbours (contiguous shift preferred, a
    large gap allowed). The marginal posterior ``gamma_i(p)`` is computed by a
    standard forward-backward; the confidence of section i is ``gamma_i`` at its
    MAP start — HIGH when the emission is peaked and the neighbours agree (a
    well-determined placement), LOW when the emission is flat (a blind / ambiguous
    region, e.g. a self-similar section or a solo). This IS the self-detection
    signal: a flat posterior means "I don't know where this goes."
    """
    M = len(placements)
    if M == 0:
        return []
    # support: window of shifts d in [-w, w] around each MAP start
    shifts = np.arange(-window, window + 1)
    S = len(shifts)
    # local emission potentials phi[i, d] = softmax over the window
    phi = np.zeros((M, S))
    for i, pl in enumerate(placements):
        em = emissions_by_step[i]
        vals = np.array([em[pl.start_beat + d]
                         if 0 <= pl.start_beat + d < len(em)
                         and np.isfinite(em[pl.start_beat + d]) else -1e9
                         for d in shifts])
        vmax = vals.max()
        phi[i] = np.exp((vals - vmax) / max(temp, 1e-6))
        phi[i] /= phi[i].sum() if phi[i].sum() > 0 else 1.0
    # transition psi[d, d']: prefer preserving the contiguous chain (d' == d), i.e.
    # a section shifting keeps its successor's relative spacing; deviation is
    # penalised by a Gaussian on |d' - d| (soft constant-tempo continuity).
    dd = shifts[None, :] - shifts[:, None]
    psi = np.exp(-0.5 * (dd / 1.5) ** 2)
    psi /= psi.sum(axis=1, keepdims=True)
    # forward
    alpha = np.zeros((M, S))
    alpha[0] = phi[0]
    alpha[0] /= alpha[0].sum() or 1.0
    for i in range(1, M):
        alpha[i] = phi[i] * (alpha[i - 1] @ psi)
        z = alpha[i].sum()
        alpha[i] /= z or 1.0
    # backward
    beta = np.zeros((M, S))
    beta[-1] = 1.0
    for i in range(M - 2, -1, -1):
        beta[i] = (psi @ (phi[i + 1] * beta[i + 1]))
        z = beta[i].sum()
        beta[i] /= z or 1.0
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-12)
    # confidence = posterior mass at the MAP shift (d == 0)
    zero = np.where(shifts == 0)[0][0]
    return [float(gamma[i, zero]) for i in range(M)]


# ═════════════════════════════════════════════════════════════════════════════
# Downbeat phase — form-refined, anticipation-aware
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DownbeatPhase:
    """Global downbeat phase after form-refinement (see module docstring)."""
    phase: int                    # the CHOSEN global phase (form-preferred)
    form_phase: int               # phase implied by the aligned chart bar-1s
    acoustic_phase: int           # phase from the acoustic global-phase resolver
    confidence: float
    anticipation: bool            # acoustic is 1 beat off form = the pop "push"
    flagged: bool                 # half-bar disagreement / weak evidence
    meter: int = _METER
    notes: list = field(default_factory=list)


def refine_downbeat_phase(form_phase: int, acoustic_phase: int,
                          acoustic_conf: float, meter: int = _METER
                          ) -> DownbeatPhase:
    """Reconcile the FORM phase (the aligned chart's bar-1 positions) with the
    ACOUSTIC global-phase resolver, anticipation-aware.

    Rules (precision-first):
      * agree                -> confident, phase = form.
      * differ by ONE beat   -> the pop "push": the chord change anticipated the
                                downbeat and fooled the acoustic harmonic-rhythm
                                evidence (Every Breath). PREFER the FORM phase;
                                mark ``anticipation``; keep a moderate confidence.
      * differ by a half-bar -> genuine ambiguity (which of the strong pair is 1);
                                FLAG. Keep the form phase as the best guess.
    """
    form_phase %= meter
    acoustic_phase %= meter
    diff = (acoustic_phase - form_phase) % meter
    notes: list[str] = []
    if diff == 0:
        return DownbeatPhase(form_phase, form_phase, acoustic_phase,
                             float(np.clip(0.5 + 0.5 * acoustic_conf, 0, 1)),
                             False, False, meter,
                             ["form and acoustic phase agree"])
    if diff == 1 or diff == meter - 1:
        notes.append(f"acoustic phase {acoustic_phase} is 1 beat off form "
                     f"{form_phase} — chord-change ANTICIPATION (pop push); "
                     f"form phase preferred")
        return DownbeatPhase(form_phase, form_phase, acoustic_phase,
                             float(np.clip(0.45 * (0.5 + acoustic_conf), 0, 1)),
                             True, False, meter, notes)
    notes.append(f"acoustic phase {acoustic_phase} disagrees with form "
                 f"{form_phase} by a half-bar — ambiguous downbeat, FLAGGED")
    return DownbeatPhase(form_phase, form_phase, acoustic_phase,
                         float(np.clip(0.2 * acoustic_conf, 0, 1)),
                         False, True, meter, notes)


# ═════════════════════════════════════════════════════════════════════════════
# The audio-facing entry (reuses brick0 for the proven non-circular front-end)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionAlignment:
    """Output of the fusion aligner."""
    song_id: str
    transpose: int
    period: float
    phase: float
    beats: np.ndarray                     # constant-tempo lattice (s)
    placements: list[Placement]
    downbeat: DownbeatPhase
    gt_chords: list[dict]
    whole_song_confidence: float
    low_confidence_regions: list[dict]
    diag: dict = field(default_factory=dict)


def _brick0():
    """Lazy import of the brick0 propose script (non-circular helpers only)."""
    import importlib.util
    import sys
    from pathlib import Path
    if "brick0_propose" in sys.modules:
        return sys.modules["brick0_propose"]
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "brick0_propose", repo / "scripts" / "brick0_propose.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["brick0_propose"] = m
    spec.loader.exec_module(m)
    return m


def _section_bass_and_changes(sec, transpose: int):
    """Per-section-beat transposed bass pc (None for no-chord) + the within-section
    beat offsets at which a new chord begins (harmonic-rhythm change beats)."""
    bc = sec.beat_chords
    bass = []
    changes = []
    prev = object()
    for j, c in enumerate(bc):
        if c is None:
            bass.append(None)
        else:
            bass.append((c.bass_pc + transpose) % 12)
        key = None if c is None else (c.root_pc, c.quality, c.bass_pc, c.bass_name)
        if key is not None and key != prev:
            changes.append(j)
        prev = key
    return bass, changes


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b — DRIFTING-τ state (bounded lattice warp) + form-vamp propagation
#
# The v1 lattice was a single constant tempo (brick0's fine-tuned grid); within-song
# tempo DRIFT was the documented remainder. Stage 2b adds it back as a STATE of the
# bar-pointer model: a BOUNDED, strongly-regularised warp of the beat lattice (the
# per-window τ adjustment), NOT a free per-section warp. We reuse brick0's proven,
# non-circular drift DETECTION (offset-ramp → windowed/whole-song classification →
# piecewise-linear knots) to PROPOSE the warp, then RE-PLACE the FUSION DP on the
# warped lattice and accept the warp only behind brick0's calibrated agreement self-
# check (coverage-weighted harmonic agreement must rise). Symmetrically, form-periodic
# VAMPS (Autumn's turnaround) are reused from brick0's ``propagate_form_vamps`` (large
# pause-gaps only, its own agreement self-check) with the schedule re-tiled onto the
# FUSION emission. Both refinements can only IMPROVE agreement or REVERT — a clean
# constant-tempo song is never warped (the STOP-rule: an overfit drift is worse than
# none). See ``docs/fusion_aligner_design.md`` (2026-07-24 checkpoint).
# ═════════════════════════════════════════════════════════════════════════════

def _grid_streams(grid, *, frames, ftimes, bass_ch, beat_period, transpose,
                  sections, salience_weighting, w_bass, w_hr, b0, bassmod, dbmod):
    """Recompute the four streams + the per-section fused-emission and harmony-only
    agreement curves on a (possibly warped) beat ``grid``. Returns
    ``(emit_by_si, harm_by_si, salience, aux)`` where the curves are indexed by
    section (identical across choruses, so tiled by ``_tile_order``) and ``aux``
    carries the bass/flux streams for reuse. This is the v1 emission block factored
    out so the drift stage can rebuild it on each candidate warped grid."""
    N = len(grid)
    cn = b0._centre_norm(b0.beat_sync_chroma(frames, ftimes, grid))
    salience = (np.clip(b0._chart_fill(sections, cn), 0.0, None)
                if salience_weighting else None)
    bounds = np.concatenate([grid, [grid[-1] + beat_period]])
    audio_bass_pc, bass_rel = bassmod.bass_pc_series(bass_ch, bounds)
    flux = dbmod.harmonic_change_evidence(frames, ftimes, grid)
    emit_by_si: list = []
    harm_by_si: list = []
    for s in sections:
        harm = fused_agreement_curve(s.template, cn, salience)
        sec_bass, changes = _section_bass_and_changes(s, transpose)
        bcurve = bass_match_curve(sec_bass, N, audio_bass_pc, bass_rel)
        hcurve = hr_match_curve(changes, s.n_beats, N, flux)
        emit_by_si.append(fuse_emissions(harm, bcurve, hcurve, w_bass=w_bass, w_hr=w_hr))
        harm_by_si.append(harm)
    aux = dict(cn=cn, flux=flux, audio_bass_pc=audio_bass_pc, bass_rel=bass_rel)
    return emit_by_si, harm_by_si, salience, aux


def _tile_order(sections, emit_by_si, N):
    """Tile the chart form across the lattice -> (order, emissions) for the DP; the
    same section's fused-emission curve is reused for every chorus."""
    chorus_beats = sum(s.n_beats for s in sections) or 1
    maxchor = max(1, int(np.ceil(N / chorus_beats)) + 1)
    order: list = []
    emissions: list = []
    for c in range(maxchor):
        for si, s in enumerate(sections):
            order.append((c, si, s))
            emissions.append(emit_by_si[si])
    return order, emissions


def _attach_harm(placements, harm_by_si):
    """Attach each placement's harmony-only agreement (for the reproduction diff, the
    vamp seed/tail-trim, and the confidence fit)."""
    for pl in placements:
        h = harm_by_si[pl.sec_idx]
        v = h[pl.start_beat] if 0 <= pl.start_beat < len(h) else float("nan")
        pl.harm = float(v) if np.isfinite(v) else 0.0


def _place_fusion(sections, emit_by_si, harm_by_si, beats, beat_period, seed_s, b0):
    """Fusion-DP placement on a lattice: tile the form, run the bar-pointer Viterbi
    MAP, attach the harmony-only agreement. Returns ``(placements, p0, start_diag)``."""
    N = len(beats)
    order, emissions = _tile_order(sections, emit_by_si, N)
    p0, start_diag = b0._first_section_start(harm_by_si[0], beats, seed_s)
    placements = viterbi_bar_pointer(order, emissions, N, p0, _min_gap_beats(beat_period))
    _attach_harm(placements, harm_by_si)
    return placements, p0, start_diag


def _place_from_starts(sections, emit_by_si, harm_by_si, starts, N):
    """Tile each chorus contiguously from its start beat with the FUSION emission —
    the vamp schedule sets the per-chorus ``starts``; emission/harm come from the
    fused streams. Mirrors brick0's ``_placements_from_starts`` but keeps the fusion
    emission as the score field (tail ~noise placements dropped, like brick0)."""
    pls: list[Placement] = []
    for c, st in enumerate(starts):
        bb = st
        for si, s in enumerate(sections):
            if bb + s.n_beats > N:
                break
            em, hh = emit_by_si[si], harm_by_si[si]
            e = float(em[bb]) if 0 <= bb < len(em) and np.isfinite(em[bb]) else 0.0
            pl = Placement(c, si, s.label, bb, s.n_beats, e)
            v = hh[bb] if 0 <= bb < len(hh) and np.isfinite(hh[bb]) else 0.0
            pl.harm = float(v)
            pls.append(pl)
            bb += s.n_beats
    while pls and pls[-1].harm < _MIN_FIT:
        pls.pop()
    return pls


def _cov_agr(placements, sections, frames, ftimes, beats, beat_period, b0):
    """Coverage-weighted whole-song harmonic agreement (brick0's calibrated, UNWEIGHTED
    self-check objective) — the gate that accepts a warp/vamp only if it genuinely
    improves alignment. Unweighted on purpose: a salience-weighted objective is biased
    toward warping (the drift ramp maximises local agreement) and falsely accepts a
    clean song's spurious warp."""
    return float(b0._whole_song_agreement(placements, sections, frames, ftimes,
                                          beats, beat_period, 0.0))


def _form_vamp_stage(placements, sections, emit_by_si, harm_by_si, frames, ftimes,
                     beats, beat_period, base_agr, b0):
    """STAGE 2b (A) — form-periodic VAMP propagation as discrete large-gap transitions.

    Reuse brick0's ``propagate_form_vamps`` (seed the vamp from the one large gap the
    bar-pointer DP already opened, propagate it after each chorus, snap to the low-
    agreement turnaround; kept only behind its own coverage-weighted agreement self-
    check). The schedule (per-chorus start beats) is then re-tiled onto the FUSION
    emission. Only LARGE pause-gaps (>= a real multi-second vamp); no sub-second
    catch-up gaps. Returns ``(placements, report)`` — inputs unchanged if not accepted."""
    N = len(beats)
    fv_pl, _fv_diag, report = b0.propagate_form_vamps(
        sections, frames, ftimes, beats, beat_period, placements,
        {"song_score": base_agr})
    if not report.get("accepted"):
        return placements, report
    starts = sorted({p.start_beat for p in fv_pl if p.sec_idx == 0})
    new_pl = _place_from_starts(sections, emit_by_si, harm_by_si, starts, N)
    return new_pl, report


def _drift_stage(placements, sections, beats, beat_period, base_agr, *, frames,
                 ftimes, bass_ch, transpose, salience_weighting, w_bass, w_hr,
                 seed_s, b0, bassmod, dbmod):
    """STAGE 2b (B) — the DRIFTING-τ state (a bounded, regularised lattice warp).

    Reuse brick0's drift DETECTION (offset-ramp -> whole-song <=quadratic AND windowed
    piecewise-linear classification) to PROPOSE candidate warped lattices, RE-PLACE the
    FUSION DP on each, and accept the best only if the coverage-weighted harmonic
    agreement rises by >= ``_WIN_ACCEPT_EPS`` (brick0's calibrated self-check). The warp
    is bounded to the classified drift/flat spans (constant-tempo prior + a tracked
    drift, never a free per-section warp), so a constant/rubato song is never warped.
    Returns ``(placements, beats, emit_by_si, harm_by_si, report)``; on reject the
    inputs' grid is kept and the caller keeps its existing curves."""
    ramp_rows, per_ch = b0.offset_ramp(placements, sections, frames, ftimes,
                                       beats, beat_period)
    ws = b0.detect_drift(ramp_rows, beat_period)
    wd = b0.detect_windowed_drift(ramp_rows, beat_period)
    report = dict(classification=wd.get("classification"), windowed=bool(wd.get("windowed")),
                  granularity="chorus" if per_ch else "section", ws_apply=bool(ws["apply"]),
                  wd_apply=bool(wd["apply"]), accepted=False, model=None,
                  pre_agr=round(float(base_agr), 4), candidates=[])
    cands: list = []
    if ws["apply"]:
        cands.append(("wholesong", b0.drift_grid(beats, ws["coeffs"])))
    if wd["apply"]:
        cands.append(("windowed" if wd.get("windowed") else "windowed1",
                      b0.windowed_drift_grid(beats, wd["knots"])))
    best = None                              # (model, wbeats, w_pl, agr, emit, harm)
    for name, wbeats in cands:
        emit_w, harm_w, _sal, _aux = _grid_streams(
            wbeats, frames=frames, ftimes=ftimes, bass_ch=bass_ch,
            beat_period=beat_period, transpose=transpose, sections=sections,
            salience_weighting=salience_weighting, w_bass=w_bass, w_hr=w_hr,
            b0=b0, bassmod=bassmod, dbmod=dbmod)
        w_pl, _p0, _sd = _place_fusion(sections, emit_w, harm_w, wbeats,
                                       beat_period, seed_s, b0)
        agr = _cov_agr(w_pl, sections, frames, ftimes, wbeats, beat_period, b0)
        report["candidates"].append(dict(model=name, agr=round(float(agr), 4),
                                          delta=round(float(agr - base_agr), 4)))
        if best is None or agr > best[3]:
            best = (name, wbeats, w_pl, agr, emit_w, harm_w)
    if best is not None and best[3] > base_agr + b0._WIN_ACCEPT_EPS:
        report.update(accepted=True, model=best[0], post_agr=round(float(best[3]), 4),
                      delta=round(float(best[3] - base_agr), 4))
        return best[2], best[1], best[4], best[5], report
    return placements, beats, None, None, report


def align_fusion(audio, song_cfg: dict, *, features: Optional[dict] = None,
                 w_bass: float = _W_BASS, w_hr: float = _W_HR,
                 salience_weighting: bool = True) -> FusionAlignment:
    """Align a chart to audio with the fusion DBN.

    ``song_cfg`` is a brick0 BATCH1-style dict (``song_id``, ``ireal_file``,
    ``tune_title``, optional ``human_anchor``). ``features`` may pre-supply the
    expensive extracts (``frames``, ``ftimes``, ``beats``, ``downbeats``,
    ``beat_period``, ``drum_track``, ``bass_chroma``, ``dur``) to skip
    re-extraction; otherwise they are computed via brick0 + the align instruments.
    Returns a ``FusionAlignment`` with the placements, the form-refined downbeat
    phase, the GT chord timeline, and the posterior confidence.
    """
    from harmonia.align import bass_salience as bassmod
    from harmonia.align import downbeat as dbmod
    from harmonia.align import drum_pattern as drummod
    b0 = _brick0()

    song_id = song_cfg["song_id"]
    seed_s = song_cfg.get("human_anchor")

    # ── front-end (proven, non-circular): chart parse, grid, transpose ──────────
    chart = b0.parse_chart(song_cfg["ireal_file"], song_cfg["tune_title"])
    if features is None:
        features = extract_features(audio, song_cfg)
    frames, ftimes = features["frames"], features["ftimes"]
    beats_bt = features["beats"]
    beat_period0 = features["beat_period"]
    dur = features["dur"]
    drum_track = features["drum_track"]
    bass_ch = features["bass_chroma"]

    mean_chroma = frames.mean(axis=0)
    ssum = mean_chroma.sum()
    mean_chroma = mean_chroma / ssum if ssum else mean_chroma

    # transpose on the coarse grid (a pitch decision, ~independent of the octave)
    coarse_phase = b0._phase_base(seed_s, beat_period0, beats_bt)
    coarse_grid = b0._const_grid(beat_period0, coarse_phase, dur)
    cn0 = b0._centre_norm(b0.beat_sync_chroma(frames, ftimes, coarse_grid))
    tr, _pl0, _d0 = b0.propose_transpose(chart, cn0, coarse_grid, mean_chroma, seed_s)
    transpose = tr["transpose_semitones"]

    # joint tempo-octave + fine tempo -> the constant-tempo lattice (brick0's grid,
    # so the ear-approved timelines reproduce)
    grids = b0.tempo_grid_hypotheses(beats_bt, beat_period0)
    octave, beat_period, phase, tempo_report, octave_scores = b0.select_octave_and_tempo(
        chart, transpose, frames, ftimes, grids, seed_s, dur)
    beats = b0._const_grid(beat_period, phase, dur)             # (N,) lattice
    N = len(beats)
    cn = b0._centre_norm(b0.beat_sync_chroma(frames, ftimes, beats))
    sections = b0.deconstruct_sections(chart, transpose)

    # ── the four per-lattice-beat streams + per-section fused-emission curves ────
    streams_kw = dict(frames=frames, ftimes=ftimes, bass_ch=bass_ch,
                      beat_period=beat_period, transpose=transpose, sections=sections,
                      salience_weighting=salience_weighting, w_bass=w_bass, w_hr=w_hr,
                      b0=b0, bassmod=bassmod, dbmod=dbmod)
    emit_by_si, harm_by_si, salience, aux = _grid_streams(beats, **streams_kw)
    mgb = _min_gap_beats(beat_period)

    # ── v1 placement: fusion bar-pointer Viterbi on the constant-tempo lattice ──
    placements, p0, start_diag = _place_fusion(sections, emit_by_si, harm_by_si,
                                               beats, beat_period, seed_s, b0)

    # ── ONE global sub-beat phase refinement (v3, reused) BEFORE drift/vamp: slide
    #    the whole rigid lattice by a single offset that maximises coverage-weighted
    #    agreement. A global phase slide (never a per-section warp), so the placement
    #    indices are unchanged; we refresh the emission curves on the shifted grid so
    #    the vamp seed + drift ramp measure agreement on the final lattice (matches
    #    brick0's phase-then-rescore order).
    dphase, _pa = b0.refine_global_phase(placements, sections, frames, ftimes,
                                         beats, beat_period)
    if abs(dphase) > 1e-6:
        beats = beats + dphase
        phase = (phase + dphase) % beat_period
        emit_by_si, harm_by_si, salience, aux = _grid_streams(beats, **streams_kw)
        _attach_harm(placements, harm_by_si)

    # ── STAGE 2b — subsume brick0's drift/vamp inside the model (each behind the
    #    calibrated agreement self-check; a clean constant-tempo song is never warped)
    base_agr = _cov_agr(placements, sections, frames, ftimes, beats, beat_period, b0)
    #    (A) form-periodic VAMPS as discrete large-gap transitions (Autumn's turnaround)
    placements, vamp_report = _form_vamp_stage(
        placements, sections, emit_by_si, harm_by_si, frames, ftimes, beats,
        beat_period, base_agr, b0)
    if vamp_report.get("accepted"):
        base_agr = _cov_agr(placements, sections, frames, ftimes, beats, beat_period, b0)
    #    (B) the DRIFTING-τ state: a bounded windowed/whole-song lattice warp
    placements, beats, warp_emit, warp_harm, drift_report = _drift_stage(
        placements, sections, beats, beat_period, base_agr, frames=frames,
        ftimes=ftimes, bass_ch=bass_ch, transpose=transpose,
        salience_weighting=salience_weighting, w_bass=w_bass, w_hr=w_hr,
        seed_s=seed_s, b0=b0, bassmod=bassmod, dbmod=dbmod)
    if drift_report.get("accepted"):
        emit_by_si, harm_by_si = warp_emit, warp_harm     # curves on the warped grid
        aux["flux"] = dbmod.harmonic_change_evidence(frames, ftimes, beats)
    N = len(beats)
    flux = aux["flux"]
    _attach_harm(placements, harm_by_si)
    _assign_occurrences(placements, sections)

    # ── attach fused reliability per placement (final lattice) ──────────────────
    drum_rel = drum_track.reliability_at(beats)
    for pl in placements:
        sl = slice(pl.start_beat, pl.start_beat + pl.n_beats)
        rels = []
        if salience is not None:
            rels.append(float(np.mean(salience[sl])))
        rels.append(float(np.mean(drum_rel[sl])) if pl.start_beat < N else 0.0)
        pl.reliability = float(np.mean(rels)) if rels else 0.0

    # ── posterior confidence per section = CONFIRMABILITY x placement SHARPNESS ──
    # The forward-backward posterior alone conflates a slow harmonic rhythm (a
    # locally flat emission on a CORRECT clean-pop placement) with a genuinely
    # uncertain one. So we blend it with the per-song-normalised harmonic FIT: how
    # well the chart CONFIRMS this placement acoustically, relative to THIS song's
    # own best-aligned section (the ceiling). A solo (harmony dead) or a flat jam
    # then reads LOW (correct self-detection: "placed by the form prior, cannot
    # confirm"); a clean pop chorus reads HIGH.
    emissions_by_step = {i: emit_by_si[pl.sec_idx] for i, pl in enumerate(placements)}
    posteriors = forward_backward_confidence(placements, emissions_by_step, N)
    harms = np.array([max(p.harm, 0.0) for p in placements]) if placements else np.array([0.0])
    ceiling = float(np.percentile(harms, 90)) if len(harms) >= 5 else float(harms.max())
    ceiling = max(ceiling, 1e-3)
    for pl, post in zip(placements, posteriors):
        pl.posterior = float(post)
        pl.fit = float(np.clip(max(pl.harm, 0.0) / ceiling, 0.0, 1.0))
        # fit is the base (confirmability); the posterior sharpens it (unambiguity)
        pl.confidence = float(np.clip(pl.fit * (0.55 + 0.45 * post), 0.0, 1.0))

    # ── downbeat phase (form-refined, anticipation-aware) ───────────────────────
    downbeat = _resolve_downbeat_phase(placements, beats, drum_track, frames,
                                       ftimes, bass_ch, dbmod, flux, mgb, beat_period)

    # ── GT timeline (reuse brick0's builder; overrides are downstream) ──────────
    gt_chords = b0.build_gt_chords(placements, sections, beats, beat_period,
                                   transpose, frames, ftimes, dur)

    # ── whole-song confidence + low-confidence regions ──────────────────────────
    whole_conf, low_regions = _aggregate_confidence(
        placements, beats, beat_period, downbeat, gt_chords)

    diag = dict(octave=octave, octave_scores=octave_scores, n_placed=len(placements),
                coverage=round(sum(p.n_beats for p in placements) / N, 3) if N else 0,
                mean_harm=round(float(np.mean([p.harm for p in placements])), 4)
                if placements else 0.0,
                mean_reliability=round(float(np.mean([p.reliability for p in placements])), 4)
                if placements else 0.0,
                start=start_diag, transpose_evidence=tr.get("evidence"),
                w_bass=w_bass, w_hr=w_hr, salience_weighting=salience_weighting,
                vamp=vamp_report, drift=drift_report)

    return FusionAlignment(
        song_id=song_id, transpose=transpose, period=beat_period, phase=phase,
        beats=beats, placements=placements, downbeat=downbeat, gt_chords=gt_chords,
        whole_song_confidence=whole_conf, low_confidence_regions=low_regions, diag=diag)


def _assign_occurrences(placements: list[Placement], sections: list) -> None:
    seen: dict = {}
    for pl in placements:
        ck = sections[pl.sec_idx].content_key
        pl.occ = seen.get(ck, 0)
        seen[ck] = pl.occ + 1


def _resolve_downbeat_phase(placements, beats, drum_track, frames, ftimes,
                            bass_ch, dbmod, flux, mgb, beat_period) -> DownbeatPhase:
    """Compute the acoustic global phase on THIS lattice + the form phase from the
    placement, and reconcile them (anticipation-aware)."""
    N = len(beats)
    meter = _METER
    if not placements:
        return DownbeatPhase(0, 0, 0, 0.0, False, True, meter, ["no placements"])
    form_phase = placements[0].start_beat % meter
    # acoustic evidence on this lattice: harmonic-rhythm + bass root-on-1
    raw_bass, bass_rel = dbmod.bass_beat_evidence(bass_ch, beats)
    bass_ev = dbmod._peak_pick(raw_bass)
    bass_gate = float(np.mean(bass_rel)) if len(bass_rel) else 0.0
    strong_mask = None
    sm = getattr(drum_track, "strong_beat_mask", None)
    bt = getattr(drum_track, "beat_times", None)
    if sm is not None and bt is not None and len(sm) == len(bt) and len(bt):
        idx = np.clip(np.searchsorted(bt, beats), 1, len(bt) - 1)
        left = idx - 1
        pick = np.where(np.abs(bt[idx] - beats) < np.abs(bt[left] - beats), idx, left)
        strong_mask = np.asarray(sm)[pick]
    strong_conf = float(getattr(drum_track, "strong_beat_confidence", 0.0))
    # chart bar-1 hits from the placement (structural prior)
    chart_hits = np.zeros(N)
    for pl in placements:
        for b in range(0, pl.n_beats, meter):
            j = pl.start_beat + b
            if 0 <= j < N:
                chart_hits[j] += 1.0
    res = dbmod.resolve_phase(beats, meter, harm=flux, bass=bass_ev,
                              bass_weight=bass_gate, chart_hits=chart_hits,
                              strong_mask=strong_mask, strong_conf=strong_conf)
    return refine_downbeat_phase(form_phase, res.phase_offset, res.confidence, meter)


_ABS_AGR_REF = 0.45     # absolute harmonic agreement that reads as "fully confirmed"


def _aggregate_confidence(placements, beats, beat_period, downbeat, gt_chords):
    """Whole-song confidence + the low-confidence region list.

    Whole-song confidence pools four signals: the per-section posterior confidence
    (WITHIN-song, per-song-normalised — for locating a wrong region), the ABSOLUTE
    harmonic agreement (CROSS-song — a chroma-flat jam like Blue Bossa reads low
    even though its own best sections look locally fine), the fused reliability, and
    the downbeat confidence. The absolute term is what stops a globally-ambiguous
    song from looking confident just because it is internally uniform. A region is
    FLAGGED when its blended confidence OR its fused reliability is low."""
    if not placements:
        return 0.0, []
    w = np.array([p.n_beats for p in placements], float)
    conf = np.array([p.confidence for p in placements], float)
    rel = np.array([p.reliability for p in placements], float)
    abs_agr = np.array([max(p.harm, 0.0) for p in placements], float)
    sec_conf = float(np.average(conf, weights=w))
    rel_mean = float(np.average(rel, weights=w))
    abs_conf = float(np.clip(np.average(abs_agr, weights=w) / _ABS_AGR_REF, 0.0, 1.0))
    whole = float(np.clip(0.35 * sec_conf + 0.25 * abs_conf + 0.20 * rel_mean
                          + 0.20 * downbeat.confidence, 0.0, 1.0))
    low = []
    for pl in placements:
        if pl.confidence < _CONF_FLAG or pl.reliability < 0.10:
            t0 = float(beats[pl.start_beat]) if pl.start_beat < len(beats) else 0.0
            end = pl.start_beat + pl.n_beats
            t1 = float(beats[end]) if end < len(beats) else (
                float(beats[-1]) + (end - len(beats) + 1) * beat_period)
            low.append(dict(label=f"{pl.label}{pl.occ + 1}", t0=round(t0, 2),
                            t1=round(t1, 2), confidence=round(pl.confidence, 3),
                            reliability=round(pl.reliability, 3),
                            reason="flat posterior" if pl.confidence < _CONF_FLAG
                            else "low reliability"))
    return whole, low


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction (expensive; cache-friendly) — reuses brick0 + the instruments
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(audio, song_cfg: dict) -> dict:
    """Extract the expensive per-song features once (chroma, Beat This! grid, drum
    track, bass chroma). Reuses brick0's non-circular loaders + the align modules.
    ``audio`` is a path to the source m4a/mp3/wav."""
    import tempfile
    from pathlib import Path
    from harmonia.align import bass_salience as bassmod
    from harmonia.align import drum_pattern as drummod
    b0 = _brick0()

    audio = Path(audio)
    workdir = Path(tempfile.mkdtemp(prefix="fusion_"))
    wav = b0.decode_wav(audio, workdir)
    dur = b0.audio_duration(audio)
    bt = b0.beat_this_full(wav)
    frames, ftimes = b0.load_chroma_frames(wav)
    drum_track = drummod.track_from_audio(str(wav), beat_this_beats=bt["beats"],
                                          run_beat_this=False)
    bass_ch = bassmod.bass_chroma(str(wav))
    return dict(frames=frames, ftimes=ftimes, beats=bt["beats"],
                downbeats=bt["downbeats"], beat_period=bt["beat_period"],
                dur=dur, drum_track=drum_track, bass_chroma=bass_ch,
                wav=str(wav))
