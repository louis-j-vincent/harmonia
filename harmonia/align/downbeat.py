"""The GLOBAL-PHASE downbeat resolver of the Bayesian fusion aligner.

DESIGN (Louis's simplification, 2026-07-23 — see ``docs/fusion_aligner_design.md``
"DOWNBEAT = ONE GLOBAL PHASE"). Meter is constant 4/4 and the BEAT grid is
already locked (``harmonia.align.drum_pattern``), so the downbeat is NOT a
per-bar detection problem. It is a SINGLE discrete global phase offset
``phi in {0..meter-1}`` — which beat of the metrical cycle is beat 1 — held for
the whole song. We pick the ONE phase that maximises AGGREGATE downbeat evidence
pooled over every bar. A per-bar signal that is only ~53% becomes a STRONG global
estimate aggregated over ~100 bars.

The four premise-checks (all in) settled which evidence, and its ranking:

  1. **Harmonic rhythm — chord CHANGES land on beat 1** (PRIMARY, GT-confirmed
     0.93-1.0). The beat-synchronous chroma FLUX concentrates on the true
     downbeat. Aggregated over the whole song, the comping sections dominate and
     solos contribute little — which is fine. This is what CARRIES swing tunes
     (Autumn) where the bass says nothing.
  2. **Chart-bar alignment** — the aligned chart's bar-1 positions are downbeat
     candidates (a structural prior; optional, present in the full fusion run).
  3. **Bass root-on-1** (SECONDARY, pop/soul only) — the bass states a NEW root
     on beat 1. Read from ``bass_salience`` and weighted by its OWN per-span
     reliability, so it is near-useless on walking jazz bass (auto-downweighted)
     and strong on pop. It must NOT dominate — weight = its own reliability.
  4. **Drum strong-beat PAIR** (``drum_pattern``) — narrows the phase to the 2
     candidates on the strong (1&3) grid; the harmonic/bass terms then pick beat
     1 from beat 3. Drum TIMBRE is NOT a downbeat signal (Stage 0b/0b-bis:
     chance on swing) — no drum-timbre resolver exists.

Each term is RELIABILITY-WEIGHTED by its OWN self-reported concentration, so the
genre is handled automatically: on Autumn the bass term self-zeroes and harmonic
rhythm carries it; on Stand By Me the bass term reinforces the harmony.

PRECISION-FIRST (the dataset gate discipline). A weak winning margin, a mid-song
phase-evidence FLIP (the tell-tale of a dropped/added beat, which flips the phase
downstream), or no usable evidence at all -> ``confidence`` is set low and the
song is FLAGGED rather than asserted. A correctly-flagged song is a SUCCESS, not
a miss: we would rather say "I don't know the downbeat here" than place it wrong.

API
---
* ``resolve_downbeat(beat_track, chart_alignment, chroma, bass, meter=4)`` — the
  audio-facing entry: derives the per-beat evidence from a ``DrumBeatTrack`` +
  raw CQT chroma + a ``BassChroma`` (+ optional chart bar-1 times) and resolves
  the global phase. Returns a ``DownbeatResult``.
* ``resolve_phase(beat_times, meter, harm, bass, chart_hits, strong_mask, ...)``
  — the pure per-beat-evidence core (audio-free; what the unit tests drive).
* ``beat_chroma_flux`` / ``bass_beat_evidence`` — the two evidence extractors.

NON-CIRCULARITY. Chroma flux is raw CQT; bass is raw CQT; chart hits come from
the chart alignment, never the model's decode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from harmonia.align.bass_salience import BassChroma, bass_pc_series

# ── fusion weights (term priors) + gates ─────────────────────────────────────
_W_HARM = 1.0      # harmonic rhythm is PRIMARY
_W_BASS = 1.0      # bass is scaled by its own reliability (-> secondary in practice)
_W_CHART = 0.6     # chart bar-1 is a structural prior (discounted: alignment slack)
_STRONG_NARROW_CONF = 0.15   # drum strong-pair confidence to narrow to 2 phases
_FLAG_CONF = 0.12            # below this normalised margin -> FLAG (precision-first)
_MIN_TERM_MASS = 1e-9
_PEAK_FRAC = 0.5            # keep the top-half local-maxima of a change signal
_LATTICE_REFINE = 0.02     # constant-lattice period refine half-range (+/- 2%)
_FLIP_CONC = 0.40          # per-half concentration floor for the phase-flip flag:
                           # a real dropped-beat slip leaves each half internally
                           # CLEAN (~0.5+) but offset; a merely diffuse half (0.25-
                           # 0.35, e.g. a fast backing track) is ambiguity, not a slip


# ═════════════════════════════════════════════════════════════════════════════
# Constant-tempo lattice (the CLEAN-count grid the global phase lives on)
# ═════════════════════════════════════════════════════════════════════════════
# The raw tracked beat array jitters and occasionally inserts/drops a beat; on
# that grid a global metrical PHASE slips after every insertion, so the aggregate
# evidence smears across phases (verified: peak-flux concentration collapses from
# ~0.6 on a clean grid to ~0.07 on the jittery drum grid). The global-phase model
# REQUIRES a clean, constant-count grid — exactly the constant-tempo lattice
# brick0 v3 lays chords on. So we rebuild one from the drum tracker's octave-
# locked PERIOD (its most reliable output) + a single global phase snapped to the
# tracked beats, and resolve the downbeat on THAT. (A song whose tempo genuinely
# drifts beyond the refine band is caught by low coverage -> FLAG.)

def constant_lattice(beat_times: np.ndarray, period: Optional[float] = None,
                     refine: bool = True) -> tuple[np.ndarray, float]:
    """Build the constant-tempo beat lattice ``phi0 + k*P`` spanning the tracked
    beats. ``period`` defaults to the tracked beats' median spacing; ``phi0`` is
    the circular-mean phase of the tracked beats (robust to a minority of
    inserted/dropped beats). With ``refine`` the period is swept +/-2% to best
    fit the tracked beats (reduces end-of-song drift). Returns ``(lattice, P)``.
    """
    b = np.asarray(beat_times, float)
    if len(b) < 2:
        return b.copy(), (period or 0.5)
    P0 = float(period) if period and period > 0 else float(np.median(np.diff(b)))

    def _fit(P: float) -> float:
        phi = _snap_phase(b, P)
        lat = phi + np.arange(int((b[-1] - phi) / P) + 1) * P
        if len(lat) == 0:
            return -1.0
        d = np.abs(b[:, None] - lat[None, :]).min(axis=1)
        return float(np.mean(d < 0.15 * P))

    P = P0
    if refine:
        cand = np.linspace(P0 * (1 - _LATTICE_REFINE), P0 * (1 + _LATTICE_REFINE), 41)
        fits = np.array([_fit(c) for c in cand])
        # among the near-best fits, prefer the period CLOSEST to the tracker's
        # P0 (regularise: the drum tempo is already octave-locked; don't stray on
        # a tie — an over-eager refine drifts a long song's grid off, e.g. Let It Be)
        near = cand[fits >= fits.max() - 1e-9]
        P = float(near[np.argmin(np.abs(near - P0))])
    phi0 = _snap_phase(b, P)
    n = int((b[-1] - phi0) / P) + 1
    lat = phi0 + np.arange(max(n, 1)) * P
    lat = lat[(lat >= b[0] - P) & (lat <= b[-1] + P)]
    return lat, P


def _snap_phase(beats: np.ndarray, P: float) -> float:
    """Global sub-beat phase in [0, P): the circular mean of ``beats mod P``."""
    ph = (beats / P) % 1.0
    mu = np.angle(np.exp(2j * np.pi * ph).mean()) % (2 * np.pi)
    return (mu / (2 * np.pi)) * P


# ═════════════════════════════════════════════════════════════════════════════
# Evidence extractors
# ═════════════════════════════════════════════════════════════════════════════

def _beat_sync(frames: np.ndarray, ftimes: np.ndarray,
               beat_times: np.ndarray) -> np.ndarray:
    """Mean of ``frames`` within each beat interval [b[k], b[k+1]) -> (K, D)."""
    n = len(beat_times)
    D = frames.shape[1] if frames.ndim == 2 else 1
    out = np.zeros((n, D))
    med = float(np.median(np.diff(beat_times))) if n > 1 else 0.5
    for k in range(n):
        t0 = beat_times[k]
        t1 = beat_times[k + 1] if k + 1 < n else t0 + med
        lo = int(np.searchsorted(ftimes, t0))
        hi = int(np.searchsorted(ftimes, t1))
        if hi > lo:
            out[k] = frames[lo:hi].mean(axis=0)
    return out


def _tv_flux(bc: np.ndarray) -> np.ndarray:
    """Per-row total-variation distance to the previous row of an L1-normalised
    (K, 12) matrix, in [0, 1]. Row 0 = 0 (no predecessor)."""
    s = bc.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = bc / s
    flux = np.zeros(len(bc))
    if len(bc) > 1:
        flux[1:] = 0.5 * np.abs(p[1:] - p[:-1]).sum(axis=1)
    return flux


def _peak_pick(f: np.ndarray, frac: float = _PEAK_FRAC) -> np.ndarray:
    """Keep only LOCAL-MAXIMA beats whose value is in the top ``frac`` of the
    positive values; zero the rest. This is what turns a diffuse beat-to-beat
    flux (melody contamination on every beat) into a SPARSE chord-CHANGE signal
    that concentrates on beat 1 — the single biggest lever for phase resolution
    (raw flux conc ~0.15 -> peak-picked ~0.6 on the clean grid)."""
    f = np.asarray(f, float)
    out = np.zeros_like(f)
    pos = f[f > 0]
    if len(pos) == 0:
        return out
    thr = float(np.quantile(pos, 1.0 - frac)) if len(pos) > 2 else float(pos.min())
    for k in range(len(f)):
        lo = f[k - 1] if k > 0 else -np.inf
        hi = f[k + 1] if k + 1 < len(f) else -np.inf
        if f[k] >= lo and f[k] >= hi and f[k] >= thr:
            out[k] = f[k]
    return out


def beat_chroma_flux(frames: np.ndarray, ftimes: np.ndarray,
                     beat_times: np.ndarray) -> np.ndarray:
    """Per-beat harmonic-rhythm flux (RAW): the L1 total-variation distance
    between consecutive beats' L1-normalised chroma, in [0, 1]. A chord CHANGE at
    beat k -> large flux[k]; a held chord -> ~0. flux[0] = 0. The PRIMARY
    downbeat evidence before peak-picking (see `harmonic_change_evidence`)."""
    return _tv_flux(_beat_sync(frames, ftimes, beat_times))


def harmonic_change_evidence(frames: np.ndarray, ftimes: np.ndarray,
                             beat_times: np.ndarray) -> np.ndarray:
    """The PRIMARY downbeat term: PEAK-PICKED beat-chroma flux — sparse chord
    changes, which concentrate on beat 1. This is `beat_chroma_flux` passed
    through `_peak_pick`."""
    return _peak_pick(beat_chroma_flux(frames, ftimes, beat_times))


def bass_beat_evidence(bass: BassChroma, beat_times: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat BASS root-on-1 evidence + per-beat bass reliability.

    Beat-syncs the flat bass chroma, takes its total-variation FLUX (a new bass
    root stated -> a spike; bass roots change on beat 1) and WEIGHTS it by the
    beat's bass reliability (bass-band energy x chroma concentration). Walking /
    absent bass has low concentration -> low reliability -> ~zero evidence
    (Autumn self-downweights); a clear held/repeated root -> high reliability
    (Stand By Me). Returns ``(evidence, reliability)``, both length K.
    """
    n = len(beat_times)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    bc = _beat_sync(bass.chroma, bass.times, beat_times)          # (K, 12)
    be = _beat_sync(bass.energy.reshape(-1, 1), bass.times, beat_times).ravel()
    flux = _tv_flux(bc)
    s = bc.sum(axis=1)
    top = np.divide(bc.max(axis=1), s, out=np.zeros(n), where=s > 0)
    conc = np.clip((top - 1.0 / 12.0) / (1.0 - 1.0 / 12.0), 0.0, 1.0)
    energy_norm = np.clip(be / bass.energy_ref, 0.0, 1.0)
    rel = energy_norm * conc
    return flux * rel, rel


# ═════════════════════════════════════════════════════════════════════════════
# Result
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DownbeatResult:
    """Output of the global-phase downbeat resolver.

    Attributes
    ----------
    phase_offset : int
        The winning global phase: beat index ``k`` is a downbeat iff
        ``k % meter == phase_offset`` (indices into ``beat_times``).
    downbeat_times : (D,) float
        Predicted downbeat times = the beats on the winning phase.
    confidence : float in [0, 1]
        Normalised margin of the winning phase over the runner-up (0 = tie,
        1 = maximal separation). LOW when the evidence is ambiguous.
    flagged : bool
        Precision-first flag: weak margin / mid-song phase flip / no evidence.
        A flagged song asserts NO downbeat (treat ``downbeat_times`` as a best
        guess, not a claim).
    meter : int
    phase_scores : (meter,) float
        Total weighted (concentration-centred) score per candidate phase;
        ``-inf`` for phases excluded by the drum strong-pair narrowing.
    candidate_phases : list[int]
        Phases actually scored (all ``meter`` unless narrowed by the drum pair).
    per_term_contributions : dict
        Per evidence term: ``weight`` (the reliability weight it entered with),
        ``phase_fracs`` (fraction of the term's mass on each phase's downbeats,
        baseline ``1/meter``), ``favored_phase``, and ``concentration``. This is
        the audit trail that SHOWS which evidence carried the song (e.g. bass
        weight ~0 on Autumn, high on Stand By Me).
    notes : list[str]
        Human-readable flags/reasons.
    """
    phase_offset: int
    downbeat_times: np.ndarray
    confidence: float
    flagged: bool
    meter: int
    phase_scores: np.ndarray
    candidate_phases: list
    per_term_contributions: dict
    notes: list = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# The pure per-beat-evidence core
# ═════════════════════════════════════════════════════════════════════════════

def _phase_fracs(evidence: np.ndarray, residues: np.ndarray, meter: int
                 ) -> tuple[np.ndarray, float]:
    """Fraction of an evidence term's mass landing on each phase's downbeats +
    the term's concentration (how far the best phase exceeds the 1/meter flat
    baseline, in [0, 1] — this is the term's self-reported reliability)."""
    total = float(evidence.sum())
    if total <= _MIN_TERM_MASS:
        return np.full(meter, 1.0 / meter), 0.0
    fracs = np.array([evidence[residues == phi].sum() / total
                      for phi in range(meter)])
    conc = float(np.clip((fracs.max() - 1.0 / meter) / (1.0 - 1.0 / meter),
                         0.0, 1.0))
    return fracs, conc


def _strong_parity(strong_mask: np.ndarray, meter: int) -> Optional[int]:
    """The index parity (0 or 1) carrying the STRONG (1&3) drum grid, or None if
    unresolved. Only meaningful for even meters (the strong-beat pair)."""
    if strong_mask is None or len(strong_mask) == 0 or meter % 2 != 0:
        return None
    idx = np.flatnonzero(strong_mask)
    if len(idx) == 0:
        return None
    par0 = int(np.sum((idx % 2) == 0))
    return 0 if par0 >= len(idx) - par0 else 1


def resolve_phase(beat_times: np.ndarray, meter: int = 4, *,
                  harm: Optional[np.ndarray] = None,
                  bass: Optional[np.ndarray] = None,
                  bass_weight: Optional[float] = None,
                  chart_hits: Optional[np.ndarray] = None,
                  strong_mask: Optional[np.ndarray] = None,
                  strong_conf: float = 0.0,
                  weights: Optional[dict] = None) -> DownbeatResult:
    """Resolve the global downbeat phase from per-beat evidence arrays (the pure,
    audio-free core; ``resolve_downbeat`` builds the arrays from audio).

    Parameters
    ----------
    beat_times : (K,) float
        The locked beat grid.
    harm, bass, chart_hits : (K,) float, optional
        Per-beat nonnegative evidence: harmonic-rhythm flux (PRIMARY), bass
        root-on-1, chart bar-1 hits. Any may be omitted.
    bass_weight : float, optional
        Overall bass-stream reliability gate (mean per-beat bass reliability);
        multiplies the bass term so a walking/absent bass self-zeroes. Defaults
        to 1.0 when bass evidence is given without an explicit gate.
    strong_mask : (K,) bool, optional
        Per-beat strong (1&3) grid membership from the drum tracker; used to
        NARROW the candidate phases to the strong pair when ``strong_conf`` is
        high enough.
    strong_conf : float
        Drum strong-beat-pair confidence (gates the narrowing + a soft term).
    weights : dict, optional
        Override the term priors {"harm","bass","chart"}.
    """
    beat_times = np.asarray(beat_times, float)
    K = len(beat_times)
    w = dict(harm=_W_HARM, bass=_W_BASS, chart=_W_CHART)
    if weights:
        w.update(weights)
    residues = np.arange(K) % meter

    # candidate phases: narrowed to the drum strong pair when the pair is confident
    parity = _strong_parity(strong_mask, meter)
    notes: list[str] = []
    if parity is not None and strong_conf >= _STRONG_NARROW_CONF:
        candidates = [phi for phi in range(meter) if phi % 2 == parity]
        notes.append(f"drum-pair narrowed to phases {candidates} "
                     f"(strong_conf={strong_conf:.2f})")
    else:
        candidates = list(range(meter))

    terms: dict[str, tuple[np.ndarray, float, float]] = {}   # name -> (fracs, weight, conc)
    contrib: dict[str, dict] = {}

    def _add(name: str, evidence: Optional[np.ndarray], base_w: float,
             gate: float = 1.0):
        if evidence is None:
            return
        ev = np.asarray(evidence, float)
        if len(ev) != K:
            raise ValueError(f"{name} evidence length {len(ev)} != {K} beats")
        fracs, conc = _phase_fracs(ev, residues, meter)
        weight = base_w * gate * conc          # self-reliability weighting
        terms[name] = (fracs, weight, conc)
        contrib[name] = dict(weight=round(float(weight), 4),
                             concentration=round(float(conc), 4),
                             phase_fracs=[round(float(x), 4) for x in fracs],
                             favored_phase=int(np.argmax(fracs)))

    _add("harm", harm, w["harm"])
    bw = 1.0 if bass_weight is None else float(np.clip(bass_weight, 0.0, 1.0))
    _add("bass", bass, w["bass"], gate=bw)
    _add("chart", chart_hits, w["chart"])

    # soft drum term: rewards the strong-parity phases (does not pick 1 vs 3)
    if parity is not None and strong_conf > 0:
        drum_fracs = np.array([1.0 if phi % 2 == parity else 0.0
                               for phi in range(meter)])
        drum_fracs = drum_fracs / drum_fracs.sum()
        terms["drum"] = (drum_fracs, float(strong_conf), float(strong_conf))
        contrib["drum"] = dict(weight=round(float(strong_conf), 4),
                               concentration=round(float(strong_conf), 4),
                               phase_fracs=[round(float(x), 4) for x in drum_fracs],
                               favored_phase=int(parity))

    # weighted, concentration-CENTRED score per candidate phase (uniform terms -> 0)
    total_w = sum(weight for _f, weight, _c in terms.values())
    scores = np.full(meter, -np.inf)
    for phi in candidates:
        s = 0.0
        for _name, (fracs, weight, _c) in terms.items():
            s += weight * (fracs[phi] - 1.0 / meter)
        scores[phi] = s

    if not np.any(np.isfinite(scores)) or total_w <= _MIN_TERM_MASS:
        # no usable evidence at all
        return DownbeatResult(
            phase_offset=candidates[0] if candidates else 0,
            downbeat_times=beat_times[residues == (candidates[0] if candidates else 0)],
            confidence=0.0, flagged=True, meter=meter, phase_scores=scores,
            candidate_phases=candidates, per_term_contributions=contrib,
            notes=notes + ["no usable downbeat evidence"])

    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
    winner = int(order[0])
    best = float(scores[winner])
    runner = float(scores[order[1]]) if len(order) > 1 and np.isfinite(scores[order[1]]) else best
    margin = best - runner
    scale = total_w * (1.0 - 1.0 / meter)              # max attainable separation
    confidence = float(np.clip(margin / (scale + _MIN_TERM_MASS), 0.0, 1.0))

    # mid-song phase-flip check on the PRIMARY term (dropped/added-beat tell). A
    # genuine dropped beat makes each half concentrate STRONGLY on a DIFFERENT
    # phase; require a real concentration floor in both halves so a diffuse
    # peak-flux (noise) can't false-positive a clean song (bb_backing regressed
    # this way — prec 1.0 yet a spurious flip).
    flip = False
    if harm is not None and K >= 4 * meter:
        h = np.asarray(harm, float)
        halves = []
        for lo, hi in ((0, K // 2), (K // 2, K)):
            seg_res = residues[lo:hi]
            fr, conc = _phase_fracs(h[lo:hi], seg_res, meter)
            cand_fr = [(phi, fr[phi]) for phi in candidates]
            halves.append((max(cand_fr, key=lambda x: x[1])[0], conc))
        if halves[0][1] > _FLIP_CONC and halves[1][1] > _FLIP_CONC \
                and halves[0][0] != halves[1][0]:
            flip = True
            notes.append(f"mid-song phase-evidence FLIP "
                         f"(1st half phase {halves[0][0]} -> 2nd half {halves[1][0]}) "
                         f"— possible dropped/added beat")

    flagged = bool(confidence < _FLAG_CONF or flip)
    if confidence < _FLAG_CONF:
        notes.append(f"weak margin (confidence={confidence:.3f} < {_FLAG_CONF})")
    if flip:
        confidence = min(confidence, _FLAG_CONF)

    return DownbeatResult(
        phase_offset=winner, downbeat_times=beat_times[residues == winner],
        confidence=confidence, flagged=flagged, meter=meter, phase_scores=scores,
        candidate_phases=candidates, per_term_contributions=contrib, notes=notes)


# ═════════════════════════════════════════════════════════════════════════════
# The audio-facing entry
# ═════════════════════════════════════════════════════════════════════════════

def _chart_hits(chart_alignment, beat_times: np.ndarray) -> Optional[np.ndarray]:
    """Turn chart bar-1 times into a per-beat hit array (1.0 at the nearest beat
    to each bar-1 onset). Accepts an array of times, an object exposing
    ``bar1_times`` / ``downbeat_candidate_times``, or None."""
    if chart_alignment is None:
        return None
    times = None
    if isinstance(chart_alignment, (list, tuple, np.ndarray)):
        times = np.asarray(chart_alignment, float)
    else:
        for attr in ("bar1_times", "downbeat_candidate_times", "downbeat_times"):
            if hasattr(chart_alignment, attr):
                times = np.asarray(getattr(chart_alignment, attr), float)
                break
    if times is None or len(times) == 0 or len(beat_times) == 0:
        return None
    hits = np.zeros(len(beat_times))
    for t in times:
        j = int(np.argmin(np.abs(beat_times - t)))
        hits[j] += 1.0
    return hits


def _map_strong_mask(beat_track, lattice: np.ndarray) -> Optional[np.ndarray]:
    """Carry the drum tracker's per-tracked-beat STRONG (1&3) mask onto the
    constant lattice by nearest tracked beat."""
    bt = np.asarray(getattr(beat_track, "beat_times", []), float)
    sm = getattr(beat_track, "strong_beat_mask", None)
    if sm is None or len(sm) != len(bt) or len(bt) == 0 or len(lattice) == 0:
        return None
    idx = np.clip(np.searchsorted(bt, lattice), 1, len(bt) - 1)
    left = idx - 1
    pick = np.where(np.abs(bt[idx] - lattice) < np.abs(bt[left] - lattice), idx, left)
    return np.asarray(sm)[pick]


def resolve_downbeat(beat_track, chart_alignment=None, chroma=None,
                     bass: Optional[BassChroma] = None, meter: int = 4
                     ) -> DownbeatResult:
    """Resolve the global downbeat phase from the fusion instruments.

    The tracked beat array jitters and inserts/drops beats, which slips a global
    metrical phase; so we first rebuild a CLEAN constant-tempo lattice from the
    drum tracker's octave-locked period (`constant_lattice`) and resolve the
    downbeat on THAT (see the lattice section's rationale). ``downbeat_times`` are
    returned on the lattice.

    Parameters
    ----------
    beat_track : DrumBeatTrack
        The locked beat grid (``harmonia.align.drum_pattern``). Provides
        ``beat_times``, ``period``, ``strong_beat_mask``, ``strong_beat_confidence``.
    chart_alignment : optional
        Chart bar-1 onset times (array), or an object exposing them, or None.
        The structural-prior term; omit it to resolve from acoustics alone.
    chroma : (frames, times) tuple, optional
        Raw CQT chroma frames (F, 12) + frame times, for the PRIMARY harmonic-
        rhythm flux term (peak-picked into chord changes).
    bass : BassChroma, optional
        The bass-salience stream (``harmonia.align.bass_salience.bass_chroma``)
        for the bass root-on-1 term.
    meter : int
        Beats per bar (4/4 -> 4).
    """
    tracked = np.asarray(beat_track.beat_times, float)
    period = getattr(beat_track, "period", None)
    lattice, P = constant_lattice(tracked, period)

    harm = None
    if chroma is not None:
        frames, ftimes = chroma
        harm = harmonic_change_evidence(np.asarray(frames, float),
                                        np.asarray(ftimes, float), lattice)

    bass_ev = bass_gate = None
    if bass is not None:
        raw, bass_rel = bass_beat_evidence(bass, lattice)
        bass_ev = _peak_pick(raw)
        bass_gate = float(np.mean(bass_rel)) if len(bass_rel) else 0.0

    chart_hits = _chart_hits(chart_alignment, lattice)
    strong_mask = _map_strong_mask(beat_track, lattice)
    strong_conf = float(getattr(beat_track, "strong_beat_confidence", 0.0))

    res = resolve_phase(lattice, meter, harm=harm, bass=bass_ev,
                        bass_weight=bass_gate, chart_hits=chart_hits,
                        strong_mask=strong_mask, strong_conf=strong_conf)

    # coverage sanity: the lattice must actually span the tracked beats (a badly
    # drifting tempo the constant lattice can't cover -> flag, precision-first)
    if len(tracked) and len(lattice):
        span = (lattice[-1] - lattice[0])
        cover = span / max(tracked[-1] - tracked[0], 1e-9)
        if cover < 0.85:
            res.flagged = True
            res.notes.append(f"lattice covers only {cover:.0%} of the tracked beats "
                             "— tempo may drift beyond a constant grid")
    if not getattr(beat_track, "octave_locked", True):
        res.flagged = True
        res.notes.append("beat grid not octave-locked — phase may be unreliable")
    res.notes.append(f"resolved on constant lattice P={P:.4f}s ({len(lattice)} beats)")
    return res
