"""The strict per-segment confidence gate — the crux of the harvest pipeline.

A harvested song is a sequence of *segments* (one per proposed chord span). For
each segment the aligner (``scripts/brick0_propose.py``, imported read-only by
``harvest.py``) gives us a bundle of **non-circular** signals — harmonic
agreement, coverage, transpose margin, cross-repetition consistency /
divergence, mid-span split — and the Stage-1 drum tracker
(``harmonia.align.drum_pattern``) gives a beat-lock reliability. This module
turns those signals into a **3-way decision**:

  * ``CLEAN``  (bucket 1) — CONFIDENT and the chart MATCHES the recording. Emit
    to the clean GT manifest with the chart label. This is the high-precision
    training bulk; keep precision here VERY high (precision >> recall).
  * ``REVIEW`` (bucket 2) — the span is confidently placed in TIME but the chart
    chord does NOT match the audio (a repeated substitution / reharmonisation,
    e.g. Georgia's charted F#dim played as B7). This is *valuable*, not noise:
    do NOT emit the chart label as GT, do NOT discard — route to a separate
    review/substitution queue for later ear-labelling (active learning).
  * ``DROP``   (bucket 3) — low confidence NOT attributable to a clean
    substitution (weak/ambiguous alignment, gap, weak transpose, drum-less
    rubato). Discard.

The discriminator between REVIEW and DROP is exactly the aligner's divergence
detectors: a span that is time-aligned (beat-locked, cross-rep consistent,
covered, transpose-clear) but harmonically off = REVIEW; a span that is merely
poorly/ambiguously aligned = DROP.

DESIGN NOTE — purity. This module is deliberately **audio-free and pure**: it
operates on a :class:`SegmentSignals` bundle of already-normalised floats /
bools, so it is fully unit-testable with no audio, no models, no I/O. All
per-song normalisation (e.g. turning the drum tracker's per-song-relative
reliability curve into a [0, 1] ``beat_lock``) happens upstream in
``harvest.py``. The single most important knob — the harmonic-agreement
threshold ``agr_keep`` — is swept during calibration (see the package README and
``scripts``-free calibration harness) and frozen at a very-high-precision
operating point.

DOWNBEAT INTEGRATION POINT. The beat-lock input is the drum tracker's
``reliability`` today (``SegmentSignals.beat_lock``). When the dedicated
downbeat model (``harmonia/align/downbeat.py``, not built yet) lands, its
per-span downbeat confidence multiplies into ``beat_lock`` in ``harvest.py`` —
``gate.py`` needs no change (it already consumes a single [0, 1] beat-lock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Bucket(str, Enum):
    """The 3-way gate outcome."""
    CLEAN = "clean"     # bucket 1 — emit to the clean GT manifest
    REVIEW = "review"   # bucket 2 — substitution / to-decide queue
    DROP = "drop"       # bucket 3 — discard


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the gate. Defaults are the calibrated very-high-precision
    operating point (see ``harmonia/dataset/README.md`` for the precision/recall
    curve they were chosen from). ``agr_keep`` is THE swept knob.

    All thresholds are on signals normalised the same way ``harvest.py`` emits
    them, so the meaning is stable across songs.
    """
    # ── harmonic agreement (the primary precision knob; swept in calibration) ──
    agr_keep: float = 0.34
    """Per-chord Pearson agreement (audio CQT chroma vs chart chord-tones) must
    be >= this for a CLEAN emit. Raw agr scale is roughly [-0.1, 0.7]; a
    well-aligned chord typically sits 0.35-0.6. Higher => higher precision,
    lower recall."""

    agr_norm_keep: float = 0.60
    """Secondary, per-song-normalised agreement floor (agr / this-song's
    agreement ceiling) — guards against a song whose absolute agr scale is high
    but whose *worst* spans are still relatively bad."""

    # ── beat / downbeat lock ───────────────────────────────────────────────────
    beat_lock_min: float = 0.45
    """Per-song-normalised drum-tracker reliability over the span (1.0 == this
    song's steadiest drum region). Drops drum-less intros / rubato ballad tails.
    The downbeat model's confidence will fold into this upstream."""

    require_octave_locked: bool = True
    """The drum tracker's period must be within the tempo-octave band of the
    Beat This! prior (no 2x/0.5x slip) for a CLEAN emit."""

    # ── coverage / placement ───────────────────────────────────────────────────
    require_coverage_full: bool = True
    """The span must lie inside a placed chart section (not an unlabeled
    intro/vamp gap)."""

    drop_near_gap: bool = True
    """Segments abutting an unlabeled gap have ambiguous onsets — exclude from
    CLEAN (still eligible for REVIEW is disabled; they DROP)."""

    # ── transpose ──────────────────────────────────────────────────────────────
    transpose_margin_min: float = 0.03
    """Song-level margin (best transpose score - runner-up). Below this the key
    is ambiguous and every label is suspect => nothing CLEAN/REVIEW for the song."""

    # ── cross-repetition ───────────────────────────────────────────────────────
    require_xrep_consistent: bool = True
    """For a span inside a repeated section, its agreement must be *consistent*
    across occurrences (low cross-occurrence variance). Inconsistent => the
    placement is unstable => not CLEAN (DROP)."""

    # ── song-level broad-misalignment veto ─────────────────────────────────────
    frac_low_veto: float = 0.35
    """If more than this fraction of the song's labeled duration is below the
    agreement noise floor, the whole alignment is broadly off — emit NOTHING
    (neither CLEAN nor REVIEW) for the song."""

    # ── bucket-2 (substitution) routing ────────────────────────────────────────
    review_requires_time_lock: bool = True
    """A segment may only be routed to REVIEW if it is confidently time-aligned
    (beat-locked, octave-locked, covered, transpose-clear, cross-rep consistent).
    Otherwise a merely-misaligned span would masquerade as a substitution."""

    split_review_margin: float = 0.18
    """A mid-span split may corroborate a REVIEW only if the alternative chord
    beats the charted chord's 2nd-half agreement by >= this. The cross-rep
    uniform-low divergence flag is the *primary* REVIEW signal (reliable across
    repeats); a lone split-fire is treated as spurious (melody flux) unless it is
    this strong AND the span is otherwise well time-aligned."""


@dataclass(frozen=True)
class SegmentSignals:
    """Per-segment inputs to the gate — all pre-normalised by ``harvest.py``.

    A "segment" is one proposed chord span (``t0``, ``t1``, ``label``). Signals
    are non-circular: none of them is the shipped model's own chord decode.
    """
    # identity / label (carried through, not gated on)
    t0: float
    t1: float
    label: str

    # harmonic agreement
    agreement: float               # raw per-chord Pearson agr
    agreement_norm: float          # agr / this-song's agreement ceiling, clipped [0,1]

    # beat / downbeat lock
    beat_lock: float               # per-song-normalised drum reliability over span, [0,1]
    octave_locked: bool            # song-level: drum period octave-locked to prior

    # coverage / placement
    coverage_full: bool            # span inside a placed section (not a gap)
    near_gap: bool                 # span abuts an unlabeled gap

    # transpose
    transpose_margin: float        # song-level best-minus-runnerup transpose score

    # cross-repetition
    xrep_consistent: bool          # consistent agreement across occurrences (or n/a)
    xrep_divergence: bool          # flagged uniform-low (chart!=recording, repeated)

    # mid-span split (secondary substitution signal)
    split_fire: bool = False       # a mid-span split fired on this span
    split_margin: float = 0.0      # alt-2nd-half agr minus charted-2nd-half agr

    # song-level context
    frac_low_song: float = 0.0     # fraction of song labeled-duration below noise floor

    # optional substitution suggestion (for the review queue)
    substitution_suggestion: str | None = None


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for one segment."""
    bucket: Bucket
    confidence: float              # [0,1]
    reasons: tuple[str, ...] = field(default_factory=tuple)


# ── confidence model ─────────────────────────────────────────────────────────

def _squash(agr: float) -> float:
    """Map raw agreement (~[-0.1, 0.7]) to [0, 1] for the confidence product.
    Linear ramp 0.10 -> 0.60 saturating at both ends (0.10 == noise floor,
    0.60 == a confidently-fit chord)."""
    lo, hi = 0.10, 0.60
    if agr <= lo:
        return 0.0
    if agr >= hi:
        return 1.0
    return (agr - lo) / (hi - lo)


def _confidence(sig: SegmentSignals) -> float:
    """A smooth [0, 1] confidence = harmonic fit x beat-lock, tempered by
    transpose decisiveness and cross-rep consistency. Reported alongside every
    emitted / reviewed segment (drops report their would-be confidence too)."""
    c = _squash(sig.agreement)
    c *= 0.35 + 0.65 * min(max(sig.beat_lock, 0.0), 1.0)      # beat-lock weight
    c *= 0.60 + 0.40 * min(max(sig.transpose_margin / 0.15, 0.0), 1.0)
    if not sig.xrep_consistent:
        c *= 0.6
    if sig.near_gap:
        c *= 0.85
    if not sig.octave_locked:
        c *= 0.7
    return round(float(min(max(c, 0.0), 1.0)), 4)


# ── the gate ─────────────────────────────────────────────────────────────────

def _time_aligned(sig: SegmentSignals, cfg: GateConfig) -> tuple[bool, list[str]]:
    """Is the segment CONFIDENTLY placed in TIME (independent of whether the
    chart *label* is right)? This is the shared precondition for both a CLEAN
    emit and a REVIEW route — the discriminator that keeps merely-misaligned
    spans out of the substitution queue."""
    fails: list[str] = []
    if sig.beat_lock < cfg.beat_lock_min:
        fails.append(f"beat_lock {sig.beat_lock:.2f}<{cfg.beat_lock_min}")
    if cfg.require_octave_locked and not sig.octave_locked:
        fails.append("octave_unlocked")
    if cfg.require_coverage_full and not sig.coverage_full:
        fails.append("uncovered")
    if cfg.drop_near_gap and sig.near_gap:
        fails.append("near_gap")
    if sig.transpose_margin < cfg.transpose_margin_min:
        fails.append(f"transpose_margin {sig.transpose_margin:.3f}<{cfg.transpose_margin_min}")
    if cfg.require_xrep_consistent and not sig.xrep_consistent:
        fails.append("xrep_inconsistent")
    return (not fails), fails


def gate_segment(sig: SegmentSignals, cfg: GateConfig = GateConfig()) -> GateDecision:
    """Classify one segment into CLEAN / REVIEW / DROP (precision-first).

    Order of decision:
      1. Song-level broad-misalignment veto (``frac_low_song``) — emit nothing.
      2. Time-alignment precondition (beat-lock, octave, coverage, gap,
         transpose margin, cross-rep consistency).
      3. If time-aligned AND a divergence is detected (repeated uniform-low, or a
         strong mid-span split) => REVIEW (chart != recording; label withheld).
      4. Else if time-aligned AND harmonic agreement is high => CLEAN.
      5. Else => DROP.
    """
    conf = _confidence(sig)

    # (1) whole-song veto: a broadly-off alignment can produce locally-plausible
    # spans; refuse to emit anything from such a song.
    if sig.frac_low_song > cfg.frac_low_veto:
        return GateDecision(Bucket.DROP, conf,
                            (f"song frac_low {sig.frac_low_song:.2f}>{cfg.frac_low_veto} "
                             "(broad misalignment veto)",))

    # (2) time-alignment precondition
    time_ok, time_fails = _time_aligned(sig, cfg)

    # (3) divergence => REVIEW (only when confidently time-aligned)
    strong_split = sig.split_fire and sig.split_margin >= cfg.split_review_margin
    is_divergence = sig.xrep_divergence or (strong_split and sig.agreement < cfg.agr_keep)
    if is_divergence and (time_ok or not cfg.review_requires_time_lock):
        why = "xrep uniform-low (repeated substitution)" if sig.xrep_divergence \
            else f"strong mid-span split (+{sig.split_margin:.2f} over chart)"
        return GateDecision(Bucket.REVIEW, conf,
                            (f"DIVERGENCE: {why}; well time-aligned -> review queue",))

    # (4) CLEAN emit — precision-first: all locks + high agreement
    if time_ok:
        agr_ok = sig.agreement >= cfg.agr_keep and sig.agreement_norm >= cfg.agr_norm_keep
        if agr_ok and not sig.xrep_divergence:
            return GateDecision(Bucket.CLEAN, conf,
                                (f"agr {sig.agreement:.2f}>={cfg.agr_keep} "
                                 f"(norm {sig.agreement_norm:.2f}); all locks",))
        # time-aligned but agreement not high enough, and not a clean divergence
        return GateDecision(Bucket.DROP, conf,
                            (f"time-aligned but agr {sig.agreement:.2f}<{cfg.agr_keep} "
                             "(not a clean substitution) -> drop",))

    # (5) not time-aligned -> DROP
    return GateDecision(Bucket.DROP, conf, ("misaligned/ambiguous: " + ", ".join(time_fails),))


__all__ = ["Bucket", "GateConfig", "SegmentSignals", "GateDecision", "gate_segment"]
