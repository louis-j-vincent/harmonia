"""No-chord (N) emission policy — kill-switched brick, default OFF.

╔══════════════════════════════════════════════════════════════════════════════╗
║ 2026-07-27 — READ THIS BEFORE THE REST.  The "+2.10 pp" below is RETRACTED.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Two independent things went wrong with that number, and the second is the one
that matters.

**(1) It does not reproduce on the current pipeline.**  Re-measured 2026-07-27
with a bit-identical OFF control (pooled root 0.7367 / partial 0.6419, per-song
exact) and the same intersect/suppress masks applied at the same hook:

    metric   recorded 2026-07-24   re-measured 2026-07-27
    root         +2.10 pp                +0.98 pp
    partial      +1.67 pp                +0.10 pp

``blue_bossa`` reproduces exactly (suppress root .6503 vs the recorded .650);
``stand_by_me`` does not (.8735 vs the recorded .988).  **Root cause, visible in
the logs:** suppressing N over Stand By Me's bass-only intro turns 41 chart cells
into 87 — twelve chords in a 10 s intro (``D:dim/D#``, ``G#:sus4``, ``B:7/F#``,
``A:dim``, ``E:dim`` …), one per beat, because the per-beat argmax root flaps
when there is nothing above the bass to read.  That junk drops the OCCAM
loop-family coverage from 0.98 to 0.97 and the simplicity post-pass **GATE
REJECTS** ("post-pass discarded, chart left unchanged"), so the flicker survives
into the final chart.  Before the ChordHead consolidation (``62e3f40``,
2026-07-26) it did not.  So this brick now actively fights the Occam pass —
i.e. it fights the project's own stated simplicity principle.  Error-pattern #6:
a component swap moved something other than the target metric.

**(2) Even the residue scores for the WRONG REASON.**  Louis, on the two worst
slices in the benchmark (Blue Bossa's contrabass solo, Stand By Me's verses):

    "only the bass is present — it is enough to deduce the chord from the
     general harmony but no chord is being played"

**The model is acoustically correct there; the reference asserts the IMPLIED
harmony.**  So every pp this brick earns in those regions is arithmetic against
a reference that is wrong there.  Scored against the repaired overlay
(``golden/frozen_parity/gt_repair_2026-07-27/``), which makes those regions
unscoreable, intersect/suppress go **NEGATIVE on the headline metric**:

    reference                                     partial     root   scoreable
    R_raw  frozen reference, all time            +0.0010  +0.0098    1653.8 s
    R_lab  retimed + WRONG dropped               -0.0010  +0.0064    1453.9 s
    R_rep  + EXCISE + IMPLIED dropped (L4)       -0.0044  +0.0009    1413.9 s

**VERDICT: do NOT wire ``suppress`` or ``intersect``.**  The wiring hook below is
kept for the record, not as a recommendation.  What the regions actually need is
the IMPLIED-harmony policy at the bottom of this module — and that, measured
honestly, is also not an accuracy win (see MEASURED, IMPLIED POLICY).

The pre-2026-07-27 rationale follows unchanged, for provenance.

WHY THIS EXISTS (as written 2026-07-23/24 — superseded by the block above)
--------------------------------------------------------------------------
On the FROZEN Brick-0 real-audio benchmark (jazz/pop standards, iReal-derived +
ear-verified GT), the shipped nnls24 path OVER-emits no-chord (N).  The GT for
these chord-continuous standards has ZERO N spans, yet the model marks segments N
via the UNION of two sources in ``_infer_nnls24``:

  * ``musx_bass.no_chord_per_segment`` — music-x-lab's own "N"/"X" token, and
  * ``_nnls_no_chord_segs`` — a raw-NNLS-energy gate (fallback when musx-N absent).

Because it is a UNION (either source can fire), it is precision-poor.  Measured on
the 4 cache-warm frozen songs (blue_bossa, bein_green, georgia_on_my_mind,
stand_by_me; 948s), EVERY predicted N was wrong (61.2s = 6.45pp of the pooled
score; music-x-lab alone marked 108/900 segments N on blue_bossa).  Suppressing
all N (labelling every segment) recovers (brick-ON minus brick-OFF, 2026-07-23):

    mode="suppress"  root +3.61pp  majmin +3.10  7ths +2.33  partial +2.88
                     strict +2.33  bass +3.53   (pooled, N=4 frozen songs)
    per-song root: stand_by_me +13.5pp, blue_bossa +2.6pp, bein_green/georgia 0.0

Realized +3.61pp vs the 6.45pp ceiling ⇒ ~56% of the un-N'd spans still land on a
wrong root (the recovered label is not always right); the gain is real but partial
and concentrated in stand_by_me (a doo-wop bass-loop whose sparse segments read as
no-chord).  See docs/research_sessions/brick0_chord_accuracy_2026-07-23.md +
docs/research_sessions/brick0_diagnosis_2026-07-23.png.

WHAT THIS IS *NOT* (CLAUDE.md #4 — state the unsolved remainder)
---------------------------------------------------------------
- mode="suppress" is UNSAFE as a global default.  The N mask exists precisely for
  pop songs with genuinely chordless regions (intros/breakdowns/spoken word — the
  2026-07-19 user report of an invented "C" over a chordless intro).  On the
  Brick-0 set those don't occur, so suppress-all only helps here; on a song WITH
  real silence it would INVENT chords and HURT.  Do NOT flip suppress-all ON for
  the production server without a repertoire guard.
- It does NOT fix the root ERRORS themselves (the dominant loss: root_lost is 21-70%
  of duration per family; the bottleneck is upstream NNLS/musx root labelling, not
  the N gate).  It only reclaims the 6.45pp that N-masking threw away outright.
- mode="intersect" (require BOTH musx-N AND nnls-energy-N — the principled precision
  fix that keeps genuine silence) was VALIDATED end-to-end on the full 7 frozen songs
  (Wave-2, 2026-07-24): it is BIT-IDENTICAL to mode="suppress" on the benchmark
  (root +2.10pp 0.7367->0.7577, bass +2.05pp, partial +1.67pp, ZERO per-song
  regressions) because the only spans it keeps as N (17.1s across 7; blue_bossa +
  stand_by_me each keep 1) fall OUTSIDE the GT-scored windows (pre-GT intros / post
  song tail).  Unlike suppress it is production-SAFE: on a synthetic clip
  [music | 15s pure silence | music], suppress labels 0% of the silence N (invents
  G#:hdim7/C, E:maj/D# over silence) while intersect labels 92% of it N.  So intersect
  recovers ALL of suppress's benchmark gain while keeping the genuine-silence valve.
  See docs/research_sessions/brick_wave2_bricks_2026-07-24.md +
  brick_wave2_A_suppressed_spans.html + brick_wave2_A_timeline.png.  The two masks
  ARE computable at one hook (see INTEGRATION HOOK below).  intersect is the
  recommended shippable variant.
- Validated on N=4 frozen songs only (3/7 were disk-blocked from fresh music-x-lab
  decode this session).  Few-song finding — treat as a hypothesis until the full 7
  (or a broader chord-continuous set) confirm it (CLAUDE.md #5).

INTEGRATION HOOK (default OFF; wiring is the orchestrator's call — needs an edit to
the concurrently-owned chord_pipeline_v1.py, so it is NOT wired here)
-----------------------------------------------------------------------------------
In ``chord_pipeline_v1._infer_nnls24``, immediately BEFORE the final
``labeled = _label_segments(... seg_no_chord=seg_no_chord)`` call (grep the symbol,
not a line number — the file is churning), wrap the mask.  For the SHIPPED path
(quality_frontend="musx"), ``seg_no_chord`` there is the music-x-lab N mask and the
raw-NNLS-energy mask is NOT yet computed (the ``_nnls_no_chord_segs`` fallback is
skipped when musx supplied its own N).  So the intersect wiring must compute the
second mask explicitly (``arr``/``times``/``bt``/``segs`` are all in scope)::

    from harmonia.models.no_chord_policy import gated_no_chord_mask
    _nnls_nc = _nnls_no_chord_segs(arr, times, bt, segs)   # the raw-energy silence gate
    seg_no_chord = gated_no_chord_mask(seg_no_chord, _nnls_nc)  # env HARMONIA_NC_POLICY

``gated_no_chord_mask`` returns ``seg_no_chord`` UNCHANGED when the env flag is
unset/"off" (behavioural no-op).  With HARMONIA_NC_POLICY="intersect" it ANDs the two
masks (recommended, safe: +2.10pp on the benchmark, keeps genuine silence); with
"suppress" it drops all N (also +2.10pp but UNSAFE on real silence).  Computing
``_nnls_no_chord_segs`` unconditionally adds only a cheap per-beat energy loop.
Validated via runtime monkeypatch at this exact hook (Wave-2 scratchpad measure_A.py).
"""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "no_chord_policy", "gated_no_chord_mask",
    "implied_segments", "state_implied_mask", "hold_through_implied",
    "implied_policy_mode",
]


def no_chord_policy(
    musx_mask: np.ndarray,
    nnls_mask: np.ndarray | None = None,
    mode: str = "off",
) -> np.ndarray:
    """Apply a no-chord emission policy to a per-segment N mask.

    Args:
        musx_mask: (n_segs,) bool — the segment N mask the pipeline would emit
                   (music-x-lab-derived, or the raw-energy fallback).
        nnls_mask: (n_segs,) bool — the raw-NNLS-energy N mask, if separately
                   available (only used by mode="intersect").
        mode:      "off"       — return ``musx_mask`` unchanged (exact no-op).
                   "suppress"  — never emit N (all-False). +3.61pp root on the
                                 Brick-0 frozen set; UNSAFE where real silence exists.
                   "intersect" — N only where BOTH masks agree (precision fix;
                                 requires ``nnls_mask``; falls back to "off" if None).
                                 DESIGNED, not yet validated end-to-end.

    Returns:
        (n_segs,) bool mask.  Reference behaviour is bit-identical to the input
        when mode == "off".
    """
    musx_mask = np.asarray(musx_mask, dtype=bool)
    if mode == "suppress":
        return np.zeros(musx_mask.shape, dtype=bool)
    if mode == "intersect":
        if nnls_mask is None:
            return musx_mask
        return musx_mask & np.asarray(nnls_mask, dtype=bool)
    return musx_mask  # "off" / unknown → passthrough


def gated_no_chord_mask(
    musx_mask: np.ndarray,
    nnls_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Env-controlled drop-in for the per-segment N mask (default OFF).

    - ``HARMONIA_NC_POLICY`` (str, default "off"): one of {off, suppress, intersect}.
      "off" returns ``musx_mask`` unchanged (behavioural no-op).
    """
    mode = os.environ.get("HARMONIA_NC_POLICY", "off").strip().lower()
    return no_chord_policy(musx_mask, nnls_mask, mode)


# ═══════════════════════════════════════════════════════════════════════════
# IMPLIED-harmony policy — the STATED/IMPLIED rebuild (2026-07-27)
# ═══════════════════════════════════════════════════════════════════════════
#
# THE QUESTION.  ``harmonia.models.harmonic_texture`` says *where* the harmony is
# only implied (bass-only passages: a contrabass solo, a doo-wop intro).  What
# should the chart SHOW there?  Three behaviours were on the table:
#
#   (a) emit N          — acoustically honest, musically useless to a reader
#   (b) hold the previous chord
#   (c) emit a chord but MARK IT INFERRED so the UI can render it differently
#
# PREMISE CHECK, run before implementing (CLAUDE.md #2) — (b)'s stated
# justification is FALSE.  The claim was "holding is what a real chart does".
# The reference *is* the chart, and over the 65.8 s of detected implied time it
# **cycles**, it does not hold: 88.9% of implied time (58.5 s) spans ≥2 distinct
# chords.  Blue Bossa's contrabass solo runs the whole 16-bar form
# (Cm7 / Fm7 / Dm7b5 G7 / Cm7 / Ebm7 Ab7 / Dbmaj7 …) across 35 s.  Holding one
# chord through it is not a chart, it is a stuck chart.  Only 7.3 s (11%) of
# implied time is a genuine single-chord hold.
#
# MEASURED, IMPLIED POLICY.  Inside the implied spans only (65.8 s of 1653.8 s
# = 4.0% of the benchmark), agreement with the reference and the number of chart
# cells emitted:
#
#     behaviour                      root   family   cells   cells/s
#     (a) N            (baseline)    0.30     0.24      11      0.17
#     (b) hold                       0.48     0.39      11      0.17
#     (c) state, texture-gated       0.59     0.39      76      1.16
#         suppress, ungated          0.54     0.37      85      1.29
#
# **Under the headline metric — family-level partial credit — (b) and (c) TIE at
# 0.39.**  Stating the acoustically-decoded chord wins on ROOT (0.59 vs 0.48,
# because a walking bass plays roots) and loses it all again on QUALITY, for
# SEVEN TIMES the chart clutter.  On Blue Bossa, the longest and most important
# implied passage, holding is outright BETTER on family credit (0.23 vs 0.15)
# with 6 cells against 69.  The true harmonic rhythm there is ~0.8 chords/s;
# (c) emits 2.0/s.
#
# POOLED, three references (headline partial_credit; scoreable seconds shown
# because no row may buy a gain by scoring less time):
#
#     behaviour     R_raw (1653.8 s)   R_lab (1453.9 s)   R_rep (1413.9 s)
#     (b) hold          +0.0071            +0.0033            +0.0000
#     (c) state         +0.0069            +0.0051            +0.0007
#     suppress          +0.0010           -0.0010            -0.0044
#
# R_rep is the repaired overlay, which makes the implied regions unscoreable —
# so ~0.0000 there is the CORRECT reading, not a failure to measure.  It says
# plainly: **the entire gain of every one of these behaviours lives inside
# regions the repaired reference declares it cannot adjudicate.**  Nothing here
# clears the project's ~2 pp bar for keeping a brick.
#
# IS THERE A SAFE VERSION OF HOLDING?  No — and this was measured, not assumed.
# Holding is safe only while the form cannot have moved, so the right test is
# not "how many seconds" but "could a chord have gone by": hold across an
# implied span only if it is no longer than the song's OWN median stated chord
# duration (``hold_short_implied``).  Result:
#
#     song                budget   spans held
#     bein_green           3.22 s      0 / 1
#     blue_bossa           1.40 s      0 / 5
#     close_to_you         2.71 s      0 / 2
#     stand_by_me          2.01 s      0 / 3
#
# **Zero spans qualify.**  Every span the detector flags is longer than a chord
# in that song — which is structural, not luck: ``harmonic_texture`` enforces a
# 3 s minimum span to avoid slivers, and a 3 s chord is already slow for this
# repertoire.  ``hold_short_implied`` therefore degenerates to today's behaviour
# exactly (+0.0000 on all three references) with **0 misleading seconds**, while
# plain ``hold_through_implied`` asserts a chord it cannot know across **58.5 s
# of 65.8** (88.9%).  So there is no regime in which holding is defensible here.
#
# RECOMMENDED BEHAVIOUR: (c)'s *structure*, and NOT its filling — mark the
# region IMPLIED and do not invent a chord in it.  Concretely: keep the cell's
# label as-is (N) but change its MEANING and its rendering from "nothing here"
# to "the changes continue, we cannot hear them".  This is the only candidate
# that is:
#   * +0.0000 on every reference, because it rewrites no label — zero accuracy
#     risk, nothing to regress;
#   * 0 misleading seconds, unlike hold (58.5 s) and unlike the acoustic
#     re-decode (which prints 2.0 chords/s over a passage whose real harmonic
#     rhythm is 0.8/s);
#   * strictly more informative than today, because it separates genuinely
#     CHORDLESS from IMPLIED — on the 7 frozen songs those are distinguishable:
#     Blue Bossa's solo is 83% agreed implied, while georgia's 4.2 s of N is 0%
#     implied, i.e. really chordless.
# It is a **UI-honesty feature, not an accuracy brick.**  Default OFF, NOT wired.
# ``hold_through_implied`` and ``state_implied_mask`` are kept as the measured
# alternatives and as the substrate a future form-filler will reuse — not as
# recommendations.
#
# WHAT WOULD ACTUALLY FIX THESE REGIONS: continue the FORM.  Cheap check — take
# the model's OWN chart one form-period earlier (period estimated from the
# model's own label self-similarity over stated regions; no reference consulted)
# and read it into the implied span: root agreement **0.49**, versus 0.48 hold /
# 0.59 state / 0.30 N, and it is the only candidate that is right for the right
# reason.  It needs a form/section model that does not exist yet; finding #9
# already named it ("repetition/bass is the only honest path for them").
#
# UI HOOK NEEDED FOR (c) — PROPOSAL ONLY, NOT EDITED HERE.  That surface has
# regressed silently before, so it is described, not touched:
#   1. ``harmonia/output/chart_model.py``, cell builder (~L150-200): mirror the
#      existing ``if is_nc: entry["nc"] = True`` with
#      ``if c.get("inferred"): entry["inferred"] = True``.  One line.
#   2. ``harmonia/output/chart_interactive.py`` glyph renderer (~L1142, next to
#      the existing ``if(d.nc)`` branch): render an inferred cell in the normal
#      chord style but parenthesised / lightened.  The confidence colour ramp
#      (``colour(key, conf)``) already exists and could carry it instead.
#   Neither file is touched by this module.


def implied_segments(
    seg_bounds: "list[tuple[float, float]]",
    implied_spans: "list[list[float]] | list[tuple[float, float]]",
) -> np.ndarray:
    """(n_segs,) bool — segment midpoint falls inside an IMPLIED span.

    Midpoint, not overlap: an implied span's edges are soft to roughly ±1 s (see
    ``harmonic_texture`` limitation 5), so an overlap test would drag whole
    neighbouring chords in.
    """
    out = np.zeros(len(seg_bounds), dtype=bool)
    for i, (a, b) in enumerate(seg_bounds):
        mid = 0.5 * (a + b)
        out[i] = any(x <= mid < y for x, y in implied_spans)
    return out


def state_implied_mask(
    musx_mask: np.ndarray,
    seg_bounds: "list[tuple[float, float]]",
    implied_spans: "list[list[float]] | list[tuple[float, float]]",
) -> np.ndarray:
    """Behaviour (c): clear N *only* inside IMPLIED spans (texture-gated).

    Everywhere else the N mask is untouched, so genuine silence outside an
    implied passage still renders N.  This is the narrow, right-for-the-right-
    reason version of ``mode="suppress"``: it fires exactly where a chart would
    still be showing changes over a bass-only passage.

    NOTE the label it produces there comes from the acoustic head reading a
    bass-only spectrum — see the module docstring's MEASURED section for what
    that is actually worth.
    """
    musx_mask = np.asarray(musx_mask, dtype=bool)
    if not implied_spans:
        return musx_mask
    return musx_mask & ~implied_segments(seg_bounds, implied_spans)


def hold_through_implied(
    chords: "list[dict]",
    implied_spans: "list[list[float]] | list[tuple[float, float]]",
    *, no_chord_label: str = "N",
) -> "list[dict]":
    """Behaviour (b): relabel N cells inside IMPLIED spans as the chord before.

    Pure post-hoc rewrite of a finished chart (no re-decode).  A leading N with
    no predecessor is left alone.  Returns a new list; the input is not mutated.

    **This behaviour is REFUTED for anything longer than a bar or two** — see the
    module docstring.  Kept because it is the honest baseline for "hold" and
    because it is correct on the short spans (11% of implied time here).
    """
    out = [dict(c) for c in chords]
    prev = None
    for c in out:
        mid = 0.5 * (c["start_s"] + c["end_s"])
        inside = any(a <= mid < b for a, b in implied_spans)
        if c["label"] != no_chord_label:
            prev = c["label"]
        elif inside and prev is not None:
            c["label"] = prev
            c["inferred"] = True
    return out


def stated_chord_duration(
    chords: "list[dict]",
    implied_spans: "list[list[float]] | list[tuple[float, float]]",
    *, no_chord_label: str = "N",
) -> float:
    """Median duration of a chord in the STATED parts of this chart.

    The song's own harmonic rhythm, measured where it is audible.  Used to
    decide whether an implied span is short enough that holding is safe.
    Returns ``0.0`` when there is nothing to measure.
    """
    d = [c["end_s"] - c["start_s"] for c in chords
         if c["label"] != no_chord_label
         and not any(a <= 0.5 * (c["start_s"] + c["end_s"]) < b
                     for a, b in implied_spans)]
    return float(np.median(d)) if d else 0.0


def hold_short_implied(
    chords: "list[dict]",
    implied_spans: "list[list[float]] | list[tuple[float, float]]",
    *, no_chord_label: str = "N", factor: float = 1.0,
) -> "list[dict]":
    """Behaviour (b) restricted to spans too short to contain a chord change.

    Holding is safe only while the form cannot have moved.  The test is not
    "how many seconds" but "how many chords would normally have gone by": hold
    across an implied span only when it is no longer than ``factor`` x the
    song's own median stated chord duration.  Longer spans are left as they
    were (a hole), because asserting one chord across a cycling form is worse
    than admitting ignorance — see the module's PREMISE CHECK.

    ``factor`` is a shape parameter, not a fitted one; 1.0 means "at most one
    chord could have passed".
    """
    if not implied_spans:
        return [dict(c) for c in chords]
    budget = factor * stated_chord_duration(chords, implied_spans,
                                            no_chord_label=no_chord_label)
    short = [s for s in implied_spans if (s[1] - s[0]) <= budget]
    return hold_through_implied(chords, short, no_chord_label=no_chord_label)


def implied_policy_mode() -> str:
    """``HARMONIA_IMPLIED_POLICY`` (default ``"off"``) — the kill switch.

    ``"off"`` (default) means every function above is either not called or
    called with an empty span list, i.e. an exact behavioural no-op.  Other
    values: ``"state"`` (c), ``"hold"`` (b).  Nothing in the shipped pipeline
    reads this yet — this brick is NOT wired (see INTEGRATION HOOK).
    """
    return os.environ.get("HARMONIA_IMPLIED_POLICY", "off").strip().lower()
