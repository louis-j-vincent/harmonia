"""No-chord (N) emission policy — kill-switched brick, default OFF.

WHY THIS EXISTS
---------------
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

__all__ = ["no_chord_policy", "gated_no_chord_mask"]


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
