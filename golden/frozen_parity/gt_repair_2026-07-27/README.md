# Frozen-benchmark repair overlay — 2026-07-27

**This is an OVERLAY, not a replacement.** `golden/brick0/*.gt.json` is owned by
the alignment lane and was **not modified**. Each file here carries, per song:

- `gt_chords[]` with both `t0_orig`/`t1_orig` (the frozen reference) and
  `t0`/`t1` (re-laid on the detected beat grid). Labels are untouched.
- `regions.IMPLIED` / `.WRONG` / `.UNCERTAIN` / `.EAR_OK` / `.EXCISE` spans.
- `retime{}` — the guards, the tempo ratio, and how far each boundary moved.
- `detector{}` — the definition + calibration of the STATED/IMPLIED detector.

Consumers should read the overlay and mask; the scorer does not read it yet
(`scratchpad/gtrepair_score.py` shows the masking).

## Why the re-lay is not circular

The new times come from **Beat This!** beats captured from the shipped pipeline's
beat stage, which runs before any chord decode and never sees a chord label. The
chart's beat-relative positions are preserved: every boundary is snapped to its
own nearest detected beat, capped at half a beat, monotonicity enforced.

## Guards (both must pass, else `retimed: false`)

| guard | rule | who failed |
|---|---|---|
| tempo octave | detected/GT beat-period ratio in [0.85, 1.18] | none |
| beat-grid coverage | <10% of expected beats missing from the tracker | **blue_bossa (25.2% missing)** |

`blue_bossa` is therefore **not** re-laid; its original timing is kept.

## The finding that matters more than the repair

The shipped pipeline quantises its chord boundaries onto its own rigid
constant-tempo grid (`bt = arange(phase, dur, period)`): median distance of a
predicted boundary to that grid is **0.2 ms**, to the real detected beats
**60–104 ms**. The reference is on a *different* synthetic lattice. Re-laying
only the reference costs −0.6 pp; re-laying **both** returns exactly the
baseline. So the lattice defect is real but currently self-cancelling, and the
fix belongs in the model (`beat_period_mode`), not only in the benchmark.

## What the alignment lane would need to change (proposal, not an edit)

1. `golden/brick0/{every_breath_you_take,bein_green,georgia_on_my_mind,close_to_you}.gt.json`
   — chord **labels** in the `WRONG` spans are rejected by Louis's ear. They need
   re-transcription, not re-timing. Until then they are unscoreable.
2. `golden/brick0/blue_bossa.gt.json` — the contrabass solo (~397–440 s) should
   be dropped from the benchmark ("à virer, ça perturbe à chaque fois").
3. All seven — consider adopting `t0`/`t1` from this overlay **only if** the
   model's boundaries are moved to the detected grid at the same time.
4. `SCHEMA.md` says `downbeat_times` are "INDEPENDENT Beat This! downbeats".
   In the files they are exactly periodic (bein_green: 3.213 s apart to the
   millisecond) — i.e. lattice-derived. The schema text and the data disagree.
