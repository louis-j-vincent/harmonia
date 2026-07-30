# Handoff — displayed-confidence recalibration (2026-07-30, mid-task)

Written before a context compaction. Everything committed is pushed
(`feat/harmonic-key`, up to date with origin). This is the one task left open.

## The decision already made (don't re-litigate)

Louis chose target **(b)**: the percentage should mean **P(root + parent family
are right)** — partial credit — not P(exact chord). Reason: you play Cm over Cm7
and you are fine, so a missing 7th should not cost the reader's attention.

**The target was never the problem.** `scratchpad/nnls24_conf_calibration.py`
already fits root + parent family (its docstring says "7-family" but its code
compares `maj/min/dom/hdim/dim/aug/sus`, identical to
`accuracy_score._FAMILY`). The FITTING CONFIG is the problem.

## The measured bug

`_get_nnls24_conf_map` applies an isotonic map fitted on the NNLS-24 heads at
ORACLE GT blocks over 100 RWC songs. Production moved to music-x-lab for
root/quality/bass/segmentation and the map never moved with it. Measured on
4 verified Brick-0 songs, shipped config, duration-weighted:

| song | shown | strict | partial |
|---|---|---|---|
| stand_by_me | 0.672 | 0.743 | 0.743 |
| bein_green | 0.463 | 0.487 | 0.661 |
| georgia_on_my_mind | 0.493 | 0.309 | 0.530 |
| close_to_you | 0.468 | 0.246 | 0.751 |
| **POOLED** | **0.519** | — | **0.669** |

Under-reports partial by ~15 pp pooled while OVER-reporting strict by up to
22 pp on 2 of 4 songs. The code comment claiming it "errs low, never high" is
false against strict. Full writeup in `docs/known_issues.md` (2026-07-30).

## What is ready to run

`scratchpad/refit_conf_shipped.py` — written, NOT yet executed. It:
1. runs the shipped pipeline (fold ON) on the 7 verified Brick-0 songs,
2. attributes per-predicted-chord correctness (root + family vs the GT chord
   holding most of the span),
3. fits an isotonic map raw→P(correct), validated **leave-one-song-out**,
4. writes `scratchpad/nnls24_conf_calibration_refit.npz` ONLY if LOSO beats the
   deployed map on the same data.

**Do not deploy a fit that only wins in-sample.** 7 songs vs the deployed map's
100 — a curve that wins in-sample and loses LOSO is an overfit and must be
rejected. If LOSO does not beat it, the honest outcome is "the deployed map
stays; we need audio for a bigger corpus", and RWC audio is NOT on disk (only
`data/cache/rwc/rwc_nnls24.npz`, features only).

Deploy path if it wins: overwrite `harmonia/models/nnls24_conf_calibration.npz`
(that is where `chord_pipeline_v1` reads it from — note the old script writes to
`data/models/`, a different path). Keep `HARMONIA_NNLS24_CALIB=off` working.

## Second-order effect worth keeping in view

The vocabulary fold lowers displayed confidence while RAISING accuracy: Misery's
median went 0.399 → 0.342 after re-baking while Brick-0 gained +2.12 pp partial.
Averaging posteriors flattens the peak and the stale map turns that into a lower
percentage. Any refit must be done with the fold ON, which the script does.

**Rejected alternative, recorded so it is not retried:** replacing the calibrated
probability with a peak-minus-runner-up margin. It survives averaging better but
is uncalibrated — it would look nicer and mean less.

## Also in flight

A background agent is fixing the form-timeline highlight (stuck on "A", does not
follow playback) in `harmonia/output/app_shell.html`. It was told to first
determine whether the BAR highlight also sticks — if it does, the suspect is
`chart_display._clip_spans_in_play_order` (added today), and it was told to
report rather than fix that file.

## Uncommitted in the working tree (NOT mine, do not sweep into a commit)

`harmonia/output/chart_model.py` carries ~300 lines of chordtone/section-repr WIP,
plus `irealb_fetcher.py`, `theory/local_key.py` and their tests. Today's chip fix
was staged as explicit hunks around them; keep doing that.
