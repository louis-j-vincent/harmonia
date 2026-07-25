# Occam / form-repetition vote for the ±P5/P4 fifth-bleed — SCREEN + BUILD (null)

**2026-07-24. Disk-free (~1.0 GiB, below 1.5 floor — no decode; reused
captured predictions, cached `bt`, `fifth_errors.json`).** The one remaining
motivated lever after the two acoustic levers failed (novelty-snap −1.20pp,
min→dom corrector ~1% precision): use the song's repeating FORM (out-of-frame,
non-acoustic) to majority-vote outlier bars. Verdict: **refuted as a
fifth-bleed brick; the only measurable gain is a single-song, oracle-dependent,
sibling-regressing general Occam denoiser.** No brick shipped.

## STEP 1 — premise screen (fold GT+pred by detected form period, classify each fifth-error span)

Per-form-position: is the model correct in a MAJORITY of repeats (VOTABLE) or
wrong in most (SYSTEMATIC)? Pooled over the 4 fifth-error songs (134.8 s):

| class | seconds | % | meaning |
|---|---|---|---|
| **VOTABLE** | 34.2 | **25.4%** | form-vote *could* recover (ceiling) |
| SYSTEMATIC | 20.7 | 15.3% | model wrong here in the majority → vote can't help |
| FORM-BROKEN | 35.1 | 26.0% | GT itself varies across repeats → vote ill-defined |
| NO-PERIOD | 44.9 | 33.3% | no repeating form (bein_green, georgia) |

Per song: blue_bossa P=16 (GT rec 0.96) 42% votable / 29% sys / 29% broken;
blue_bossa_backing P=16 27% votable / **69% form-broken**; bein_green &
georgia **P=None** (through-composed — lever structurally inapplicable, 45 s).
Ceiling if every votable second were perfectly corrected: **+2.07 pp**
(34.2/1653.8 s scored). Optimistic (ignores collateral + vote fragility) but
non-trivial → worth building.

Mapping to the isolation doc: VOTABLE ≈ the chorus-inconsistent boundary bleed
(25% MUSX-BOUND); SYSTEMATIC ≈ the every-chorus-identical min→dom label bleed
(56% MUSX-LABEL). The systematic min→dom errors are, as LEVER 2 proved, the same
in every chorus (consistent acoustic ambiguity) → the form cannot rescue them.

## STEP 2 — build the corrector, measure end-to-end (control 0.7367 reproduced exactly)

Recall-preserving surgical splice on the shipped pred: detect period (≤40 bars),
per-position modal root+qual, snap outlier bars where the majority frac ≥ τ.

**Production (period self-detected from the noisy pred):** every τ/policy nets
**≤ 0.** blue_bossa self-recurrence @P16 = **0.69 < 0.70 threshold → P=None → never
fires** on the only beneficiary. blue_bossa_backing (self-detects P=16) regresses
every setting (−0.4 to −1.6 pp).

**Oracle form (P=16 handed in for both blue-bossa tunes) = the reachable ceiling:**

| policy | best τ | pooled Δroot | blue_bossa | blue_bossa_backing |
|---|---|---|---|---|
| FIFTH (targeted) | 0.8 | **+0.0018** | +0.007 | −0.002 |
| ALL (general denoise) | 0.6 | **+0.0069** | **+0.033** | **−0.016** |

- The **targeted fifth-only** policy — the actual motivated lever — recovers
  ~1/8 of its own 2.07 pp ceiling: **+0.18 pp pooled**, and even that is
  single-song with the sibling regressing. Refuted.
- The only real number (+0.69 pp) comes from the **ALL** policy = a *general*
  form-denoiser, not a fifth-bleed fix. It is **one song** (blue_bossa +3.3 pp);
  its structural twin **blue_bossa_backing regresses −1.6 pp** with the identical
  P=16 form. Sign-inconsistent across the only 2 form-songs on the bench =
  textbook single-song mirage (error-pattern #5). Fails "no per-song regression."

## Why blue_bossa helps but backing hurts (general principle)

Form-vote wins only at the intersection: base already bad (blue_bossa root 0.624
→ lots of chorus-inconsistent noise to clean) AND form detectable AND errors
inconsistent. Where the base is already clean (backing 0.879) the snap mostly
overwrites correct-but-non-modal bars → collateral, exactly like LEVER 1. That
intersection is a single song here, and even there needs an oracle to detect.

## Relation to the existing Occam pass

`occam_compress_bars`/`detect_loop_pattern` already do per-position modal snap,
but cap at **max_period=8** (blue_bossa's form is 16 → never folded) and gate
snaps on per-bar confidence (high-conf min→dom deviations are kept by design).
Extending it to P=16 would capture blue_bossa's +3.3 pp *and* backing's −1.6 pp —
not a safe change.

## Bottom line

Form/Occam cannot rescue the fifth-bleed. The votable mass is real but small
(25% ceiling), single-song, and mostly recoverable only by a general denoiser
that damages its structural sibling and can't self-detect the form in production.
The systematic min→dom core is chorus-consistent by construction — it needs a
**different musx front-end** (the parallel bake-off), not post-processing.

**New files (nothing staged, no brick):**
`scratchpad/occam_formvote_screen.py`, `scratchpad/occam_formvote_build.py`,
this note.
