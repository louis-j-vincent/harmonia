# Part 15 — The diagnostics were wrong; the fix was real anyway

## A design brief built on numbers that didn't exist

This session started with a detailed brief: build `beat_seq_model_v3` to beat a
POP909 baseline of *overall 77.6% root, interior 93.6%, boundary 64.1%, beat-1
62.2%*. The architecture followed from those numbers — the model is "saturated on
interior beats," downbeats are "the worst," so add beat-position encoding, a
canonical-form scorer, a wider window.

Rule #2 says screen the premise before implementing. So before writing any model,
I re-measured the baseline on POP909 001-005 under one consistent harness. None of
the four numbers reproduced, and two were *reversed*:

| metric | brief | measured (v2) |
|---|---|---|
| interior | 93.6% | 83.9% |
| boundary | 64.1% | 72.8% |
| beat-1 (downbeat) | 62.2% (worst) | 84.9% (best, on the librosa grid) |

A provenance hunt found why: the brief's numbers were **real numbers from unrelated
evals, relabeled**. 77.6% was the jazz-corpus standalone-disjoint root (wrong
corpus). 93.6% was a *majmin oracle-ceiling*. 62.2% was literally the existing v3's
own majmin CV, lifted from a commit message. And the canonical-form scorer the brief
called "the biggest gain" was **already built** — and net-neutral vs v2. This is
issue #17's calibration-warning family, and it's now issue #18.

## The one real diagnostic, and the reframe

One claim survived: ~46-51% of root errors are ±5/±7 semitones — root confused with
its 4th/5th. Confirmed on the canonical jazz corpus (51%), where walking bass makes
it worse. The pessimistic read is "that's Basic Pitch's bass wall." But a cheap
rescuability probe killed that: of the 5th-apart errors, the true root is in the
model's **top-2 for 86%**, and is an adjacent beat's prediction for **82%** — 92.5%
rescuable. The right answer is sitting in the neighbourhood; it just loses the
per-beat argmax. So the lever isn't better single-beat acoustics — it's harmonic
*context*.

## A vacuum bake-off, and two good ideas

I built a leak-free bake-off: irealb/jazz1460, disjoint even/odd song split, every
model trained on train songs only, one metric. On oracle chord segments (mean chroma
per chord ±4 chords context):

- **Within-chord mean chroma alone → 92.7%.** Averaging over a chord's beats makes
  the root dominate the 1-then-5 bass — exactly as expected.
- **Key-agnostic canonical scorer → 95.8%,** halving the 5th-error share.
- The human's **bass-anchored** idea — fix the rotation on the *observed* bass note
  (bass→C), predict root as an offset from the bass, recover the absolute root from
  the known anchor — hit 94.7% standalone and, ensembled with the canonical view,
  **96.1%: the best.** The two rotation strategies (search-all-12 vs bass-observed)
  are complementary.

Then the per-beat bake-off, which is what deployment actually sees. Here the spread
opened up: the v2-style key-biased LR scored **86.7%**; the canonical scorer **92.9%**;
canonical ⊕ bass-anchored **93.3%**. +6.6pp, and it holds at the boundary beats.

## Shipped: beat_seq_model_v4

Trained canonical (±4, key-agnostic MLP) ⊕ bass-anchored (LR) on jazz1460 + POP909,
20 epochs — 40 overfit and silently regressed pop, caught by re-checking held-out
POP909 (rule #1). Held out, v4 beats v2 on *both* domains: jazz +6.6pp on the clean
split, POP909 79.4→80.4 (boundary 72.8→75.1). Wired into `chord_pipeline_v1`
(`_get_beat_seq` prefers v4→v2→v1); 223/223 tests green.

## What it doesn't solve

Two honest caveats. This is **per-beat root** only — end-to-end MIREX with v4 in the
loop is not yet run. And the deeper result: root-ID is **~96% given correct
boundaries**. So the dominant remaining root error is now **segmentation**, not the
root model. The next lever moved.
