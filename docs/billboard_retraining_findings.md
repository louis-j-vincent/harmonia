# Billboard retraining — strategy, verification & "What We Learned"

*2026-07-15. Supersedes the 4-line `billboard_training_results_v2.md`.*

This document was written to design a retraining campaign (Part 1), but the
premise-check step (per CLAUDE.md rule 2: "screen the premise cheaply before
implementing") found that the shipped Billboard v2 quality model rests on a
ground-truth collapse bug. The finding reframes everything downstream, so it
leads.

## TL;DR

- **Root model v2 is real:** song-grouped 5-fold CV = **76.2% ± 1.6%**, matching
  the headline 76.5%. Honest number.
- **Quality head v2 "81.7%" is an artifact.** The corpus was built with
  `BillboardDataset(chord_type="majmin")`, which collapses every `dom/hdim/dim`
  into maj/min/N. The "5-way" head saw **0 dom, 0 hdim, 0 dim** examples
  (maj 83,638 · min 26,162 · rest 0). Its 77.9% CV ≈ the 76.2% majority floor;
  minor-recall is **0.21** (79% of minors called major). It is a maj-predictor.
- **Corrected-GT retry (`chord_type="full"`, 97,686 chords):** a class-weighted
  5-way head reaches **balanced acc 43.8%** (chance 20%) — real signal in every
  class — but **dom recall is only 0.31**, confused ~evenly with maj/min. This
  reproduces issue #19's dom→maj/min confusion on a second, independent dataset.
- **These are all oracle-boundary numbers** (features are mean chroma over the GT
  `[t0,t1]`), and on **McGill NNLS chroma**, not the production Basic-Pitch
  feature space. Neither model is drop-in for `chord_pipeline_v1`.

## Verified facts about the shipped assets

| Asset | What it actually is |
|---|---|
| `billboard_root_model_v2.npz` | 12→12 logistic regression on McGill NNLS chroma (A-referenced). Real, ~76% CV at oracle boundaries. |
| `billboard_quality_head_v2.pt` | 12→64→32→5 MLP, but trained on majmin-collapsed GT → only maj/min ever seen. Last 3 logits are dead. |
| `billboard_training_corpus_v2.npz` | 124,781 rows; `quality_idx ∈ {0(maj),1(min),-1(N)}` only. Features = per-chord mean of McGill `bothchroma.csv` (first 12 of 24 cols). |
| Billboard audio | **Not present.** Only pre-extracted chroma CSVs. End-to-end eval on Billboard is impossible without downloading audio. |

## Error analysis (Part 3)

Interactive diagnostic: `docs/plots/billboard_error_analysis.html`
(two quality confusion matrices + root confusion + hypotheses).

**Root errors are structured, not noisy.** Top off-diagonal confusions are almost
all ±5/+7 semitones (the fourth/fifth): D→A (1140), A→D (868), F→C (791),
G→D (716), C→G (649). A triad and its dominant share 2 of 3 pitch classes, and
mean-pooled chroma bleeds in the neighbouring chord's bass. → a bass-register
prior or a key/transition prior targets exactly this (cf. the POP909 oracle
sprint: bass evidence moved root 53%→83%, known_issues §1).

**Quality (corrected full GT) confusion:** maj recall 0.47 / min 0.56 / dom 0.31 /
hdim 0.39 / dim 0.46. dom is the hard class — it smears into both maj and min,
i.e. the model hears "a triad" and can't reliably find the b7. Same bottleneck
the whole project keeps hitting: **quality/7th discriminability, not decoding.**

## Strategy assessment (Part 1) — revised by the findings

1. **GT source priority.** Billboard is a fine *root/majmin* teacher (large,
   clean, real audio) but a **weak quality teacher**: even with `full` GT, dom is
   ~11% and hdim/dim <0.5% of chords, and it's pop/rock, not jazz. For Autumn
   Leaves / jazz-7th quality, corrected iRealb + the real-audio YouTube corpus
   (issue #19) remain the right teachers. **Recommendation: don't retrain the
   production quality head on Billboard.** Use Billboard to pressure-test the
   root stage and as majmin-domain augmentation only.
2. **Feature-domain gap is the real blocker to integration.** Billboard models
   live on McGill NNLS chroma; production uses 48-dim root-shifted BP chroma. A
   model trained on one will not transfer to the other without a bridge (re-extract
   BP-style features from Billboard audio — which we don't have — or retrain on
   the production feature pathway). This must be resolved before any "wire it in."
3. **Ordering, if we proceed:** (a) fix the corpus to `chord_type="full"`
   [done here, as a CV probe]; (b) decide feature space; (c) only then touch the
   decoder/duration prior. Retraining the decoder on collapsed GT would have been
   wasted effort — exactly the trap the mission set out to avoid, in reverse.

## Dead-ends retried (Part 4)

The mission's thesis — "prior dead-ends failed because of broken GT, retry them"
— is sound as a discipline, but the specific new asset (Billboard quality) was
itself broken by a GT bug. Net finding on the dead-ends:

- **dom-class collapse ("dom7 0%")** is *refuted again*, consistent with issue #19:
  with correct GT dom is learnable (recall 0.31, balanced-acc well above chance),
  it's a *confusion* problem (dom↔maj↔min), not a *collapse*.
- **Emission discriminability is still the binding constraint** (known_issues §1,
  §5). Nothing here contradicts it; the root/quality factorization (root≈bass,
  quality≈chroma-template) holds on Billboard too.

## Ranked next steps

1. **Decide the feature space** for any Billboard use (BP-style vs NNLS). Without
   this, the v2 models cannot enter production. (blocker, cheap decision)
2. **Retrain quality on corrected GT where it matters:** the real-audio YouTube +
   corrected-iRealb jazz corpus, `full` vocabulary, class-weighted, report
   *balanced* accuracy and per-class recall — never overall-acc against an
   imbalanced set again.
3. **Attack dom↔maj/min directly:** the b7 is present but low-contrast (issue #19).
   Contrast features (HPSS/whitening) or a bass-anchored root + separate 7th
   detector, per the POP909 factorization finding.
4. **Root stage:** fold a bass-register / key prior to kill the fifth/fourth
   confusions — highest-leverage, cheapest, well-supported win.
5. **Rewrite `billboard_training_results_v2.md`** (the 4-line file) with methodology,
   or delete it — it currently overstates the result.

## Reproduction
- Root/quality CV + confusions: `scratchpad/cv_eval.py`
- Corrected-GT (`full`) re-extraction + CV: `scratchpad/reextract_full.py`
- Shipped trainer (the one with the majmin bug): `scripts/train_billboard_from_features.py`
  line 99 `BillboardDataset(chord_type="majmin")` is the root cause.
