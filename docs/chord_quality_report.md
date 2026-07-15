# Chord quality disambiguation + harmonic-prior ablation (2026-07-15, Agent 2)

Flat 5-class quality head (maj/min/dom/hdim/dim) and a test of whether trigram
harmonic context priors improve minority-class (dom/min) recall.
Repro: `scratchpad/agent2_quality_trigram.py`. Model:
`data/models/chord_quality_disambiguator_v1.pt`. Metrics JSON:
`docs/chord_quality_report.json`.

## Data / setup

- Corpus `billboard_training_corpus_full.npz`: 114,741 chords, 887 songs, 12-d
  McGill-NNLS chroma, oracle boundaries. Class mix maj 73072 / min 26182 /
  dom 14975 / hdim 200 / dim 312 (heavily maj-skewed).
- Song-stratified 80/10/10 (train 709 / val 89 / test 89 songs).
- **Frame = GT-root-relative** (`np.roll(chroma,-root)`). This is the mission's
  "evaluate quality only where the bass/root is correct" constraint: Agent 1's
  `bass_predictions_train_val_test.npz` was **not present**, so the GT root is
  used as the correct-bass proxy. Numbers therefore isolate quality-vs-quality
  discrimination from root errors — realistic cascade will be lower (Phase 2B).
- Head: 12→64→32→5 MLP (LayerNorm+GELU+Dropout0.3), inverse-freq class weights,
  AdamW, cosine schedule, 120 epochs.

## Acoustic-only result (already clears every target)

| class | recall | target | met |
|---|---|---|---|
| maj  | 0.752 | >0.70 | ✓ |
| min  | 0.824 | >0.60 | ✓ |
| dom  | 0.665 | >0.60 | ✓ |
| hdim | 0.688 | — | |
| dim  | 0.786 | — | |

Balanced acc **0.743**, raw acc 0.757. Confusion:
`docs/plots/chord_quality_confusion_matrix.png`. dom errors split maj 0.18 /
min 0.13 — the classic "is there a ♭7, and is the 3rd major or minor" ambiguity,
not a structural collapse. On the correct-root frame the acoustic head alone is
already excellent by the mission's bar.

## Harmonic-prior ablation — NEGATIVE result

Built root-relative context priors P(quality | prev_root−root, next_root−root)
from train only (Dirichlet-smoothed to the global marginal) and combined as
`logits + λ·log P(q|ctx)`. Plot: `docs/plots/harmonic_prior_ablation.png`.

| config | λ | bal acc | dom rec | min rec |
|---|---|---|---|---|
| acoustic-only | 0 | **0.743** | **0.665** | 0.824 |
| prev-bigram | 0.5 | 0.721 | 0.528 | 0.867 |
| next-bigram | 0.5 | 0.697 | 0.542 | 0.869 |
| trigram | 0.5 | 0.736 | 0.537 | 0.876 |
| trigram | 1.0 | 0.623 | 0.370 | 0.864 |
| trigram | 2.0 | 0.464 | 0.135 | 0.801 |

**Every prior monotonically lowers dom recall and balanced accuracy**, while
*raising* raw accuracy (up to 0.847) by shifting ambiguous chords onto the
maj-heavy marginal. This is the same mechanism as Phase 2A (known_issues
2026-07-15): the corpus is dominated by tonic/subdominant/dominant fifth-moves,
so a context prior reinforces the majority (maj/min) at the expense of exactly
the minority (dom) discrimination it was meant to help. Priors here trade the
metric we care about (balanced / dom recall) for the one we don't (raw acc).

## Error analysis / why priors can't help dom here

dom's context is *not* distinctive enough at the root-relative-interval level:
a V7 (prev ii = +5 up... i.e. −5, next I = +5) shares its context signature with
plenty of maj chords on the same scale degrees, so P(dom|ctx) never dominates
P(maj|ctx). The discriminating evidence for dom is acoustic (the ♭7 at index 10
of the root-relative chroma), which the acoustic head already reads. Context adds
prior mass, not new evidence, and the prior is maj-biased.

## Recommendation

- Ship the acoustic-only head (`chord_quality_disambiguator_v1.pt`); it meets all
  targets on the correct-root frame. Do **not** wire the trigram prior as a
  posterior multiply — documented negative.
- The remaining, real bottleneck is the **root frame in realistic cascade**
  (Phase 2B: rotating by a *wrong* predicted root drops balanced acc to ~0.52).
  The lever is a bass/lowest-note feature for the root model (known_issues Phase
  2A/2B), or folding quality-under-root-uncertainty into the #27 joint Viterbi —
  not more quality-side context priors.
- Domain caveat (CLAUDE.md rule 6): all numbers are oracle-boundary McGill-NNLS;
  the gap to production BP48 chroma (#31) still blocks drop-in wiring.
