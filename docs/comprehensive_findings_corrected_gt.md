# Comprehensive findings — corrected-GT retraining & error analysis

*2026-07-15. Synthesis for the 4-task mission (BP48 extraction, 3-representation
training, vocab/feature normalisation, error analysis + dead-end retries).*

**Read `docs/billboard_retraining_findings.md` first** — a concurrent session found
the root cause (issue #31, the majmin-collapse) earlier today. This document does
**not** re-litigate that; it (a) turns that transient CV probe into **durable,
reproducible artifacts**, (b) adds a **controlled root-feature ablation** that yields
a free +4 pts, (c) delivers the **vocabulary alignment** (Task 3), and (d) states
plainly which mission sub-tasks are blocked and why.

## What actually got built (durable artifacts)

| Artifact | What it is |
|---|---|
| `data/cache/billboard/billboard_training_corpus_full.npz` | 114,741 chords · 887 songs · corrected 5-way GT from `full.lab` · NNLS **bass+treble** chroma (24→12 summed). |
| `data/models/quality_head_nnls_full.pt` (+`.json`) | 12-64-32-5 class-weighted MLP. Balanced acc 0.41. |
| `data/models/root_model_nnls_full.npz` | 12→12 balanced logistic regression, z-norm baked in. Balanced acc **0.84**. |
| `docs/plots/billboard_corrected_gt_analysis.html` | Interactive: quality confusion, per-class recall, root-error-by-interval, feature ablation. |
| `docs/chord_vocab_alignment.md` | iRealb ↔ Billboard ↔ Q5 mapping (Task 3). |

Repro: `scratchpad/billboard_full_corpus.py` → `train_eval_full.py` → `root_ablation.py` → `make_dashboard.py`.

## Task-by-task status

### Task 1 — BP 48-dim extraction: **BLOCKED (documented)**
Billboard **audio is not present** — only McGill's pre-extracted NNLS chroma CSVs.
Disk is at **99% (3.9 GB free)**; downloading ~890 tracks would risk the exact
disk-full incident CLAUDE.md flags. Per mission fallback rules, dropped in favour of
NNLS. Consequence: **BP 12-dim and BP 48-dim representations, and every transfer
experiment (A–D), are unreachable** without audio + disk headroom. Only the **NNLS
12-dim** representation is trainable, so the "3-representation bake-off" collapses to
one honest representation. Do not simulate the other two.

### Task 2 — Training (NNLS 12-dim only)
Song-stratified 80/10/10 (709/89/89 songs, zero song leakage). Corrected 5-way GT.

- **Root (LR, class-weighted):** test acc 0.837, **balanced 0.839**. Real signal in
  all 12 classes.
- **Quality (MLP, class-weighted):** test acc 0.454, **balanced 0.410** (chance 0.20,
  maj-floor 0.635). Per-class recall maj 0.56 / min 0.34 / **dom 0.15** / hdim 0.20 /
  dim 0.80. hdim/dim recalls are noise (≤5 & ≤39 test chords). The v2 "81.7% 5-way"
  head is confirmed an artifact — it never saw a dom/hdim/dim example.

### Task 3 — Vocabulary unified; feature merge deliberately withheld
Vocabulary alignment across all three dialects is done and verified (see the
alignment doc). The cross-dataset **feature merge is BLOCKED on purpose**: iRealb has
no native chroma; its audio form is 48-dim Basic-Pitch, a different sensor from
Billboard's NNLS. z-normalising each to N(0,1) masks but does not remove that domain
gap. Shipping a pooled `combined_…npz` across NNLS↔BP would be a silent-calibration
trap (rule 1). Merge only *within* a feature domain.

### Task 4 — Error analysis + dead-ends
Two structured error patterns dominate, both now quantified:

1. **Root: the fifth/fourth confusion.** **45% of all root errors are a P4/P5 apart**
   (P4 462 + P5 370 of 1,854). A triad shares 2/3 pitch classes with its dominant,
   and mean-pooled chroma bleeds the neighbour's bass. → a **bass-register / key /
   transition prior** targets this precisely (POP909 oracle: bass evidence moved
   root 53%→83%).
2. **Quality: dom↔{maj,min}.** dom (recall 0.15) smears ~evenly into maj and min —
   the b7 is present but low-contrast. This is issue #19's bottleneck reproduced on
   an independent, second dataset. **Discriminability, not decoding.**

## Dead-ends retried with corrected GT

| Prior claim ("failed") | Result now | Why |
|---|---|---|
| "5-way quality head works (81.7%)" | **Was an artifact** — collapsed GT | v2 built with `chord_type="majmin"`; 0 dom/hdim/dim examples. |
| "dom7 collapses to 0% recall" | **Refuted again** — dom is *learnable* (recall 0.15–0.31, balanced ≫ chance) | It's a *confusion* (dom↔maj↔min), not a *collapse*. Consistent with issue #19. |
| "bass chroma is enough for root" (v2's implicit choice) | **Beaten** — bass 0.798 → bass+treble **0.840** (+4 pts) | v2 discarded the treble half of `bothchroma.csv`. z-norm neutral; the gain is the extra register. Controlled, same-split ablation. |
| "just z-normalise and merge datasets" | **Correctly abandoned** | NNLS↔BP domain gap survives per-feature z-norm; merge would blend two sensors. |

The mission's thesis — *dead-ends may have failed on broken GT* — held in one
direction and reversed in another: the newest asset (Billboard quality) was itself
broken by a GT bug, and fixing GT did **not** rescue dom quality on NNLS (still
low-contrast). GT correction is necessary but not sufficient; the emission
discriminability constraint is still binding.

## Ranked next improvements (impact × effort)

1. **Bass/key/transition prior on the root stage** — *high impact, low effort,
   well-supported.* Directly attacks the 45%-of-errors P4/P5 confusion; POP909
   precedent is strong (53%→83%). Cheapest real win.
2. **Fold treble into the shipped `billboard_root_model_v2`** — *medium impact,
   trivial effort.* It currently uses bass-only; +4 pts is free. (New root model
   `root_model_nnls_full.npz` already does this.)
3. **Attack dom↔maj/min with contrast features** — *high impact, medium effort.*
   HPSS/whitening or a bass-anchored root + a *separate* b7 detector (per the POP909
   root/quality factorisation). The recurring project-wide bottleneck.
4. **Resolve the feature-domain question before any Billboard→production wiring** —
   *blocker, cheap decision.* Pick BP-style vs NNLS; re-extract accordingly. Until
   then keep Billboard as a root/majmin teacher only — **do not** wire the NNLS
   quality head into `chord_pipeline_v1`.
5. **Train jazz-7th quality where it's teachable** — corrected-iRealb + YouTube
   real-audio (BP domain), `full` vocab, class-weighted, report *balanced* acc +
   per-class recall. Billboard is pop and 7th-poor; it can't teach jazz quality.
6. **Delete/rewrite the 4-line `billboard_training_results_v2.md`** — it still
   overstates the collapsed-GT result.

## Standing caveats
All numbers are **oracle-boundary** (mean chroma over GT `[t0,t1]`) on **McGill NNLS**
chroma — **not drop-in** for the production 48-dim Basic-Pitch pipeline. Single
80/10/10 split (not CV) for the new models; the concurrent session's 5-fold CV
corroborates the root/quality magnitudes.
