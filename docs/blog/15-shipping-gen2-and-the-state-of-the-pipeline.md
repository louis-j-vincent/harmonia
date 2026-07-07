# 15 — Shipping Gen-2 and the State of the Pipeline

*2026-07-08*

## The synthesis problem

After several parallel agent sessions, the codebase had accumulated three
competing pipeline implementations, the best models (beat-sequence, ctx MLP)
were undeployed, and both production entry points (server + CLI) were still
running Gen-1. This session was about cutting through that and shipping a
coherent Gen-2 baseline.

## What Gen-2 is

`harmonia/models/chord_pipeline_v1.py` — a clean standalone module that wires
together every validated improvement from the last two weeks:

1. **Tempo-grid de-jitter** — librosa's tempo is accurate ±1% but per-beat
   times jitter; imposing a uniform grid at the detected tempo + circular-mean
   phase recovers ~20pp majmin on metronomic audio.
2. **Beat-sequence model** (`beat_seq_model.npz`, 88.3% CV) — windowed LR
   with ±2-beat context. Beats degrade gracefully on real audio; the segment
   model wins on clean, beat-seq wins under degradation.
3. **Trained family + seventh classifiers** — norm_blocks, root-shift, SUM
   pooling (not mean). All four calibration bugs fixed by design.
4. **Ctx family model** (`ctx_family_model.npz`, 86.9% CV) — entropy-gated
   MLP that blends a LogReg baseline logit with an MLP context-window logit.
   Auto-loaded when the file is present.

## Numbers

| pipeline | root | majmin |
|---|---|---|
| Gen-1 (frozen HMM) | ~33% | ~29% |
| Gen-2 v1 | **60.5%** | **39.1%** |

Roughly double the Gen-1 baseline on POP909. Segmentation threshold barely
matters (theta 0.04–0.18 all within noise) — the same-label coalescence step
handles over-segmentation cleanly.

The ctx model is neutral on POP909 (it was trained on MMA jazz, tested on
pop) — expected. It should show gains on jazz audio.

## What's next (two models, one winner)

The user asked for both the base family model and the ctx model to coexist as
"two competing models with good results." The v1 module already supports this:
it uses ctx when `ctx_family_model.npz` is present, falls back to base
otherwise. A proper A/B on real jazz audio (not POP909) is the next evaluation.

The other pending item is the accomp_db regen with fixed vary_voicings — disk
is currently full (100%), which blocks it. Clean up `data/cache/accomp_varied/`
(347MB stale) to unblock.

## The calibration-bug pattern, again

Every session finds one. This time it was in the previous codebase, not in
what we built: the earlier vary_voicings created "independence" by omitting
pitch classes, which thinned the harmony and confused the chord classifier.
The fix (issue #13) was committed, but the impact won't show until DB regen.

The six error patterns in CLAUDE.md continue to hold: silent calibration bugs,
no premise-screening, GT trusted, partial fixes, single-song generalization,
cross-stage confounds. The counter-rules are the right habits — use them.
