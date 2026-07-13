# 17 — The reranker that wasn't, and confidence you can trust

*2026-07-13*

## The audit

After a week of fast, scattered parallel sessions, a four-agent audit swept the
whole project: architecture and redundancy, history and dead ends, empirical
biases, and — the real question — whether we have the ingredients for the
principled Bayesian model this project is aiming at.

The audit's buried lede: **we already built that model, then froze it.**
`chord_hmm.py` is a coherent generative semi-Markov HMM — emissions, key prior,
transitions, explicit-duration Viterbi, forward–backward posteriors. It was
frozen because its *emissions* (fixed templates) were the bottleneck, not its
structure. Gen-2 fixed the emissions with trained classifiers but replaced the
probabilistic backbone with a stack of greedy, hand-weighted rerankers. The plan
that fell out: re-couple the two halves. Full trail in
`docs/handoff_2026-07-13_project_overview_and_bayesian_plan.md`.

Then the plan met reality faster than expected.

## The harness was measuring a different pipeline

A nightly session had just shipped the 801d two-pass ctx classifier (+2.7pp
majmin over 684d) and noted, honestly, that `eval_irealb_e2e.py` — the harness
behind the progression reranker's "default ON, +1.0pp" decision (#21) —
**bypasses the ctx family classifier entirely**. That is the audit's
contamination pattern in live form: a proxy harness quietly standing in for the
production path (it had happened twice before, issues #6 and #11 in the
contamination table; one of those reversed on re-measurement too).

So before flipping any defaults, we filled in the missing cells of the 2×2 on
the *real* path (`infer_chords_v1`, jazz1460 held-out, n=25):

| ctx variant | reranker | root | majmin | 7ths |
|---|---|---|---|---|
| 684d | OFF | 88.7% | **84.0%** | **59.2%** |
| 684d | ON (prod default) | 88.7% | 80.4% | 56.7% |
| 801d two-pass | OFF | 88.7% | **84.0%** | **59.2%** |
| 801d two-pass | ON | 88.7% | 83.1% | 57.7% |

The bypass harness's +1.0pp is **−3.6pp on the real path.** The default-ON
decision doesn't survive — it reverses. And the two corollaries sting more:

1. **801d's +2.7pp was recovery, not gain.** With the reranker off, 801d's
   output is byte-identical to 684d — its refined q5 distribution only ever
   flowed *into* the reranker. It was winning back part of a self-inflicted
   wound.
2. **On POP909 the reranker never fires at all** — ON vs OFF byte-identical on
   all 5 songs. The A/B that "validated" it there was vacuous.

`use_progression_prior` is now default OFF (issue #25). The encoder is not
dead: its information belongs in a joint decode as a real transition factor,
which is exactly build-order step 2 — in progress as this posts.

The general lesson, added to the pile: **a harness is part of the model.** Every
number from `eval_irealb_e2e.py` (ProgressionEncoder gain, phase-fix e2e
effects) now reads "on the bypass path", not "in prod".

## The confidence shown wasn't the decision shown

The app's whole premise is a model that tells you where it's unsure. The audit
found the confidence pipeline undermined that in three stacked ways:

1. **Stale after rerank** (real bug, now fixed): the 8a/8b rerankers flipped a
   chord's quality but carried the *pre-rerank* confidence into the output. The
   fix makes both rerankers return the posterior of the decision they actually
   made (the normalized 2-way winner for the diatonic prior;
   `softmax(log_aco + w·log_enc)` at the chosen quality for the progression
   rerank), consumed at flipped positions. Red-first tests.
2. **Root-blind**: the quality heads never see the root, so a confidently-wrong
   root surfaced as a confident chord. Output chords now carry `root_conf`
   (span-mean beat-seq posterior at the label's root) and the displayed
   confidence fuses both.
3. **Uncalibrated**: max-softmax, never validated. Now: an isotonic map fitted
   on held-out jazz1460 songs (disjoint from every eval split), auto-loaded
   like every other model artifact, applied at output assembly only — labels
   and gates by construction untouched.

First fit: raw fused ECE 0.191 → calibrated 0.056 on a disjoint test block —
a 3.4× improvement that still missed the 0.05 gate. The miss itself was
informative: the fit block was much easier than the test block (82.4% vs 74.4%
base accuracy), so the map transferred optimistically. Same-pool interleaved
splits (still song-disjoint) are refitting as this posts; the residual
question — how well calibration transfers across song difficulty — is real and
worth keeping visible rather than tuning away.

## Where this leaves the model

The three stacked rerankers are now one OFF-by-default legacy path and one
honest baseline: **root 88.7 / majmin 84.0 / 7ths 59.2 on held-out jazz1460**,
from the classifiers alone. Everything the rerankers were trying to add —
progression grammar, local key, and the 801d key-relative context — re-enters
through the joint decode over root × quality, where the audit's rescuable-root
finding (true root in top-3 for 95.2% of errors) can actually be cashed in,
and where a user's confirmed chord can propagate to its neighbours through the
transition factor instead of dying at the display layer.
