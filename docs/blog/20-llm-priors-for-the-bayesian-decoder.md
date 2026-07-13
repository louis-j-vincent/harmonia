# 20 — LLM priors for the Bayesian decoder (Mission 5)

The idea: let an LLM analyst read a song and hand the Bayesian decoder good
priors — key, section structure, per-position quality expectations, transition
bias — so it lands on the right answer instead of leaning on generic corpus
statistics. Design + prototype + a convergence study.

## What got built

- `scripts/llm_chord_priors.py` — an analyst that emits a priors JSON
  (structure / `P(q|root)` / `P(root|prev)` / confidence). Real path calls
  `claude-opus-4-8` with structured output; an offline rule-based analyst emits
  the identical schema so the whole thing runs with no network (and doubles as
  the baseline the LLM must beat). `to_bayesian_factors` maps the JSON onto the
  decoder's *existing* factor hooks.
- `scripts/eval_llm_priors.py` — a controlled convergence sim.
- `docs/mission_5_*` — research, architecture, integration.

## Three honest corrections to the brief

1. **"Bayesian models are slow to converge from random init" doesn't fit our
   decoder.** `joint_decode` is an *exact* Viterbi — no init, nothing to
   converge. So I built the convergence study against the solver class where it
   *is* real (coordinate-ascent / EM warm-start), as a labelled simulation, and
   found the LLM's payoff is **final accuracy and robustness-under-noise, not
   fewer iterations**. On Autumn Leaves: +35pp root / +40pp (root,q5) at σ=1.2,
   growing to +45pp at σ=1.6, shrinking to +6pp at σ=0.8 — i.e. worth the most
   exactly in the degraded-audio regime that's our real bottleneck (#19).

2. **Don't sell a generic transition grammar.** That slot is saturated (#27 —
   key-local bigram, encoder fusion, density-ratio fusion all net-negative).
   What the LLM adds that a corpus bigram can't is *song-specific* structure:
   this tune is Gm, these two spans repeat, this V is a `7b13`.

3. **The LLM can't hear.** Its input is the iReal chart + metadata (+ optional
   numeric front-end summary). Strong case: a known standard with a chart. The
   circularity that implies (chart-derived prior vs same-chart audio) is why the
   eval scores convergence/robustness, never chart-agreement.

## The nice part: it's the user-constraints interface

The LLM is just "an automated annotator." Its key→`tonic`, repeats→`pool_groups`,
`P(q|root)`→`q5_bonus` map onto the exact hooks `user_constraints.py` already
feeds `joint_decode`. So: no new decoder recursion; the "never pool blindly"
rule (#28 vs Candidate C #1) is inherited for free; and a human user's 40-nat
confirm always overrules the LLM's ≤8-nat tilt. Confidence → strength (nats) is
the honesty knob — a low-confidence read on an unknown tune is near-inert.

Offline analyst on Autumn Leaves reads it correctly (Gm; A→ø, D→dom7, G→min,
Bb→maj; ii-V-i; pool group bars 1-8≈9-16). On All The Things You Are it
correctly declines to pool (repeats are transposed, not identical) and reports
lower confidence (0.78) for the more chromatic harmony.

## Not claimed

Any real-audio end-to-end number — gated on the Mission 1 benchmark like every
other real-audio claim here. Ship gate when it lands: `use_llm_priors=True` iff
Δ(root+q5) ≥ +2pp with 0 regressions on held-out jazz.
