# Mission 5 — Bayesian integration: exact seams

How the LLM priors enter the decoder, line by line against the current code.
Companion: `mission_5_llm_priors_architecture.md` (what/why),
`scripts/llm_chord_priors.py` (`to_bayesian_factors` produces these objects).

## 0. The convergence framing — read this first

The mission is framed as "Bayesian models are slow to converge from random
init; LLM bootstrap fixes that." That framing needs an honest correction for
*this* codebase:

- **The shipped labeler does not iterate.** `joint_decode.joint_decode` is an
  **exact** segment-chain Viterbi (`O(T·K²·25)`), and `semi_markov_decode` is an
  exact explicit-duration Viterbi. There is no random init, no EM loop, nothing
  to "converge" — the MAP is found in one pass. So "LLM init → faster
  convergence" is **not literally measurable** against the production decoder.
- **Where convergence *is* real:** any *learned per-song* component — fitting a
  song-specific transition matrix by EM, or a Gibbs/coordinate-ascent labeler
  over a coupled graph (the shape a future per-song refinement would take). Our
  eval (`eval_llm_priors.py`) exercises exactly that solver class as a
  controlled simulation.
- **What the eval found (Autumn Leaves, synthetic emission):** the LLM
  warm-start did **not** cut sweep count (uninformed coordinate-ascent
  fixed-points in 1 sweep — with a uniform transition prior there is nothing to
  propagate), but lifted final accuracy **+35pp root / +40pp (root,q5)** at
  σ=1.2, growing to **+45pp** at σ=1.6 and shrinking to +6pp at σ=0.8. So the
  integration is sold on **accuracy and robustness-under-degradation**, not
  iteration count — and it pays off most in the degraded-audio regime that is
  our real bottleneck (#19). Claiming "faster convergence" would be the kind of
  plausible-but-wrong number `CLAUDE.md` #1 warns about.

The integration below is therefore built to inject priors as **decode-time
factors** (which improve the MAP directly), and is *also* forward-compatible
with a future iterative learner (same factors as a warm start).

## 1. The three seams, all already present

`infer_chords_v1(...)` → `joint_decode(...)` already exposes every hook needed.
No new decoder code; only a glue function that fills existing arguments.

### Seam A — key/mode → `tonic` / diatonic prior

```python
# today
inferred = infer_key(...)              # noisy: #0 relative maj/min confusion
# with LLM prior (confidence-gated override)
tonic = factors.tonic if factors.confidence >= KEY_TRUST else inferred.tonic
res = joint_decode(segs, beat_proba, classify_fn, tonic=tonic, ...)
```

`joint_decode` already takes `tonic` and an optional per-segment `local_tonic`.
The LLM tonic also feeds `apply_diatonic_prior` (#20) with a *correct* key
instead of the inferred one — the exact condition (#20) says the diatonic prior
needs to fire.

### Seam B — `P(q|root,pos)` → `q5_bonus` callback

`joint_decode(..., q5_bonus=cb)` already folds an additive `(5,)` per-q5
log-score into the emission *before* the joint argmax (the slot the encoder's
H2 shallow fusion used, #27). The LLM prior is a drop-in:

```python
def q5_bonus(seg_idx: int, root: int) -> np.ndarray:
    row = np.zeros(5)
    for q5, nats in factors.quality_bonus.get(root, {}).items():
        row[q5] = nats                 # strength·(p − 1/5), already centred
    return row
res = joint_decode(segs, beat_proba, classify_fn, tonic, q5_bonus=q5_bonus, ...)
```

Root-dependent by construction — exactly the callback's contract. This targets
the live lever: q5 acc is 44% on real audio (#19).

### Seam C — repeats → `pool_groups`

`joint_decode(..., pool_groups=[[seg_i, seg_j, ...], ...])` ties segment groups
and **sums** their emission (P3 √N denoising, #28, +10pp q5). The LLM's
`structure.repeats` (bar-spans) map to segment-index groups via the existing
`user_constraints.build_pool_groups` / segment index lookup:

```python
pool_groups = bars_to_segment_groups(factors.pool_group_bars, segs, beat_times)
res = joint_decode(..., pool_groups=pool_groups)
```

This reuses `user_constraints.py` wholesale — the LLM's assertion enters the
*same* path a user's "these sections are the same" merge does, at lower trust.

### Seam D (optional) — transition bias → `bigram_logp`

`joint_decode(..., bigram_logp=table)` takes the 60×60 transition table.
Fold the LLM's per-song root bias additively onto the fitted bigram:

```python
table = load_bigram().copy()
for pr, dist in factors.root_transition_bias.items():
    for nx, nats in dist.items():
        table[pr*5:(pr+1)*5, nx*5:(nx+1)*5] += nats
```

Kept **default-off / low-strength** — the generic grammar slot is saturated
(#27, all three fusions net-negative). Only enable once measured; expect small.

## 2. Confidence → prior strength (the honesty knob)

`to_bayesian_factors(analysis, max_nats=8.0)`:

```
strength = max_nats · confidence          # 0 … 8 nats
quality_bonus[root][q5] = strength · (p − 1/5)
root_transition_bias[prev][next] = strength · p
```

- **Ceiling 8 nats vs user `CLAMP_NATS ≈ 40`** — the LLM is ~5× weaker than a
  human confirm, in the *same* log-space, so a user always overrules the LLM and
  the LLM always overrules a diffuse prior but not sharp acoustic evidence.
- **Linear in confidence** — a 0.3-confidence analysis contributes ≤2.4 nats
  (barely perturbs the decode); a 0.85 analysis contributes ≤6.8 (meaningful,
  never pinning). This is the mechanism that makes "honest about what it knows"
  operational: a hallucinating LLM on an unknown tune reports low confidence and
  is automatically down-weighted to near-inert.

## 3. Precedence when user + LLM + priors all speak

All are additive log-terms on the same emission cells, so precedence is just
magnitude:

```
user confirm (≈40) ≫ LLM prior (≤8) ≫ diatonic/duration priors (≈2–4) ≳ 0
                                        ▲
                         acoustic emission (data-driven, unbounded)
```

Strong acoustic evidence can still beat an 8-nat LLM tilt — which is correct:
when the recording plainly plays something other than the chart, the ear wins
(trust hierarchy caveat, `CLAUDE.md` #3, "a strong music-theoretic hypothesis is
allowed some leeway against a single disputed label" — here the roles reverse,
and that's the safe direction).

## 4. Forward-compatibility with an iterative learner

If a future component fits a per-song transition matrix (EM) or runs a
Gibbs/coordinate-ascent labeler, the *same* `factors` object is the warm start:
init labels = LLM MAP, init transition = fitted bigram + LLM bias, tied groups =
pool spans. `eval_llm_priors.py` already implements that solver and shows the
warm start dominates on final accuracy — so the integration serves both the
current exact decoder (as decode-time factors) and any later iterative one (as
init) with no schema change.

## 5. Validation status (honest)

- **Done, in-repo, reproducible:** offline analyst on Autumn Leaves emits a
  functionally-correct prior (Gm; A→ø, D→dom7, G→min, Bb→maj; ii-V-i; pool group
  bars 1-8≈9-16). `eval_llm_priors.py` shows +35pp root / +40pp (root,q5) on a
  controlled synthetic-emission convergence sim, gap growing with noise.
- **Designed, not yet wired:** the four seams above are argument-fills on
  existing `joint_decode` hooks — a ~30-line glue function in `infer_chords_v1`,
  gated behind a `use_llm_priors` flag defaulting off (same discipline as every
  other opt-in prior).
- **Deliberately not claimed:** any real-audio end-to-end number. Gated on the
  Mission 1 benchmark (`data/real_audio_benchmark/`, #20/#28), like every other
  real-audio claim in this repo. Stopping criterion when it lands: ship
  `use_llm_priors=True` iff Δ(root+q5) ≥ +2pp with 0 regressions on the held-out
  jazz set — mirroring the #27/#28 gate style.
```
