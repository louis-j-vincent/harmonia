# Mission 5 — Research: how existing systems bootstrap chord structure

Prior art on getting a chord-recognition / Bayesian model off the ground with
good structure, and what that implies for an LLM-as-prior design here. Scoped to
what actually informs our decoder (`joint_decode.py`, `semi_markov_decode.py`);
not a literature dump.

## 1. Commercial chord apps (Chordify, Chord AI, "Spotify chord" features)

There is no published architecture for these, so this is inference from their
behaviour + the ISMIR/ACR literature they cite, flagged as such:

- **They do not bootstrap from an LLM.** The pipeline is a trained acoustic
  model (CNN/CRNN over CQT or chroma) → a **CRF / HMM decoder** whose transition
  matrix is a *fixed, corpus-learned* prior, not a per-song one. The "structure"
  they exploit is (a) a beat/downbeat grid (madmom-style) and (b) a global key
  estimate that biases the chord vocabulary. This is exactly our stack minus the
  per-song semantics.
- **Key is the highest-value cheap prior.** Every system estimates key first and
  uses it to down-weight out-of-key chords — the single most consistent
  structural prior in the field. We have this (`infer_key`) but it is
  error-prone (relative maj/min confusion, `known_issues.md` #0) and the
  *inferred* local key isn't reliable enough to exploit the diatonic prior
  end-to-end (#20). **An LLM that names the key/mode of a known standard
  outright removes that error source** — the clearest single win available.
- **Repetition structure is used for smoothing, not labelling.** Where these
  apps expose "sections", it is post-hoc (segment the SSM, copy the label of the
  most similar prior segment). This is the same P3 "parallelism as denoising"
  idea we validated (`known_issues.md` #28, +10pp q5 on real audio by pooling
  repeats) — and the same failure mode we hit blindly (Candidate C, #1: pooling
  *non-identical* repeats hurt quality). The lesson both directions agree on:
  **pool repeats only when something external asserts they are identical.** An
  LLM asserting "these two 8-bar strains are the same" is exactly that external
  signal, and it is higher-level than a chroma-SSM peak (which fires on shared
  *texture*, not shared *harmony* — the precise reason Candidate C failed).

## 2. Music LLMs / generative models (Jukebox, MusicLM, MusicGen, chord-LMs)

- **Jukebox / MusicLM / MusicGen** are *generative* audio models. They encode
  harmony only implicitly (in VQ-VAE codes / semantic tokens); none exposes a
  calibrated `P(chord | context)` you can lift out as a prior. They are the
  wrong tool for this — we need an explainable distribution, not a sampler.
- **Symbolic chord language models** (Hu­ang et al.'s transformer chord models;
  our own `progression_encoder.py`) *are* the right shape — a learned
  `P(quality | ±context)` — but our own experience with them is cautionary:
  the encoder reranker was **net-negative on the real path** (`known_issues.md`
  #21/#25) because it was wired as a greedy post-hoc override, and even as a
  shallow-fusion transition factor (#27, H2) it drove the optimum weight to 0.
  The transition *grammar* slot is saturated on jazz (#27, Mission 1: key-local
  bigram, encoder fusion, density-ratio fusion **all** net-negative).
- **Implication for Mission 5 — the honest one.** A generic learned progression
  prior does not help our decoder; that slot is full. The thing an LLM adds that
  a corpus bigram cannot is **song-specific** structure: *this* tune is in Gm,
  *these* two spans repeat, at *this* bar the V is a `7b13` not a plain 7. That
  is not "a better grammar" (which fails); it is per-song evidence the generic
  models have no access to. Mission 5 must sell song-specificity, not grammar.

## 3. What prior information actually helps a Bayesian decoder converge/land right

From the Bayesian-model and warm-start literature, and cross-checked against our
own ablations:

| Prior | Value here | Evidence |
|---|---|---|
| **Global key / mode** | High | removes #0's maj/min confusion; enables the #20 diatonic prior with a *correct* key instead of a noisy inferred one |
| **Repeated-section assertion** (parallelism) | High, *conditional on being identical* | #28 (+10pp q5 pooling identical repeats) vs #1 Candidate C (pooling non-identical repeats hurt) |
| **Per-position quality expectation** `P(q\|root,pos)` | Medium | targets the live lever — quality head q5 acc is only 44% on real audio (#19); a prior "V→dom7, vi→min" is exactly where it fails |
| **Per-song root transition bias** `P(root\|prev)` | Low–Medium | the generic grammar slot is saturated (#27), but a *song-specific* ii-V-I bias is not the generic bigram — worth testing, expect small |
| **Duration / boundary shape** | Already owned | semi-Markov duration prior shipped (#27 Mission 2, +1–2pp); LLM section lengths can sharpen boundary placement |

**Convergence, specifically.** The classic result (warm-starting EM / variational
inference / MCMC from an informed point) is that a good init mainly improves the
**basin of attraction and final quality**, and only *sometimes* the iteration
count — for a non-convex objective a warm start avoids bad local optima more than
it shortens the path. Our own controlled simulation (`eval_llm_priors.py`)
reproduces exactly this: LLM priors did **not** reduce sweep count on Autumn
Leaves (the uninformed coordinate-ascent trivially fixed-points in one sweep
because, with a uniform transition prior, there is nothing to propagate), but
lifted final accuracy by **+35pp root / +40pp (root,q5)** at σ=1.2, and the gap
**grew with noise** (+5.6pp at σ=0.8 → +44.7pp at σ=1.6). This is the essential
research finding for the design: **for this decoder the LLM's payoff is accuracy
and robustness-under-degradation, not fewer iterations** — and it is largest
exactly in the degraded-audio regime that is our real bottleneck (#19).

## 4. Can the LLM "listen"? (a hard constraint on the whole design)

Claude has no audio modality in this pipeline. Every design below feeds the LLM
**symbolic** input (the iReal chart, title, style) plus optionally a compact
*numeric* summary of our own audio front-end (detected key, tempo, per-beat root
posterior). Consequences:

- The strong case is a **known standard with a chart** (Autumn Leaves): the LLM
  recognises the tune and its functional harmony. This is a real, common
  production case (user picks a standard).
- The weak case is **unknown audio with no chart**: the LLM can only reason over
  our front-end's own (noisy) summary, so it cannot add semantics the front-end
  didn't already surface. Its confidence should — and in the prototype does —
  drop here, and the confidence-gated strength (below) makes that self-limiting.
- **Circularity guard** (`CLAUDE.md` #3, trust hierarchy iReal > tabs > model):
  priors derived from a tune's chart must not be scored as an *audio* result
  against that same chart. The eval measures convergence/robustness on held-out
  synthetic emission, never chart-agreement, for this reason.

## 5. Design takeaways (feed into the architecture doc)

1. Lead with **key/mode** and **repeat-structure** — the two priors with the
   strongest independent support (#0, #28) and the ones the LLM is best at.
2. Treat `P(q|root,pos)` as the second tier — it targets the live q5 weakness
   (#19) but must enter *softly* (the quality head is the thing being corrected,
   not replaced).
3. Do **not** pitch a generic transition grammar — that slot is saturated
   (#27); only a *song-specific* transition bias, and only softly.
4. **Confidence → strength**, always. The LLM is an automated annotator with
   lower authority than a human user; its priors tilt but never pin the decoder.
5. Reuse the **existing factor interface** (`user_constraints.py`) — the LLM is
   "an automated annotator", mechanically identical to a user confirm/merge but
   at a lower `bonus`. No new decoder recursion.
