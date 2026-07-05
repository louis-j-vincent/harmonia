# Joint structure ↔ chord co-inference (the EM loop) — design + idea log

Design captured 2026-07-05 from a design conversation. **Not fully implemented yet
— logged here so the ideas aren't lost** (per user request). Implement after the
current structure-folding experiment is fixed, logged and committed.

## The core idea (user's): certainty and structure validate each other

Structure S (which slots are "the same") and chords C are two coupled latent
variable sets. Neither is decided first — they consolidate by iteration to a
mutually-consistent fixed point, with **calibrated certainty as the currency**
flowing both ways:

- **structure → chords**: fold confirmed repeat-groups (certainty-weighted) → sharper C.
- **chords → structure**: certainty-weighted cross-repeat *agreement* confirms or
  **refutes** a structure hypothesis. Two slots that should match but disagree *with
  high certainty* → the hypothesis is wrong. Disagreement between *uncertain* chords
  proves nothing. **Certainty turns agreement into evidence.**

### The loop (variational EM / loopy belief propagation)
1. decode chords independently → C₀ + per-chord certainty.
2. propose structure hypotheses (periodicity + form prior {8,16 bars}); score each by
   certainty-weighted agreement; keep those clearing a shuffle null.
3. fold confirmed groups (certainty-weighted) → C₁ sharper.
4. re-score structure with C₁ (now some hypotheses confirm/refute decisively).
5. iterate to fixed point — only mutually-consistent (S, C) survive.

### Why robust / self-correcting
A wrong structure hypothesis groups non-identical chords → low agreement → refuted →
no destructive fold. This is exactly the guard the earlier Candidate-C fold lacked.

### Factor-graph form (how "they respond to each other")
- nodes: a chord distribution Cₜ per beat (soft, never a hard choice).
- audio factors: P(audio_t | Cₜ).
- equivalence edges: soft "slot i ~ slot j" beliefs from structure hypotheses.
- messages chord→edge = agreement (update structure belief); edge→chord = fold
  (update chords). **Certainty (inverse entropy of the marginals) modulates message
  strength in both directions.**

## Multi-level / nested folding (user's addition)

Folding must happen at **several structural levels at once** — within one A section
there are also repetitions (sub-phrases, 2-bar cells, single bars), not just A≈A≈A.
So the structure hypotheses form their own hierarchy:

    song form (A B A …) → section → phrase (4/2-bar) → bar → beat-cell

Fold at every level where agreement holds; the levels are **nested**. A robust
scheme folds bottom-up: confirm the tightest repeats (bar-cells) first, then phrases,
then sections — each level's folded (cleaner) chords feed the agreement test at the
next level up.

### SSM / co-similarity matrix as the WHERE-to-fold prior (user's idea)
The self-similarity matrix (`structure.build_ssm`, built at the very start of the
project) is exactly a prior over *which slots are equivalent* — its bright
off-diagonal bands ARE candidate fold-groups. Concretely:
- **Matrix reduction** (user's question — could it work?): yes — a low-rank / block
  factorization of the SSM (e.g. NMF or spectral clustering of the beat-similarity
  graph) recovers the repeat structure as blocks = the equivalence classes to fold.
  The SSM gives soft, multi-scale block structure → the nested fold levels fall out
  of the factorization at different ranks/resolutions.
- **Chord-probability matrix at each hierarchy level** (user's idea): build a second
  matrix whose entries are agreement between the *inferred chord distributions*
  (not raw chroma) at slots i,j — at each level of the chord tree (family / seventh /
  exact). This is the CRHA signal as a matrix, and it *validates* the SSM's proposed
  blocks: fold a block only where the chord-agreement matrix confirms it, at the
  deepest tree level where it still agrees. **Two matrices: SSM = where repeats might
  be (from audio surface); chord-agreement = where they really share harmony (from
  the model). The second gates the first — the same certainty↔structure loop, in
  matrix form.**

## The nested hierarchies — orthogonal in places, complementary in others

Three hierarchies interlock; being explicit about how matters:
1. **Structure hierarchy** (form → section → phrase → bar → beat) — *where* to fold.
2. **Chord tree** (family → seventh → exact) — *how deep* to trust a chord / a fold.
3. **Certainty** — the *weight/gate* on both.

They are **orthogonal** in that structure-depth and chord-depth are independent axes
(a bar-level fold can still only confirm the family; a section-level fold might
confirm the exact chord). They are **complementary** in that certainty couples them:
a fold at structure-level L is trusted down to chord-tree-level D only where the
certainty clears the bar. The design must not collapse these two depths into one —
they're a 2-D grid (structure level × chord-tree level), gated by certainty.

## Guard-rails (confirmed with user)
1. never fold **below the chord-tree level where the group agrees** (fold family if
   repeats agree on family but not seventh).
2. agreement must beat a shuffle **null** (don't trust coincidental agreement).
3. everything **soft** — probabilities, no hard early decisions (what propagated
   errors before).
4. certainty must be **calibrated** — CONFIRMED: `docs/plots/certainty_calibration.png`
   (ECE family 0.012 / seventh 0.032 / exact 0.021, near-diagonal, especially at high
   confidence where most chords sit). The loop can trust the certainty.

## Empirical note on the synthetic data (2026-07-05)

Folding on varied-jazz hard audio: family 77.6→79.2 (+1.6), seventh 61.5→65.0
(+3.5) — real but modest, and `certainty-fold ≈ mean-fold`. Reason: even with
per-occurrence comping voicing variation, the **full-mix** chroma BP hears stays
~0.998 similar across repeats, because bass + drums + the (per-render-fixed) melody
dominate the spectrum and do NOT vary per occurrence. So there's little independent
variation for certainty-weighting to exploit; the small gain is mostly noise
averaging. **To exploit structure folding here, the whole mix must vary per
repeat** (melody line, bass, timing), not just the comping — i.e. render each
section occurrence like a genuinely different take, or move to real recordings.
This bounds the measurable value of the EM loop on the current synthetic data;
the *mechanism* (certainty-weighted agreement selecting/refuting structure) is
still testable independently of repeat variation.

## EM discriminator — validated (2026-07-05)

`scripts/experiment_em_structure.py` tested the loop's core (structure ← chords),
independent of repeat variation, on 90 multi-section varied-jazz songs:
- certainty-weighted within-group agreement: **TRUE structure 0.798 vs RANDOM 0.720**
  (+0.077 margin); certainty adds +0.022 over uniform.
- the agreement tracks real sharing: TRUE groups are one GT family **94.1%** of the
  time vs 45.6% random.
- TRUE beats RANDOM on **86%** of songs.
**Conclusion: the certainty-weighted agreement discriminator works** — the loop can
confirm real structure and refute wrong hypotheses. This is the E-step; the M-step
(certainty-weighted fold) was tested in experiment_certainty_folding.py. Remaining
build = wire them into the iterative loop (bottom-up over nested levels, SSM-block
proposals + chord-agreement-matrix gating).

## Implementation order
1. ✅ voicing variation so repeats differ at chroma level (done).
2. structure-folding experiment on varied audio (in progress) — the empirical check.
3. SSM-block + chord-agreement-matrix prior for fold-groups (multi-level).
4. the EM / message-passing loop, bottom-up over the nested levels, certainty-gated.
5. evaluate end-to-end on hard/varied audio (where it should help most).
