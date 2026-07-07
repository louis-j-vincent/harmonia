# Part 13 — Context, gates, and honest labels

## The goal

Push family classification accuracy above the 79.8% baseline on hard audio
(oracle boundaries, GT root shift). Two threads: richer features from context,
and a visual diagnostic we could actually trust.

---

## What we built

### LTAS chord-quality tree (recap)

A three-level hierarchy — 5 families → 14 base7 → 18 exact qualities — each
node described by a diagonal Gaussian over 12-dim root-shifted LTAS chroma.
Greedy tree search scores 10 LL calls vs 216 exhaustive. With corpus-only data,
accuracy was 59.1% / 40.9% / 27.4% at family / base7 / exact.

---

### Key×family LL matrix as input

Instead of the flat 5d max-LL vector, compute a full (5, 12) matrix: for every
(family, root-candidate) pair, one Gaussian LL score. Then take a
softmax-attention aggregate over frames (confident frames dominate). This
captures the full key ambiguity, not just the argmax.

---

### Context window classifier

Extend each segment's feature to a (9, 5, 12) tensor: ±4 neighbours, each a
(5, 12) key×family LL matrix. Models compared (5-fold CV, 40 songs, hard audio):

| Model | Acc | Δ base |
|---|---|---|
| logreg_base (17d) | 79.8% | — |
| logreg_ctx (552d) | 84.0% | +4.3pp |
| MLP | 84.8% | +5.1pp |
| CNN | 84.1% | +4.3pp |
| LSTM | 70.0% | −9.7pp |
| **entropy gate** | **85.9%** | **+6.1pp** |

LSTM needs much more data or pre-training to utilise sequence structure; MLP
and CNN are both useful. LSTM was dropped.

**Rare-class regression.** Aug and sus accuracy dropped from ~75–83% to ~33–39%
when context was added — neighbouring chords are unrelated to the rare class,
so context adds noise. The entropy gate partially recovers this.

---

### Key unification (important fix)

The original context tensor stored each neighbour's (5, 12) matrix in its own
root frame. This means the model had to learn "Dm next to Am vs Dm next to Gm"
as separate patterns — it couldn't generalise ii-V-I across keys.

Fix: roll neighbour j's ll_mat by `-(root_j - root_i) % 12` before inserting
into the tensor. Column 0 of every context position then means "same root as
the target chord." After this fix:

| Model | Before | After | Δ |
|---|---|---|---|
| logreg_ctx | 84.0% | 84.9% | +0.9pp |
| MLP | 84.8% | 86.4% | +1.6pp |
| CNN | 84.1% | 85.4% | +1.3pp |
| LSTM | 70.0% | 76.9% | +6.9pp |
| **entropy gate** | **85.9%** | **87.5%** | **+1.6pp** |

LSTM improved the most (+6.9pp) because it is the most sensitive to relative
key structure across time steps.

**Note on oracle assumption.** The root shift currently uses GT root. At
inference, the predicted root comes from argmax over ll_mat columns —
introducing ~5–10% root-prediction noise into context alignment.

---

### Entropy gate

```
α = sigmoid(w · H(p_base) + b)
p_final = α · p_base + (1-α) · p_ctx
```

Learned w=−0.49, b=−0.29: high entropy (ambiguous base model) → α≈0.25 → lean
on context; low entropy (confident base model) → α≈0.43 → trust point estimate.
The range is moderate (not 0→1 saturating) because rare classes have high
entropy from limited training data, so the gate can't fully isolate them.

---

### Failure correlation analysis

Per-segment metrics correlated with correct/fail (N=1621, point-biserial r):

| Metric | r | sig |
|---|---|---|
| LL margin (1st − 2nd) | +0.215 | *** |
| Chroma temporal variance | −0.210 | *** |
| Audio RMS | +0.100 | *** |
| Chroma peakedness | +0.089 | *** |
| Duration | +0.087 | *** |
| Context same-fam fraction | −0.050 | * |

**LL margin** is the single best predictor — directly motivates the entropy gate
(low margin ↔ high entropy ↔ α→0 ↔ trust context). **Temporal chroma variance**
is equally strong: if pitch content shifts within the half-bar (arpeggio, passing
melody note), the LTAS mean smears and classification degrades. This is a signal
problem, not a model problem. Short, quiet segments also fail more.

Surprisingly, **context diversity is slightly positive**: isolated chords
classify *better*, not worse. The trouble is harmonic similarity at the maj/min
boundary, not context noise.

---

## What didn't work

- **LSTM (pre-unification):** 70.0% (−9.7pp) — too few samples (~1600) to learn
  sequential dependencies; MLP/CNN are strictly better here.
- **Entropy gate full recovery of rare classes:** aug recovered to 40%, sus to
  55.6% — still well below their base-model accuracy (66.7%, 83.3%). This is
  a data-scarcity problem (aug n=15, sus n=18 in 5-fold splits), not a gating
  failure.

---

## Bugs caught and fixed

**Root label in diagnostic HTML** was using GT root for both the GT chord name
and the predicted chord name. Bug: `pred_tok = NOTE[seg["root"]] + suffix`.
Fix: `pred_root = (seg["root"] + seg["keys5"][seg["pred_fam_i"]]) % 12`.

**LL bar normalisation** was cross-segment min-max, so the argmax of the bar
chart within a card didn't match the argmax of the actual LL vector. Fix:
within-segment softmax, so the tallest bar is always the model's argmax.

**Second bar row** (LogReg pred_proba) added so LL signal and final decision
are visible together — they can diverge because the 12d chroma adds information
the bare LL ignores.

All root-relative arithmetic is now covered by 17 unit tests in
`tests/test_root_arithmetic.py`. Tests verify: chroma root-shift, LL bar key
labels, predicted root in card header, ctx tensor key unification, ll_mat
column semantics, max_ll_over_keys end-to-end, and explicit before/after
comparison of the old label bug.

---

## State at end of session

Best result: **87.5% family accuracy** (entropy gate, key-unified context MLP,
hard audio, oracle bounds, GT root, 5-fold CV, 40 songs).

Baseline was 79.8%. Gap to close: rare-class recovery (aug/sus), and eventual
move off oracle root shift.
