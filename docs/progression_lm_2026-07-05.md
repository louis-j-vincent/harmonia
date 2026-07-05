# Chord-progression language model over scale-relative degrees (2026-07-05)

User's idea: drop the rigid chord-transition table; instead, given the correctly
identified scale, feed a sequence model the last few chords as **scale-relative
degrees** (their function within the key, e.g. `II:min7 → V:dom7 → I:maj7`) and
let it learn the grammar of progressions. Test whether it's a good indicator and
whether a sequence model beats a bigram.

Scripts: `scripts/train_progression_lm.py` (the LM),
`scripts/experiment_progression_plus_audio.py` (combined with audio).

## 1. Is progression predictable, and does a sequence model beat a bigram?

Full jazz corpus, symbolic (1,454 songs, 63,380 transitions), split by song,
next-chord prediction. Tokens = (scale degree, quality), 157 types.

| model | next top-1 | next top-3 | next root (degree) | next quality |
|---|---|---|---|---|
| unigram (most common) | 12.5% | — | — | — |
| bigram (the old test) | 35.1% | — | — | — |
| LSTM (full history) | 41.1% | 59.5% | 52.0% | 54.9% |
| **MLP (last 4 chords)** | **42.6%** | **60.5%** | **54.3%** | **55.5%** |

The sequence model beats the bigram by **+7 points** — longer progression context
carries real signal, confirming the idea. Two honest notes:
- **MLP(last-4) ≈ LSTM** (even slightly better here): jazz progression context is
  mostly local (the last ~4 chords — ii-V-I's, turnarounds), not long-range, so
  the recurrent memory doesn't add over a 4-chord window on this corpus.
- The next chord's **root** is predictable 54% of the time from progression alone
  — independent evidence from the bass, useful where the bass is ambiguous.

## 2. Does it help the audio model? (the test the bigram failed)

Combine, on the rendered songs, `log P(quality | audio) + w · log P(quality |
history, degree)`, LM trained only on songs held out from the test, weight swept:

| level | audio alone | + progression LM (best weight) |
|---|---|---|
| family | 95.8% | **96.4%** (w=0.2) |
| seventh | 91.2% | **92.0%** (w=0.2) |

**+0.6 to +0.8 points, at a low weight (0.2).** Small but real and positive —
whereas the earlier crude bigram / raw-feature progression was redundant or
harmful (−3). The difference is exactly the user's proposal: scale-relative
functional tokens + a sequence model, instead of a one-step quality table. The
low best-weight is the same "gentle nudge" the key prior needed.

## Takeaways

- Progression *is* usable evidence once represented right (scale-relative,
  sequence model) — it just adds modestly because a strong audio model already
  explains most of the variance. It earns its keep in the ambiguous cases and,
  notably, for the **root** (54% next-root prediction) where it complements the
  bass.
- A 4-chord MLP is enough; the LSTM's extra memory isn't needed for jazz here.
- Prerequisite the user flagged: this used the **true** scale. With an inferred
  scale the gain will shrink; getting the key right is upstream of this paying
  off — another reason key inference matters.

## Next

- Combine the progression LM as a low-weight transition prior in the actual HMM
  decode (replacing/augmenting the hand-coded `jazz_priors` transition table),
  and re-evaluate end-to-end.
- Use the LM's next-**root** prediction to help the bass-derived root on
  ambiguous/weak-bass beats (its most complementary signal).
