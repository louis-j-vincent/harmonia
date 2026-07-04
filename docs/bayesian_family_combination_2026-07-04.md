# Combining clues for the chord family — measured weights + interactions

`scripts/experiment_bayesian_family.py`, 2026-07-04. The skeleton of the
Bayesian chord model, tested on the family decision (the third: major / minor /
dim / aug / sus) because that is the bottleneck. Each clue produces a
probability distribution over families; they are pooled log-linearly,
`score(f) = Σ_clue w_clue · log P_clue(f)`, and the weights `w` are FIT on
training songs — that fit is the estimate of how much to trust each clue.

Ground truth (key, root, structure) is known; audio is real Basic Pitch on the
rendered pilot. 4,900 chords, split by song.

## The clues

- **AUDIO** — P(family | notes heard), softmax of the template match (the likelihood).
- **KEY** — P(family | scale-degree, key), learned per-degree table (the "third" lever).
- **PROG** — P(family | previous-root → this-root), learned (the ii-V-I / progression signal).
- **FOLD** — AUDIO averaged over the repeats of this slot in the song form
  (multiple observations of the same chord).

## Results

| clue | alone (test acc) |
|---|---|
| AUDIO | 81.7% |
| FOLD | 82.6% |
| KEY | 74.5% |
| PROG | 68.4% |

**Fitted weights** (normalized share of trust): AUDIO **0.66**, KEY 0.11,
PROG 0.11, FOLD 0.11 — i.e. audio is the likelihood and the priors are light
nudges at roughly a **5 : 1** ratio (audio ≈ 1.0, each prior ≈ 0.2). This is the
"gentle nudge, not a rule" principle, now a measured number.

**Combined accuracy: 88.4%** (audio alone 81.7%, so **+6.7 points**).

### How the clues interact (leave-one-out — unique information each adds)

| drop | combined acc | unique contribution |
|---|---|---|
| — (full model) | 88.4% | — |
| without AUDIO | 79.4% | **−9.0** (the likelihood, essential) |
| without KEY | 86.0% | **−2.4** (the biggest prior) |
| without PROG | 87.9% | −0.5 (mostly redundant here) |
| without FOLD | 87.9% | −0.5 (mostly redundant here) |

Two interaction findings, which are the point of the exercise:

1. **KEY carries real unique information (−2.4)** — it is the prior worth wiring
   in. **PROG is largely redundant with KEY** (both condition on scale degree),
   and **FOLD is largely redundant with AUDIO** (it *is* audio). The 80/20 model
   is **AUDIO + KEY at ~5:1**; PROG and FOLD are optional.
2. This redundancy is level-specific. PROG (the ii-V-I signal) is about the
   *seventh* (a V is a dominant-7th), not the third — so its unique value should
   reappear at the seventh level, where KEY is weaker. Testing PROG at the
   seventh level is the natural next experiment.

### Diatonic vs out-of-scale (the modulation signal)

| chords | audio alone | combined |
|---|---|---|
| diatonic (root in key, 2178) | 81.5% | 88.2% |
| out-of-scale / chromatic (366) | 82.8% | 89.9% |

Crucially, the combined model does **not** hurt chromatic chords — because the
key weight is low, a genuine out-of-scale chord (a jazz A7 in C major) is still
decided by the audio, which correctly hears the chromatic third. That "audio
overrides the key prior" event is exactly the signal that the chord is borrowed
or the key is changing — the same low weight that protects accuracy here is what
makes an out-of-scale chord detectable as a modulation cue rather than an error.

## Caveats

- FOLD is likely *underestimated* here: MMA renders each song deterministically,
  so repeated sections are near-identical audio and averaging them denoises
  little. Real performances vary between repeats, so folding would add more — but
  its *unique* value is still bounded by how much AUDIO+KEY already capture.
- Weights fit with true key + true root; the real pipeline's inferred key
  (4/5 songs) and bass-derived root will shrink the gain, though the ordering
  (audio ≫ key > prog ≈ fold) should hold.
- Family only. The seventh level (dom7 vs maj7) is the next target; the same
  machinery applies, and PROG should matter more there.

## Recommendation → the Bayesian model, concretely

For the family (third) decision:

    score(family) = 1.0 · log P(family | audio)
                  + 0.2 · log P(family | key, scale-degree)      ← the third lever
                  [+ 0.2 · log P(family | progression)]           optional
                  [+ 0.2 · log P(family | folded audio)]          optional

Start with AUDIO + KEY at 5:1 (captures +5 of the +6.7). This is a targeted,
low-weight realization of `key_prior_per_beat`, aimed at the family/third
decision specifically rather than the full quality — which is likely why the
earlier full-weight version regressed on song 001.
