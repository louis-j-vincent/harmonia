# Harmonia — Architecture Extensions Spec

Brainstorm output to guide implementation. Existing pipeline (Basic Pitch → madmom →
SSM segmentation → Krumhansl-Schmuckler → Bayesian HMM) stays intact. These are
additive refinements, ordered simplest → most complex.

## 1. Soft hierarchy principle (guiding rule for everything below)

Priors at every level (scale transitions, chord n-grams, timing) should **regularize,
not override**. If the soft note-probability tensor from Basic Pitch strongly supports
a reading that contradicts a prior (e.g. an out-of-scale jazz chord), the data wins.
Priors are there to resolve ambiguity, not to censor valid acoustic evidence. All
priors should have a tunable weight so this balance is adjustable empirically.

## 2. Bass-anchored root inference (near-term, simple)

Before inferring full chord quality, use the lowest sounding pitch in each beat
window as a strong prior on the chord root. Two-stage inference:
1. Bass note → root candidate (e.g. G in bass → root is G).
2. Given root, infer quality (major/minor/etc.) from the rest of the pitch content.

Slash chords / inversions are a later extension — not needed for Phase 1.

## 3. Modulation detection via accidentals (replaces/augments confidence-gap method)

Instead of (or in addition to) the KL-divergence confidence-gap method already in
Stage 4, treat out-of-scale notes as **hard evidence** for modulation:
- One accidental (e.g. F# while nominally in C major) → candidate key is G major.
- Two consistent accidentals (F#, C#) → candidate key is D major.
- General rule: infer the minimal key change that explains the new accidental(s).

Longer-term extension: allow a fully chromatic "scale" (12-tone) as an escape valve
for passages that don't fit any diatonic key — treat out-of-scale color tones as
first-class events with their own prior weight rather than forcing a key label.

## 4. Hierarchical n-gram priors, learned from data

Extend the existing scale-agnostic (interval, quality) chord priors to also include:
- **Scale-to-scale transition priors** (e.g. likelihood of modulating a 5th up,
  relative major/minor, etc.), learned empirically rather than hand-specified.
- **Chord n-gram priors**: bigram and trigram progressions (e.g. ii-V-I), stored as
  scale-agnostic interval/quality tuples as already designed. Skip 4-chord+ n-grams
  for Phase 1 — revisit later.
- Weight training examples by the confidence score of the chord detector that
  produced them, so weak/uncertain detections don't pollute the learned priors.

## 5. Learnable timing-deviation model (rubato/swing)

Replace the current rigid beat-window averaging with a two-stage model:
1. Fixed tempo/beat grid from madmom (unchanged).
2. A learned **timing offset distribution** per player/style — how far ahead of or
   behind the beat notes tend to land, and whether that offset is systematic
   (e.g. consistently early) or high-variance. This is inferred, not hand-tuned.
3. Use this distribution to reweight frame probabilities around the nominal beat
   time before pooling into beat-level note probabilities, instead of a flat window
   average.

This is the most architecturally invasive change — expect it to interact with
everything else (more free parameters, harder joint convergence). Build it last,
after the rigid-grid version is working end-to-end.

## 6. Multi-resolution training data from personal recordings

Plan: record ~1 hour of solo piano improv (free playing, deliberately drifting
between key centers) plus synthetically generated MuseScore progressions (chord
sheet → full arrangement, randomized instrumentation) as augmentation. Use source
separation (e.g. Spleeter or similar) to isolate the piano track from any
augmented/multi-instrument versions.

Critical detail: **segment length must match what you're trying to learn**:
- **~20–30s windows** for learning key/scale stability and modulation patterns —
  short windows are too statistically noisy to infer a stable scale.
- Slice those same long segments into **shorter windows (2–5s)** for learning chord
  transition priors, emission probabilities, and timing deviations.
- One recording session, multiple derived granularities — don't record separately
  for each.

Held-out validation: keep a portion of personal recordings unseen during training;
also cross-check generalization against POP909 (acknowledging it's pop, not jazz,
so treat it as a sanity check rather than primary validation for jazz-specific
priors).

## 7. Compute plan

- MacBook M4 is fine for inference (Viterbi decoding, running the pipeline).
- Use cloud GPU (e.g. Colab Pro, ~$20–50/mo) for training the EM/joint inference
  over n-gram priors and timing deviations — local overnight training would take
  weeks to converge.
- Budget: ~$50–100/month is acceptable.

## 8. Known risk areas

- **Basic Pitch**: not trained specifically for polyphonic piano across full
  register range — expect degraded performance on low/sustained notes; the soft
  probability tensor should mitigate this somewhat since no hard thresholding
  decisions are made early.
- **madmom**: trained on pop/electronic music; likely to snap-to-grid on jazz
  ballad rubato. The timing-deviation model (Section 5) is the fix, applied as a
  post-process rather than replacing madmom itself.
- **Krumhansl-Schmuckler profiles**: from 1980s tonal-music research; expect
  misclassification on modal jazz / modal interchange passages. No fix specified
  yet — flagged for future work.
- **Joint convergence risk**: as more learnable levers are added (n-gram priors,
  timing deviations, modulation priors), joint EM convergence gets harder. Build
  incrementally — get the rigid/simple version working fully before adding each
  new learnable layer.

## Suggested build order

1. Bass-anchored root inference (cheap, immediate win).
2. Accidental-based modulation detection.
3. Record + generate training data at multi-resolution (20–30s + sliced shorter).
4. Learn chord bigram/trigram priors + scale-transition priors from that data.
5. Learn timing-deviation distribution; integrate into beat-level pooling.
6. Revisit Krumhansl-Schmuckler suitability for modal jazz once the above is stable.
