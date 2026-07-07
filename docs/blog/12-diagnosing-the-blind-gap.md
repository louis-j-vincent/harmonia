# Part 12 — Diagnosing the blind gap: degradation, not segmentation

## The question

Three evaluation points sat on the board:

| Eval | Root acc | What it measured |
|---|---|---|
| Beat-seq model | 88.9% | Per-beat root · clean audio · oracle GT boundaries |
| Root model (oracle) | ~89% | Segment root · clean audio · oracle boundaries |
| Blind hard-audio (Anthropology) | ~44–48% | Degraded multi-stem · blind segmentation · blind root |

A ~40 pp gap between the oracle and blind conditions. The obvious suspects were:
1. Audio degradation (phone-level noise, stem masking, band-limiting)
2. Blind segmentation (novelty detector instead of GT chord boundaries)
3. The root estimator itself (bass argmax vs. trained model)

Before building anything, I wanted to know *which* of these was actually responsible.

## The ablation

I wrote [scripts/ablation_gap.py](../../scripts/ablation_gap.py) — a 2×2 grid crossing two audio conditions (clean full-MIDI render vs. degraded multi-stem mix) with two segmentation conditions (oracle GT spans vs. chroma-novelty blind detector). Root classifier held constant: bass argmax throughout, same as `eval_blind_motif_accuracy.py`. This kept the ablation clean — each cell differs by exactly one variable.

Ran on 30 songs, seed 42, time-varying SNR (3–20 dB, same as the blind eval).

## Results

| Condition | Root | Family | Seventh | Exact |
|---|---|---|---|---|
| Clean + Oracle segs | 57.5% | 66.4% | 52.1% | 49.7% |
| Clean + Blind segs | 62.9% | 70.6% | 62.6% | 58.7% |
| Degraded + Oracle segs | 24.6% | 51.3% | 23.6% | 20.9% |
| Degraded + Blind segs | 28.6% | 52.8% | 27.9% | 25.7% |

Gap decomposition (root accuracy, clean+oracle as anchor):

- **Total gap to degraded+blind: −29 pp**
- **Degradation cost (clean+oracle → degraded+oracle): −33 pp** — explains the entire gap
- **Segmentation cost (clean+oracle → clean+blind): +5 pp** — blind segs are *better*, not worse

## The diagnosis

**Audio degradation is entirely responsible for the gap. Segmentation is not the problem.**

Blind segmentation marginally outperforms oracle boundaries in both audio conditions (+5 pp root on clean, +4 pp on degraded). The reason is likely that the novelty detector naturally forms longer, more stable segments, which pool more beats and produce a cleaner chroma estimate. Oracle boundaries sometimes create very short spans (2-beat passing chords) where the pooled activations are too sparse for a reliable root read.

The −33 pp from degradation makes sense mechanistically: the phone-style filtering plus nonuniform SNR plus stem-melody masking directly corrupts the bass register that the root estimator reads from (`_reg(son, 0, 52)`). Family accuracy is more robust to this (~51% vs 66% clean) because the LR quality classifier operates on relative chroma shape, which survives spectral smearing better than peak-finding in the bass.

## What this means for next steps

The fix target is now precise: **root estimation under degraded audio**, not segmentation, not chord boundaries.

Three candidates, in order of cheapness to test:

1. **Use the trained root models on the degraded path.** `debug_blind_pipeline.py` already runs the beat-seq model (88.9% on clean oracle) alongside the bass argmax — it just doesn't use it in `eval_blind_motif_accuracy.py`. Plugging it in is a one-line change. The question is whether its 88.9% clean accuracy degrades as badly as bass argmax under phone noise.

2. **Mid-range register as root source.** `_reg(son, 52, 72)` avoids both the kick drum energy that pollutes low bass *and* the melody content that dominates the treble. Worth a cheap swap-and-measure.

3. **Calibrate the degradation.** The SNR range (3–20 dB) may be harsher than the actual YouTube target. Worth checking what SNR a real phone capture of a jazz performance actually has — if it's 15–25 dB, the degradation sim is a 10 dB pessimist.

One caveat: the clean+oracle anchor here (57.5%) is lower than the 89% beat-seq oracle from last session. Both use the same GT boundaries and clean audio; the difference is the root estimator — bass argmax (~57%) vs. trained beat-seq model (~89%). The gap decomposition is self-consistent within this ablation, but the 57% anchor shouldn't be used to compare against that 89% headline number.
