# Priors reclaim their value when the audio degrades (2026-07-05)

Every experiment so far found the key/progression priors add little — but on
*clean synthetic audio with complete voicings*. The user's point: real recordings
and added instruments drop chord tones (the 3rd, 5th, 7th aren't always sounding),
weakening the audio likelihood, and then the priors matter. Quantified here
(`scripts/experiment_priors_when_audio_degrades.py`).

Audio model (LR) trained on FULL voicings, evaluated on progressively degraded
ones (train-simple / deploy-subtle), family level, key prior weight tuned:

| voicing condition | audio alone | audio + key prior | key recovers |
|---|---|---|---|
| full (clean) | 94.4% | 94.5% | +0.1% |
| missing 5th | 64.6% | 88.5% | **+23.9** |
| missing 3rd | 65.6% | 75.8% | +10.2 |
| missing 7th | 84.9% | 92.6% | +7.7 |
| + other-instrument noise | 47.6% | 52.5% | +4.8 |
| root + color only (3/5/7 gone) | 33.5% | 65.2% | **+31.7** |

## Reading

- On a **complete** voicing the key prior is nearly free (+0.1). On **degraded**
  voicings it recovers **10–32 points**. The prior's value scales inversely with
  the audio's completeness — exactly the pattern seen before (key was worth +6
  with weak fixed templates, ~0 with a strong trained model).
- **Missing the 3rd** is the signature case: the audio can no longer tell major
  from minor, and the key — which *predicts* the third from the scale degree —
  fills precisely that gap (+10).
- The prior recovers even the near-hopeless "root + color only" case from 34% to
  65%, because the key still constrains what the family can be.

## Caveat (honest)

The large audio-alone drops are partly distribution shift (trained on full,
tested degraded). Training with tone-dropout augmentation would make the audio
model more robust and shrink some of the gap. But the core conclusion is
information-theoretic and survives that: **a chord tone that isn't in the signal
cannot be recovered by any audio model — only a prior can supply it.** The right
system trains the audio model with realistic voicing dropout AND leans on the
priors, with a weight that rises as the audio confidence falls.

## Implication for the architecture

This is the argument for the whole Bayesian-combination direction. The priors
(key for the third, progression/structure for the seventh and root, the BiLSTM's
bidirectional context) are cheap insurance that pays out exactly when the audio is
incomplete — i.e. on the real, multi-instrument, subtly-voiced material the clean
pilot doesn't contain. Concretely:
- make the prior weight a function of audio certainty (the BiLSTM already outputs
  a calibrated one) — lean on priors only when the audio is unsure;
- train the audio model with voicing dropout so it degrades gracefully;
- validate on real recordings, where these gains are expected to be large, not the
  +0.4 the clean pilot shows.
