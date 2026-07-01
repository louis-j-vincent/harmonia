# Harmonia — Model Improvement Suggestions

Suggestions are ordered by impact and grouped by pipeline stage. None of these have been implemented yet. Each entry states the problem, the proposed fix, and the rationale from a training perspective.

---

## Stage 1 — Basic Pitch: Thresholds and Normalization

### Problem: Fixed onset threshold does not generalise across songs

**Current behaviour:**  
`onset_threshold = 0.3` is subtracted from raw sigmoid outputs and clipped. This is a global constant that ignores:
- Per-song loudness / velocity distribution (forte playing pushes more values above 0.3 than pianissimo)
- Soundfont response curves (VintageDreams vs GeneralUser GS will have different activation ranges)
- Note density (uptempo bebop has many more onsets per second than a slow ballad)

A fixed threshold means: the same note played softly may not survive, while background resonance in a louder passage does. This introduces spurious correlations between loudness and chord identity during training.

**Proposed fix: per-song percentile threshold**

```python
raw_onset = model_output["onset"].astype(np.float32)
onset_threshold = float(np.percentile(raw_onset, 95))  # keep top 5% per song
onset_probs = np.clip(raw_onset - onset_threshold, 0.0, None)
```

This guarantees a constant sparsity level (5% active) regardless of song loudness or instrument. The 95th percentile should be tuned empirically — start at 95, check that the resulting beat vectors look sparse and musically meaningful on 5–10 songs.

Same logic applies to `frame_threshold` on the `"note"` channel.

**Why this matters for training:**  
If beat vectors have systematically different sparsity between songs (dense for forte songs, sparse for pianissimo), a model trained on mixed data will learn loudness as a proxy for harmonic content. Per-song normalisation removes this nuisance variable.

---

### Problem: `quantise_frames()` sums over a beat window — scale depends on beat duration

**Current behaviour:**  
`beat_probs[b] = onset_probs[mask].sum(axis=0)` — the value of each key accumulates over however many frames fall in that beat. At 43 Hz, a beat at 60 BPM has ~43 frames; at 180 BPM it has ~14 frames. Slow songs produce beat vectors ~3× larger in magnitude than fast songs, even for identical note content.

**Proposed fix: L1-normalise each beat vector after quantisation**

```python
# After quantise_frames():
row_sums = beat_probs.sum(axis=1, keepdims=True)
beat_probs = beat_probs / np.where(row_sums > 0, row_sums, 1.0)
```

This turns each beat vector into a proper distribution over the 88 keys — it sums to 1 on beats with any onset signal, and stays at zero for silent beats.

**Why this matters for inference:**  
The HMM emission is `log(beat_probs @ emission.T + eps)`. If `beat_probs` magnitudes vary with tempo, the log-likelihood magnitudes also vary, which shifts the effective balance between emission and transition terms in Viterbi. Normalising makes the balance explicit and hyperparameter-controlled.

**Why this matters for training:**  
Normalised vectors are what you'd want as inputs to any learned chord template model. Without this, the model would need to separately learn loudness → class relationships that we do not want.

**Caveat on zero beats:**  
Silent beats (no onsets — sustained chord) will have `beat_probs[b] = 0`. Do not normalise these to uniform — leave them as zeros. The HMM should handle zero observations via the noise floor in the emission matrix (`noise_floor` in `build_emission_matrix`). If zero-observation beats cause Viterbi instability, increase `noise_floor` slightly (e.g. from 0.05 to 0.1).

---

### Problem: `onset` only fires at note attacks — sustained notes produce zero observations

**Current behaviour:**  
A whole note held over 4 beats will only produce onset signal on beat 1. Beats 2–4 have near-zero onset_probs, so the HMM receives no harmonic evidence for those beats and must rely entirely on the transition prior. For slow jazz ballads with long sustained chords, this means Viterbi is essentially guessing based on progressions alone.

**Proposed fix: hybrid observation combining onset and sustained note signal**

```python
# In stage1_pitch.py / extract():
onset_probs = np.clip(raw_onset - onset_threshold_per_song, 0.0, None)
note_probs  = np.clip(raw_note  - note_threshold_per_song,  0.0, None)

# Combine: onset dominates at attacks; note fills in sustain
alpha = 0.15  # tunable — note channel contributes weakly
combined = onset_probs + alpha * note_probs
```

Then pass `combined` to `quantise_frames()`. The `alpha` weight should be small so the flat note channel doesn't overwhelm the sparse onset signal, but large enough to give the HMM evidence during sustained beats.

`alpha = 0.0` is the current state (onset only). `alpha = 1.0` is the broken original state (note only). The right value is somewhere in 0.05–0.20 and should be treated as a hyperparameter to sweep.

---

### Problem: Basic Pitch sensitivity is uneven across the frequency range

**Current behaviour:**  
The `(F, 88)` output is not uniformly calibrated — the model is less sensitive at the extreme low and high ends of the piano range (A0–C2 and above C7), which is typical for neural pitch estimators trained on mixed data. This means bass notes and very high treble notes are systematically underrepresented in `beat_probs`, which biases chord root inference upward.

**Proposed fix: per-key calibration via song-level statistics**

```python
# Compute per-key mean activation across the whole song
key_mean = onset_probs.mean(axis=0)  # (88,)
key_mean = np.where(key_mean > 1e-6, key_mean, 1.0)
onset_probs_calibrated = onset_probs / key_mean  # per-key z-score (simplified)
```

This is a rough equalisation that amplifies underrepresented keys relative to the song average. It should be applied before `quantise_frames()`.

This is a long-term fix — validate after the end-to-end pipeline is working first.

---

## Stage 2 — Beat Quantisation: Pooling strategy

### Problem: SUM pooling is noise-accumulating; MAX pooling may be more robust for onsets

**Current behaviour:**  
`quantise_frames()` sums onset values over all frames in a beat window. For onset signals (which fire for 2–3 frames per note attack), summing over 43 frames accumulates noise from the 40+ non-onset frames even at threshold 0.3.

**Alternative: MAX pooling**

```python
if mask.any():
    beat_probs[b] = onset_probs[mask].max(axis=0)
```

MAX extracts the peak onset response per key in the beat window, which is more robust to the exact frame count. It also makes the output independent of beat tempo (no accumulation over longer beat windows).

**Tradeoff:**  
MAX ignores how many notes in the same pitch class are struck within one beat (e.g., arpeggios). SUM captures this. For chord recognition (not note counting), MAX is probably better. For future n-gram training where harmonic rhythm matters, SUM carries more information.

**Recommendation:** Try both. Log both `beat_probs_sum` and `beat_probs_max`, compare their sparsity patterns on a few songs visually before committing.

---

## Stage 5 — HMM: Emission / observation model

### Problem: emission matrix rows sum to 1 but observations are not normalised to the same scale

The emission matrix `E[c, k]` is row-normalised (each chord's distribution over keys sums to 1). The observation `beat_probs[b]` is currently unnormalised. The inner product `beat_probs @ E.T` gives different magnitude outputs depending on the scale of `beat_probs`.

This is addressed by the L1 normalisation of `beat_probs` proposed above. Once that is in place, `beat_probs[b] @ E[c]` is interpretable as the expected activation under chord `c` weighted by observed key salience — essentially a likelihood score with consistent units.

---

## Implementation order

1. Per-song percentile threshold (biggest impact, one-liner, no architectural change)
2. L1 normalisation of beat vectors after `quantise_frames()` (makes emission scale-consistent)
3. Hybrid onset + note observation with tunable `alpha` (addresses sustained-note blind spot)
4. MAX vs SUM pooling experiment (after end-to-end baseline is established)
5. Per-key calibration (after enough songs are validated to estimate typical key sensitivities)
