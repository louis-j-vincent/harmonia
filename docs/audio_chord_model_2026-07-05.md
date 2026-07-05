# Bridging the audio → ground-truth gap with a trained model (2026-07-05)

The bottleneck is Basic Pitch's smeared audio→notes: the third and seventh are
under-detected, so fixed chord templates misfire. Fix: stop using fixed
templates and *train* a classifier on real Basic Pitch evidence with the
perfect-MIDI chord as the label — it learns what BP's smear actually looks like.

Data: 120 rendered pilot tracks (60 jazz songs × clean + a varied/noisy/
transposed copy), 4,900 chord instances, cached BP activations. Features are
**root-relative** (so we learn *quality given the root*; the root itself comes
from the bass, separately and reliably). Evaluation is 5-fold cross-validation
**grouped by song** — no song appears in both train and test, so these are
honest generalization numbers, not memorization.

Scripts: `scripts/build_audio_chord_features.py` (features → npz),
`scripts/train_audio_chord_model.py` (train + evaluate + `--save`). Saved
models: `data/models/audio_chord_{family,base7,exact}.joblib`.

## The gap, bridged, at all three tree levels

| level | fixed templates (now) | **trained audio model** | perfect-notes ceiling | gap closed |
|---|---|---|---|---|
| **family** (maj/min/dim/aug/sus) | 79.8% | **94.3%** | 99.5% | **73%** |
| **seventh** (base 7th chord, 14) | 64.4% | **87.9%** | 98.6% | **69%** |
| **exact** (18 qualities incl. 6ths, alt-dom) | 57.4% | **83.6%** | 98.0% | **62%** |

The trained model closes roughly **two-thirds of the audio→MIDI gap at every
level** — the biggest single move on the chord-quality problem so far. These are
quality-given-root; the full chord is this × the bass-derived root.

## What carries the signal (actionable: the pipeline feeds the wrong thing)

The pipeline currently pools everything into one 12-note chroma from the onset
channel. Training on richer audio features (still all from BP):

| level | onset only (pipeline default) | + sustain channel | + register split (bass/treble) |
|---|---|---|---|
| family | 90.5% | 90.3% | **93.5%** |
| seventh | 81.0% | 81.8% | **87.9%** |
| exact | 74.5% | 75.7% | **83.6%** |

Two takeaways:
1. **Just training a model on the existing onset chroma already beats fixed
   templates by a lot** (exact 57→75%). The template *scoring* was leaving
   points on the table, independent of any new features.
2. **Splitting the chroma by register (bass vs treble) is worth ~9 more points
   at the exact level.** Knowing *where* a note sits helps identify the third
   and seventh (a third in the tenor voice reads differently from bass noise).
   The sustain channel adds almost nothing. **Recommendation: feed the model
   register-split chroma, not one collapsed chroma.**

## Two priors that stop helping once the audio model is good

- **Key / scale-degree context adds ~0** on top of the trained model (family
  +0.8, seventh/exact ~0). This is the important interaction: the key prior was
  worth +6 with *fixed templates* (it substituted for the third we couldn't
  hear) — but a trained model already extracts the third from the audio, so the
  key's job is mostly done. Priors matter most exactly when the likelihood is
  weak.
- **Progression (previous chord + root motion, the ii-V-I signal) *hurts*** when
  added as raw features (family −1.5, seventh −3.9, exact −3.2). Under
  grouped-by-song CV it overfits song-specific progressions that don't transfer.
  The ii-V-I signal, if used at all, must be a heavily-regularized low-weight
  prior — never a free feature — consistent with the earlier Bayesian finding
  that progression was redundant.

## Honest caveats

- **Synthetic audio.** MMA renders are cleaner and more uniform than real
  recordings, so absolute numbers will drop on real audio. The *relative*
  results — trained ≫ fixed templates, register split helps, priors redundant
  once the model is good — are the transferable part. The saved models are
  fit on synthetic audio and should be **re-fit on real recordings** before
  production use.
- **Quality given root.** Root/family from the bass is a separate (strong)
  signal; this measures the quality decision.
- **Ceiling is a trained classifier on perfect notes** (98–99.5%), higher than
  the earlier nearest-template ceiling (~95%) — a better scorer raises the roof
  too.

## Update — 150 songs (2.5× data) + CNN + structure stacking

Rendered to 150 songs (12,524 chord instances) and re-ran everything.

### More data closed the gap further, most at the hard levels

| level | gap closed @ 60 songs | gap closed @ 150 songs |
|---|---|---|
| family | 73% | 79% |
| seventh | 69% | 75% |
| exact | 62% | 71% (LR) / **74% (CNN)** |

### The user's CNN architecture wins at every level

Circular 1-D convolution over the 12-note chroma (onset+note channels,
transposition-aware) + an MLP branch on the bass/treble register chroma (the
"who's on the bottom" signal). Trained in PyTorch, same grouped-by-song CV
(`scripts/train_audio_chord_cnn.py`):

| level | logistic regression | **CNN + bass-MLP (user's idea)** |
|---|---|---|
| family | 94.3% | **95.0%** |
| seventh | 87.9% | **89.1%** |
| exact | 83.6% | **86.4%** (+2.8) |

The circular-conv + register-MLP inductive bias helps most at the hardest level
(exact), exactly where the fine 3rd/7th structure matters. This is now the best
model; the LR remains a strong, interpretable, cheap fallback.

### Progression: the user's overfitting hypothesis was right — but it still adds ~nothing

As a *weighted prior* with the weight tuned (not raw features), on the bigger
data (`scripts/experiment_progression_prior.py`): best weight is very low
(w≈0.1) and the gain is family +0.0, seventh +0.0, exact +0.2. So more data +
low weight took progression from **harmful** (−3.2 at exact on 60 songs, raw
features) to **neutral**. Confirmed: it was overfitting on too little data. But
even done right it adds essentially nothing on top of a strong audio model — the
model already extracts what the ii-V-I context would tell us. The best weight
being ~0.1 is the same "gentle nudge" the key prior needed.

### Structure stacking (AABA): the mechanism works, but needs repeats that actually vary

Combining a chord's repeats before naming it (`scripts/experiment_structure_stacking.py`):

| stacking | family | seventh | exact |
|---|---|---|---|
| single observation | 78.8% | 59.6% | 54.1% |
| structural (section repeats, 1 render) | 79.0% | 59.5% | 54.1% |
| cross-render (2 independent renders) | **85.2%** | **67.4%** | **62.6%** |

The structural repeats within one MMA render are near-identical (mean cosine
**0.93**), so averaging them denoises nothing — **~0 benefit**, as expected for
deterministic synthetic audio. But when the repeats are *independently corrupted*
(two different renders — different soundfont/noise, mean cosine 0.90), stacking
is a **+6 to +8.5 point** win. That is the real-performance case: on a real
recording, each time the A section comes around it's played and recorded a little
differently, so song-structure stacking should recover most of that cross-render
gain. **The idea is validated; exploiting it needs real repeat-to-repeat variation
(real audio, or renders with per-repeat variation), not MMA's identical repeats.**

## Recommendation / next

1. **Replace fixed-template emission scoring with the trained per-level model**
   (family → seventh → exact), fed register-split chroma. This is the concrete
   Stage-1 upgrade; wire the saved models into the emission step or re-fit on a
   larger, more realistic render set first.
2. **Render more + more realistic audio** (more songs, more soundfonts, real
   reverb/room, dynamics) and re-fit — the model is data-hungry and the pilot is
   only 60 songs.
3. Fold the trained model into the hierarchical reporter so `reported_label`
   descends on the model's calibrated confidence, not template margins.
