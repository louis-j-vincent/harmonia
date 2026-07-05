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
