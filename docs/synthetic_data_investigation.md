# Synthetic training-data investigation (2026-07-17)

**Question (user proposal):** generate our own training data by (a) programmatically
creating chord-progression MIDI with exact ground truth (root/quality/inversion), then
(b) rendering to realistic audio — to sidestep the ground-truth-alignment problems that
have repeatedly burned real-audio corpora (Billboard/YouTube offsets, POP909 dropping
inversions). If it transfers, it's unlimited perfectly-aligned data on demand.

## What synthesis is actually runnable here (2026)

- **No local neural audio synthesis.** DDSP, MusicGen/AudioGen and successors, Stable
  Audio, MT3-family, diffusion MIDI-renderers — all require GPU + model weights and/or a
  paid API. None is runnable in this environment today without incurring cost. They remain
  literature-only for this machine. (A parallel dataset-survey agent found **AAM**, Zenodo
  5794629 — 3,000 *pre-rendered* professional-synthesis tracks with aligned chords AND
  isolated bass stems, CC-BY-4.0. That is the stronger "someone else already did good
  synthesis" path and should be the next comparison — see Next steps.)
- **Runnable path tested here: high-quality sample-based rendering.** `fluidsynth`
  (installed) + two real quality soundfonts already on disk (`MuseScore_General.sf2`,
  `GeneralUser.sf2`). Fully scriptable, zero cost, and we control the exact label
  distribution by construction. This is what the numbers below use.

## Pipeline (feature parity with real corpora — no reimplementation)

`scratchpad/synth_gen.py` generates progressions: 7-way quality vocab
(maj/min/dom/hdim/dim/aug/sus) weighted to RWC frequencies, inversions as Harte degree
tokens at RWC's ~12.4% rate and degree mix (/3 /5 dominant). Bass note is rendered to
match the label's sounding bass. Diversity: 8 comping GM programs, 4 bass programs, tempo
72–152, velocity/voicing/onset jitter, both soundfonts. `build_midi` → `fluidsynth` → WAV.
Features extracted with the EXACT real pipeline: `PitchExtractor` (Basic Pitch) +
`seg_feature_clipped`/`seg_feature_abs_clipped` (bleed-fixed frame-clip pooling). Corpus
written to `corpus_schema` format (`data/cache/synth/synth_bp48.npz`, all REQUIRED_KEYS).
WAVs deleted immediately after extraction (disk discipline). Corpus: 60 songs, **1440
chords**, inversion frac **0.128** (target 0.124 ✓), quality dist maj/min/dom-dominated ✓.

## Premise check — realism via chroma entropy (proxy)

Chroma norm-entropy (identical computation both sides), synthetic vs real RWC:

| block   | synthetic | real RWC |
|---------|-----------|----------|
| onset   | 0.700     | 0.885    |
| note    | 0.996     | 0.999    |
| bass    | 0.560     | 0.830    |
| treble  | 0.629     | 0.814    |
| full-48 | 0.846     | 0.931    |

**Synthetic audio is consistently SHARPER/cleaner (lower entropy)** — the exact
"too clean, won't transfer" domain-mismatch risk CLAUDE.md flags. Gap is moderate
(~9% full-48), worst on **bass** (0.560 vs 0.830) — a clean single bass note vs real
bass buried in mix harmonics/drums. This is a proxy; the transfer test below is the arbiter.

## Transfer + augmentation (the real arbiter)

Single-segment MLP (48→128→64→out), identical for all conditions, class-weighted loss,
song-level 80/20 RWC splits, 3 seeds. Evaluated on held-out **real** RWC. `acc` = raw,
`bal` = balanced (mean per-class) accuracy. (`scratchpad/synth_transfer.py`.)

**Quality (7-way, root-relative feat48):**

| condition                  | acc            | balanced       |
|----------------------------|----------------|----------------|
| A synth-only → real        | 0.425 ± 0.045  | 0.398 ± 0.054  |
| B real-only (baseline)     | 0.651 ± 0.034  | 0.499 ± 0.098  |
| C real + synth (augment)   | 0.644 ± 0.038  | 0.522 ± 0.083  |

**Root (12-way, feat48_abs):**

| condition                  | acc            | balanced       |
|----------------------------|----------------|----------------|
| A synth-only → real        | 0.519 ± 0.016  | 0.524 ± 0.009  |
| B real-only (baseline)     | 0.620 ± 0.018  | 0.619 ± 0.012  |
| C real + synth (augment)   | 0.619 ± 0.012  | 0.616 ± 0.004  |

### Reading

- **Pure synthetic→real transfer works for ROOT, not for QUALITY.** Root: synth-only
  reaches 0.519 = **84% of the real-trained baseline** (0.620), far above 1/12 chance —
  identifying the dominant pitch class is a low-level acoustic task robust to the
  too-clean gap. Quality: synth-only 0.425 is *below* the majority-class (maj≈0.5) raw
  baseline — the subtle interval cues (is the 3rd major/minor, is a b7 present) live in
  weaker upper-partial energy that clean synthesis renders differently from real audio.
  The entropy gap being worst exactly where transfer is worst (bass/treble harmonic
  detail) is consistent.
- **Augmentation is ~neutral.** Raw accuracy unchanged within noise for both targets.
  Quality balanced-accuracy nudges +0.024 (minority qualities benefit from synthetic's
  controllable class balance) but within one std — weak, not a win to bank on.

## Honest ceiling assessment

Soundfont-rendered synthetic data does **not** close the acoustic domain gap enough to
replace real data, and augmenting real RWC with it gives at best a marginal minority-class
benefit. What the synthetic data is *missing* is precisely what makes RWC/JAAH hard and is
absent from a clean 2-instrument fluidsynth render: **drums/percussion, vocals, real
multi-instrument mixing/mastering, and per-note timbral variation** — all of which spread
and roughen the chroma (raising entropy toward 0.93) and are where a real chord-recognizer
must be robust. The perfect-alignment advantage is real, but it buys little if the audio
distribution the model learns is the wrong one. Root is the exception worth remembering:
that low-level task transfers well enough that synthetic pretraining/augmentation for root
could still be worth a look.

## Causal realism test — does raising synthetic entropy recover transfer? (DONE)

Built a "rich" variant (`--rich`): added a busy melody/arpeggio layer (chord + passing
tones) and post-render band-limited broadband noise (SNR 12–20 dB) to push synthetic
chroma entropy toward real. `data/cache/synth/synth_bp48_rich.npz`, same 1440 chords.

Entropy moved only partway: onset 0.710, note 0.999, **bass 0.504** (got *sharper* — my
noise was mostly high-freq, the bass block is register-gated <52 MIDI), treble 0.718 (up
from 0.629, toward real 0.814), **full-48 0.854** (barely up from 0.846; real is 0.931).

Transfer on the rich corpus (vs plain synth in parentheses):

| target  | condition   | rich acc | (plain) |
|---------|-------------|----------|---------|
| quality | synth→real  | 0.416    | (0.425) |
| quality | augment     | 0.640    | (0.644) |
| root    | synth→real  | 0.530    | (0.519) |
| root    | augment     | 0.632    | (0.619) |

**Result: cheap realism tricks did NOT recover quality transfer** (0.416, still failed) —
and only nudged root up ~0.01. Since noise+melody only partially closed the entropy gap
and transfer barely moved, the quality gap is **largely structural**: the too-clean
render fundamentally under-represents real harmonic complexity, and post-hoc noise/extra
layers don't synthesize the missing upper-partial timbral detail. The remaining lever is
genuinely better *synthesis* (professional/neural — AAM), not louder noise. Root's small
consistent bump is the one place added spectral density helped.

## Next steps (ranked)

1. **AAM subset comparison** (Zenodo 5794629): professional synthesis + isolated bass
   stems, exactly where the fluidsynth gap is worst. Pull a small subset (disk permitting —
   currently ~7 GiB free, tight), run this SAME entropy + transfer harness, compare to the
   fluidsynth numbers above. Likely a better synthetic source for the quality target.
2. **Causal realism test** (below, in progress): add broadband noise / reverb / extra
   layers to raise synthetic entropy toward real, re-measure transfer — does closing the
   entropy gap recover quality transfer? If yes, realism is the bottleneck (favors AAM);
   if no, the gap is structural (label-vs-audio) and no soundfont trick helps.
3. **Root-only synthetic augmentation** at larger scale, since root is the one target that
   transferred.

Artifacts: `scratchpad/synth_gen.py`, `synth_premise.py`, `synth_build_corpus.py`,
`synth_transfer.py`; `scratchpad/synth_transfer_{quality,root}.json`;
`data/cache/synth/synth_bp48.npz`.
