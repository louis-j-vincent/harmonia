# Chord ai vs Harmonia — Architecture Comparison

Reconstructed from public literature (Chord ai is closed-source; see
`docs/chord_ai_reverse_engineering.md` for the trust caveat). "Chord ai" column
= the SOTA structured-DL class it belongs to (McFee/Bello 2017 → ChordFormer 2025).

| Component | Chord ai (structured DL class) | Harmonia |
|---|---|---|
| Audio→MIDI front end | Spotify Basic Pitch | Spotify Basic Pitch (**same**) |
| Source separation | 4 stems inc. dedicated **bass stem** | none |
| Feature | **CQT, 24–36 bins/octave (192–252 bins)** — octave-resolved | Basic-Pitch 88-dim activations **folded to 12-dim chroma** (octaves summed) |
| Emission model | **Learned** by the network from labelled audio | **Hand-built template** scored by dot (default) / cosine |
| Bass / inversion | **Dedicated bass head** + bass stem; slash chords first-class | **Missing (Task 1)**; folded chroma has deleted octave info |
| Chord representation | **Factored heads**: root • bass • quality-bitmap • 7/9/11/13 | single label via template match; `maj→dom` hierarchy still TODO |
| 7th handling | **Dedicated 7th/extension head** learned from data | **Confused with maj** (1-pc difference lost in 12-dim dot product) |
| Temporal model | CRF / Semi-CRF or self-attention (BTC, ChordFormer) | **HMM** (transition matrix + key prior) |
| Prior / correction | LLM post-pass (ChordMini) for tonal/structural correction | Harmonic prior (didn't help); Mission-5 LLM priors (in progress) |
| Vocabulary | 170 (McFee) → **301 (ChordFormer)**, inc. slash chords | 25-class base; larger vocab is a stated goal |
| Training corpus | McGill Billboard (~890), Isophonics, RWC, +synthetic (AAM) | POP909 GT (**discards `/bass` inversions**) |
| Augmentation | Pitch-shift ±3–5 semitones, noise, batch-norm | none of note |
| Reported accuracy | ~83–85% MajMin / ~83.6% MIREX large-vocab (ChordFormer) | not directly comparable (different GT/metric) |

## The three architectural gaps, in one sentence each

1. **Feature**: Harmonia sums octaves into 12 bins, which is mathematically
   incapable of representing a bass note; the SOTA keeps 24–36 bins/octave.
2. **Emission**: Harmonia scores a hand-built template by inner product; the SOTA
   *learns* P(obs|chord) from thousands of labelled, pitch-shifted songs.
3. **Output structure**: Harmonia predicts one chord label; the SOTA predicts
   root, bass, and each extension as separate jointly-trained heads, which both
   fixes maj-vs-7th and lets rare chords borrow strength from common ones.

## What's genuinely shared (don't rebuild these)

- Basic Pitch front end — identical choice, already validated.
- Beat/downbeat tracking + HMM/temporal smoothing — Harmonia's is fine; the
  weak link is the emission, not the transition model.
- LLM correction pass — Harmonia's Mission-5 is on the same track as ChordMini.
