# Chord ai — Reverse-Engineering Notes

**Agent 3 mission, 2026-07-15.** Goal: understand why "Chord ai"
(chordai.net) recognises chords so well and what Harmonia can adopt.

## Caveat on sources (trust-order discipline)

Chord ai is a **closed-source commercial app** (chordai.net, iOS/Android). Its
own pages disclose almost nothing technical — only marketing copy: "in-house
state-of-the-art deep learning models trained on thousands of songs that we
have meticulously labelled over the years," running "on-device within seconds."
No architecture, no feature spec, no benchmark numbers are published by them.

So this document reconstructs the *class of system* Chord ai belongs to, using
the open, peer-reviewed literature it is built on plus open-source clones that
implement the same recipe. Everything attributed to "Chord ai" specifically is
flagged as **inferred**; everything else is a cited public result. This is a
reconstruction, not a leak.

## What Chord ai actually discloses (verified)

- **Pipeline components it advertises** (chordai.net): chord recognition, beat
  + downbeat tracking, 4-stem source separation (bass / vocals / drums / other),
  audio→MIDI via **Spotify Basic Pitch** (same front-end as Harmonia), lyrics
  via **OpenAI Whisper**, key detection. On-device inference.
- **Chord vocabulary** (from their feature list): free tier = maj, min, aug,
  dim, 7, maj7, sus2, sus4. Pro tier adds half-dim, dim7, m7b5, 6, 6/9, 9,
  maj9, 11, maj11, 13, maj13, add9/add11/add#11/addb13/add13, 7#5, 7b5, 7#9,
  7b9, and **slash chords (C/E, Am/C)**. That slash-chord support means a real
  **bass model** — inversions are first-class, not discarded.
- Accuracy claim: "beyond trained human level." No number given. Ignore this;
  it is unfalsifiable marketing.

**Key takeaway from disclosures alone:** Chord ai and Harmonia share the exact
same front end (Basic Pitch). The divergence is entirely in what happens *after*
pitch extraction — and that is where all of Chord ai's advantage lives.

## The system class: structured deep chord recognition

Chord ai's public description ("deep learning models," slash-chord vocabulary,
thousands of labelled songs) places it squarely in the lineage that starts with
McFee & Bello (ISMIR 2017) and runs through BTC (2019) to ChordFormer (2025).
The defining idea of this lineage — and the single most important thing Harmonia
does not do — is **structured decomposition of the chord label**.

### 1. Structured output heads (the core idea)

McFee & Bello, *Structured Training for Large-Vocabulary Chord Recognition*
(ISMIR 2017), 170-class vocabulary. Instead of one softmax over N chord classes,
the network emits **several parallel heads over shared features**:

- a **root** head (13-way: 12 pitch classes + N),
- a **bass** head (13-way — this is the inversion / slash-chord model),
- a **pitch-class "quality" bitmap** (12 independent sigmoids = which pcs sound),
- plus the joint chord-label head for the final decode.

The heads are trained *jointly*. The bitmap and root/bass heads share structure
across chords (a Cmaj7 and a Cmaj share 3 of 4 notes), so rare classes borrow
statistical strength from common ones — the fix for large-vocab data scarcity.

ChordFormer (arXiv 2502.11840, 2025) is the current SOTA and makes the
decomposition explicit with **six components**: (1) root+triad quality,
(2) **bass note / inversion**, (3) 7th, (4) 9th, (5) 11th, (6) 13th extension.
This is a factored representation of the chord: predict the pieces, assemble the
label. It is exactly the "chord tree / hierarchy" Harmonia has been sketching in
`docs/chord_tree_2026-07-04.md`, but *learned end-to-end and supervised on audio*.

**Why this beats a 12-dim template match at the maj-vs-dom problem:** Cmaj7 and
C7 differ in exactly one pitch class (B vs Bb). Folded into a normalised 12-dim
chroma and scored by a dot/cosine against a template, that one-note difference is
a tiny geometric perturbation easily swamped by Basic-Pitch noise, octave
leakage, and the other 3 shared notes. A dedicated **7th head** trained on
labelled data learns the specific cue that disambiguates them, instead of hoping
it survives an inner product. That is Harmonia's known "7th confused with maj"
failure, and the structured head is the literature's answer to it.

### 2. Features: CQT, not folded 12-dim chroma

The modern systems do **not** fold to 12 bins. ChordFormer uses a **Constant-Q
Transform**: 22050 Hz, hop 512, C1–C8, **36 bins/octave → 252 CQT bins**. The
open-source `accordoai` M3 model uses **192 CQT features (24 bins/octave × 8
octaves)** and clearly outperforms its own 12-chroma variants (M1/M2).

Why this matters: 36 bins/octave preserves *register* and *intra-note detail*.
The bass note lives in a specific octave; a 12-dim chroma that sums octaves
(exactly what Harmonia's `_fold_to_chroma` does) **destroys the bass-vs-upper-
voice distinction by construction**. You cannot recover a slash-chord bass from
a representation that has already summed all octaves together. Chord ai keeping
octave resolution (via CQT and/or its bass stem) is a prerequisite for its
slash-chord vocabulary.

### 3. Bass handled two ways (both of which Harmonia lacks)

1. **Source separation**: Chord ai advertises a dedicated **bass stem**. The bass
   line can be tracked in isolation (near-monophonic → easy, robust root/bass),
   then fused with the harmonic content of the "other" stem.
2. **Dedicated bass output head** (McFee/Bello, ChordFormer). Even without
   separation, the model predicts bass as its own 13-way classification.

Harmonia has **neither**: no stem separation, no bass head, and a feature
(folded chroma) that has thrown the octave information away. This is Task 1
("bass model missing") and it is architecturally, not incrementally, missing.

### 4. Inference: learned emissions + light temporal model

- The neural net *is* the emission model — P(observation | chord) is learned from
  audio, not a hand-built template. Temporal smoothing is either a CRF /
  Semi-CRF layer (autochord uses Bi-LSTM-**CRF**; ChordFormer uses a Neural
  Semi-CRF for precise chord-interval boundaries) or self-attention (BTC,
  ChordFormer) that provides long-range context directly.
- ChordMini (open-source, chordmini.me / github ptnghia-j/ChordMiniApp) adds an
  **LLM post-pass** for tonal correction and structural labelling — the same role
  as Harmonia's Mission-5 LLM priors.

Contrast: Harmonia's HMM is a *hand-specified* emission (chroma template dot- or
cosine-product) plus a transition prior. The HMM is fine; the emission is the
weak link, because it is not learned.

### 5. Training data / corpus

Public datasets this class of model is trained on (Chord ai's "thousands of
songs, meticulously labelled" is an in-house superset of the same idea):

- **McGill Billboard** (~890 annotated songs, Harte-notation chord labels with
  inversions) — the standard large-vocab benchmark.
- **Isophonics** (Beatles / Queen / Zweieck, ~300 songs) — `accordoai` trains
  M2/M3 here.
- **RWC-Pop**, **USPop**, **Schubert Winterreise**, plus newer synthetic sets:
  **AAM (Artificial Audio Multitracks)** and *artificially generated audio*
  (arXiv 2508.05878, 2025) to beat the audio-annotation scarcity problem.
- **Augmentation is standard and heavy**: pitch-shift ±3–5 semitones (multiplies
  data ~10×, teaches transposition invariance), Gaussian noise, batch-norm.

Crucially these are **Harte-notation** labels that **retain `/bass` inversions** —
the opposite of POP909, whose labels *discard* inversions (per CLAUDE.md /
known-issues #3). So Chord ai is both trained and evaluated on bass; Harmonia's
main GT can't even score it.

## Public benchmark numbers (so "works well" is quantified)

MIREX Weighted Chord Symbol Recall (WCSR), large-vocab, on Billboard-style test:

| Model (year)              | Root  | MajMin | MIREX (large-vocab) |
|---------------------------|-------|--------|---------------------|
| CNN + BLSTM (baseline)    | 83.4% | 82.6%  | 81.5%               |
| **ChordFormer (2025 SOTA)** | **84.7%** | **84.1%** | **83.6%** |

BTC (Bi-directional Transformer, Park 2019) sits in the same ~83% MajMin band.
Chord ai is not publicly benchmarked, but its vocabulary and feature set match
this cohort, so ~83–85% MajMin / low-80s large-vocab MIREX is the honest
estimate of the ceiling it is chasing. Note the **class-wise** (rare-chord) score
is far lower (~0.39–0.45 frame-wise), which is exactly why structured training
exists — and where a naive template match collapses entirely.

## What they do that we don't (the 3 that matter)

1. **Structured, learned output heads** (root / **bass** / quality-bitmap / 7-9-11-13
   extensions) trained jointly on audio — vs Harmonia's single hand-built 12-dim
   template scored by dot/cosine.
2. **Octave-preserving features** (CQT, 24–36 bins/octave) **+ a bass stem** — vs
   Harmonia's `_fold_to_chroma` that sums octaves and deletes bass information.
3. **Large labelled corpus with inversions + heavy pitch-shift augmentation** —
   learned emissions from data, vs a corpus (POP909) whose labels discard bass.

## What Harmonia could adopt (ranked; see chord_ai_insights.txt)

See `docs/chord_ai_insights.txt` for the impact-ranked list and
`docs/plots/chord_ai_vs_harmonia_comparison.md` for the side-by-side table.

## Sources

- Chord ai — https://chordai.net/ and https://www.chordai.net/next-level-chord-recognition/
- McFee & Bello, *Structured Training for Large-Vocabulary Chord Recognition*,
  ISMIR 2017 — https://brianmcfee.net/papers/ismir2017_chord.pdf ,
  code https://github.com/bmcfee/ismir2017_chords
- ChordFormer, arXiv 2502.11840 (2025) — https://arxiv.org/html/2502.11840
- BTC, Park et al. 2019 — https://arxiv.org/pdf/1907.02698
- accordoai (open-source Bi-LSTM, 4 heads) — https://pypi.org/project/accordoai/ ,
  https://github.com/NightKing-V/Chord-Classification-Model-accordo.ai-
- autochord (Bi-LSTM-CRF, 25-class) — https://github.com/cjbayron/autochord
- ChordMini (open-source, DL + LLM) — https://github.com/ptnghia-j/ChordMiniApp
- *Training chord recognition models on artificially generated audio*,
  arXiv 2508.05878 (2025) — https://arxiv.org/abs/2508.05878
