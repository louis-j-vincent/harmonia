# 16 — Closing the Domain Gap: Training on YouTube + iReal Pro

*2026-07-09*

## The problem statement

Everything in this project so far has been trained on synthetic audio — MMA-rendered General MIDI piano
comping, built from iReal Pro chord charts. The models work on that domain. But real recordings sound
nothing like that: live piano, ensemble textures, reverb, rubato, all the things that make jazz recordings
sound like jazz rather than a sound bank.

The question: can we build a paired dataset of (real YouTube audio, iReal Pro ground truth) and train a
quality classifier that actually generalises to real recordings?

## The setup

New file: `harmonia/data/yt_chord_corpus.py`. It chains:

1. **Download** audio via yt-dlp → ffmpeg (mono, 22050 Hz)
2. **Infer** a chord chart via `infer_chords_v1` (our existing pipeline)
3. **Align** the iReal Pro GT to the inferred chart via DTW (`irealb_aligner.align_irealb_to_inferred`)
4. **Extract** per-segment features: 48-dim root-shifted BP chroma (same pathway as `_FamilyClassifier`)
   plus 12-dim root-shifted CQT chroma (librosa, 36 bins/octave)
5. **Label** each segment with the aligned iReal root + quality

Quality scheme: 7 classes — maj / min / dom / hdim / dim / aug / sus. Dom is a new class (the existing
`_FamilyClassifier` merges dom into major — fatal for jazz where V7 vs Imaj7 is the most important
distinction).

## Pilot: 10 songs, reality check

10 jazz standards (Autumn Leaves, Bye Bye Blackbird, All The Things You Are, …): 1337 total records,
872 clean (alignment exact or family match).

Leave-one-song-out cross-validation with logistic regression: **49.4% quality accuracy** (RF: 56.3%).

Feature diagnostics: BP chroma m3/M3 ratio = **2.89 for min** vs **0.73 for dom**. The 1-semitone
major/minor third distinction IS captured at the mean level. The failure isn't features — it's data
diversity. With 10 songs (1 held out for val) the model memorises recording-specific patterns.

Confusion breakdown:
- min/dom is bidirectional and massive: 85 min→dom + 65 dom→min in LOSO
- hdim: 17% (only 41 records); dim: 0%
- Random Forest (56.3%) beats logistic regression significantly — the boundary is non-linear

## 50 songs: first meaningful numbers

50 jazz standards from the top of jazz1460. **7195 total records, 4257 clean.**

Song-level split (7 held-out songs, 534 val records):

| Model | Quality val | Root val |
|---|---|---|
| 7-class MLP, no context | 53.0% | 65.9% |
| 7-class MLP, ±1 context | 53.6% | 63.7% (overfit!) |
| **3-class MLP** (maj/min/dom only) | **61.4%** | 66.2% |

Context windowing (±1 segment neighbor) barely helps quality (+0.6pp) and badly overfit the root head
(train=97%, val=64%). The 180-dim model has too many parameters relative to 50 songs.

The 3-class simplification (drop hdim/dim/aug/sus entirely) is the bigger win: **+8pp quality accuracy**.
Rare classes (hdim=2.6%, dim=1.6%) bleed noise into the 7-class objective without having enough data
to learn from. With 200+ songs hdim may be worth adding back.

## Direct comparison: yt model vs existing FamilyClassifier

On the same 7 val songs:

| Class | yt 3-class | existing (synth-trained) |
|---|---|---|
| maj | 60% | **87%** |
| min | 63% | 46% |
| dom | 59% | 0% (merges to maj!) |

The existing classifier never predicts dom — it's structurally blind to the dominant/major distinction.
The yt model trades some maj accuracy for dom/min discrimination. In jazz, that's the right trade.

Note the existing classifier's "lenient" accuracy (crediting dom→maj predictions) is 60.3%, vs yt's 61.4%
strict. But strict dom recognition at 59% vs 0% is the real story.

## What limits us

Train/val gap at 50 songs: 88.9% train vs 61.4% val (quality). This is overfitting from limited
training diversity. The gap should narrow with more songs.

The harder problem: jazz chord quality labels are ambiguous. A dom7 chord with b9, #11, b13 alterations
has a completely different pitch-class distribution than a plain dominant 7th. Both are labelled "dom"
in iReal Pro. A minor chord voiced as 1-3-7-9 (omitting the 5th) looks like a half-diminished from a
chroma perspective. These are inherent ambiguities — the ceiling for this task on real audio is probably
70-75% even with infinite data.

## Next: 200 songs

The 200-song corpus build is running now. Projecting ~28k total records, ~17k clean.
Expectation: quality val should clear 65-68% for 3-class, narrow the train/val gap to <20pp.
After that: integrate the best quality head into `chord_pipeline_v1` as a real-audio branch, and
run end-to-end MIREX on held-out YouTube songs.

**Models saved:** `harmonia/models/yt_chord_model.npz` (7-class, 60-dim), 
`harmonia/models/yt_chord_model_3cls.npz` (3-class, 60-dim).
