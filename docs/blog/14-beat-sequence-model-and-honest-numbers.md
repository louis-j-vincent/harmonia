# Part 14 — The beat-sequence model and honest numbers

## What are we actually measuring?

Before logging any more numbers, worth being explicit about what the eval harness
tests — because "88% majmin" means very different things depending on the data and
metric.

**Metric: MIREX weighted-overlap accuracy.** For each predicted chord interval, the
score is the fraction of its duration that overlaps the correct GT chord, weighted
by duration. Three levels:
- `root` — root pitch class only (C, C#, …)
- `majmin` — root + major/minor quality
- `sevenths` — root + extended quality (maj7, min7, dom7, …)

This is a duration-weighted overlap metric, so a predicted chord that spans the
right region and has the right label counts fully; one that's right but a beat
early counts partially. It's the standard MIREX ACE metric.

**Data: synthetic jazz1460 corpus.** iReal Pro chord charts → MMA (Musical MIDI
Accompaniment) renderer → FluidSynth with MuseScore General soundfont → WAV.
Ground truth is the chord chart itself, parsed into time-aligned spans via
`song_chord_spans()`.

Key properties of this data:
- **Metronomic.** MMA renders at a fixed tempo with no rubato. Beat grid is exact.
  This is best-case for beat-synced chord inference.
- **Synthetic instruments.** No real ensemble noise, room acoustics, or performance
  variation between occurrences of the same section.
- **Jazz vocabulary.** ii-V-I progressions, walking bass, extended chords. Much
  richer than POP909 (where the bottleneck was timing; here it's labeling the
  right root over walking bass).

**What's NOT being tested:**
- Real recorded audio (rubato, reverb, real instruments, imperfect performance)
- Chord changes that anticipate the beat (common in jazz comping)
- The fully degraded condition (`--hard-degrade`) — that was tested earlier at
  ~64% majmin, showing the segmentation breaks first under heavy distortion

---

## The honest standalone number

Numbers on the "15 clean songs" eval have mild train/eval overlap (models trained on
the full corpus, eval on a subset). The **disjoint held-out** number — train on
even-numbered songs, eval on 30 odd songs, detected tempo grid, fully standalone —
is **~75% majmin** (root 77.6%). That's the number to compare against real-audio
systems. The 88–89% figures are the ceiling under ideal conditions (known beat grid,
no train/test leak from the song-level split).

---

## This session: the beat-sequence model

### The bottleneck we were attacking

Per-beat root accuracy at 67% was blocking everything fine-grained: within-cell
chord splits, segmentation under distortion, soft probabilistic labeling. But the
67% was a red herring — it was the *segment-trained* model misapplied to single
beats (distribution mismatch). A model actually trained on per-beat data gets
**85.5%**. With ±2 neighbouring beats' features concatenated (480d → 240d windowed
LR), it reaches **88.9%** on clean audio and **86.0%** on degraded.

This was validated in `per_beat_context_experiment.py` and motivated building a
production-ready version: `train_beat_seq_model.py`.

### What we built

`scripts/train_beat_seq_model.py`: windowed logistic regression over per-beat
chroma features.

- **Features per beat**: 48d — `chroma88(onset)` full, note, bass register (MIDI
  21–52), treble register (MIDI 60+). Same as the per-beat experiment.
- **Context window**: ±2 neighbouring beats concatenated → 48 × 5 = 240d. Each
  beat "sees" its context within the song; edges are zero-padded.
- **Training**: 50 songs, clean + degraded augmentation (100 renders total),
  ~12,928 beat samples.
- **CV result**: 88.3% 5-fold per-beat root accuracy (matches the 88.9% from
  the 15-song experiment).
- **Saved to**: `harmonia/models/beat_seq_model.npz`

Wired into `chord_change_engine.py` via `--beat-seq`:
- Runs once per song → `(n_beats, 12)` soft root probability matrix
- **Labeling**: for each segment, sum per-beat proba over the segment's beats →
  argmax. Fully soft — no argmax until the last step.
- **Within-cell split**: same trigger as before (both beats in a 2-beat cell have
  different argmax AND both > `split_conf`), but now using per-beat proba from
  the sequence model.
- **EM refinement**: per-segment soft proba = normalized sum of per-beat proba →
  feeds the Viterbi progression prior.

### Results

| condition | root | majmin | seg/GT | notes |
|---|---|---|---|---|
| `--root-model` clean | 86.0% | 88.7% | 1.04 | segment-level baseline |
| `--beat-seq` clean | 85.1% | 88.0% | 1.02 | nearly neutral |
| `--root-model` degraded | 84.3% | 86.0% | 1.18 | segment model degrades |
| **`--beat-seq` degraded** | **87.0%** | **88.9%** | 1.14 | **+2.7 root, +2.9 majmin** |
| `--beat-seq --oracle-bounds` | 87.6% | 91.1% | 0.94 | labeling ceiling |
| `--beat-seq --split-conf 0.7` | 83.8% | 86.9% | 1.21 | over-segments |
| `--beat-seq --split-conf 0.85` | 84.2% | 87.2% | 1.10 | still over-segments |

The degraded result is the key finding: **beat-seq matches the clean baseline even
under degradation** (88.9% degraded ≈ 88.7% clean segment). Temporal context is
doing what it should — averaging out degradation artifacts across neighbouring beats
makes each per-beat estimate more stable than a single pooled-segment signal.

On clean synthetic data it's neutral to slightly behind. That's expected — the
segment model sees a fully pooled oracle signal optimally calibrated for the
"pool everything then classify" path. Beat-seq's edge is in the real-audio regime.

### What still doesn't work: within-cell split

The within-cell split (deciding whether a 2-beat cell contains 1 or 2 chords) still
over-fires at every tested threshold. The LR softmax max-probability reaches 0.7–0.9
even for ordinary beats where there's no real ambiguity — it's not a calibrated
"I'm really sure" signal. Any threshold below ~0.95 fires too often.

This is a calibration problem, not a threshold problem. The fix would be:
- Temperature-scaling the LR outputs post-training
- Or a dedicated binary "same/different" head trained on adjacent-beat pairs

For now the split remains off by default.

### EM + beat-seq interaction

`--beat-seq --refine` is also slightly worse than `--beat-seq` alone (86.7% vs 88.0%
majmin). Cause: summing per-beat distributions over a segment produces a flatter
probability vector than the segment model's single-shot softmax. The EM conf-gate
(≥0.6) was calibrated for sharper segment-model distributions, so it misclassifies
too many segments as "fuzzy" and Viterbi-decodes them unnecessarily. The conf-gate
would need to be recalibrated for beat-seq proba distributions.

---

## This session: vary_voicings fixed

The structure-fold experiment (`--fold --vary`) was showing −13 majmin despite folding
helping in isolation. The culprit: `vary_voicings()` was creating "independence" by
*omitting* upper pitch classes per repeat occurrence — this changed the chroma vector
(thinned the harmony), confusing the chord classifier.

The fix: remove all note-dropping. Vary only the audio surface:
- Octave shifts (30% per non-bass note, ±1 octave)
- Velocity swings (±25%)
- Micro-timing jitter (±15ms)

Independence now comes from different Basic Pitch onset errors per repeat (same
chords, different waveform surfaces), not from changing the harmony.

Verified: zero pitch-class diff, same note count.

The `--fold --vary` combination should now be net-positive on the engine — pending
a full re-eval.

---

## Where things stand

Best numbers:
- **GT grid, coarse, θ=0.08, segment root model**: root 86.0% / majmin 88.7%
- **GT grid, coarse, θ=0.08, beat-seq, degraded**: root 87.0% / majmin 88.9%
- **Standalone disjoint (30 odd songs, tempo grid)**: root 77.6% / majmin 74.9%

The standalone 75% majmin is the honest floor. The 88–89% figures are ideal-condition
ceilings. Real audio with rubato and ensemble noise will sit somewhere between them —
the beat-seq model should help close that gap.

Next levers in roughly priority order:
1. `--fold --vary` re-eval with fixed vary_voicings
2. Calibrate per-beat proba for the split decision (temperature scaling or binary head)
3. DB regen with fixed vary_voicings for training fold-robust models
4. Real audio evaluation
