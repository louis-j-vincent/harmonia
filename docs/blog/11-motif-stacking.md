# Part 11 — Motif stacking: a tune is three shapes in different keys

A jazz standard looks complex on paper — 54 chords in Anthropology, 58 in Satin
Doll. But a musician never thinks "54 separate events." They think "it's all
ii–V's in different keys, plus a turnaround and a dominant cycle."

This session's work teaches Harmonia to see that too.

## The insight

Take Anthropology (rhythm changes in B♭, AABA 32 bars). The model infers 54
chord slots. But if you ask "how many *distinct shapes* are there," the answer
is three:

1. **ii–V** — a min7 followed by a dom7 a fourth up. `Cm7 F7`, `Dm7 G7`,
   `Fm7 Bb7` — same shape, three keys. Appears **16 times**.
2. **I-VI** — the `B♭maj7 G7` turnaround. **3 times**.
3. **V/V-V** — a dominant resolving to another dominant. The bridge's cycle
   (`D7 G7 C7 F7`) is just this shape chained. Multiple occurrences.

The entire 54-chord tune compresses to **29 units using only 3 unique motifs.**
At the literal level (no transposition), it's 20 units with 4 unique patterns.

## Corpus-wide numbers

Across the 150-song corpus:

| View | Mean compression | Median | Unique motifs/song |
|------|-----------------|--------|-------------------|
| Shape (transpose-invariant) | **51%** | 51% | 4.2 |
| Exact (literal repeats) | **50%** | 52% | 3.8 |

Half the chord stream is redundant. On average, a song with 47 chords reduces
to 22 meaningful units. Best cases hit 84% compression (Alfie's Theme: 67
chords → 11 units). Only 3 songs in 150 compress less than 20% — and those are
the through-composed modal pieces (El Gaucho, Peace, Prism) where non-repetition
is the point.

## What this means for the pipeline

The motif detector operates *after* chord inference, on the decoded stream. It
finds patterns by two mechanisms:

- **Exact:** literal chord sequences that recur (`Cm7 F7` appearing 9 times).
- **Shape:** transposition-invariant — comparing the *interval + quality* skeleton
  regardless of key. Every ii–V is the same shape; a dominant cycle is a chain of
  V/V–V shapes.

Patterns are bar-aligned (respecting barlines, not straddling them) and greedy
(longest, most-compressive patterns tile first).

The interactive charts colour-code each motif occurrence and highlight all copies
on hover — you can visually verify that the model is finding real structure, not
noise. Switch between shape and exact views via the dropdown.

## Does motif voting improve accuracy?

Measured across the 150-song corpus, comparing three conditions with the same classifier:

| Condition | Family | Seventh | Exact |
|-----------|--------|---------|-------|
| Audio only | 96.6% | 93.4% | 91.3% |
| Motif fold | 96.8% | 93.0% | 90.6% |
| GT fold (oracle) | 96.7% | 93.7% | 91.6% |

**Delta (motif vs audio): family +0.2%, seventh −0.4%, exact −0.7%.**  
**GT-fold ceiling delta: family +0.1%, seventh +0.3%, exact +0.3%.**

The result is clear: on clean MMA renders, motif-based voting neither helps nor
hurts meaningfully. The oracle GT fold ceiling is itself only +0.1–0.3% — there's
simply no accuracy gap to close at this audio quality level. Family accuracy is
already 96.6%, meaning the classifier rarely makes an error; averaging across
motif copies doesn't fix what's not broken.

**Hard-audio blind experiment (N=150 songs, full multi-stem mix + SNR 3–20 dB):**

| Condition | Family | Seventh | Exact |
|-----------|--------|---------|-------|
| Blind audio | 54.3% | 25.2% | 22.4% |
| Blind + motif fold | 54.3% | 24.4% | 21.1% |

**Delta: family ±0%, seventh −0.8%, exact −1.3%.**

The motif fold *still doesn't help* on hard audio — it mildly hurts. This is the
key negative result: when inferred chords are only 54% correct, motif grouping
averages noise with noise. The detector groups "iii-I" errors together across six
occurrences and reinforces the wrong answer rather than correcting it.

**Why decision-level voting fails here:** the inferred chords are too noisy for the
grouping to be meaningful. The fix is *feature-level averaging* — pool the raw BP
activations across all instances of the same motif position *before* classifying,
not after. That averages evidence, not confusion. Not yet implemented.

## Files

- `harmonia/models/motif.py` — the detector (both views, bar-aligned, greedy)
- `harmonia/models/block_fold.py` — section-level folding (experimental, meanARI
  0.49 on clean GT — the ceiling is low because chord-repeat ≠ section labels)
- `scripts/demo_motif.py` — CLI demo on any tune
- `scripts/render_motif_chart.py` — interactive HTML charts
- `docs/plots/motif_*.html` — rendered examples (Anthropology, Satin Doll, Bye
  Bye Blackbird, Blue Skies)
