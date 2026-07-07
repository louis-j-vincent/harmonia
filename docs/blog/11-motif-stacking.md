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

## What it doesn't (yet) do

This is a post-hoc analysis of the decoded chords. It doesn't yet feed back into
inference (voting across motif copies to fix errors). On our clean MMA renders,
family accuracy is already ~100%, so there's no headroom for voting-to-fix. On
harder input (real recordings, degraded audio where family drops to 70–85%), motif
voting across copies could be the lever — that's the next test.

## Files

- `harmonia/models/motif.py` — the detector (both views, bar-aligned, greedy)
- `harmonia/models/block_fold.py` — section-level folding (experimental, meanARI
  0.49 on clean GT — the ceiling is low because chord-repeat ≠ section labels)
- `scripts/demo_motif.py` — CLI demo on any tune
- `scripts/render_motif_chart.py` — interactive HTML charts
- `docs/plots/motif_*.html` — rendered examples (Anthropology, Satin Doll, Bye
  Bye Blackbird, Blue Skies)
