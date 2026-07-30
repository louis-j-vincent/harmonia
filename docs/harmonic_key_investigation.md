# Harmonic-key investigation (branch `feat/harmonic-key`)

Goal: given NNLS chroma (+ section folding later), detect which *scale* the
song is in at each moment — e.g. This Love alternating C harmonic minor
(B natural, on the G7 bars) vs C natural minor (Bb).

## v1 — binary scale masks vs per-bar chroma (2026-07-30)

Script: `scratchpad/key_scale_dotprod.py` →
plot `scratchpad/key_scale_dotprod_this_love.png`.

Setup: per-beat NNLS treble chroma (C-first, live `pool_beats`), pooled to
4-beat bars (anchor chosen to align bar lines with the baked chart's chord
onsets: offset 2, 74/117 onsets within 120 ms). For each C-minor variant
(natural / harmonic / melodic / dorian), score = share of the bar's chroma
mass inside the 7-pc mask.

Findings:

1. **The alternation Louis hears is clearly recoverable** — but the signal
   lives almost entirely in the two discriminating pitch classes, not the
   full mask. Per-bar Bb-vs-B share flips exactly with the chart's G/G7
   bars (B ≈ 0.20–0.29 there, ≈ 0.02 elsewhere); verse 1 is solidly
   leading-tone (harmonic-minor colour), the Bb-chord sections solidly
   natural. B > Bb in 40% of bars; A > Ab in 32% (F-7/C-7 bars).
2. **The full 7-pc mask dot product is diluted**: variants share 6 of 7 pcs,
   so mean shares sit within 0.05 of each other (natural .768, harmonic
   .747, dorian .735, melodic .714) and the top-panel curves overlap. A
   detector should score the *contrast pcs* between hypotheses (pairwise
   likelihood-ratio), not the whole mask.
3. **Noise floor**: ~11.6% of chroma mass is outside *every* C-minor variant
   (Db+E+Gb) — NNLS bleed + vocals. Any threshold must clear this.
4. Cosine (Louis's normalized-dot formulation) vs L1 mass share: r = 0.65–0.91
   per variant — same information, L1 share easier to read as "% in scale".

Open design questions (for Louis):
- Granularity: per-bar scale flips on every G7 — do we want that, or a
  two-level output (key = C minor constant; scale *colour* tracked per
  bar/chord)?
- Include the bass half? The chart literally writes G7/B — the leading tone
  is often *in the bass*.
- Next: sticky HMM over scale states on the contrast-pc evidence, vs
  section-fold the chroma first and decide per section group.
