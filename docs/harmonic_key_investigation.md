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

Design calls (Louis, 2026-07-30): key stays C minor; track the *colour* per
chord; include the bass half; sticky HMM first, section folding after.

## v2 — per-chord colour via sticky HMM (2026-07-30)

Script: `scratchpad/colour_hmm_this_love.py` →
plot `scratchpad/colour_hmm_this_love.png`.

Model: states = {natural, harmonic, dorian, melodic} = joint setting of the
two contrast degrees (Ab/A × Bb/B). Per chord span (chart t0/t1): chroma =
L1(treble) + L1(bass), evidence per degree = (mass on the pair, raised
share), Bernoulli-style emission (Q=0.85) weighted by GAIN=25 × mass,
Viterbi with STAY=0.92. Chords with no contrast-pc mass are carried by
stickiness ("hold until forced", same philosophy as local key).

Results (120 chords): natural 25, harmonic 40, dorian 55, melodic 0;
11 switches; 6/7 G-root chords get the raised 7th.

1. **The 7th-degree axis is crisp and matches the ear**: verses decode
   harmonic (B share 0.7–0.93, big evidence), flat-7 sections decode
   clean. This axis is reliable.
2. **The 6th-degree axis is the weak one**: through the chorus the A share
   hovers 0.55–0.65 with modest mass, so the HMM calls **dorian**, not
   natural. Two readings: (a) the chorus really has F major/F7 (A natural)
   and the chart's `F-` label is wrong — then this is a genuine catch; or
   (b) A is harmonic-series bleed (A = 3rd partial of D, and D is in every
   Bb chord). Needs Louis's ear / iReal to adjudicate.
3. **The bass half matters a lot**: 44/120 chords change colour without it;
   it is what pushes the chorus to dorian.

Open: make 6th-degree evidence chord-aware (only count it when the sounding
chord can contain a 6th degree) or raise its evidence bar; then fold
evidence by section group.
