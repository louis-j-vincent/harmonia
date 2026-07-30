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

## v2b — ear check + bass-as-sounding-note fix (2026-07-30)

**Ear GT (Louis)**: chorus F at ~42.7s is definitely **F minor** → the v2
dorian chorus was a false positive. But the *last 4 bars* of the B section
do have **F major** → dorian there is real. (Chart agrees: it writes `F^7`
at 57.9 / 108.4 / 158.9 s — the end of each B occurrence — and plain `F` at
179.1 / 199.3 s.)

**Mechanism found** (diagnostic in the script): on the F- chord the *bass
half's* top pitch classes are A (0.14) and E (0.14) — partial-series junk
(A is the 5th partial of F) — while the treble correctly has Ab ≥ A. Adding
the bass half as a mass blob (v2 `l1` mode) injects a fake A natural under
every F bass. **Fix**: bass enters as the *sounding-bass pc only* (argmax
one-hot, +0.3 mass) — the same way the live pipeline uses the bass half
(`nnls_features.py`: "UNTRAINED argmax"). Ledger note: ear-correction →
general rule, *never use the NNLS bass half as mass; argmax only*.

Decode comparison (120 chords):

| colour   | argmax (new) | l1 (v2) | treble-only |
|----------|-----|-----|------|
| natural  | 78  | 25  | 69   |
| harmonic | 27  | 40  | 36   |
| dorian   | 11  | 55  | 15   |
| melodic  | 4   | 0   | 0    |

Chorus now decodes natural (matches ear); verses stay harmonic.

Remaining gaps:
1. Of the 5 chart F-major chords, only #102 (179.1s) flips to dorian — a
   single-chord inflection loses to stickiness (2 switch penalties ≈ 7).
   Design question: should a one-chord borrow flip the sticky track, or be
   a per-chord *inflection flag* layered on the prevailing colour?
2. A 4-chord **melodic** block appeared near the bridge (around Ab→G7) —
   raised 6+7 simultaneously; check real vs bleed.
3. Treble-half bleed remains on Bb chords (A = partial of D; #22: treble
   A=.050 vs Ab=.016) — feeds the pre-chorus dorian patch. Chord-aware
   evidence is the likely guard, but the 6th-degree pair is genuinely
   quality-entangled on Bb roots (Bb7 has Ab, Bbmaj7 has A).
