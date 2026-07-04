# What the accompaniment DB teaches the inference models — 2026-07-03

Cheap symbolic experiments on the new (structure, chords, bass, midi) database
(`docs/accompaniment_db_2026-07-03.md`), targeting Harmonia's three documented
struggles: chord-change detection, chord priors, and the note→chord emission
bottleneck. All experiments are noise-free (chart = ground truth by
construction) — they measure *ceilings* and *signal locations*, separating
"the model is wrong" from "the evidence is weak".

Scripts: `scripts/analyze_accomp_priors.py` (db.jsonl only),
`scripts/analyze_accomp_emission.py` (MIDI voicings),
`scripts/analyze_accomp_structure.py` (form recovery).

**Standing caveat**: MMA voicings are algorithmic groove libraries, not human
performances — learned numbers are cleaner than reality and describe *this
renderer's* accompaniment style. Directions and orderings transfer; exact
magnitudes should be re-fit if the input distribution changes. Also, within-bar
placement of 3+-chords-per-bar is our even-split assumption (exact for the
dominant 1–2 chords/bar case).

---

## 1. Emission weights: the hand templates guess wrong about the 5th, but template geometry is NOT the bottleneck

Measured from ~67k chord instances (full jazz corpus, duration-weighted pitch
classes of comping tracks during each ground-truth chord span):

- **The 5th is fully present** in accompaniment voicings (empirical weight
  ≈1.0 vs the hand template's 0.3–0.35 "jazz shell voicing" assumption). Every
  core interval of every quality sounds in ≈99% of instances — MMA (like most
  real-world comping guitar/piano) plays complete voicings.
- **Oracle quality-classification ceiling** (perfect MIDI notes, true root
  given, 14 qualities, train/test split by song): hand templates **90.3%**,
  learned templates **95.4%**; collapsed to maj/min/dom/dim: **~98%** for both.
- Confusions concentrate exactly where `known_issues.md` #1 predicted:
  `dom7alt→dom7` dominates (extensions share the core tetrad), then
  `maj7↔maj`, `min7↔min`, `min7→7sus4`.

**Implication (important):** with *perfect* note evidence the fine-quality
problem is ~95% solvable and majmin is ~98% solvable — so the real pipeline's
~30% majmin is overwhelmingly an **evidence-quality problem (audio→notes)**, not
template geometry. Swapping learned templates in (`build_emission_matrix`)
is a real but modest win (+5 points at the ceiling); the big lever remains
Stage-1 observation quality (`docs/suggestions.md` items). Learned-vs-hand
cosine table shows learned templates *increase* separation for
`min7 vs m7b5` (0.86→0.74) but qualities like `dom7 vs dom7alt` (0.95) are
intrinsically near-identical — supporting a scoring vocabulary that merges
alt-dominants into dom7 until Stage 1 improves.

## 2. Register: root is a bass question, quality is a comping question — now with numbers

Share of each interval's total evidence originating in the bass track:
**root 24.7%, 5th 8.4%, 3rd 0.5%, b7 0.4%, maj7 0.7%**. Within the bass track
itself, 75.6% of its mass is the root. This is the two-stage bass-anchored
design's ablation finding (86.8% experiment), measured directly: bass carries
root information and essentially zero quality information.

## 3. Bass reliability is beat-phase-dependent — learned weights for the bass-anchored scorer

P(bass interval | beat phase), walking-bass grooves, on-beat notes (~94k notes):

| phase | root | 3rd | 5th | 7th |
|---|---|---|---|---|
| beat 0 | **97.2%** | 0.7% | 1.0% | 0.4% |
| beat 2 | 48.7% | 0.5% | **45.9%** | 0.7% |
| beat 3 | 78.8% | 0.5% | 17.3% | 1.0% |

A downbeat bass note is a near-certain root read; a beat-2 bass note is a
root/fifth coin flip. The bass-anchored scorer should weight bass evidence by
phase instead of uniformly. (Synthetic-clean caveat applies — real bassists
are looser — but jazz pedagogy agrees directionally.)

## 4. Jazz harmonic rhythm: sharper and *safer* than POP909's

- **Duration PMF** (60k merged events): d=2 beats **56.4%**, d=4 **31.6%**,
  d=8 5.5%, d=1 4.7%, d=3 ≈ **0%**. Compare POP909's 15/49/9/26. Jazz chart
  durations live almost entirely on {2, 4, 8} — a semi-Markov decoder for
  jazz-style input can use support {2,4,8} outright.
- **P(change | beat phase)**: 86.6% / 2.4% / 38.4% / 0.7% for beats 0–3.
  Per-song downbeat advantage **+0.86 ± 0.17** (range +0.05..+1.00) — unlike
  POP909's +0.28 ± 0.53, this prior is *consistent across songs*, so the
  per-song-adaptive weighting `architecture_extensions.md` #9 demands for pop
  is unnecessary for jazz: changes essentially only happen on beats 0/2.
- **Cadence acceleration is real but tiny** (change rate 34.0% in a section's
  last 2 bars vs 31.1% elsewhere) — not worth a prior.
- **"ii-V-I implies longer tonic" is falsified**: tonic duration after a ii-V
  3.70 beats vs 3.76 otherwise, near-identical PMFs (n=3.5k/6.1k). This was
  `structure_trigram_design_2026-07-04.md`'s Section 2 test-before-building
  condition — the trigram-conditioned duration model should **not** be built.

## 5. Jazz n-grams: measured, and very different from POP909

Top bigrams over 57k transitions (scale-degree relative to song key,
qualities maj/min/dom):

| jazz1460 | share | | POP909 | share |
|---|---|---|---|---|
| iim → V7 | **9.81%** | | V → I | 9.58% |
| V7 → I | 7.57% | | IV → V | 5.21% |
| VI7 → iim | 3.53% | | I → IV | 4.56% |
| iiim → VI7 | 2.94% | | I → I(qual) | 4.27% |

Dominant-heavy, ii-V saturated — the vocabulary the hand-written
`PROGRESSIONS` assumed but POP909 couldn't supply. Two negative results that
save work:

- **Trigram context adds little**: P(I | ii,V) = 58.9% vs P(I | V) = 56.1% —
  the bigram already knows. With median 6 obs/context (only 24% of contexts
  ≥20 obs), trigram tables buy sparsity without predictive gain. **Use
  bigrams + duration prior; skip trigrams.**
- Explicit tritone-sub (bII7→I 0.35%) and backdoor (bVII7→I 0.56%) are rare
  *as notated* — keep their hand-specified prior weights modest.

## 6. Structure: form is recoverable from harmony alone — but only with the right representation, and period selection is the real bottleneck

Test: MIDI-derived beat chroma → existing `build_ssm`/`score_periods` →
greedy clustering → ARI against ground-truth per-bar section labels
(200 multi-section jazz songs; MMA's constant per-song groove means the SSM
sees *harmony only* — the clean version of Candidate C's confound).

| configuration | median ARI | ≥0.5 |
|---|---|---|
| pooled window chroma @ GT period (current prototype's method) | **0.00** | 32% |
| **position-wise sequence** chroma @ GT period | **1.00** | **78%** |
| sequence chroma @ `score_periods` top-1 | 0.34 | 39% |
| sequence chroma @ cluster-quality-selected period | 0.39 | 35% |

Three hard conclusions:

1. **Pooled section chroma is the wrong representation** — same-key sections
   share aggregate pitch content; form identity lives in the chord *sequence*.
   `illustrate_form_clustering` (and any CRHA-gated folding) should compare
   windows position-wise, not by mean chroma. This one change moves median
   ARI from 0.00 to 1.00 at the true period.
2. **Period selection is now the bottleneck**: `score_periods` top-1 matches
   the true section length only 29% of the time; "smallest/longest within ε"
   rules and cluster-quality selection all land ≤29% top-1 / ≤0.49 mean ARI,
   vs 0.76 achievable at the true period. SSM diagonal scores are nearly flat
   across harmonically related lags (groove-level repetition saturates short
   lags) — the same failure `known_issues.md` diagnosed for Candidate C,
   quantified. Selectors on the same score won't fix it.
3. **A learned section-length prior exists now**: GT dominant section length
   is 8 bars in 65% of songs, 16 bars in 18% — restricting hypotheses to
   {8, 16} bars covers 83% of the corpus. Combined with multi-hypothesis
   carry-forward (already the design in `structure_trigram_design`), this is
   the practical path: enumerate {8,16}-bar hypotheses, validate by
   position-wise sequence agreement, never trust a single detected period.

## 7. Audio round-trip: quantifying (and partially recovering) the Stage-1 gap

Follow-up run the same day: rendered a 60-song pilot to audio with controlled
variation (transpose ±5 st with GT shifted accordingly, 2 soundfonts, reverb
on/off, pink noise at 15/8 dB SNR — 120 renders,
`scripts/build_accomp_audio.py`), ran Basic Pitch, aligned to the *known* beat
grid (fixed tempo — zero beat-tracking noise), and re-ran the oracle
quality-classification test on real BP evidence
(`scripts/learn_stage1_mapping.py`). 4,896 chord instances.

**Evidence degradation, isolated:**
- Per-beat chroma agreement with GT: cosine **0.81** on clean canonical
  renders — even pristine audio loses a fifth of the signal. Added noise costs
  surprisingly little more (0.77 at 8 dB SNR); soundfont/transposition variants
  ~0.78. BP's problem is baseline smear, not noise fragility.
- Per-key on/off activation ratios are dire: 0.8–1.2 in the main registers,
  0.4 in the high register (harmonic bleed activates keys that aren't
  sounding). The evidence is key-level mush that only becomes usable after
  pooling to chroma.

**Quality classification (true root given, train/test split by song):**

| templates | perfect MIDI notes | real BP evidence |
|---|---|---|
| hand (current) | 90.3% | **42.1%** |
| MIDI-learned | 95.4% | 45.2% |
| **noisy-learned** (fit on BP features) | — | **55.2%** |
| collapsed maj/min/dom/dim (any) | ~98% | ~78% |

Three conclusions:
1. The audio round-trip costs ~40 points of fine quality — confirming Stage 1
   as the dominant bottleneck, now with a number.
2. **Fitting templates on noisy BP evidence recovers +13 points over hand
   templates for free** (42→55%). The emission model should be *learned in
   the observation space it actually sees*, not specified in note space.
   This is the v0 learned audio→chord mapping, and the pilot pipeline can
   generate unlimited training pairs for it.
3. Confusions on real evidence are semitone-smear shaped (`dom7→7sus4` = 3rd
   vs 4th, `min7→min` = lost 7th) — precisely what a trained per-key
   denoising/calibration model could target next.

**Two negative results worth keeping:**
- The naive hybrid onset+α·note observation (suggestions.md item 3) does
  nothing for chroma agreement once the note channel is mass-normalized
  (0.808 vs 0.814) — and without normalization it's actively harmful (the
  sustain channel swamps the onset signal).
- Naive per-key sensitivity equalization (divide by μ_on, suggestions.md's
  per-key calibration) *collapses* accuracy (55→26%): it amplifies exactly
  the noise-dominated high-register keys. Any per-key calibration must be
  discriminative (on/off contrast), not gain-based.

**Recommendation:** the infrastructure now generates unlimited
(BP-activations, perfect-pianoroll) pairs with augmentation. Next step in
order: (1) adopt noisy-learned emission templates in the pipeline (cheap,
+13 validated); (2) train a small per-key discriminative calibration
(logistic per key on BP activations → GT-on) locally; (3) only if that
plateaus, a proper learned audio→pianoroll correction model on Colab per the
compute plan.

## Integration priority (proposed)

1. **Cheap + validated**: jazz duration prior {2,4,8} + beat-phase change
   prior (beats 0/2 only) into the change-detector plan; per-phase bass
   weights into the bass-anchored scorer. All three are direct parameter
   swaps in already-planned components.
2. **Learned emission templates**: export per-quality learned weights and use
   them in `build_emission_matrix`; merge alt-dominants into dom7 in the
   scoring vocabulary. Modest but free.
3. **Structure**: switch form clustering to position-wise sequence vectors;
   adopt {8,16}-bar hypothesis set with sequence-agreement validation.
4. **Do not build**: trigram tables, trigram-conditioned durations, cadence
   priors — all measured too weak to pay for their complexity.
5. The dominant open lever is unchanged and now sharply bounded: **Stage-1
   audio→notes quality** — the gap between the 95%/98% oracle ceilings and
   the real pipeline's ~30% majmin lives entirely there.
