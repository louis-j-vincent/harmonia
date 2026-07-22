# Bayesian Fusion Aligner — design (greenlit 2026-07-23)

**Status:** Louis greenlit "la fusion avec batterie en bayésien total" (2026-07-23),
superseding the earlier "finish the 8 songs first" call. The 5 frozen Brick 0 songs
(Stand By Me, Bein' Green, Blue Bossa backing, Every Breath, Blue Bossa) + Close To You
become the **validation set**, not wasted work.

## Why we're here (the wall the threshold-stack hit)
We built good single instruments (harmonic agreement, offset-ramp, windowed drift,
form-vamp) combined by hand-tuned thresholds. Two failures show that's not robust:
- **Autumn** "part en cacahuète" in the solos: comping drops → chroma agreement becomes
  noise → vamp/section placement drifts (metric r rose 0.228→0.273 while the ear said
  WORSE — a metric-up/ear-down proof that a single signal + thresholds is fragile).
- **Let It Be** (held-out): good at the start then gets lost, and the DP opened a phantom
  4.2s gap at 40–44s that Louis confirms is NOT there. Same root: harmony-only loses the
  grid where the mix thickens.

Louis's fix: navigate by the **DRUM rhythm** — the kit holds a steady pattern and marks
the downbeat clearly; learn that pattern on the confident opening and carry it through the
solos. Generalize: **fuse several complementary signals, each weighted by local
reliability**, in one Bayesian model — not a pile of thresholds.

## Model: a Bayesian bar-pointer state-space over the beat grid
Latent state per beat *k*: `{ t_k (time), τ_k (beat period, slowly drifting),
b_k (metrical position; downbeat = b_k==0), s_k (section label) }`.

**Observation likelihood at beat k = Σ_i w_i(t_k) · log L_i(t_k)** — reliability-weighted
fusion of complementary streams:

| i | stream L_i | strong where | weight w_i(t) driver |
|---|---|---|---|
| 1 | **harmonic agreement** chroma(t) vs chart chord template @ (s_k, position) | comping present | comping/chroma salience (LOW in solos) |
| 2 | **drum pattern** local percussive/onset envelope vs a template learned on the confident opening (per-metrical-position onset signature + downbeat accent) | everywhere incl. solos | drum-track energy/steadiness (HIGH in solos) |
| 3 | **harmonic rhythm** chord-change (chroma-flux) landing on strong metrical positions | clear changes | flux salience |
| 4 | **downbeat accent** low-band/percussive accent at b_k==0 | clear groove | accent contrast |

The **reliability weighting is the Bayesian win** ("se complémentent et se renforcent"):
in a solo, w_harm↓ and w_drum↑ automatically — impossible with fixed thresholds.

**Priors:** τ continuity (random-walk, small variance) + rare discrete jumps = pause-gaps
(vamps/turnarounds, only LARGE ones); section transitions from the CHART form (order +
section lengths, e.g. Georgia 8-bar A, form A-A-B-A); chart as a strong prior on chord
identity per (section, position).

**Inference:** discretize (time × tempo × metrical × section) → HMM/DBN → **Viterbi** for
the MAP alignment + **forward-backward** for the per-beat/region **posterior confidence**
(= self-detection, for free). Particle filter if the discretized state is too large.
Lineage: Whiteley/Krebs/Böck bar-pointer beat/downbeat trackers, plus the chart prior.

## Bayesian first, ML later (deliberate)
No labeled training data yet — we're literally building the benchmark. Interpretable
Bayesian components encode the structure without training and give **calibrated
confidence**. When the frozen benchmark supplies data, an ML tracker can follow, with these
fused signals as its **features**.

## The deep reason this generalizes to INFERENCE (Louis's meta-point)
- **Alignment** = this model with chord identity **OBSERVED** (from the chart) → solve
  timing/beats/sections.
- **Inference** (the product) = the **same** model with chord identity **LATENT** → it
  recognizes chords. So the alignment instruments ARE the downstream inference pipeline.
Order (firm): rock-solid alignment first, then extract the inference variant.

## Staged, validated build (each brick falsifiable)
- **Stage 0 — DRUM PREMISE CHECK (cheapest falsifier, FIRST; rule #2):** on Autumn (solo
  failure) + Let It Be (dense mix), learn a drum/downbeat template from the confident
  opening bars and test whether it TRACKS the beat+downbeat through the regions where
  harmonic agreement collapses. Falsifies the whole premise before we build. Guard against
  calibration bugs (verify the template's period matches the known tempo).
- **Stage 1 — drum instrument:** standalone beat/downbeat likelihood + tracker (new module,
  not brick0_propose.py).
- **Stage 2 — fusion DBN:** streams 1–4 + tempo/section priors, reliability-weighted;
  Viterbi MAP + forward-backward confidence.
- **Stage 3 — validate:** must reproduce the 5 frozen + CTY; must FIX Autumn's solos, Let
  It Be (kill the phantom 40–44s gap, stop getting lost), and place Georgia's body.
- **Stage 4 — inference variant:** chords latent.

## Validation
Non-circular agreement + coverage + the frozen GT (5 frozen + CTY) as alignment ground
truth; report whether posterior confidence predicts where the alignment is wrong
(calibration) — the real test of self-detection.
