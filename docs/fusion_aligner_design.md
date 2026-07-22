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
| 2 | **drum BEAT/tempo** percussive/onset envelope vs a beat template learned where DRUMS are steady (octave-anchored) — pins beat phase + tempo + the strong-beat PAIR; **NOT the downbeat** (Stage 0 finding) | everywhere incl. solos | drum energy/steadiness (HIGH in solos) |
| 3 | **harmonic rhythm** chord-change (chroma-flux) landing on strong metrical positions | clear changes | flux salience |
| 4 | **bass** low-freq pitch salience — root lands on the downbeat; also IS the sounding-bass-root GT target | most tunes | bass-band energy |

**Downbeat: drums give the PAIR, resolve beat 1 by SIGNATURE + fusion.** Drums give the
strong-beat PAIR (kick 1&3 / snare 2&4) but MAGNITUDE doesn't say which is beat 1 (Stage 0).
Two complementary resolvers:
- **(A) Per-song downbeat SPECTRAL SIGNATURE (Louis, 2026-07-23) — a new instrument.** The
  downbeat is marked not by loudness but by TIMBRE: a specific drum voice / frequency profile
  recurs on beat 1 (e.g. always a snare, an open hi-hat, a crash, a specific spectral shape).
  LEARN that signature PER SONG once ONE downbeat is anchored (from the head via chart/form),
  then CONVOLVE / matched-filter it along the beat grid to pick out the subsequent downbeats
  (incl. solos, where the groove signature persists even as harmony dies). Tested in Stage 0b
  — **PREMISE FAILS (2026-07-23), see Stage 0b RESULT below.** Resolver A is dropped; the
  downbeat leans on (B) the fusion prior (chart/form + bass root-on-1 + harmonic rhythm).
- **(B) Fusion prior:** CHART/FORM periodicity (survives solos) + BASS (root on 1) + harmonic
  rhythm. (A) is the per-song acoustic evidence, (B) the structural prior; the DBN fuses both.

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
  **RESULT (2026-07-23) = PARTIAL, and it sharpened the design:**
  - **BEAT/tempo: PREMISE HOLDS.** Drums keep the beat through the solo/dense regions —
    Autumn 65%→**78%** of grid beats land on a drum onset (BETTER in the solo than the head),
    Let It Be 75%→**98%**; phase err ≤0.08 beat. Complementarity is REAL: on Let It Be drum
    clarity RISES 0.17→0.34 exactly as harmonic agreement FALLS 0.81→0.57. The two streams
    fail in DISJOINT places — the whole justification for fusion.
  - **DOWNBEAT: PREMISE FAILS for drums alone.** No band shows a real beat-1 accent (Autumn
    b1/mean=0.84 — beat 1 is *softer*; swing accents 2&4); downbeat hit-rate at chance
    everywhere; even Beat This!'s NN downbeat is ~1 beat off in Autumn's solo. It's the
    backbeat's 2-beat symmetry, not a weak template → downbeat must be a fusion output.
  - **Calibration caveat:** drum-only tempo octave-slips on 2/4 measures (subdivision/3-beat)
    → octave-anchor the drum tempo from Beat This!/head, never naive peak-pick.
  - **Scoping catch:** the harmony-confident region ≠ the drum-confident region (Let It Be's
    clean intro is solo piano, drum clarity 0.17) → learn the drum model where DRUMS are
    steady, decoupled from harmonic confidence.
  Plots: `docs/brick0_review/drum_premise_{autumn_leaves,let_it_be}{,_overview}.png`.
- **Stage 0b — DOWNBEAT SPECTRAL-SIGNATURE PREMISE CHECK (Louis's resolver A; cheapest
  falsifier, rule #2). RESULT (2026-07-23) = PREMISE FAILS — resolver A dropped, NOT built.**
  Test: golden downbeats define a 4/4 beat grid (3 beats interpolated per bar → position 0 =
  beat 1, position 2 = beat 3); per beat, the HPSS-percussive onset-peak log-mel spectrum,
  L1-normalised to a loudness-free SHAPE ("which voice"); learn the beat-1 vs beat-3 shape on
  the head, classify 1-vs-3 through the REST by nearest learned shape (balanced acc, chance 50%).
  - **Targets ≈ CHANCE.** Autumn (solo) 51–53% and Let It Be (dense) 45–53% across EVERY
    representation tried (mel-128 / coarse-16 / log-profile / mean-window / spectral centroid) —
    incl. the solo/dense regions; magnitude ≈ chance too. Head in-sample "separation" is pure
    overfit: Autumn head AUC 0.81 but leave-one-out **44.5%** (below chance) on 63 beats; Let
    It Be head AUC 1.00 on only 14 beats. No generalising per-song downbeat voice exists.
  - **Where any 1-vs-3 signal exists (frozen songs), MAGNITUDE ≥ timbre — contradicting the
    premise** ("timbre not loudness"): Blue Bossa magnitude **70%** vs best-timbre 68%; Stand
    By Me magnitude **66%**, timbre mostly 52–54% (one magnitude-correlated mean-window rep 70%).
    The learned beat-1/beat-3 profiles overlap in shape and differ only by a vertical dB shift
    (loudness), not a distinct spectral voice.
  - **Calibration guard passed:** tempos match bar/4 (177/70/171/120 bpm); the grid is coherent
    (magnitude splits 1-vs-3 at 66–70% on the frozen songs — impossible on a wrong grid). Autumn
    beat-1 lands on a drum-onset peak only 38% of the time — consistent with the substantive
    reason it fails: swing often has NO drum voice on beat 1 (Stage 0: beat 1 is *softer*).
  - **Implication for the design:** the downbeat is a **fusion output** (chart/form periodicity
    + bass root-on-1 + harmonic rhythm), not a standalone acoustic drum-timbre instrument. Drums
    still contribute the strong-beat PAIR + tempo/phase (Stage 0 HOLDS), never beat-1 identity.
  Plot: `docs/brick0_review/downbeat_signature_premise.png` (gitignored). No module written.
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

---

## AUTONOMOUS RUN — 2026-07-23, budget 3 days (Louis asleep, full autonomy)

**MORNING SUMMARY (newest at top — read this first on return):**
- [run just started] 3 agents in flight: Stage 0b (downbeat signature), Stage 1 (drum beat
  tracker), Georgia v6c. Benchmark: 5 frozen, Autumn unfrozen (needs fusion), CTY mirror + Georgia
  await your ear. Disk 7.0Gi free (97%) — watching.

**Rules I'm holding (self-imposed, from CLAUDE.md):**
- Every number from a real run. Premise-check before any big build (rule #2). Calibration guard
  first (rule #1). Diff all intermediate outputs after a component swap (rule #6). Log findings
  immediately to this doc; commit at each green gate (specific files, never `git add -A`, no
  `--no-verify`, foreground).
- **I will NOT auto-freeze GT.** `verified=true` encodes YOUR ear approval, which I can't fabricate.
  I build + prepare `verified=false` proposals + review pages; you freeze on return. The 5 frozen
  stay; Autumn stays unfrozen; CTY-mirror + Georgia wait for your A/B.
- **Disk floor**: check `df` each cycle; if free < 3Gi, clean scratchpad `.npz` caches + stop
  emitting 11MB embedded-audio HTMLs until cleared. Currently 7.0Gi.
- Keep the pipeline full (always ≥1 agent in flight so the task-notification chain keeps driving
  the run); disjoint file ownership across concurrent agents.

**Ordered plan (drives the run):**
1. [in flight] Stage 0b downbeat-signature premise check → if holds, build resolver A
   `harmonia/align/downbeat_signature.py`; else downbeat leans on form+bass in the DBN.
2. [in flight] Stage 1 drum beat tracker `harmonia/align/drum_pattern.py` → beat likelihood + API.
3. [in flight] Georgia v6c → `verified=false` proposal + page for your ear.
4. BASS instrument (stream #4 + the sounding-bass GT target): premise-check a bass-salience/pitch
   extractor → likelihood term + downbeat/root evidence.
5. Stage 2 — fusion DBN: bar-pointer state-space fusing harmony + drum-beat + downbeat-sig + bass +
   form/tempo priors, reliability-weighted; Viterbi MAP + forward-backward confidence. Add one
   stream at a time, validate each.
6. Stage 3 — validate: reproduce the 5 frozen + CTY alignments (must match); then FIX Autumn solos
   + Let It Be (kill phantom 40–44s gap, stop getting lost) + place Georgia body. Review pages +
   a confidence-calibration report (does posterior confidence predict where it's wrong?).
7. Stage 4 — inference variant (chords latent) if budget remains.

**Running checkpoint log (newest first):** — updated as stages land —
