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

**DOWNBEAT = ONE GLOBAL PHASE (Louis, 2026-07-23 — the simplification that makes it easy).**
Meter is constant (4/4), so once the BEAT grid is locked (Stage 1, works well), the downbeat is
NOT a per-bar detection problem — it's a SINGLE discrete global phase offset ∈ {0..barlen-1}
(which beat of the 4-beat cycle is beat 1), constant across the whole song. So: don't classify
each bar (Stage 0b's framing — the hard, wrong one). Instead pick the ONE global phase that
maximizes AGGREGATE downbeat evidence over the WHOLE song — bass-root-on-1 + harmonic-rhythm
(chord changes concentrate on 1) + chart-bar alignment — narrowed to 2 options by Stage 1's
strong-beat PAIR. A per-bar signal that's only ~53% (near-useless per bar) becomes a STRONG
global estimate aggregated over ~100 bars. This is why the downbeat is cheap once the beat is
solid: one argmax over a handful of phase hypotheses, pooling all evidence. **Caveats
(precision-first):** the constant-delta assumption needs (i) a clean beat COUNT (a dropped/added
beat flips the phase after it — Stage 1's octave-lock guards this) and (ii) constant meter (flag
3/4 bridges / metric modulation / added bars). Where the aggregate phase evidence is weak/ambiguous,
FLAG the song rather than guess — same discipline as the dataset gate.

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
**DELIVERED (committed):**
- **Dataset pipeline** `harmonia/dataset/` (b2c690c) — mandates #3+#4 v1 ✅. 3-way gate (clean GT /
  substitution-review / drop) calibrated to **~95.5% precision** on the frozen songs; demo harvested
  **448 clean (segment→chord) pairs / 18.2 min + 33 substitution candidates**; `add_song(chart,youtube)`
  works (yt-dlp+irealb). Manifests gitignored. Precision is capped by 2 ALIGNER confidently-wrong regions
  (every_breath outro over-extension; bein_green one misplaced section), NOT the gate → those are the
  next alignment fixes (mandate #2). Autumn refused entirely (transpose margin 0.027); Georgia A/C# → REVIEW ✅.
- **Drum beat tracker** `harmonia/align/drum_pattern.py` (4d823cf) — beats Stage-0 targets, octave-locked,
  clean DBN API. Stream #2 ✅.
- **Georgia v6c** (6f29229, verified=false) — form A-A-B-A (2:17=B), F#dim→B7 + A/C# split overrides, tail
  truncated 166.2s, body agr 0.424→0.490. Awaits your ear.
- **4 premise-checks done** (Stage 0/0b/0b-bis/bass) → downbeat design fully determined: GLOBAL-PHASE argmax
  (harmonic-rhythm PRIMARY + chart-bar + bass-root-on-1 self-weighted + drum pair), no drum-timbre resolver.

**IN FLIGHT:** Downbeat model `harmonia/align/{bass_salience,downbeat}.py` — priority #1, building now.

**AWAITS YOUR EAR (nothing frozen without you — 5/8 frozen):** Georgia (overrides+truncation); CTY mirror
A/B (`close_to_you_mirrorfix.html` — chroma-positive + matches your description, likely the accept); Autumn
is UNFROZEN (its solos need the downbeat/fusion model).

**NEXT (autonomous):** validate downbeat model → plug into dataset gate (integration point ready → lifts
Georgia/Let It Be) → re-harvest → fix the 2 aligner confidently-wrong regions.

Disk 6.8Gi (watching). Rules held: no GT frozen without your ear; serving/refactor lane untouched (clean
modules + documented integration points); every number from a real run.

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
- 2026-07-23 — Stage 0b downbeat-signature: PREMISE FAILS (chance on targets; magnitude≥timbre on
  frozen). Resolver A dropped. Downbeat = fusion output. BASS instrument promoted. No module/commit.
- 2026-07-23 — Georgia v6c LANDED (6f29229, verified=false, ready for Louis's ear): form A-A-B-A
  (2:17=B, +out-head B-A rotation), F#dim→B7 ×6 + A/C#→Cmaj|A7/C# split ×4 overrides, split
  detector (3/7 confirmed), tail truncated 166.2s. Body agr 0.424→0.490. 28 tests. Frozen untouched.
- 2026-07-23 — BASS premise-check dispatched (scratch-only, parallel with Stage 1): does low-freq
  bass salience (a) recover the sounding-bass/root vs GT, (b) concentrate root-on-downbeat above
  chance, (c) persist through solos? Bass = stream #4 + the GT target + the promoted downbeat resolver.
- 2026-07-23 — Stage 1 drum beat tracker LANDED (4d823cf, harmonia/align/drum_pattern.py): beat-lock
  BEATS Stage-0 targets (Autumn solo 85.9% on-drumhit, Let It Be 99.5%; octave anchor holds on all 7,
  0 slips; clean DBN API: beat_likelihood/reliability/strong-beat-pair/local_tempo). 15 tests. Stream #2 ✅.
- 2026-07-23 — DOWNBEAT reframed by Louis = ONE GLOBAL PHASE over the locked beat grid (constant 4/4),
  not per-bar detection. Pick the single phase offset maximizing AGGREGATE evidence (bass-root-on-1 +
  harmonic-rhythm + chart-bar align), narrowed to 2 by Stage-1's strong-beat pair. Weak per-bar signal
  → strong global estimate over ~100 bars. Build after bass premise lands. Caveat: clean beat-count +
  constant meter; flag ambiguous (precision-first).
- 2026-07-23 — OVERNIGHT MANDATE (Louis): (1) best downbeat model, clean/modular, integrates w/ the
  refactor; (2) continue alignment; (3) high-PRECISION training dataset (audio-segment→GT-chord) via our
  aligner — few FPs (FP=bad label); (4) modular add-song pipeline (irealb+YouTube). Dataset design in
  docs/dataset_harvest_design.md. Dataset-pipeline scaffold dispatched.
- 2026-07-23 — Stage 0b-bis (Louis's MAINTENANCE framing) RESULT: partially VINDICATES Louis. His
  continuity framing ≠ Stage 0b classification and is NOT universally dead — Stand By Me: a per-song
  BASS-band downbeat signature arrests a +3% drift (0.480→0.145 beat, 24 slips→0). BUT on the swing/dense
  TARGETS (Autumn, Let It Be) it fails on maintenance terms too — physical: swing beat-1 is symmetric
  with beat-3 (both ~80% onset), nothing beat-1-specific. The signal lives in the BASS root-on-1, not
  drum snare/hi-hat (Louis's literal band is chance everywhere; Stand By Me bass recurrence AUC 0.949).
  DESIGN: no drum-timbre resolver A. Downbeat term = BASS-ROOT-ON-1 weighted by its own per-song
  recurrence AUC (≈1 non-swing, ≈chance swing → auto-downweighted) → feeds the GLOBAL-PHASE argmax with
  form + harmonic-rhythm. On swing, downbeat leans on form + bass. Plot: downbeat_maintenance_phase_error.png.
- 2026-07-23 — BASS premise-check RESULT (stream #4): bass-PC vs sounding-bass GT = Stand By Me 97.5%,
  Every Breath 82.5%, Bein' Green 74.5% (root-oriented pop/soul, 1.9–2.6× baseline); Let It Be 54.8%,
  Blue Bossa 45.6% (mod/weak); Autumn 25.4% = BELOW chance (walking jazz bass, no signal). Split is
  INSTRUMENT-driven, not solo-vs-head. P2 root-on-1 FAILS as a general downbeat resolver (only Stand By
  Me 0.92). P3 persistence: HOLDS on dense Let It Be (bass 32→62% as harmony 75→66% — complementary),
  FAILS on walking-bass Autumn. Verdict: bass earns (a) a reliability-weighted stream for the sounding-bass
  target (weight = bass-band energy × chroma-argmax CONCENTRATION → Autumn's low concentration
  auto-downweights = self-detection), NOT (b) a downbeat oracle. Recipe for bass_salience.py: C1–C4 CQT →
  flat bass chroma; duration-integrate over the SPAN (never the downbeat instant — attack masks pitch ~1
  beat); reliability = concentration; fifth/harmonic guard (28–48% of misses are 3rd/5th, mostly the 5th).
- 2026-07-23 — ALL 4 premise-checks in. DOWNBEAT = global-phase argmax over the beat grid, aggregating:
  harmonic-rhythm/chord-change-on-1 (PRIMARY, GT-confirmed 0.93–1.0) + chart-bar alignment + bass-root-on-1
  (self-weighted, pop only) + drum strong-beat-pair (narrows to 2). Reliability weighting auto-handles genre.
  Building harmonia/align/{bass_salience.py, downbeat.py} now.
- 2026-07-23 — DATASET PIPELINE LANDED (b2c690c, harmonia/dataset/): 3-way gate calibrated ~95.5% strict
  precision (agr_keep=0.34, plateau) on the 5 frozen; 3/5 songs perfect. Demo: 448 clean pairs / 18.2 min +
  33 substitution candidates. add_song(chart,youtube) via yt-dlp+irealb works. Precision capped by aligner
  confidently-wrong regions (every_breath outro over-extend; bein_green 1 misplaced section) NOT the gate —
  those + the downbeat model are the path higher (mandate #2). Downbeat integration point wired (its per-span
  confidence will multiply into beat_lock → lifts Georgia/Let It Be). 22 tests. Frozen/other-lanes untouched.
- 2026-07-23 — DOWNBEAT MODEL LANDED (870d247, harmonia/align/{downbeat,bass_salience}.py) = priority #1 ✅.
  Global-phase resolver (rebuilds a clean constant-tempo lattice from the drum tracker's octave-locked
  period — the raw tracked grid jitters and slips the phase; peak-picked flux = biggest lever). Per-song:
  phase correct 5/7, correct-OR-flagged 6/7. The 3 clean-tempo pop songs (Stand By Me, Bein' Green, Blue
  Bossa backing) = prec/rec 1.00, confident, UN-flagged = "marche nickel". Ambiguous ones correctly FLAG
  not guess (Autumn swing conf 0.00, Blue Bossa jam conf 0.04, Let It Be tempo-drift). Bass auto-downweights
  on Autumn walking bass (w_bass 0.003 vs 0.04–0.06 pop) — self-detection works. 42 tests. Other lanes untouched.
  FLAG FOR LOUIS'S EAR: every_breath — model+all evidence agree phase 2, golden says phase 1 (possible
  anticipation/half-bar offset); golden wins per trust order, counted a miss — worth an ear check.
- 2026-07-23 — Integration dispatched: wire downbeat per-song/per-span confidence into the dataset gate's
  beat_lock (harvest.py) + re-harvest; report the HONEST lift (expect: more recall on confident pop, correctly
  still-conservative on flagged jazz; precision must NOT drop below ~95.5%).
