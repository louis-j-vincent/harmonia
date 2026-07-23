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
**ALL 4 OVERNIGHT MANDATES DELIVERED ✅ (committed + tested):**
- **#1 DOWNBEAT MODEL** `harmonia/align/{downbeat,bass_salience}.py` (870d247) — global-phase resolver.
  "Marche nickel" where it should: Stand By Me / Bein' Green / Blue Bossa backing = prec/rec **1.00**,
  confident. Correctly FLAGS the ambiguous (Autumn swing, Blue Bossa jam, Let It Be drift) instead of
  guessing. Bass stream self-downweights on walking bass (Autumn w_bass 0.003). 42 tests.
- **#3 HIGH-PRECISION DATASET** `harmonia/dataset/` (b2c690c + faec07f) — 3-way gate (clean GT /
  substitution-review / drop), **~95% strict precision** on the frozen set, **449 clean (segment→chord)
  pairs / 18.3 min** + **33 substitution candidates** (your alteration corpus). Downbeat confidence folded
  in (boost confident, ABSTAIN on flagged — precision held). Manifests in gitignored `data/chord_dataset/`.
- **#4 ADD-SONG PIPELINE** — `harmonia.dataset.ingest.add_song(chart_ref, youtube_ref)` via yt-dlp + irealb.
- **#2 ALIGNMENT** (d9ce696) — fixed the 2 precision-cappers. **KEY FINDING:** the section-**skip** branch
  (added in Georgia v6c) had **silently regressed 3 songs** (error-pattern #6). Skip-OFF-by-default fixed
  them AND bonus-fixed **Autumn 0.44→1.00** and **Close To You 0.84→1.00**; every_breath outro overshoot
  +18s→+1s; bein_green 0.80→**1.00**. Dataset lift: bein_green 0.82→**1.00**, every_breath 0.85→**0.98**.
  3 perfect-frozen unchanged; all 5 frozen goldens byte-identical.

**⚠️ NEEDS YOUR EAR (nothing frozen without you — still 5/8 frozen):**
1. **Georgia** — its out-head **B-A rotation is now in question**: it was produced by the harmful skip
   branch. Skip-off reverts it to contiguous A-A (raw agr 1.00→0.79). **Ear-check whether the out-head is
   real** — if yes I re-enable `allow_skip` for Georgia (it stays wired). (F#dim→B7 + A/C# + 166s truncation
   also still await you.)
2. **CTY mirror** — `close_to_you_mirrorfix.html` vs `close_to_you.html`: the mirror matches your
   description AND is chroma-positive → likely the accept, then I freeze CTY (→ 6/8).
3. **every_breath downbeat** — model + all evidence say phase 2, golden says phase 1 (possible
   anticipation/half-bar). Worth a listen.
4. **Autumn** — UNFROZEN; skip-off now aligns it raw 1.00, but its solos still want the fusion model.

**⚠️ DISK: system volume at 98%, ~4.1 Go free.** The drop is system/other-session activity, NOT my caches
(`data/cache` is yesterday's, unchanged). The machine is nearly full — flagging it. Regenerable reclaim if
needed: `data/cache` (1.2 Go). I did NOT delete shared/other-lane/system data.

**Serving/refactor lane untouched** — clean modules (`harmonia/align/`, `harmonia/dataset/`) with documented
integration points; wiring into the refactor is a reconcile step for when the lanes converge (I did not edit
`serving/*` to avoid clobbering the concurrent session). Rules held: no GT frozen without your ear; premise-check
before every big build (killed 2 dead ends cheaply); every number from a real run.

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
- 2026-07-24b — PRODUCTIONIZED THE FUSION DBN AS A REFACTOR-COMPATIBLE ALIGNER BRICK
  (`harmonia/align/chart_aligner.py` + `tests/test_chart_aligner.py` NEW; `harmonia/dataset/harvest.py`
  wired). Louis's "hybrid": OUR brick, the refactor's conventions (ABC + factory + dataclass-at-every-
  seam), a pure adapter in between — inserted WITHOUT touching any refactor file (serving/stages/core
  byte-unchanged; `fusion.py` + `brick0_propose.py` + every golden byte-unchanged, confirmed
  `git diff --stat`).
  **(1) THE SEAM / ADAPTER.** `class ChartAligner(ABC)` with `align(self, audio, chart, *, features=None)
  -> ChartAlignment`; `class FusionChartAligner(ChartAligner)` is a THIN wrapper = `align_fusion(...)`
  then a PURE, audio-free adapter `_to_chart_alignment(FusionAlignment) -> ChartAlignment` (a total
  function of the fusion output — unit-testable with a hand-built alignment, no re-computation).
  `@dataclass SectionMarker{t0,t1,label,mma,section_idx,bar,conf}` carries the UNION of the per-chord
  fields BOTH serving consumers need, so either payload builds from a `ChartAlignment` losslessly:
  `serving/loaders._load_ireal_alignment`'s `{i,bar,beat,section,label,t0,t1,match,conf}` (i/beat/match
  derived at build time) AND the server section-route's `markers=[{t0,t1,mma,sectionIdx}]` +
  `sections=[{...,label,t0,t1,acceptedWithModel}]`. `mma:=label` (fusion carries the shipped-schema label,
  not the raw iReal token; serving uses mma for display only + `_load_ireal_alignment` reads `label` —
  lossless for both). `@dataclass ChartAlignment{sections:[{label,t0,t1,markers:[SectionMarker],
  confidence}], beats, downbeat_phase, tempo_bpm, whole_song_confidence, low_confidence_regions,
  downbeat_confidence, downbeat_flagged}` — the last two expose the fusion downbeat posterior so a
  consumer never reaches back into `FusionAlignment` (the adapter is the single seam source).
  Chords partition over sections by MAX-OVERLAP (`_assign_section`), not midpoint containment — a
  boundary/merged span (build_gt_chords merges identical chords across placement boundaries) is a
  chord<->marker BIJECTION with no double-count and no gap-drop (midpoint containment double-counted 1
  chord on let_it_be; caught + fixed).
  **LOSSLESS VALIDATION — 9 benchmark songs:** adapter reproduces `align_fusion` EXACTLY — section↔
  placement (label/confidence/span), gt_chords↔markers bijection (t0/t1/label; all 9: 41/64/286/156/63/
  62/82/254/125 markers == gt_chords), downbeat phase/conf/flag + whole-song conf + low-conf regions +
  beats + tempo all byte-match. `FusionChartAligner.align` wrapper == direct adapter (deterministic). +12
  unit tests.
  **(2) DATASET HARVEST re-verify — PRECISION HELD, RECALL UP.** Wired `compute_beat_lock(chart=song)` →
  `_resolve_downbeat_via_fusion` → `FusionChartAligner`, so the harvest's downbeat confidence now comes
  from the productionised DBN's FORM-refined, anticipation-aware phase (the bare resolver was passed
  `chart_alignment=None` — never saw the chart). Surgical swap: same `track_from_audio` reliability curve,
  ONLY `db_confidence`/`db_flagged` change; distinct cache tag so the bare-resolver cache is never
  clobbered; `chart=None` fallback keeps existing callers/tests byte-identical (51 gate+fusion tests
  green). RESULT vs bare-resolver baseline: **FROZEN-5 CLEAN strict precision 0.9454 → 0.9459 (HELD, no
  drop)**, partial 0.9492 → 0.9497; **clean pairs 449 → 461 (+12 recall)**. Per-frozen-song precision
  IDENTICAL (blue_bossa/bb_backing/stand_by_me 1.000, bein_green 0.670, every_breath 0.852); songs whose
  downbeat was already confident have byte-identical counts — the fusion form-refinement UN-flags
  blue_bossa/georgia/close_to_you/let_it_be (bare resolver flagged them), and the boost-only + flagged→
  abstain fold turns that into +12 correct pairs with no precision cost (blue_bossa +4 clean, P held
  1.000). This is the intended lift (a better downbeat → more recall, precision-first fold protects the
  bar), not a regression.
  **(3) FRONTEND DECOUPLING — DEFERRED (rule #4 remainder).** `align_fusion` depends on 25 distinct brick0
  symbols; only 4 are the named front-end (parse/chroma/grid/transpose) — the other 21 are drift/vamp
  detection + `build_gt_chords` + `_whole_song_agreement`, transitively pulling brick0's Chart/Section/
  BarChord dataclasses, the iReal parser, and its constants. A faithful move = duplicating a large subset
  of the FROZEN 184KB `brick0_propose.py` (byte-compared reference for all 9 goldens; rule #6 risk) —
  AND the concurrency boundary forbids editing brick0, which is where the true fix lives. A thin re-export
  shim would still import from `scripts/` under the hood (cosmetic, not the stated goal). Per the task's
  own guidance ("DEFER if it risks reproduction"), DEFERRED — `fusion.py` untouched, so per-song root
  scores stay byte-identical to `3c13a2a` trivially. TRUE fix: relocate brick0's front-end into a shared
  `harmonia` module both brick0 and fusion import — needs brick0 unfrozen + coordinated with its lane.
  **(4) THE ONE SERVING-SIDE CHANGE STILL NEEDED (coordinate-with-Louis, kill-switched).** The live
  section-align page (`scripts/harmonia_server.py` ~L2925) still calls `align_tune_sections_to_audio`
  (whole-song rigid bar-walk + DTW). Swapping that ONE call for `FusionChartAligner().align(...)` +
  reading `ChartAlignment.sections`/`markers` (already the payload shape the route builds + the UI + the
  `_load_ireal_alignment` sidecar consume) routes the live page through the productionised DBN + its
  self-detection confidence. NOT done here (harmonia_server.py is outside our lane); it must be
  kill-switched (feature-flag both aligners, A/B the markers) and is a coordinate-with-Louis item.
- 2026-07-24 — STAGE 2b LANDED (`harmonia/align/fusion.py` + `tests/test_fusion.py`; 24 pure-core
  tests, +5 new). The DBN now SUBSUMES brick0's drift/vamp. New: `_drift_stage` (bounded lattice-warp
  drifting-τ state, brick0 detection reused, fusion-DP re-place, `_whole_song_agreement` gate),
  `_form_vamp_stage` (brick0 `propagate_form_vamps` re-tiled onto the fusion emission), and the
  `_grid_streams`/`_place_fusion`/`_place_from_starts`/`_attach_harm`/`_tile_order` helpers factoring the
  v1 emission block so it rebuilds on a warped grid. Order = phase-refine → vamp → drift (matches brick0).
  **RESULTS (`score_timeline` root, v1 → 2b, 9 songs on the warm cache):**
  | song | v1 | 2b | what fired |
  |---|---|---|---|
  | stand_by_me (frozen) | 1.000 | **1.000** | drift DETECTED, self-check REJECTS (Δ<0) — held |
  | bein_green (frozen) | 1.000 | **1.000** | flat — nothing fired |
  | blue_bossa_backing (frozen) | 1.000 | **1.000** | flat — nothing fired |
  | close_to_you | 0.962 | 0.962 | flat — unchanged (mirror-fix residual downstream) |
  | georgia | 0.710 | 0.710 | erratic (rubato) — correctly NOT chased |
  | blue_bossa (frozen) | 0.711 | **0.948** | WHOLESONG drift accepted (Δ+0.033); residual = onset-nudge |
  | every_breath (frozen) | 0.817 | **0.893** | WINDOWED drift accepted (Δ+0.056) — hits the logged ≈0.893 |
  | autumn (unfrozen) | 0.350 | **1.000** | 8 form-VAMPS accepted → reproduces its golden |
  | let_it_be (unfrozen) | 0.354 | 0.354 | drift DETECTED, self-check REJECTS (Δ-0.043) — honest |
  **STOP-RULE HELD:** the 3 perfect frozen stay 1.000; nothing regressed (close_to_you/georgia unchanged);
  no metric-up/ear-down. **let_it_be does NOT rise** — brick0's own self-check rejects its drift and its
  golden is verified=False (starts at t0=-0.6s); forcing a warp would overfit a bad golden. The drift/vamp
  is REPRODUCTION of the frozen goldens' documented refinements (blue_bossa/every_breath drift, autumn vamp),
  not a new alignment. **CONFIDENCE still calibrated** (whole-song): clean high — blue_bossa_backing 0.82,
  close_to_you 0.80, bein_green 0.76, every_breath 0.73 (↑ from 0.52: drift made it confirmable), georgia
  0.70, stand_by_me 0.69, blue_bossa jam 0.65; hard lower w/ region flags — let_it_be 0.70/7-low (flags the
  lost middle), autumn 0.48/4-low (swing solos: aligned by form+vamp+drum but honestly "cannot acoustically
  confirm"). Concurrency boundary HELD: only fusion.py + test_fusion.py + this doc touched.
- 2026-07-24 — STAGE 2b DISPATCHED (drifting-τ state + vamp/gap propagation; extend fusion.py only).
  **Design (locked after a premise-check, before editing):** the drift becomes a BOUNDED, regularised
  WARP of the bar-pointer LATTICE (the drifting-τ state), reusing brick0's windowed/whole-song drift
  DETECTION (`offset_ramp` → `detect_drift`/`detect_windowed_drift` → `drift_grid`/`windowed_drift_grid`)
  unchanged; the FUSION DP (`viterbi_bar_pointer`) re-places on each candidate warped grid and the warp
  is accepted only if it RAISES the coverage-weighted harmonic agreement (`b0._whole_song_agreement`
  on the fusion placements) by ≥ `_WIN_ACCEPT_EPS`. The vamp is brick0's `propagate_form_vamps`
  (form-periodic large pause-gaps, its own agreement self-check) with the schedule re-tiled onto the
  FUSION emission. **KEY PREMISE-CHECK FINDING (rule #3 — GT is a measurement):** the frozen-vs-unfrozen
  split matters. `blue_bossa` + `every_breath` are `verified=True` and brick0's OWN calibrated self-check
  ACCEPTS their drift (Δ song_score +0.033 / +0.056) → they reproduce toward golden. `stand_by_me`
  (frozen) + `let_it_be` (verified=False, held-out) both get Δ<0 → brick0 REJECTS the warp; the fusion
  gate reproduces both rejections (stand_by_me Δ=-0.009, let_it_be Δ=-0.043). So `let_it_be` does NOT
  rise from an honest drift — its unfrozen golden even starts at t0=-0.6s; forcing a warp there is exactly
  the v2→v3 overfit the STOP-rule forbids. `autumn_leaves` (verified=False) rises via the vamp 0.34→0.72.
  Gate objective = `_whole_song_agreement` (UNWEIGHTED, brick0's measurement) NOT a salience-weighted
  cov_harm — the latter is biased toward warping (the ramp maximises local agreement) and falsely accepted
  stand_by_me (0.98→0.83 in the probe). The 3 perfect frozen never get a candidate warp/vamp → stay 1.000.
- 2026-07-23 — STAGE 2 FUSION DBN LANDED (`harmonia/align/fusion.py` + `tests/test_fusion.py`,
  first validated version). Bar-pointer state-space that FUSES the 4 streams, reliability-weighted,
  + forward-backward posterior CONFIDENCE. Reuses brick0 (import-only) for the proven non-circular
  front-end (chart parse / chroma / transpose / constant-tempo grid); the fusion owns the EMISSION,
  the bar-pointer Viterbi DP, the confidence, and the form-refined downbeat. 19 pure-core unit tests
  (numpy-only, audio-free). Concurrency boundary HELD: only the 2 new files + this doc touched;
  brick0_propose.py and every golden byte-unchanged.
  **API:** `align_fusion(audio, song_cfg) -> FusionAlignment{placements, downbeat, gt_chords,
  whole_song_confidence, low_confidence_regions}`; pure core `fused_agreement_curve` (salience-
  weighted harmony), `bass_match_curve`, `hr_match_curve`, `fuse_emissions`, `viterbi_bar_pointer`,
  `forward_backward_confidence`, `refine_downbeat_phase`.
  **THE GATE — reproduce the 7 frozen (+ Autumn/Let It Be), placement diff vs brick0:** section
  start-beats **BYTE-IDENTICAL to brick0 on all 9** songs (rule #6 diff). The fusion default
  (salience-weighting + w_bass=w_hr=0.05) was CHOSEN by a sweep: w=0.15 regresses blue_bossa's
  chroma-flat jam (placements 22->23) — so harmony stays dominant, the ear-approved alignment
  reproduces, and bass/hr corroborate. The DP's per-beat gap-cost (0.010) was added to match
  brick0's min-gap DP exactly (without it Autumn's vamp-gaps differed 37 vs 38).
  **TIMELINE vs frozen goldens (score_timeline root):** reproduces to ~1.0 EXACTLY where the golden
  is a pure constant-tempo alignment — stand_by_me **1.000**, bein_green **1.000**, blue_bossa_backing
  **1.000**, close_to_you **0.962** (residual = the downstream mirror-fix override). The 5 lower scores
  are NOT misalignment (placements identical) — they are the goldens' POST-alignment refinements this
  constant-tempo, override-free first version deliberately omits, confirmed from the golden metadata:
  every_breath **0.817** (drift=windowed), blue_bossa **0.711** (drift=wholesong + onset-nudge),
  georgia **0.710** (ear-overrides: B7 relabels+splits+truncation), autumn **0.350** (form_vamp=True),
  let_it_be **0.354** (drift=windowed). PROOF it's drift not misalignment: layering brick0's windowed
  drift back on lifts every_breath **0.817->0.893**. Documented remainder (rule #4): a slowly-drifting
  tempo state + the vamp propagation are the next brick; overrides stay downstream.
  **every_breath DOWNBEAT = phase 1 via FORM (anticipation-aware) — the target FIX:** the acoustic
  global-phase resolver says phase 2 (the pop "push" fooled harmonic-rhythm-on-1, exactly as the
  standalone downbeat model did); the aligned chart form says phase 1; the 1-beat gap is read as
  anticipation and the FORM phase wins -> **phase 1, antic=True, not flagged.** (bein_green self-
  similar section also places correctly: phase 1/1, root **1.000**.)
  **CONFIDENCE CALIBRATION (forward-backward posterior x per-song-normalised fit + absolute-agr
  cross-song term):** the 5 clean/correct songs read HIGH with ~0 flags — blue_bossa_backing **0.80**,
  close_to_you **0.78**, bein_green **0.75**, georgia **0.70**, stand_by_me **0.70** (0-1 low regions);
  the 4 hard/ambiguous read LOWER with many flags — let_it_be **0.67**/8, blue_bossa jam **0.63**/3,
  every_breath **0.52**/3 (the anticipation subtlety), autumn swing **0.50**/11. Separation is
  directional not razor-sharp (blue_bossa jam a touch high); the per-REGION low-confidence flags are
  the sharper signal and land on the genuinely hard spots (autumn solos, let_it_be dense middle).
  **Autumn:** at reproduction-safe weights the fusion == brick0 (holds the grid, coverage 0.973) and
  correctly FLAGS 11 low-confidence solo regions (self-detection = the honest answer for walking-bass
  swing where every stream is weak); heavier fusion weights re-place the solos contiguously but cannot
  be ear-validated -> deferred, not shipped as a win. No metric-up/ear-down claim.
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
- 2026-07-23 — DOWNBEAT→DATASET INTEGRATION LANDED (faec07f, harmonia/dataset/harvest.py): downbeat
  confidence folded DIRECTIONALLY — unflagged/confident → bounded beat_lock BOOST (recall); flagged →
  ABSTAIN (no boost, NO teardown). Agent correctly OVERRODE my "cap flagged low" instruction after
  measuring it deletes flagged-but-perfect Blue Bossa (frozen, P=1.0) and regresses frozen precision
  0.945→0.924 — downbeat PHASE ⊥ chord-LABEL correctness. KEPT (good call). Result: clean pairs 444→449,
  frozen strict precision HELD 0.944→0.945 (no drop); Stand By Me kept 0.575→0.650 (P=1.0), Bein' Green
  0.526→0.568 (P rose 0.644→0.670). Flagged jazz (Autumn/Blue Bossa jam/Georgia/CTY/Let It Be) correctly
  byte-identical. 27 gate tests. Georgia/Let It Be did NOT lift — the model flags them (correct, not a bug).
- 2026-07-23 — MANDATE #2 dispatched: fix the 2 RAW-aligner confidently-wrong regions capping dataset
  precision (every_breath outro over-extension past audio end; bein_green 1 misplaced high-agr section).
  General fixes, not per-song hacks. Frozen goldens untouched (freeze guard).
- 2026-07-23 — MANDATE #2 LANDED (d9ce696, aligner v6d): 2 bugs fixed generally. ROOT CAUSE: the v6c
  section-SKIP branch places a self-similar section OUT OF ORDER where agreement is higher — it mis-placed
  bein_green's B over the A theme AND had silently regressed autumn/close_to_you (error-pattern #6). FIX:
  section-skip OFF by default (contiguous is the safe general default; skip stays wired, re-enable per-song
  once a downbeat corroborates a real rotation) + intro-loopback outro guard (truncate a spurious chorus
  re-opening on an intro section). Results: every_breath overshoot +18.2→+1.2s; bein_green within-span
  0.80→1.00; BONUS autumn 0.44→1.00, close_to_you 0.84→1.00. Dataset: bein_green 0.82→1.00, every_breath
  0.85→0.98. 3 perfect-frozen unchanged; 5 frozen goldens byte-identical. 32 tests. FLAG: georgia 1.00→0.79
  — its UNVERIFIED out-head B-A rotation reverts to contiguous A-A (was a skip artifact?); needs Louis's ear,
  re-enable allow_skip for georgia if the out-head is real.
- 2026-07-23 — ALL 4 OVERNIGHT MANDATES DELIVERED. Winding down heavy work: remaining high-value steps need
  Louis's ear (Georgia rotation, CTY mirror, every_breath phase, Autumn) or more disk (system at 98%, 4.1Gi).
  Not burning compute/disk on unrequested work overnight; resume on his direction. Morning state = this summary.
- 2026-07-23 (morning) — EAR-REVIEW CLEARED → 7/8 FROZEN. Georgia FROZEN ("parfait": out-head B-A +
  B7/split overrides + 166s truncation all ear-approved → re-enable allow_skip for georgia so raw re-align
  reproduces it). Close To You FROZEN with the MIRROR head-fix ("mirrorfix est mieux": first chord
  1.18→0.59s, decaying to 0 at the C# modulation 98.3s — fixes "on commence en retard"). Every Breath
  downbeat = phase 1 KEPT (Louis: "on garde le A"). IMPORTANT: the downbeat MODEL predicted phase 2 = WRONG
  by +1 beat — a real limitation. Hypothesis: chord-change ANTICIPATION (the pop "push") fooled the
  harmonic-rhythm-on-1 evidence; the form/fusion prior should correct it (bar structure fixes phase even
  when the chord change anticipates). No golden change (every_breath stays frozen phase 1). Built a live
  A/B comparator (gitignored ab_*.html: audio-once + A/B toggle + Web-Audio metronome on downbeats) — the
  full-beat Every Breath phase test is where it shines; the CTY sub-beat diff is visible-not-audible.
  ONLY AUTUMN LEFT (unfrozen — its solos need the fusion model).
- 2026-07-23 — STAGE 2 (fusion DBN) GREENLIT by Louis + dispatched. First validated version:
  harmonia/align/fusion.py — bar-pointer state-space over the drum beat grid; observation = reliability-
  weighted fusion of {harmony agreement, drum-beat, bass root+PC, harmonic-rhythm}; downbeat phase from
  downbeat.py REFINED by chart form (anticipation-aware, to fix every_breath); section/gap alignment by
  Viterbi DP following the chart form (large pause-gaps only, constant tempo + slow drift); forward-backward
  → per-region posterior CONFIDENCE (= self-detection + dataset-gate signal). Validate: reproduce the 7
  frozen; FIX Autumn solos (drum+form carry where harmony dies), every_breath phase (form fixes anticipation),
  bein_green self-similar section. Build alongside (do NOT edit brick0_propose.py; import its agreement/chart
  helpers). Reuse caches (disk 4.1Gi). First working+validated version, incremental, stop-and-report if frozen don't reproduce.
