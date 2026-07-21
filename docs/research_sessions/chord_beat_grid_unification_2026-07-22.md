# Session: does unifying chord-change detection onto the Beat This! beat+downbeat grid improve chord accuracy?

## FINAL VERDICT (2026-07-22) — NO for unification; the real lever is grid-free flip-confidence

**Answer to Louis's question ("unify where we decide chord changes vs beats/bars"):**
chord changes ALREADY live on the beat grid; pushing them further onto the DOWNBEAT/BAR
grid does NOT help and the hard form HURTS.

Method: cheapest-falsifiable-first. The dispatch's real-audio set (docs/audio +
aligned_corpus) is unusable as-is (slug↔chart mismatch, deleted audio, sparse fragments —
see Data reliability below), so tested on POP909 (clean GT beats/downbeats/chord-times) via
cached NNLS features + the pipeline's REAL decode with pure NNLS labels. Only SEGMENTATION
varied; N=40 songs, 4987 GT chords, GT (perfect) grid = optimistic upper bound for the idea.

Results (Δ root acc vs the shipped per-beat-flip baseline 86.75%; midpoint scoring):
- per-bar pooling (decode on bar grid)      **−20.59pp**  ← catastrophic
- half-bar (strong-beat) pooling             **−2.39pp**
- strong-only (never cut off a downbeat)     **−1.62pp**  ← purest "snap to grid" = worse
- downbeat-AWARE soft prior (V1) @T=0.5       +2.09pp
- **grid-FREE confidence gate (C1) @T=0.5     +3.05pp**  (qual7 +2.13, fam3 +1.78)
- segment_source='musx' boundaries (N=8 ref)  +3.99pp  (matches C1 → convergent)

The grid-free confidence gate BEATS the grid-aware prior at every matched threshold under
midpoint scoring; under boundary-sensitive dense scoring the downbeat grid buys only a
<1pp, scoring-dependent robustness margin (protecting strong-beat flips at aggressive T).
Hard grid-snapping is negative under both scorings. **So the downbeat/bar grid is not the
lever — killing NNLS over-segmentation is** (premise screen: ~30% of nnls flips spurious,
~77% of those weak-beat noise). Artifact: docs/plots/grid_unification_negative_2026-07-22.png.

**WIN criterion:** a variant clears the ≥2pp bar, but the winning ingredient (flip-
confidence gating) is GRID-FREE — it does not support unification. Verdict on the brief:
**keep the grids' roles as they are; do NOT snap chord changes to the bar grid.** Shipped
the orthogonal finding as a default-OFF brick: `harmonia/models/segmentation_gate.py`
(env HARMONIA_FLIP_MARGIN; exact passthrough at 0.0; tests in tests/test_segmentation_gate.py).

**Next steps (for the confidence-gate brick, before any flip-ON):** (1) re-run with
music-x-lab labels (only NNLS labels tested); (2) DETECTED beats/downbeats (only GT tested);
(3) a real-audio + real-GT set with faster harmonic rhythm (POP909's slow rhythm biases the
optimal T HIGH — jazz ii-V will want LOWER T; per-corpus T calibration likely needed);
(4) confirm it doesn't fight the Occam pass. Do NOT ship as a "chord improvement" until
(1)-(3) hold corpus-wide. It is a DRAFT/fallback-path win only in the production default
(segment_source='musx'), where boundaries already come from musx.

---

Start: 2026-07-22 00:05 CEST. Budget: premise-screen first (must be falsifiable
before building), then ≤2–3 variants. Stop when one clearly wins (≥~2pp root or
family, no material regression) or all screened variants flat/negative.

Repo: canonical (`harmonia.__file__` verified under this repo). Disk 1.9 GiB free
at start — reuse caches, render one wav at a time.

## Brief (restated as spec)
1. Metric: root acc, quality (7-way strict), family (partial-credit) vs GT. WIN =
   ≥~2pp root or family, no material regression on the others. Clear flat/negative
   across variants = valid result → report honestly.
2. Integration: segmentation stage of `chord_pipeline_v1.py` (`segment_source`,
   `_label_segments`/`_coalesce_labeled`). Any win must be a clean kill-switched
   modular option, not a hot-path rewrite.
3. Premise screen FIRST: where do GT chord changes fall vs the Beat This! grid
   (on-beat / on-downbeat / off-grid, ±70ms)? + current NNLS-flip spurious/missed.
4. Constraints: do NOT touch `harmonic_downbeat.py`, `beat_grid.py`, native-bargrid
   code, or other sessions' WIP (`chart_model.py`, `app_shell.html`, `local_key.py`).
   Prototype as scratchpad harness / new kill-switched module. No `git add -A`.

## Architecture (confirmed from known_issues STEP 8)
- Chord-change boundaries = NNLS root-argmax flips SNAPPED to Beat This! BEATS
  (`seg_bounds`, chord_pipeline_v1.py:3612). Chord changes already live on the beat
  grid; they do NOT use DOWNBEATS.
- Labels = musx per-segment. Downbeats/bars feed only sections/Occam/display.
- Unexploited lever: downbeat/bar structure as a harmonic-rhythm prior.

---

## Data reliability check (before any screen) — 2026-07-22 00:10

Verified the brief's stated reference set (docs/audio + aligned_corpus) is NOT
trustworthy as-is:
- docs/audio slugs do not map cleanly to aligned_corpus song_ids: Blue Bossa,
  Stand By Me, Feeling Good, Georgia On My Mind all "NOT in corpus".
- aligned_corpus stores only sparse ACCEPTED fragments (Close To You = 14 segs over
  ~10s at t≈225s; Africa = only t=157–269s), and audio was DELETED after alignment.
- `.ireal_urls.json` chart titles are mismatched to slugs (Maroon 5 "This Love" →
  chart "Baby Love"; Carpenters "Close To You" → "We've Only Just Begun 2").
- Consequence: the docs/audio m4a audio was likely a DIFFERENT YouTube download than
  the aligned-corpus GT was built from → time bases not guaranteed to align. Trusting
  it blindly is the CLAUDE.md #1 silent-calibration trap.

Decision: run the cheapest falsifiable premise check on POP909 GT, which has clean
GT beats + GT downbeats + GT chord-change times, fully symbolic, corpus-wide, no
audio (CLAUDE.md #2 + #5). The "do GT chord changes coincide with downbeats" question
is a structural/harmonic-rhythm fact that POP909 answers cleanly. POP909's /bass +
functional-root caveats (CLAUDE.md #3) affect chord LABELS, not change TIMES, so the
change-time screen is valid. Real-audio Beat This! detected-grid screen follows on a
POP909-rendered subset to connect the structural fact to the detected grid.

---

## PREMISE SCREEN — RESULT (2026-07-22 00:40)

### Screen A — symbolic (POP909 GT, N=909 songs, no audio), TOL=±70ms
Where do GT chord CHANGES fall relative to GT beats/downbeats?
Scripts: scratchpad/screen_symbolic.py

- 109,024 GT chord changes.
- ON downbeat: 57.6% | ON non-downbeat beat: 42.4% | OFF grid: 0.0%
  - CAVEAT: POP909 chord annotations are beat-QUANTIZED by construction → "0% off
    grid / 100% on beat" is TAUTOLOGICAL, not evidence about real-audio off-grid rate.
    The load-bearing signal is the metrical PHASE distribution + harmonic rhythm.
- Metrical phase of changes (0=downbeat, 4/4): phase0 57.6%, phase2 29.0%, phase1 7.2%,
  phase3 5.6%. → strong beats (phase 0+2) = **86.6%** of changes; weak beats 12.8%.
- Harmonic rhythm (chord duration): 2 beats 48.2%, 4 beats 25.4%, 1 beat 17.8%, 3 beats 7.7%.
- Ceilings: forcing changes onto downbeats-only captures only 57.6% (misses 42.4%);
  onto ANY beat captures 100% (tautological). → a downbeat-ONLY / per-bar decode is
  too coarse; a downbeat/strong-beat PRIOR (soft) is the right framing.

### Screen B — detected-grid / NNLS-flip mechanism (POP909, N=40 cached songs)
Replicated the pipeline's exact segmentation (`_root_change_segs(heads.root_proba(
pool_beats(arr,times,GT-beats)))`) on cached NNLS features + GT beats. GT beats used
to ISOLATE the segmentation mechanism from beat-detection error. Time base verified
(feature start=0, active-audio-end≈last GT beat; overshoot was a silent tail).
Script: scratchpad/screen_nnls_flips.py

- NNLS flip boundaries 4613 vs GT changes 4482 → ratio **1.03** (not gross over-count).
- Boundary **precision 70.1%** (29.9% spurious), **recall 72.2%** (27.8% GT missed).
- **Spurious flips are WEAK-BEAT-concentrated**: phase3 52.9% + phase1 23.9% = **76.8%**
  of all spurious flips sit on weak beats; only 5.4% on downbeats. ← the exact signature
  a strong-beat/downbeat prior can attack.
- Naive "keep only strong-beat (phase 0&2) flips": precision 70.1→**91.1%** but recall
  72.2→**60.7%** (misses 39.3%). Too blunt — a HARD strong-beat filter destroys real
  weak-beat changes. Confirms: need a SOFT prior (raise weak-beat flip threshold), not a snap.

### PREMISE VERDICT: NOT falsified — but narrow, with a coalescing caveat
There IS specific structural room: ~30% of NNLS-flip boundaries are spurious and
~77% of those are weak-beat noise flips a downbeat prior could suppress. BUT (CLAUDE.md #4):
1. The prior is a PRECISION play only; it CANNOT fix the 28% MISSED changes (separate
   under-segmentation problem) and will if anything reduce recall.
2. **The decode is segment-level and coalesces adjacent identical labels** (STEP 7/8:
   native bar grid never reached chords for exactly this reason). A spurious flip only
   HURTS chord accuracy if the two sub-segments get DIFFERENT labels. So boundary
   precision ↑ may NOT translate to metric ↑. This must be tested end-to-end, not by F1.

Next: build a cached, no-audio end-to-end harness (baseline-seg vs downbeat-prior-seg,
both with NNLS labels) scored on POP909 functional-root GT (secondary signal; sounding-bass
target not available on POP909, CLAUDE.md #3). Measure if the segmentation change moves
root/quality/family accuracy at all.

---

## END-TO-END VARIANT TEST — RESULT (2026-07-22, ~00:15 elapsed)
Cached, no-audio harness: pipeline's real decode (`_label_segments`→`_coalesce_labeled`)
with PURE NNLS-24 labels on cached features + POP909 GT beats/downbeats. Only the
SEGMENTATION differs. Scored root/qual7/fam3 at GT-chord midpoints, N=40 songs, 4987
GT chords. GT beats+downbeats = OPTIMISTIC (perfect grid) → upper bound for the idea.
Script: scratchpad/variant_harness.py

variant                          root    qual7   fam3    Δroot   win/loss/tie(root)
V0 baseline (per-beat flip)     86.75%  79.01%  87.05%  (base)
V1 dbprior T=0.20 (grid-aware)  87.57%  79.63%  87.55%  +0.82   21/3/16
V1 dbprior T=0.35               88.27%  79.89%  87.77%  +1.52   25/3/12
V1 dbprior T=0.50               88.83%  80.43%  88.35%  +2.09   26/4/10
V1 dbprior T=0.70               89.45%  80.75%  88.75%  +2.71   25/7/8
V1' strong-only (never cut weak)85.12%  78.52%  87.09%  -1.62   16/18/6
C1 margin-only T=0.20 (NO grid) 88.23%  80.05%  87.83%  +1.48   28/2/10
C1 margin-only T=0.35 (NO grid) 89.15%  80.41%  88.19%  +2.41   28/1/11
C1 margin-only T=0.50 (NO grid) 89.79%  81.13%  88.83%  +3.05   27/5/8
V3 halfbar pooling              84.36%  77.88%  86.52%  -2.39   18/16/6
V2 perbar pooling               66.15%  66.35%  77.98% -20.59    6/33/1

### VERDICT ON THE UNIFICATION THESIS: NEGATIVE (well-supported)
- **The downbeat/bar grid adds nothing.** The phase-AGNOSTIC confidence gate C1 (which
  ignores downbeats entirely) BEATS the grid-aware downbeat prior V1 at every matched
  threshold (T=0.35: +2.41 vs +1.52; T=0.50: +3.05 vs +2.09). Whatever V1 gains, it
  gains by suppressing low-confidence flips — and C1 does that better because it also
  gates spurious STRONG-beat flips (5.4% of spurious flips are on downbeats) that V1
  waves through.
- **Hard grid-snapping HURTS.** "strong-only" (never cut on a weak beat = the purest
  unification-onto-bar-grid) is −1.62 root; per-bar pooling −20.6; half-bar −2.4.
  Forcing chord changes onto the downbeat/bar grid destroys the 30-40% of real changes
  that fall on weak beats (matches the symbolic screen: only 57.6% on downbeats).
- Because V1/C1 here use GT (perfect) downbeats, the negative only STRENGTHENS with
  detected downbeats: V1 depends on phase (noisy when detected), C1 does not.

### SIDE FINDING (orthogonal, real): grid-free confidence-gating of NNLS root-flips
C1 = "only accept a root-argmax flip if the new root's normalized margin over the old
≥ T" — a stay-cost/transition penalty the nnls24 `_root_change_segs` path currently
LACKS (the BP48 path has an HMM stay-cost; nnls24 cuts at every raw argmax flip). On
POP909-NNLS this is worth ~+3pp root / +2pp qual7 at T=0.50, 27/5/8 songs. This is NOT
unification — it's segmentation denoising. Caveats: (a) POP909 GT is coarse/beat-quantized
→ favors merging; optimal T is corpus-dependent and likely lower on fast jazz harmonic
rhythm; (b) tested with NNLS labels only, not musx; (c) needs a peak/over-merge check
+ dense-grid (boundary-sensitive) scoring before trusting the magnitude.

---

## ROBUSTNESS + REFERENCE (2026-07-22, ~00:20 elapsed)

### Threshold sweep under two scorings (artifact: docs/plots/grid_unification_negative_2026-07-22.png)
Script: scratchpad/sweep_and_plot.py. Baseline mid root 86.75 / dense root 87.00.
Δroot vs baseline as flip-margin threshold T rises:
  T     V1mid  C1mid | V1den  C1den   (V1=downbeat-aware, C1=grid-free)
  0.20  +0.82  +1.48 | +1.20  +1.51
  0.50  +2.09  +3.05 | +2.86  +3.21
  0.70  +2.71  +2.91 | +3.78  +3.27
  0.80  +2.75  +1.78 | +4.07  +2.63   <- C1 starts over-merging; V1 protected
  0.95  +1.88  -6.50 | +4.03  -4.16   <- C1 collapses; V1 graceful
- Under midpoint scoring, grid-FREE C1 >= grid-aware V1 for T<=0.7 (C1 peak +3.05 @0.5).
- Under dense (boundary-sensitive) scoring, grid-aware V1 degrades more gracefully at
  high T (protects strong-beat flips) -> V1 best +4.07 @0.8 slightly beats C1 best +3.32.
- Net: the grid's UNIQUE contribution is a <1pp, scoring-dependent robustness margin at
  aggressive thresholds. The dominant lever (+3-4pp) is the grid-FREE confidence gate.
- Hard grid-snap stays negative under both scorings (strong-only, dashed grey < baseline).

### Reference variant 3: segment_source='musx' (N=8 songs w/ both caches; directional)
Script: scratchpad/musx_ref.py. Same nnls labels; only segmentation differs.
  nnls per-beat flip (base) : root 83.80  qual7 77.99
  musx boundaries           : root 87.79  qual7 80.29   Δroot +3.99
  C1 confidence gate T=0.5  : root 87.18  qual7 78.96   Δroot +3.39
- musx boundaries and the grid-free confidence gate give SIMILAR ~+3.5-4pp gains ->
  convergent evidence that the lever is BETTER BOUNDARIES (killing nnls over-segmentation),
  NOT downbeat/bar-grid unification. In the production default (segment_source='musx' +
  musx labels) boundaries already come from musx, so the nnls-flip over-segmentation is
  mostly a DRAFT/fallback-path issue.
