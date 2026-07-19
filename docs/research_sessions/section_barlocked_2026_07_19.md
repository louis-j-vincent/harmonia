# Session: bar-locked, repetition-first section labeling (live path) — 2026-07-19

## UPDATE (mid-session directive): probabilistic-root similarity
The similarity input was switched from per-bar chord (root+quality) one-hot to the
per-bar **NNLS root POSTERIOR** with a **cosine** scalar product (user directive:
chord grid only, never acoustic). Wiring now pools `beat_proba`
(`_pool_root_proba_to_bars`) instead of reducing chord segments; the quality head
is dropped (root-only is more robust). Cosine (not the directive's literal "plain
dot") because a raw dot of softmaxes conflates confidence with similarity and
over-fragments on real audio — verified across the 6 gate songs. A is anchored to
the first post-intro section; adjacent same-label sections are coalesced; a single-
label collapse returns [] → defers to acoustic. Primary Mayer gate verified
END-TO-END through the live pipeline: Intro bars 0-3, first A at bar 4, 4-bar-locked;
"A most-frequent" is a 2-2 tie on the noisier NNLS-root path (clean 4×A/3×B on the
deployed musx one-hot). See the ★ STRUCTURE known_issues entry for full detail.

## Summary (top)
Fixed the user-reported section-labeling failure on Mayer Hawthorne "Just Ain't
Gonna Work Out" (E major two-chord vamp): the deployed live path filled sections
with a phase-blind ACOUSTIC detector whose boundaries landed mid-phrase (chips at
bars 11/29/45) and missed the intro. Built an OPT-IN bar-locked, repetition-first
section pass that works from the (good) predicted chords on the 4-bar grid.

- PRIMARY gate (Mayer, deployed musx chords): **PASS** — Intro bars 0-3, first A
  at bar 4 (=bar 5 1-indexed), A most-frequent (4×A/3×B), k=3, all boundaries
  4-bar-locked. Bonus B recovered.
- No-regression (5 songs): k≤5 ✓, phrase-aligned ✓, autumn_leaves A/A grouping
  preserved. aretha/let_it_be (single-chord loops) defer to acoustic fallback.
- Wired OPT-IN behind `HARMONIA_SECTION_MODE=barlocked`; acoustic stays default.
- Artifact: `docs/plots/section_barlocked_compare_2026_07_19.{html,png}`.
- Tests: `tests/test_section_structure.py` 12/12 pass (+4 new).

## Timeline / iteration

### Phase 0-1 (context + brief)
Read CLAUDE.md, known_issues (ACTIVE + all ★ STRUCTURE / GRID PHASE entries),
handoff_2026_07_18, git log. Key facts internalized: fixed-phase-0 grid is the
dominant per-bar V_F loss (oracle +0.078 but no unsupervised global-phase
selector works — CLOSED negatives); complete-linkage grain=8 groups Autumn Leaves
A/A; k≤5 hard rule; thresholds fitted on REAL AUDIO only; stem-keyed-cache trap;
bestfit bar grid is new default. The user's NEW framing (A=predominant, Intro=
prefix-before-first-A) is a different DECISION procedure than the closed global-
phase-selection negatives — not re-treading dead ends.

### Phase 2 (premise check) — CONFIRMED
`scratchpad/chart_extract.py` extracts per-bar chords + chips from cached charts
(the pass consumes only chords, so extracted-chord results == wired-path results).
- (a) OLD chips at bars 11/29/45 are OFF the 4-bar grid → phase-blind. Confirmed.
- (b) Bar-level chord SSM cleanly exposes the vamp: bars 2+ = (0,maj)/(2,min) =
  Emaj7/F#m7 cell (lag-2 0.50, lag-4 0.52, lag-8 0.38); bars 0-1 = distinct
  messy intro content (D#7, F). Premise holds.

### Phase 3 (build + iterate)
- v1 `barlocked_sections`: L via form-length prior, 4-bar-grid L-bar blocks,
  complete-linkage cluster, largest cluster=A, intro=prefix-before-first-A-block.
  FAILED on Mayer: block@0 is half-vamp so it clustered WITH the vamp →
  first_a_bar=0 → no intro, all-A (k=1). Hypothesis: a diluted opening block
  hides the intro inside the A cluster.
- v2 fix: intro detected by PER-BAR novelty (leading run of bars with cosine <
  intro_sim to the A consensus), snapped UP to a 4-bar phrase — independent of
  block clustering. RESULT: Mayer → Intro(4), A@4, B@12, A, A, B, A, B. PASS.
- v3 fix: A-letter was assigned by similarity-to-consensus, which let B dominate
  on ABBA/Autumn (violating principle 1). Changed to A = most-frequent SECTION
  cluster (excluding Intro). Now A is most-frequent on all 6 songs.

### Phase 4 (integration + evidence)
- Wired into `_infer_nnls24`: capture `seg_rq=(t0,t1,root,q5)` per segment;
  `_bar_reduce_segments` → per-bar (root_rel,q5)+times on the beat grid;
  `_barlocked_sections_or_none` gated on env flag + `is_degenerate_sections`
  guard (defers to acoustic on collapse). Unit-tested helpers reproduce the
  direct-function result on Mayer; disabled → None.
- Chrome-harness screenshot (`scripts/phone_screenshot.py`; no playwright — not
  installed, disk at 99%/2.4GiB) of a bar-axis before/after timeline with 4-bar
  gridlines for all 6 songs → `docs/plots/section_barlocked_compare_2026_07_19.png`.
- Live end-to-end run (transcoded to a UNIQUE stem per the stem-keyed-cache trap;
  temp wav + cache cleaned): no crash, key=E major, boundaries 4-bar-locked,
  Intro=4. The raw nnls24-only quality head (lighter than deployed musx) split
  the vamp into 2 clusters → first post-intro block read B; the deployed musx
  chords give A@bar4. Documented as a chord-noise sensitivity of the LETTER
  assignment (not the boundaries/intro).

## What does NOT work / limitations (for the next session)
- A/B letters are only as reliable as the chord input; a noisy quality head can
  split one section into two labels. Boundaries + Intro are robust.
- Single-chord loops (Chain of Fools, Let It Be I-V-vi-IV) have no chord-content
  contrast → collapse to one label → correctly deferred to the acoustic detector.
  A future timbre×chord fusion could separate verse/chorus there.
- Single global L per song; metre/mixed-length forms unmodeled.
- Thresholds (block_sim=0.62, intro_sim=0.55) fitted on 6 real songs, only Mayer
  has GT — do not over-trust; re-validate if more real GT appears.

## Next steps (recommended)
1. Get user sign-off on the Mayer chart, then consider flipping the default to
   `barlocked` for songs where it is non-degenerate (keep acoustic fallback).
2. To stabilize letters under chord noise: cluster on ROOT-only blocks (root head
   is far more reliable than quality — session_2026_07_17 capstone) or fuse the
   acoustic timbre SSM as a tie-breaker for single-chord loops.
