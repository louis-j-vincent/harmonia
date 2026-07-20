# Session: This Love — 2-chords/bar collapse + structure over-collapse (2026-07-20)

Budget 3.5h. Two missions (priority order): (1) FIX 2-chords/bar → 1 collapse;
(2) structure-boundary A/B split within first 32 bars.

## Brief restated (numbered spec)
1. **Target M1**: This Love chorus renders true Cm|Fm / Bb|Eb alternation
   (2 chords per ~2.53s bar) through REAL /api/analyze, 2-run stable. No
   regression on matched set (henny, just-aint, Let It Be, abba, Billie Jean,
   Stand By Me, Bein' Green, aretha, commodores). Anti-crush intact.
2. **Target M2**: This Love shows a real A/B split within first 32 bars (or a
   principled reason not to). No regression on matched-set correct forms.
3. **Integration point**: chord_pipeline_v1 `_infer_nnls24` segmentation +
   render_youtube_chart bake; section detection in chart_model.
4. **Constraints**: side ports 7773+ only, live 7771 untouched; caches
   stem-keyed; no per-song hardcoding; anti-crush ≥99.5% pop400 bars; disk
   floor 2.0 GiB; .venv/bin/python.

## Phase-0 context
- Prod config: `_ANALYZE_SEGMENT_SOURCE="nnls"`, `_ANALYZE_BASS_FRONTEND="musx"`,
  `_ANALYZE_QUALITY_FRONTEND="musx"`. So segments come from NNLS root-change
  (`_root_change_segs`), chord LABELS come from musx via midpoint lookup
  (`root_quality_per_segment`, samples at 0.5*(a+b)).

## Hypothesis M1 (before running)
The 2-chords/bar loss is NOT a musx detection failure (the .lab has Cm→Fm→Bb→Eb
at ~1.25s each, verified 33.85–51.55s). It's that prod segments on NNLS
root-change (`segment_source=nnls`), and musx labels are only sampled at
segment MIDPOINTS. When NNLS root argmax holds one root across a full bar
(under-segments), the bar becomes ONE segment and midpoint lookup picks ONE
of the two musx chords — the second is dropped. `_fit_harmonic_grid` exists
but is NOT consulted on the segment path (only `_root_change_segs` is used).

Cheapest falsifier: dump NNLS root-change segs vs musx change-times in the
chorus window and check whether NNLS under-segments there.

## CRITICAL REDIRECT (coordinator): bar-1/beat-1 anchor bug
User: "le beat 1 commence ... au premier signal audio plutôt que celui d'après —
le bestfit grid marche très bien, la détection d'où commence le premier accord non."
HARD CONSTRAINT: do NOT touch `_bestfit_beat_period` / period. Phase/anchor only.

### Premise checks (cheapest first)
1. **librosa misses beat 1? NO.** onset_detect first onset = 1.045s;
   raw_beat_times[0] = 1.091s. librosa's first beat ≈ true first onset. FALSIFIED.
2. **flux-anchor picks wrong phase? NO.** `_flux_downbeat_phase` = phi=2
   (deterministic, content-derived). Intro downbeats sit at bt[2,6,10,14];
   bar-1 downbeat = bt[2] = 1.11s ≈ true onset. Phase is CORRECT.
3. **leading-outlier trim eats first chord? NO.** chords[0]=G survives.

### ACTUAL root cause (confirmed, snap_diag.py)
- true first onset 1.045s; musx G onset 1.184s → BOTH snap to raw beat 1.091s
  (correct bar-1 downbeat).
- **NNLS root-change segmentation places G's onset at 1.420s** (0.236s late vs
  musx) → the display-snap (`_snap`, render_youtube_chart) rounds 1.420 to the
  NEXT raw beat **1.741s** (|1.42-1.091|=0.329 vs |1.42-1.741|=0.321 — tips over
  the midpoint). Result: bar-1 playhead highlights one beat late = "celui d'après".
- The bug is chord-START DETECTION (`_root_change_segs`), NOT the grid/period.
  musx (trusted, boundary-F1 0.90, already the label source) has the onset right.

### Fix (scoped to phase/anchor of chord starts)
`_refine_segs_to_musx`: snap NNLS root-change segment boundaries to the nearest
musx change-time when within ±1 beat, when quality_frontend="musx". Preserves
NNLS segment COUNT (no over-seg), corrects TIMING to the trusted source.
Applied to the FINAL pass only (after mx_labels load); draft preview untouched.

### Fix REVISED — display onset-hint (implemented)
The uniform grid's nearest beat to G's true onset (1.18) is 1.42 (grid phase put
no beat near 1.11), so a grid-level snap can't reach it. `_refine_segs_to_musx`
tried and could NOT move the opening (reverted). CORRECT fix = display layer:
`_attach_musx_onset_hints(chords_out, mx_labels, period)` attaches `onset_s`/
`offset_s` = music-x-lab change-time nearest each chord's uniform START (±1 beat
tol, else keep uniform). Renderer (`chart_to_interactive_inputs`) snaps `onset_s`
instead of the uniform time. (bar,beat) LAYOUT untouched. Kill-switch
HARMONIA_MUSX_ONSET_HINT=0.

**This Love result (offline bake, prod functions):** bar-0 G display onset
**1.741→1.091s** (= true first onset 1.045 / musx 1.184). Layout-invariance
(hint on vs off): n_bars 80=80, section_per_bar identical, (bar,beat,label)
identical across all 79 chords; ONLY 17/79 display times tighten to musx onsets.
Anti-crush unaffected by construction (change is display-only; no Occam/symbolic
path touched).

### GATE RESULTS (2026-07-20)
**Live /api/analyze, side port 7778 (7773 was taken by another session's server;
7771 untouched), 2-run stable:**
- RUN1: This Love bar-0 display onset t0=**1.077s** (was 1.42), t1=3.627, root=G,
  nBars=80, sections=[Intro, A×3].
- RUN2: **identical** (1.077 / 3.627 / G / 80 / [Intro,A×3]). (2 intervening runs
  failed on transient yt-dlp 403 — download flakiness, not code.)
Bar-1 now anchors at the true first audio onset (~1.045s / musx 1.184 → real
beat 1.077) instead of the second beat.

**Matched-set no-regression (offline bake, hint on vs off, 9 songs):**
| song | layout identical | sections identical | bar-1 onset changed | mid-song onsets tightened |
|---|---|---|---|---|
| Let It Be | ✓ | ✓ | no | 57 |
| Billie Jean | ✓ | ✓ | no | 0 |
| aretha | ✓ | ✓ | no | 0 |
| Commodores | ✓ | ✓ | no | 58 |
| Stand By Me | ✓ | ✓ | no | 9 |
| abba | ✓ | ✓ | no | 20 |
| Bein Green | ✓ | ✓ | no | 17 |
| henny | ✓ | ✓ | no | 3 |
| just-aint | ✓ | ✓ | no | 3 |

All 9: (bar,beat,label)+nBars+section_per_bar byte-identical; NO matched-set
bar-1 anchor changed (the fix fired only for This Love's drifted opening).
Anti-crush unaffected by construction (display-only; no Occam/decode path).
