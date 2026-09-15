# Bar-grid recovery from chord onsets + SSM (period & phase) — 2026-07-29

## HOW IT WORKS (plain language — the deliverable "tell me how it does it")

The beat tracker sometimes mislabels the bars, but it never moves the CHORD onsets —
those stay where the music put them (verified: This Love's onsets have 9 ms jitter).
So we throw away the tracker's bar labels and rebuild the grid from the onset times.
Five steps, no training, one picture: `scratchpad/bar_grid_recovered.png`.

1. **Find the smallest steady beat the chords tap out.** Chords change on a regular
   clock. We sweep candidate spacings and ask "which spacing do the onsets line up
   on best?" (a comb / Fourier test, weighted by how long each chord lasts). On This
   Love that smallest step is 1.26 s — the HALF-bar, because there are little passing
   chords between the main ones.

2. **Find how long the loop is.** We slide the whole chord track against a delayed
   copy of itself and find the delay where the chords come back the same. That is the
   loop. On This Love the pattern comes back every 10.1 s. (We do this on a time grid,
   not on the tracker's bad bars.)

3. **Turn the half-bar step into the real bar — using the repetition.** This is the
   crux. Fold every loop on top of each other. A real DOWNBEAT has a chord starting
   there in almost every loop; a passing chord starts there in only some loops. Keep
   the positions that fire in nearly every loop — those are the bar downbeats — and
   see how far apart they are. On This Love they sit every OTHER half-bar step, so the
   bar is 2 × 1.26 = 2.52 s. A chord that is simply HELD leaves a gap (no new onset);
   we ignore gaps, so a held chord never fools us into a finer grid.

4. **Anchor bar 1.** We drop the junk pickup (the model already flags it) and put the
   grid's first downbeat on the first real chord — G7 at 1.08 s. We nudge the bar
   lines a hair earlier so a chord that the model heard a few ms late still counts as
   its own downbeat, not the previous bar's.

5. **Read one chord per bar and re-check the loop.** Snap the chords onto the rigid
   grid (through the real `apply_rigid_grid`), take the chord on each downbeat, and
   confirm the per-bar sequence repeats. This Love → **G7 | Cm | Fm7 | Ddim**, forever.

**Where it is honest about failing:** if a song holds one chord across several bars
(Every Breath) or plays two chords per bar (Let It Be), the onsets alone can't tell
the true bar from double/half of it — you'd need the drums or the tempo to break the
tie. It nails the case Louis cares about (one chord per bar with passing chords, like
This Love) and ~11/18 pop songs; it defers on long jazz forms (too few loop repeats)
to the plain onset grid, which is right there anyway.

## RESULT SUMMARY (numbers first)
- **This Love: DONE.** bar 2.524 s (target 2.52), G7 at bar 0 (anchor 1.111 s), loop
  4 bars, per-bar **G7 Cm Fm7 Ddim** — verified end-to-end through
  `harmonia.models.rigid_grid.apply_rigid_grid`.
- **Corpus (18 songs): 11/18 recover the reference bar octave.** Failure mode is the
  known metrical-level (bar vs half/double) ambiguity, fully characterised below.
- Artifact: `scratchpad/bar_grid_recovered.png`. Finder: `scratchpad/bar_grid_v2.py`.
- Next step if resumed: pin the octave with a cheap tempo anchor (beatthis tempo, or
  a drum-onset comb) rather than chords alone — that removes the doubling/halving on
  held-chord and 2-chord-per-bar songs. Then wire `rigid_grid_for` behind
  `HARMONIA_REGRID=1` and corpus-check section boundaries downstream (rule #5).

---
## (original brief + running log below)

# Bar-grid recovery from chord onsets + SSM (period & phase) — 2026-07-29 (log)

Goal (brief): a small, explainable algorithm that finds the CORRECT rigid bar
grid per section by detecting the repetition PERIOD (bar length) and PHASE
(downbeat anchor) from chord-onset times (`t0`, seconds) + a self-similarity
matrix. Prototype only (scratchpad + writeup), no production integration.

Motivating case: This Love (Maroon 5). Verse loop `G7 | Cm | Fm7 | Ddim`, one
chord per bar, bar ≈ 2.52 s, anchored on G7 at t=1.08 s. The live pipeline's
beat grid glued G7+Cm into a ~5 s "bar 0" (thrown off by a spurious short `C`
pickup at t=0.44), so later loops appear to start on Cm and the per-bar section
detector inherits the bad grid.

Target: recover bar_len ≈ 2.52 s (±10%), phase → G7 = bar 0, per-bar loop =
`G7 Cm Fm7 Ddim`. Then validate on ≥3 more cached songs (rule #5).

## Restated spec (numbered)
1. Metric: (a) recovered bar_len within ±10% of the modal chord-onset interval;
   (b) This Love phase puts G7 at bar 0; (c) periodic sections don't fragment.
   Eval set = cached `docs/plots/inferred_*.html` payloads (symbolic, decoupled
   from server). No MIREX number here — this is a structure prototype.
2. Budget: focused sprint (research → premise-check → prototype → ~4-song
   validation → writeup). Checkpoints logged below.
3. Integration point: NONE — scratchpad prototype + this writeup only. Reads the
   `const P` payload; reuses period/phase ideas from `section_structure.py` and
   `feat/section-cnn-v2:chord_chain_structure.py` but does not import into prod.
4. Already tried (per brief + history): fixed-lag SSM detector collapses on This
   Love (recurrence 0.32/0.38 < 0.55 gate); `chord_chain_structure.detect_period`
   / `detect_downbeat_phase` exist but operate on the ALREADY-COMPUTED (corrupted)
   beat grid — the novelty here is working in the TIME domain (seconds) so the
   bad grid can't contaminate the estimate.
5. Constraints: symbolic cached payloads only; prototype only; honest failure
   reporting.

## Log

### E1 — Cheap premise check (`scratchpad/bar_grid_premise.py`)
Naive premise "modal chord-onset interval == bar length" is **FALSIFIED** on This
Love. Modal IOI = **1.26 s**, not 2.52 s. Median IOI also 1.26. Reason: This Love
has many mid-bar passing chords (Fdim @ 9.92, Bb^7 @ 20.64, and a denser chorus),
so chord changes land mostly every HALF bar (1.26 s = 2 beats). The bar (2.52 s)
is 2× the modal IOI.

What DID hold: first non-pickup chord = G7 @ 1.08 (drop the leading `nc:True`
pickup at 0.44 → phase anchor is correct). So phase-from-first-real-onset is fine;
period-from-modal-IOI is not.

Diagnosis / updated hypothesis: the onset grid has TWO levels — a fine grid unit
(1.26 s, the half-bar the passing chords use) and the BAR (2.52 s). Modal IOI finds
the fine unit. To get the BAR we must weight onsets by SALIENCE: the bar-defining
chords (G7/Cm/Fm7/Ddim) each last a full bar (~2.52 s duration), while passing
chords are short (0.64–1.26 s). So a **duration-weighted onset comb** should prefer
the 2.52 s grid. Next: test a duration-weighted comb (H2) + a grid-free time-domain
content-SSM autocorrelation for the LOOP period (H3).

### E2 — Estimator comparison (`scratchpad/bar_grid_explore.py`)
- Plain Fourier onset comb fundamental: **1.26 s** (mag 0.78). 2.52 peak weak (0.29).
- Duration-weighted comb: still fundamental 1.26 s, but the 2.52 peak strengthens
  0.29 → **0.42**. Duration-weighting helps but does NOT flip the fundamental.
- Time-domain content-SSM autocorrelation (grid-free, 0.05 s frames): loop peak at
  **10.10 s** in the 4–14 s window. loop/2.52 = 4.0 bars ✓ (loop/1.26 = 8).

So the comb alone can't separate bar (2.52) from half-bar (1.26) — both are strong
(classic tactus/meter ambiguity). Need an explicit **metrical-lift** step.

### E3 — Metrical lift (`scratchpad/bar_grid_metrical.py`) — WORKS
Fine grid → LS-refined g=**1.262 s**, phi (resid |mean|=0.044 s, max 0.611 s). Then:
assign each onset to a fine slot k=round((t−phi)/g); for bar-multiple m∈{1,2,3,4}
and downbeat offset p∈[0,m), score duration-mass on the class {k mod m == p}.

| m | bar=m·g | downbeat share | contrast (share−1/m) |
|---|---|---|---|
| 1 | 1.262 | 1.000 | +0.000 |
| **2** | **2.524** | **0.707** | **+0.207** |
| 3 | 3.786 | 0.356 | +0.022 |
| 4 | 5.048 | 0.355 | +0.105 |

Decisive: **m=2 → bar 2.524 s** (target 2.52, +0.2%), downbeat offset →
**anchor 1.122 s = G7 at bar 0** (target 1.08 s, Δ0.04 s). The long structural
chords sit on every OTHER fine slot; the passing chords fill the off-beats. That
2:1 concentration IS the bar. Chosen decision rule for the module: normalized
concentration (share−1/m)/(1−1/m), argmax over m≥2 with a floor (m=2 → 0.41 vs
m=4 → 0.14). Next: full prototype (rigid grid + one-chord-per-bar + loop period)
and validate on ≥3 more songs.

### E4 — Full prototype (`scratchpad/bar_grid.py`) — This Love PASSES all 3 criteria
Pipeline: duration-weighted comb → LS-refine → metrical lift → build rigid grid →
one-chord-per-bar → loop autocorr. Result on This Love:
- bar = **2.524 s** (target 2.52, +0.2%) ✓
- anchor (bar 0) = **1.122 s → G7 at bar 0** ✓
- loop = **4 bars** (SSM autocorr score 0.51 @ lag 4, clear winner) ✓
- per-bar loop = **G7 Cm Fm7 Ddim** ✓ ; first 12 bars: G7 Cm Fm7 Ddim | G7 Cm Fm7
  Dø7 | G7 Cm Fm Dø7 (loop repeats; 4th-chord Ddim/Dø7 and Fm/Fm7 are quality
  nuances, not grid errors).

Bug found + fixed en route: whole-bar max-overlap picked the mid-bar passing chord
(Fdim) for bar 3 because it had marginally more overlap (1.26 vs 1.23 s). Fix: label
each bar by the chord occupying its DOWNBEAT window (first ~half + small pre-roll),
so a passing chord can't steal the bar and a slightly-late onset is still credited
to its downbeat. This is the mechanism CLAUDE.md warns about (the model hears
changes late) handled explicitly. Next: validate on ≥3 more songs.

### E5 — Multi-song validation exposes a 2× METRICAL AMBIGUITY (`bar_grid_reference.py`)
Compared recovered bar vs a reference = the pipeline's OWN linear fit of onset time
vs its (bar,beat) label (octave-reliable where the beat tracker didn't fail).

| song | ref bar | my bar | m | ratio | verdict |
|---|---|---|---|---|---|
| This Love | 2.53 | 2.52 | 2 | 1.00 | ✓ |
| Billie Jean | 4.11 | 4.11 | 2 | 1.00 | ✓ |
| Katy Perry Hot n Cold | 1.82 | 1.82 | 1 | 1.00 | ✓ |
| Happy | 1.50 | 1.50 | 1 | 1.00 | ✓ |
| Stand By Me | 2.01 | 4.02 | 2 | 2.00 | doubled (defensible — slow harmonic rhythm) |
| Every Breath | 2.05 | 4.09 | 2 | 2.00 | DOUBLED (wrong — 1 chord/bar) |
| Autumn Leaves (iReal) | 1.71 | 3.44 | 2 | 2.00 | DOUBLED |
| Blue Bossa (iReal) | 1.83 | 3.71 | 2 | 2.03 | DOUBLED |
| Let It Be | 3.41 | 1.75 | 1 | 0.51 | HALVED |

Diagnosis: the duration-concentration heuristic (E3) resolves the bar-vs-half-bar
level INCONSISTENTLY — doubles when alternate chords are held slightly longer,
halves when durations are uniform. This is the classic tactus/metrical-level
ambiguity in MIR: bar length is octave-ambiguous from onsets alone.

### Coordinator steer (received) + reframe
Two traps flagged, both correct: (1) don't set bar = modal gap (confirmed E1);
derive bar = loop_period / downbeats-per-loop. (2) Don't use the per-bar SSM
(built on the broken grid) as the period signal — my E2 content-SSM autocorr is
already TIME-domain (0.05 s frames, grid-free) so it's compliant; but my downstream
`loop_period()` ran on my re-quantized bars — replacing it. Target interface:
`harmonia.models.rigid_grid.apply_rigid_grid(chords, bar_bounds_sec)` (a concurrent
session built the receiving end, commit 1728c9f) — my finder must emit
`bar_bounds_sec` (sorted bar edges in seconds).

Key insight to break the 2× ambiguity (matches Louis's "look at when the pattern
REPEATS"): **cross-loop recurrence**. A true bar-downbeat carries a chord onset in
(almost) EVERY loop; a mid-bar passing position fires in only SOME loops. So:
bar = the FINEST grid whose every downbeat slot fires in ≥ most loops. Half-bar
slots (passing) fail this test → not promoted to bars. This should fix both the
doubling (Every Breath: 4 slots all fire → n=4, no doubling) and halving (Let It
Be: passing-free, 4 slots fire → n=4, no halving). Next: E6 implement + retest.

### E6 — v2 loop-fold prototype (`scratchpad/bar_grid_v2.py`) + iteration
Built the loop-period + cross-loop design. Several sub-iterations, each a diagnosed
fix (logged in the code docstrings):
1. Cross-loop "every downbeat slot fires ≥ min_fire" → over-lifted held-chord songs
   (Stand By Me m=3) because a HELD chord leaves an empty slot, mistaken for
   "not a downbeat".
2. "Recurs at loop lag → structural" (binary) → FAILED on This Love: in a regular
   loop the PASSING chords also recur one loop later, so everything looked
   structural (modal gap = half-bar). The discriminator is recurrence FREQUENCY,
   not existence (structural ~90% of loops, passing ~40%).
3. Non-integer P/g (Let It Be 13.45/1.75=7.68) smeared the slot-fold → a single
   surviving slot → runaway m. Fixed by snapping the fold to n_fine·g + an m-cap.
Final v2 lift: fold onsets into the loop, per-slot cross-loop recurrence frequency,
largest-gap split structural vs passing, bar = modal circular spacing of structural
slots; empty (held) slots dropped; few-loops guard → fine grid.

### E7 — This Love verified END-TO-END via `apply_rigid_grid` — TARGET MET
`bars_from_apply` emits `bar_bounds_sec`, feeds the production
`harmonia.models.rigid_grid.apply_rigid_grid`, reads back per-bar. This Love:
bar 2.524 s, anchor 1.111 s (G7 at bar 0), loop = 4 bars, per-bar = **G7 Cm Fm7
Ddim** looping (first 12: G7 Cm Fm7 Ddim | G7 Cm Fm7 Dø7 | G7 Cm Fm Dø7). Fixed an
edge-binning bug: decoded downbeat onsets land ~40 ms before the ideal edge and got
binned to the previous bar (bars 4/8 showed the passing chord); fix = anchor on the
actual downbeat onsets + back off edges by 0.15·bar.

### E8 — Broad-deck validation (18 songs, `bar_grid_validate.py`) — corpus evidence
`ratio` = recovered bar / pipeline reference bar (octave-reliable where the tracker
didn't fail). **11/18 match the reference octave (ratio ≈ 1.0)**; the misses are a
KNOWN, inherent metrical-level ambiguity, not random:

| class | songs | bar vs ref | note |
|---|---|---|---|
| octave-correct | This Love, Stand By Me, Billie Jean, Katy Perry, Happy, Adele Hello, Chiquitita, Autumn Leaves, Blue Bossa, She Will Be Loved, iReal-BJ | ratio ≈ 1.0 | This Love perfect; jazz via few-loops→fine |
| doubled (2×) | Every Breath, Carpenters Close To You, Beat It | ratio ≈ 2.0 | HELD chords: dominant chord-change period = 2 musical bars |
| halved (½) | Let It Be, Sam Smith | ratio ≈ 0.5 | 2 chords per musical bar → recovers the one-chord-per-cell grid |
| skew | Jackson 5 ABC (0.80), Chain of Fools (1.54) | — | messy real-audio decode |

Autumn Leaves recovers a clean per-bar ii-V chain (Cm7 F7 Bb^7 Eb^7 Am7b5 D13 Gm6).
The `—` (empty) bars on held-chord songs are legitimate held cells, not errors.

Honest failure mode: bar length is octave-ambiguous from chord onsets ALONE when
harmonic rhythm ≠ 1 chord/bar (held chords hide the beat; 2-chord bars hide the
bar). Chords give the CHORD-CHANGE grid robustly; that equals the musical bar only
when there is ~1 chord per bar (This Love verse, most pop vamps). Breaking the
octave needs an accent/tempo cue (drums, or the beat tracker's tempo octave, which
CLAUDE.md notes is reliable on the beatthis backend — a cheap external anchor).
