# Chord confusion structure + temporal support — session log (2026-07-26)

**Status: COMPLETE.** Session 22:59–00:05 (~1 h). Baseline re-confirmed bit-exact; every
number below is from a real run on the 7 hand-verified frozen songs. Disk never went below
2.6 GiB. No commits, no edits to any guarded file. Plots:
`confusion_temporal_2026-07-26_A.png` (Part A), `confusion_temporal_2026-07-26_B.png` (Part B),
`seventh_upgrade_changes_2026-07-26.html` (every chord the new brick changes).

---

# PLAIN-LANGUAGE SUMMARY (read this first)

## 1. What do we confuse with what?

**Root (we get 73.7% of the time).** Of the 22.2% of playing time where the root is wrong,
**70% of it is the chord *next door*** — the chord that actually plays just before or just
after this one. That is *not* mostly "the model heard the wrong chord"; it is "the model put
the right chord in the wrong place". A chance control (drawing a wrong root from the song's
own chord distribution) expects only 40%, so the effect is real (lift 1.75×).
The famous "42% of root errors are fifth-related" reproduces exactly (43.6%) — **but 94% of
those fifth errors are the neighbouring chord's root.** Since jazz and pop move by fourths and
fifths, *bleed across a chord boundary is disguised as fifth confusion in an interval
histogram.* We have been reading a timing problem as an acoustics problem.
A further 3.9% of playing time is lost to the chart saying "no chord" where a chord is
playing (that one is already fixed by the dormant `no_chord_policy` brick).

**Quality (the weak metric, 7ths = 0.517).** One failure dominates: **we drop the 7th**.
61% of the quality loss is exactly that — `min7`→`min` (129 s), `7`→`maj` (49 s),
`maj7`→`maj` (43 s). Adding a 7th that isn't in the chart is only 10%. The famous "one note"
confusion (`maj7`↔`7`) is only 10%. So the chart reads like a beginner's lead sheet: right
roots, triads where the tune wants sevenths.

**Bass.** Once the root is right the sounding bass is essentially free (root-right/bass-wrong
is 0.6% of duration). Real inversions are only 2.6% of this benchmark's duration and we catch
42% of them — small numbers, don't over-read.

## 2. What slice of sound decides each chord?

Three different time scales, and they are not the ones you would guess:
- The **chroma frame** is 46 ms; a **beat** is 350–920 ms here (7.5–20 frames), and the
  per-beat feature is a **flat average of the whole beat** — no trim, no weighting, no overlap.
- The unit a chord label is attached to is a **run of beats with the same root-argmax**,
  median **1–4 beats ≈ 0.94 s**.
- **The identity itself (root, quality, bass) comes from one instant**: music-x-lab's label
  read at the **midpoint** of that run. We do not pool anything for the label. (Checked: the
  midpoint label already covers 89% of its span, so a majority vote would change 4 segments
  out of 1984 — the point sample is *not* the problem.)

And the accuracy profile inside a chord is an arch: **79.3% correct in the middle 50% of a
chord, 68.5% at its edges (+10.8pp gap), and the entry (66.1%) is worse than the exit
(70.9%)**. Short chords are where it falls apart: 0.19 accuracy under 1 beat, 0.56 at 1–2
beats, 0.72 at 4–8, 0.93 above 16.

**How much is that worth?** If you keep every label the model produced and only re-cut them
on the true chord boundaries, root goes **0.737 → 0.802 (+6.6pp)**, sevenths +5.6, bass +7.2,
and all 7 songs improve. **83% of that prize is boundaries we already emit but place one beat
off** (snapping them to the truth within ±0.8 s recovers +5.45 of the 6.56pp); only ~1pp needs
new boundaries. The boundaries are *unbiased* (median offset +0.01 s) — they are quantised
wrong, not systematically late.

## 3. What to do about it (ranked, evidence-backed)

1. **Wire the two dormant bricks — measured today, together +2.09pp root / +3.87pp sevenths,
   zero per-song regressions.** `no_chord_policy(intersect)` (+2.10pp root, re-validated
   bit-exact on the refactored pipeline) and the **new** `seventh_upgrade` (+2.59pp sevenths,
   LOSO +3.22pp, 18/19 precision where decidable). Both default-OFF. `no_chord` still needs
   its silence guard for non-benchmark material; `seventh_upgrade` is jazz-repertoire-safe
   only (it must not run on strictly triadic pop).
2. **Boundary placement is the +5.45pp prize, and neither obvious signal reaches it.**
   Picking the right beat among {b−1, b, b+1}: chroma novelty 0.547, **onset strength 0.367,
   HPSS-percussive onset 0.370, onset+chroma 0.482** — all below the shipped segmenter's
   0.554 (chance 0.333). Onset energy is blind to harmonic change (it fires on every beat), so
   "just add a percussive modality" is measured-and-refuted. The two live candidates are a
   **supervised beat-level boundary classifier** (beat features + metrical position +
   harmonic-rhythm prior) and **music-x-lab's frame-level chord posteriors, which we currently
   throw away** — we read only the argmax `.lab`. The latter is the cheapest untapped source
   in the system.
3. **Fix the Occam post-pass's fragility before any front-end work.** A 0.25-beat change to
   the pooling window swings close_to_you by **−11.5pp** through Occam alone (the same change
   is +0.06pp with `HARMONIA_OCCAM_POSTPASS=0`). Every future feature experiment is measured
   through that amplifier.
4. **Decide what a "7th" means in the target** (the `/bass` question again, CLAUDE.md #3).
   On the spans where we drop the 7th, the b7 energy is 0.168 vs 0.087 on true triads and
   0.292 where we get it right: partially sounding, AUC only 0.56. Some GT 7ths are chart
   convention, not sound. Until that is decided, part of the 7ths metric is unwinnable.
5. **Do NOT re-run these** (all measured today, all negative, mechanisms in the log):
   musx's own segmentation −1.97pp · union(NNLS, musx) −1.97pp (identical: `_coalesce_labeled`
   erases any added cut whose sides share a label) · gated under-segmentation repair −0.34pp
   (even with Occam off) · musx onset-hint retiming −0.44pp · within-beat trim/shift/weighting
   ≈ 0 · duration-majority vote instead of midpoint (dead on premise) · raw metrical snap
   −1.49pp (+0.27pp gated).

## 4. The number that reframes all of the above

Scored **raw music-x-lab output straight against the GT** (no Harmonia stage at all):
root **0.7347** vs our 0.7367, sevenths **0.5221** vs our 0.5173, partial **0.6518** vs our
0.6419. Our whole chord stage is worth **+0.2pp of root** over printing musx's `.lab`, and is
*behind* it on sevenths and partial credit. We are a beat-quantising wrapper around musx, and
the quantisation costs about what the post-processing gains — which is exactly why every
label-side lever from four sessions lands within ±2pp, and why the +6.6pp oracle-span prize
(right labels, right spans) is where the room is. Measure future chord work against the
**raw-musx baseline**, not only against our own previous chart.

**NEW FILES** (nothing staged, nothing wired, no guarded file touched):
`harmonia/models/seventh_upgrade.py`, `tests/test_seventh_upgrade.py` (12/12 green),
this log, the two PNGs and the HTML above.

## Brief, restated as a spec (Phase 1)

1. **PART A — confusion structure**, duration-weighted, on the 7 hand-verified frozen
   Brick-0 songs (`golden/brick0/*.gt.json`, `verified=true`), scored with
   `harmonia/eval/accuracy_score.py` (read-only):
   root 12×12 confusion + interval distribution (confirm/refine "42% fifth-related");
   quality confusion (the weak metric: sevenths 0.517); bass/inversion errors
   (root-right/bass-wrong and vice versa); per song and per musical context, with
   N + duration for every cell, small-N flagged.
2. **PART B — temporal support**: what audio slice actually decides each chord.
   (B1) analysis window in ms + musical units; (B2) where inside a span the evidence
   comes from (edge vs middle, accuracy vs segment duration, leakage); (B3)
   premise-screened improvement levers, measured on/off on the 7 songs, as
   NEW default-OFF bricks only.
3. **Baseline (from `docs/chord_improvement_backlog.md`, run 2, 2026-07-23):**
   root 0.737 / majmin 0.710 / sevenths 0.517 / partial 0.642 / strict 0.477 / bass 0.744.
   Must be re-confirmed before anything is changed (CLAUDE.md #1, environment drift:
   the chord stage was re-homed into `stages/chord_head.py` since that number was taken —
   commits `62e3f40`/`2c151eb`).
4. **Success bar (inferred from project convention, no threshold in the brief):** a lever
   is "keep" at ≥ +2pp pooled on its target metric with no per-song regression; anything
   smaller is reported as measured-but-below-bar. Diagnostics (Part A, B1, B2) are
   deliverables in themselves.
5. **Budget (NOT specified in the brief — inferred, flagged as such):** self-imposed
   ~4h wall clock from 22:59, checkpoints every ~30 min logged below with `df -h .`.
   Hard guardrails from the brief: no commits, no edits to the listed files, disk floor
   1.5 GiB.

## Phase 0 — what history already says (do not re-derive)

- The shipped `nnls24` path's **root/quality/bass identity comes from music-x-lab
  (`musx`)**, not from the NNLS heads (`quality_frontend="musx"`); the NNLS-24 chroma
  supplies segmentation, confidence, the min7↔hdim7 fifth-correction, and the bass veto.
- Three prior autonomous runs concluded the bottleneck is the **upstream root SOURCE**,
  not segmentation/timing/post-hoc editing. Dead: flip-margin gate (+0.17pp),
  global time-shift (0.00s), key-prior fifth resolver (−4.3pp), delta-chroma
  (NULL on real audio), coupled root×quality (−11pp), in-house learned heads (−25pp
  domain collapse), BTC front-end (lost to musx).
- Alive: no-chord `intersect` (+2.10pp root, dormant), HYBRID BTC-in-gated-regions
  (+0.048 boundary F1, dormant), `fifth_discriminator` (+0.05pp, correct but
  benchmark-limited).
- Prior pooling-window work (`leakage_robust_identity.html`) tested mean-pool vs
  clip-pool vs delta on a **kNN surrogate task**, not on the shipped pipeline: at
  windows ≥0.4 s the pooling choice was a wash. That result is about the NNLS feature
  pooling — which, per the first bullet, is *not* where the shipped identity comes from.

---

## Log


### 22:59–23:05 — Phase 2: baseline REPRODUCES exactly + fast replay harness (disk 2.6 GiB)

Ran the shipped pipeline (`accuracy_score.SHIPPED_CONFIG`, `infer_chords_v1`) on all 7
verified frozen songs. Every number a real run; all nnls/musx caches hit (no decode spike).

| song | root | majmin | 7ths | partial | strict | bass | dur(s) |
|---|---|---|---|---|---|---|---|
| bein_green | 0.630 | 0.625 | 0.535 | 0.621 | 0.450 | 0.672 | 154 |
| blue_bossa | 0.624 | 0.587 | 0.513 | 0.535 | 0.513 | 0.622 | 493 |
| blue_bossa_backing | 0.879 | 0.860 | 0.498 | 0.733 | 0.498 | 0.879 | 307 |
| close_to_you | 0.864 | 0.806 | 0.497 | 0.695 | 0.235 | 0.879 | 187 |
| every_breath_you_take | 0.747 | 0.747 | 0.519 | 0.747 | 0.519 | 0.764 | 202 |
| georgia_on_my_mind | 0.630 | 0.575 | 0.347 | 0.528 | 0.322 | 0.631 | 151 |
| stand_by_me | 0.852 | 0.852 | 0.731 | 0.731 | 0.731 | 0.852 | 161 |
| **POOLED (1654s)** | **0.7367** | **0.7102** | **0.5173** | **0.6419** | **0.4774** | **0.7440** | |

Identical to the documented 2026-07-23 baseline to 4 decimals — **no drift across the
STEP-A′/B chord-stage refactor** (`62e3f40`) that landed since. CLAUDE.md #1 discharged.

**Replay harness calibration (rule #1):** captured the chord stage's input beat grid
(`bt`, `period`, `tempo`, `duration`, `beat_times_real`) by spying on `_infer_nnls24`, then
re-ran ONLY `NNLS24ChordHead.run_full` from the captured grid. Chart JSON **byte-identical
on all 7/7** (54/255/168/67/70/87/41 chords). So variants can be measured at ~7 s/song with
zero pipeline edits and zero disk growth.
Harness: `scratchpad/harness.py` (session scratchpad, not the repo).

### 23:05–23:15 — PART A results + the finding that reframes the session

**A1 root confusion (duration-weighted, pooled 1654 s, 1421 sub-intervals).**
Root loss decomposes as: correct 1218.5 s (73.67%) · **wrong root 366.3 s (22.15%)** ·
**pred=N over a GT chord 64.4 s (3.90%)** (GT has 4.6 s of N total). Interval mix of the
wrong-root duration: P4 22.9% · P5 20.7% · m3 11.3% · m7 10.9% · TT 8.4% · m6 6.0% ·
M6 5.6% · M2 4.9% · M7 3.3% · m2 3.0% · M3 2.9%.
**Fifth-related (P4+P5) = 43.6%** — confirms the standing "~42%" number.

**A1b — the reframing: 70.5% of wrong-root duration is an ADJACENT GT chord's root**
(258.2 s of 366.3 s), against a chance expectation of 147.4 s from each song's own
duration-weighted root distribution → **lift 1.75**. And the split by interval class is
decisive: **94.3% of the fifth-related error duration (159.7 s) is the neighbour's root**,
vs 52.1% of the non-fifth error duration. Because functional harmony moves by fourths and
fifths, *neighbour bleed looks exactly like fifth confusion in an interval histogram.*
Direction is asymmetric: pred == PREVIOUS GT root 173.2 s vs pred == NEXT GT root 85.0 s
(2:1 — the chart holds a chord too long more often than it jumps early), and it is
boundary-localised: 48.5% of the "prev" mass is within 0.5 s of the span start, 71.7% of the
"next" mass within 0.5 s of the span end.

**A2 quality confusion (root-correct spans only, 1218.5 s; 7ths|root = 0.702).**
The single dominant mode is **7th OMISSION — 61.1% of the quality loss**:
min7→min 129.2 s (35.6%) · 7→maj 49.3 s (13.6%) · maj7→maj 42.7 s (11.8%).
The "one-note" maj7↔7 confusion is only 9.6% (34.7 s), and 7th *addition* is 10.5%
(min→min7 16.3 s, maj→maj7 16.2 s, maj→7 5.5 s). Recall per GT class: maj 0.92 ·
hdim7 0.77 · 7 0.74 · min 0.74 · **min7 0.59** · **maj7 0.52**.
Per song, omission share of quality loss: every_breath 100% · stand_by_me 100% ·
backing 70.5% · close_to_you 62.8% · bein_green 39.6% (small N, 5.8 s) · georgia 33.2% ·
blue_bossa 19.8% (blue_bossa's quality loss is instead maj7↔7, 12.3 s).

**A3 bass / inversions.** root-OK & bass-wrong = **9.6 s (0.6%)**; root-wrong & bass-right
21.6 s. Bass is essentially free once the root is right (confirms "bass gated by root").
GT inversions are **42.7 s = 2.6% of duration** (bein_green 13.5%, georgia 10.0%, close_to_you
3.6%, zero in the other four) — recall 0.42; the model *emits* an inversion on only 8.7 s
(precision 0.72). **SMALL-N: every inversion number here rests on <45 s of audio; do not
over-read.**

**A4 context.** Tempo/harmonic rhythm (GT median chord length): bein_green 74.6 BPM/3.0 beats ·
blue_bossa 171.7/4.0 · backing 150.2/4.0 · close_to_you 88.6/4.0 · every_breath 117.2/8.0 ·
georgia 65.2/2.05 · stand_by_me 119.5/8.0. The two songs with the *fastest* harmonic rhythm in
bars-per-chord terms (georgia 2.05 beats, bein_green 3.0) are the two worst root scores
(0.630 both) and carry all the inversions — jazz-ballad rubato + fast functional harmony.

### THE DECIDING MEASUREMENT — how much of the loss is TEMPORAL, not identity

**ORACLE-SEG** (keep the model's own labels, re-cut them on GT boundaries, each GT span gets
the duration-majority predicted label): pooled root **0.7367 → 0.8023 (+6.56pp)**,
7ths 0.5173 → 0.5731 (+5.58), bass 0.7440 → 0.8157 (+7.17), partial +6.78, strict +4.95 —
**all 7 songs improve** (backing +10.0, every_breath +9.6, georgia +8.3, close_to_you +5.7,
blue_bossa +5.5, stand_by_me +3.5, bein_green +1.6).
So **6.6 of the missing 26.3pp of root is pure span placement of labels the model already
produces**; ~19.7pp is genuine identity error that survives perfect boundaries.

**Where in that 6.6pp:** a jitter-only oracle (move each predicted boundary to the nearest GT
boundary within ±tol, never insert or delete one) gives +0.49pp @0.1 s · +1.07 @0.2 s ·
+1.94 @0.3 s · **+3.99 @0.5 s** · **+5.45 @0.8 s** (saturated). So **83% of the temporal prize
is misplacement of boundaries the model already emits**, only ~1.1pp needs new boundaries.

**Boundary geometry:** predicted-vs-GT change offsets are *unbiased* (median +0.010 s, mean
+0.020 s, 53.2% late) — confirming the earlier "no systematic offset" finding — but in BEAT
units the error is quantised: only 41% of predicted changes are within ±0.25 beat of a GT
change, with a large secondary mass at **±0.5–1.5 beats (~one beat off)**. GT changes
themselves sit a median **0.19 beat** off the pipeline's uniform beat grid (only 39.9% within
0.15 beat) — a structural ceiling for any beat-quantised chart.

### 23:15–00:25 — PART B: the temporal support, measured + 6 levers screened

**B1 — the analysis window, in ms and in musical units (measured, not inferred).**
The shipped chain is: VAMP NNLS-chroma `bothchroma` → per-beat mean → trained root head →
per-beat root-argmax change segmentation → **music-x-lab label sampled at the MIDPOINT of
each segment** → coalesce equal neighbours → Occam/2-chord/onset-hint tail.
- NNLS frame step **46.44 ms** (2048/44100), 21.5 Hz, measured on all 7 files.
- Frames per beat: **7.5** (blue_bossa, 172 BPM) … **19.8** (georgia, 65 BPM); beat = 349–920 ms.
- The per-beat feature is a **uniform mean over the whole beat** — no window, no overlap, no
  weighting, no trim (`nnls_features.pool_beats`).
- Segments (the unit a label is attached to): median **1–4 beats** (blue_bossa 1.0, backing 1.0,
  stand_by_me 1.5, bein_green/close_to_you/georgia 2.0, every_breath 4.0); 1984 segments /
  1873 s over the set ≈ **0.94 s per labelled unit**.
- **The identity decision itself has ~zero temporal extent in our code**: root, quality and bass
  come from music-x-lab's `.lab` sampled at ONE instant (the segment midpoint,
  `musx_bass.{root_quality,bass_pc,no_chord}_per_segment`). The audio slice behind that instant
  is music-x-lab's own receptive field, which we do not control. What OUR pooling window decides
  is (a) where the segment boundaries fall, (b) the confidence, (c) the min7↔hdim7 fifth
  correction, (d) the NNLS bass veto.

**B2 — where inside a span the evidence comes from (duration-weighted, pooled).**
- **Accuracy vs GT span length (beats):** <1 → **0.192** (4.8 s) · 1–2 → 0.555 (65 s) ·
  2–4 → 0.630 (416 s) · 4–8 → 0.722 (591 s) · 8–16 → **0.859** (554 s) · >16 → 0.931 (19 s).
- **Accuracy by decile of the span:** 0.576 / 0.696 / 0.765 / 0.788 / 0.800 / 0.797 / 0.797 /
  0.785 / 0.745 / **0.640** — an inverted-U.
- **MIDDLE-50% 0.793 vs EDGES-50% 0.685 → +10.78pp**, and the LEADING edge (first 25%: 0.661)
  is worse than the trailing edge (last 25%: 0.709). This is the quantitative form of Louis's
  question: *the chart is right in the middle of each chord and wrong at its edges, worst at
  the chord's entry.*
- **Is the midpoint sample representative?** Yes: the midpoint-sampled musx label covers **89.4%**
  of its segment's duration, and only **4 of 1984 segments** would change under a duration-weighted
  majority vote. → **Lever "majority vote instead of midpoint" is dead on premise** (max ≈ 18 s of
  1654 s could move). The point-sample is not the problem; the segment *edges* are.

**B3 — levers, each premise-screened then measured end-to-end (on/off, same base, 7 songs):**

| lever | screen | end-to-end Δroot | verdict |
|---|---|---|---|
| duration-majority vote vs midpoint | 4/1984 segments would change | not run | **DEAD ON PREMISE** |
| `segment_source="musx"` (musx's own change times) | musx boundary-F1 0.90 on RWC | **−1.97pp** | REJECT |
| union(NNLS cuts, musx cuts) | — | **−1.97pp, byte-identical to musxseg** | REJECT (mechanism: `_coalesce_labeled` erases any added cut whose two sides get the same midpoint label — reproduces the 2026-07-24 finding) |
| within-beat pooling: trim-lead / trim-tail / middle-50 / early-shift / ramp weights | beat-root acc 0.6233→**0.6321** and boundary F1@0.25 s 0.377→**0.413** for early-shift [−0.25,0.75] | **−1.44pp (Occam ON) / +0.06pp (Occam OFF)** | REJECT, but see the Occam finding below |
| musx onset-hints as real chord times (the pipeline already computes them, display-only) | — | **−0.44pp** (root-consistent variant −0.45) | REJECT — musx change times only recall 0.14–0.95 of GT changes @0.25 s |
| metrical snap to the bar downbeat (±1 beat) | GT changes are 54% on beat 1 pooled (up to 90%); the chart puts only 41% there and 21% on beat 4 | **−1.49pp** raw; **+0.27pp** with a self-referential gate (snap to the song's OWN dominant change-beat class, abstain if flat) | REJECT (below bar) — it is +2.3 (bein_green) / +1.2 (every_breath) where the metre is right, −7.9 (georgia, rubato) / −8.0 (stand_by_me, anchor phase off by one beat) where it is not |

**Two mechanism findings that came out of the failures**

1. **The Occam post-pass is fragile to small feature perturbations.** The early-shift pooling
   change is worth **+0.06pp with `HARMONIA_OCCAM_POSTPASS=0`** and **−1.44pp with it ON**,
   because on close_to_you it swings the song by **−11.5pp** (0.8546 → 0.7397) — Occam picks a
   different loop family off a slightly different per-bar posterior. Occam is +0.50pp on the
   shipped base but −1.00pp on the perturbed one. Any future front-end change will be measured
   through this amplifier unless it is disabled for the comparison.
2. **A chroma-novelty peak cannot pick the right beat.** Premise screen on the 581 predicted
   boundaries whose GT change is within ±1 beat: novelty-argmax over {b−1, b, b+1} picks the
   right beat **47.7%** of the time vs the shipped pipeline's **55.4%** (chance 33.3%). The
   cosine-novelty of the pooled treble chroma is *worse than what we already do* → the ±1-beat
   refinement needs a different evidence source (onset/percussive or the fusion lane's
   downbeat), not chroma self-similarity.

### 23:33–23:45 — the one lever that clears the bar: `seventh_upgrade` (NEW brick, default OFF)

Part A said the weakest metric's loss is 61% *7th omission*, dominated by `min7 → min`
(129 s). So I attacked that cell with a **span-wide** decision (the Part-B angle: musx decides
quality from one instant; this re-decides one bit from the chord's whole span).

**Premise screens first (CLAUDE.md #2), both informative:**
- On root-correct spans, the normalised b7 energy is 0.168 where the model omitted a 7th vs
  0.087 on true triads and 0.292 where it got the 7th right → the 7th IS partially sounding
  on the omitted spans, but the separation from true triads is weak (**AUC 0.56**). Window
  breakdown: first-25% 0.566 / mid-50% 0.562 / full 0.561 / **last-25% 0.488 (chance)** —
  consistent with B1's finding that a chord's evidence is front-loaded.
- An **absolute** threshold does not transfer between recordings (LOSO −0.36pp); dividing by
  the **song's own median** does (LOSO +3.22pp). Chroma "noisiness" is a property of the mix.
- Applying the same rule to major triads (maj→7 / maj→maj7) measured **−0.50pp** → dropped.
  Musically: the b7 over a minor triad is one unambiguous bin; maj/maj7/dom7 is a 3-way call
  on two weak, mutually-confusable bins.

**Result (`harmonia/models/seventh_upgrade.py`, NEW, default OFF, 12/12 tests):**

| metric | shipped | brick ON (θ=1.5) | Δ |
|---|---|---|---|
| mirex_root | 0.7367 | 0.7367 | +0.00 |
| mirex_majmin | 0.7102 | 0.7102 | +0.00 |
| **mirex_sevenths** | **0.5173** | **0.5432** | **+2.59pp** |
| partial_credit | 0.6419 | 0.6419 | +0.00 |
| strict | 0.4774 | 0.4849 | +0.75pp |
| bass_root | 0.7440 | 0.7440 | +0.00 |

Per song (7ths): backing +3.63 · close_to_you +0.73 · **every_breath +15.10** · georgia −0.13 ·
bein_green / blue_bossa / stand_by_me 0.00. θ sweep 1.0…2.0 gives +3.67…+2.10pp (broad
plateau, no spike); **LOSO on θ = +3.22pp**. OFF is a verified exact no-op end-to-end.
Inspectable artifact — every one of the 32 chords it changes, with its evidence value and the
GT label at that time: `docs/research_sessions/seventh_upgrade_changes_2026-07-26.html`.
**18 agree with GT (min7), 1 disagrees, 13 sit on spans whose root is already wrong**
(so they cannot move the sevenths metric either way) → 18/19 = 95% precision where decidable.

**Stacking with the existing dormant `no_chord_policy` (mode=intersect), re-validated today
on the refactored pipeline** (it still reproduces its documented +2.10pp root exactly):

| config | root | majmin | 7ths | partial | strict | bass |
|---|---|---|---|---|---|---|
| shipped | 0.7367 | 0.7102 | 0.5173 | 0.6419 | 0.4774 | 0.7440 |
| + no_chord intersect | 0.7577 | 0.7282 | 0.5308 | 0.6586 | 0.4909 | 0.7645 |
| **+ no_chord + seventh_upgrade** | **0.7577** | **0.7282** | **0.5560** | **0.6586** | **0.4977** | **0.7645** |
| Δ vs shipped | **+2.09** | +1.80 | **+3.87** | +1.67 | +2.03 | +2.05 |

Zero per-song root regressions. Both bricks are default-OFF and un-wired.

### 23:45 — context split (Part A4)

| context | songs | dur (s) | root | 7ths\|root | fifth share of root errors | 7th-omission share of quality loss |
|---|---|---|---|---|---|---|
| jazz (bein_green, blue_bossa ×2, georgia) | 4 | 1105 | 0.697 | 0.703 | 45.6% | 49.2% |
| pop (close_to_you, every_breath, stand_by_me) | 3 | 549 | 0.817 | 0.702 | 35.3% | 81.0% |
| fast harmonic rhythm (≤2 beats/chord) | 2 | 305 | 0.630 | 0.702 | 41.3% | 34.9% |
| 4 beats/chord | 3 | 987 | 0.749 | 0.675 | 50.3% | 56.5% |
| 8 beats/chord | 2 | 362 | 0.794 | 0.772 | 19.7% | 100.0% |

Root accuracy tracks harmonic rhythm almost monotonically (0.630 → 0.749 → 0.794) and jazz
sits 12pp below pop, but note the confound: the two fast-harmonic-rhythm songs are both jazz
ballads with rubato. **Quality accuracy is flat across jazz/pop (0.703 vs 0.702)** — the
7ths metric is equally hard on both, for different reasons (jazz: real 7ths the model reads as
triads; pop: 7ths in the chart that are barely sounding).

### 23:45 — refined novelty screen (closing the boundary-refinement direction properly)

Re-ran the ±1-beat picker with better features rather than declaring the direction dead:
treble K=1 0.461 · treble K=2 0.477 · bass K=1 0.503 · **bass K=2 0.520** · both K=1 0.525 ·
**both K=2 0.547** · root-posterior K=1 0.518 · K=2 0.515 — vs the **shipped pipeline's own
choice 0.554** (chance 0.333, N=581). Bass novelty is clearly better than treble (the bass
note IS the change cue), and pooling both halves over 2 beats nearly catches the pipeline —
but nothing in this feature family beats it. **Conclusion: the ±1-beat boundary error cannot
be fixed from the NNLS chroma we already feed the segmenter; it needs a different modality
(percussive onset / the fusion lane's downbeat), not a better chroma distance.**

### 23:50–00:05 — closing the segmentation direction properly (last lever + honesty pass)

**Gated under-segmentation repair** (the one shape not yet tried: keep the NNLS cuts, add a
musx cut *only* inside spans longer than K× the song's median span — the musx analogue of the
2026-07-24 HYBRID-BTC gate): K=1.5 **−2.15pp** · K=2 **−2.11** · K=3 **−0.45** · K=4 **−0.27**.
With `HARMONIA_OCCAM_POSTPASS=0` at K=3 it is still **−0.34pp**, so unlike the pooling-window
result this loss is genuine, not an Occam artifact. The per-song pattern is the same in every
segmentation experiment this session: **adding musx boundaries helps bein_green (+0.9…+2.6)
and blue_bossa_backing (+1.1…+3.8) and hurts blue_bossa (−1.7…−4.6) and close_to_you
(−0.3…−13.7)** — i.e. it helps where harmonic rhythm is slow and hurts where it is fast, which
is the opposite of what an under-segmentation repair is supposed to do. Direction closed.

**Reproducibility / honesty notes for this session**
- Baseline reproduced to 4 dp against the 2026-07-23 numbers, and the chord-stage replay is
  byte-identical to the full pipeline on 7/7 songs — so every on/off delta above is measured
  against the same base with only one thing changed.
- All variant machinery lives in the session scratchpad (`harness.py`, `variants.py`,
  `screen_pool.py`, `confusion.py`, `make_plots*.py`) as monkeypatches. **No file under
  `harmonia/models/chord_pipeline_v1.py`, `harmonia/stages/`, `harmonia/eval/`,
  `harmonia/align/`, `harmonia/dataset/`, `serving/` or `golden/` was modified.**
- N=7 songs. Every per-song number carries that. LOSO is reported wherever a hyperparameter
  was fitted; the one place it is not fully honest is `seventh_upgrade`'s *feature* choice
  (relative vs absolute normalisation, minor-only vs both sides), which was selected with all
  7 songs in view — only θ is LOSO-validated. Both of those choices are, however, backed by a
  mechanism (mix-dependent chroma gain; 1-bin vs 3-way decision), not just a number.
- Small-N flags: inversions (42.7 s total), bein_green's quality loss (5.8 s), the `dim`/`sus4`
  rows of the quality matrix (0 s), and the >16-beat duration bucket (19 s).

### 00:05 — ACCOUNTABILITY CHECK (added after the summary was written; it changes the framing)

Scored the **raw music-x-lab `.lab` timeline directly against the frozen GT** — no Harmonia
processing at all, just musx's own spans and labels through `score_timeline`:

| | root | majmin | 7ths | partial | strict | bass |
|---|---|---|---|---|---|---|
| our shipped chart | 0.7367 | 0.7102 | 0.5173 | 0.6419 | **0.4774** | 0.7440 |
| **raw musx `.lab`** | **0.7347** | **0.7121** | **0.5221** | **0.6518** | 0.4767 | **0.7443** |
| Δ (musx − ours) | −0.21pp | +0.19 | **+0.48** | **+0.99** | −0.08 | +0.02 |

Per song (root): backing +5.35 · bein_green +2.72 · close_to_you +1.15 · every_breath −0.95 ·
georgia −0.88 · stand_by_me −2.47 · blue_bossa −3.86.

**The entire chord stage — NNLS chroma, the trained root head, root-change segmentation,
coalescing, the fifth correction, the bass veto, the Occam post-pass, the 2-chord split, the
leading-outlier drop — nets to +0.21pp of root over simply printing music-x-lab's output, and
is behind it on sevenths (−0.48pp) and partial credit (−0.99pp).** What our stage genuinely
adds is everything the `.lab` has not got: a beat/bar grid, a readable chart layout, sections,
key, confidence. On *label accuracy* it is a wash.

That is the cleanest statement of this session's diagnosis: **we are a beat-quantising wrapper
around music-x-lab, and the quantisation costs approximately what the post-processing gains.**
It also explains why every label-side lever tried across four sessions lands within ±2pp —
they are all editing a signal whose ceiling is set elsewhere — and why the +6.6pp oracle-span
prize is where the room actually is: putting musx's labels on the right spans, not choosing
better labels.

**Caveat:** the raw `.lab` is not a shippable chart (arbitrary span edges, no bars, no beats,
no sections), so this is an accuracy accounting, not a product proposal. The actionable read is
the reverse: any future chord work should be measured *against the raw-musx baseline*, not
only against our own previous chart, or it will keep claiming credit for a wash.

**Tail-ablation (completes the accounting):** shipped 0.7367 · Occam OFF 0.7318 (**Occam
+0.50pp**) · 2-chord-split OFF 0.7346 (**+0.22pp**) · both OFF 0.7296 (**tail total +0.71pp**).
So the accounting closes exactly: **core (NNLS segmentation + midpoint sampling of musx)
0.7296 = raw musx 0.7347 − 0.51pp of quantisation loss; the tail post-passes add back +0.71pp;
net +0.21pp over raw musx.** Occam earns its keep on the shipped features (+0.50pp) — it is
its *fragility* under feature change (−11.5pp on one song for a 0.25-beat pooling shift), not
its mean effect, that is the problem.

### 00:10 — the boundary-modality question, MEASURED (revises recommendation 2)

The natural next hypothesis after "chroma novelty can't pick the right beat" was "then use a
percussive/onset modality". **Measured, and it is worse, not better.** Same 581 cases,
same ±1-beat choice:

| picker | picks the right beat |
|---|---|
| onset-strength peak (librosa, full mix) | **0.367** |
| onset-strength peak on the HPSS percussive component | **0.370** |
| chroma novelty (both halves, K=2 beats) | 0.547 |
| z-scored onset + chroma novelty | 0.482 |
| **the shipped pipeline's own choice** | **0.554** |
| chance | 0.333 |

Onset strength is essentially blind to *harmonic* change — it fires on every drum hit, i.e. on
every beat — and mixing it into the chroma score actively degrades it (0.547 → 0.482).
**So the ±1-beat boundary error is NOT recoverable from onset energy either.** The honest
remaining candidates are (a) a *supervised* beat-level boundary classifier (beat features +
metrical position + musx posterior + harmonic-rhythm prior), or (b) music-x-lab's own
**frame-level chord posteriors**, which we currently discard — we read only the argmax `.lab`.
(b) is the cheapest untapped source in the whole system and is the concrete next experiment.

### 00:10 — full accounting of where the accuracy comes from

| stage | pooled root | Δ |
|---|---|---|
| raw music-x-lab `.lab` (unquantised) | 0.7347 | — |
| naive per-beat duration-majority quantisation of musx | 0.7265 | −0.82pp (cost of the beat grid) |
| per-half-beat / per-quarter-beat quantisation | 0.7331 / 0.7337 | −0.16 / −0.10 |
| our core (NNLS root-change segs + musx midpoint sampling), tail OFF | 0.7296 | +0.31 over naive per-beat |
| + Occam post-pass | 0.7346 | +0.50 |
| + musx 2-chord-bar split | **0.7367** | +0.22 |
| **shipped total** | **0.7367** | **+0.21 over raw musx** |

Read this as: **the beat grid costs 0.8pp, and everything our chord stage does buys that back
plus 0.2pp.** It also says the sub-beat quantisers (half/quarter-beat) lose almost nothing —
if a future chart layout can carry sub-beat onsets, the quantisation cost nearly vanishes.
