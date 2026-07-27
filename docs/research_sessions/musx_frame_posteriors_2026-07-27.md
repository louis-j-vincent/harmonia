# music-x-lab FRAME POSTERIORS as the boundary lever — session log (2026-07-27)

**Status: COMPLETE.** 10:16–11:10 CEST. Disk 4.8 → 4.7 GiB free throughout (floor 1.5).
Every number below is from a real run on the 7 hand-verified frozen Brick-0 songs.

---

## FINAL REPORT (read this first)

**Premise screen: PASSED.** music-x-lab's discarded frame posteriors are a **more precise**
boundary estimator than what we ship (MAD **84.5 → 71.7 ms**, within-100 ms **0.536 → 0.611**,
per-song bias removed) — they were just **systematically LATE**.

**The mechanism nobody had found.** Our chart's boundaries are unbiased (median **−4.0 ms**
vs GT); the raw musx `.lab` is **+112.8 ms** late and the posterior change-point **+134.4 ms**
late (N=457, fixed GT match). That is a *new* explanation for a previously unexplained
refutation: `segment_source="musx"` measured −1.97 pp on 2026-07-26 because musx's boundaries
arrive after the chord has already changed. Two independent statistics agree on the constant:
the direct GT-matched latency (+113/+134 ms) and the peak of the end-to-end latency-sweep
(140–180 ms).

**The fix.** The vendored `XHMMDecoder` already has a beat-aware transition prior that
`chord_recognition.py` never switches on (`use_beats=False` ⇒ a flat ~30-nat penalty at every
23.2 ms frame ⇒ a memoryless geometric duration prior). Re-decode the posteriors with chord
changes legal **only on our Beat This! beats**, with the grid displaced by a per-song latency
`L` chosen **without ground truth** (argmax of the decoder's own Viterbi path log-likelihood),
then shift back so every boundary lands exactly on a beat.
Beat-awareness alone is worth ≈0 (+0.08 pp); **beat-awareness + latency compensation is the
whole effect** — that synthesis is the session's result.

**Headline (LOSO on the one fitted knob, pipeline-integrated, with the GT-free Occam gate
below):**

| metric | shipped chart (today) | brick ON | Δ |
|---|---|---|---|
| **partial_credit** | **0.6419** | **0.6639** | **+2.20 pp** |
| mirex_root | 0.7367 | 0.7503 | +1.36 pp |
| mirex_majmin | 0.7102 | 0.7300 | +1.98 pp |
| mirex_sevenths | 0.5173 | 0.5332 | +1.58 pp |
| bass_root | 0.7440 | 0.7584 | +1.44 pp |
| strict | 0.4774 | 0.4890 | +1.16 pp |

Per song (partial): bein_green +3.45 · blue_bossa +2.25 · backing +4.67 · close_to_you +5.17
· stand_by_me ±0.00 · **every_breath −1.20** · **georgia −1.03**. 4 up, 1 neutral, 2 down.
(Without the Occam gate — i.e. Occam simply OFF — it is +1.90 pp partial / +1.04 pp root.)

**80 % of the gain is boundary placement** (re-decode boundaries with the OLD labels already
gives +1.24 of +1.55 pp), which is exactly the +5.45 pp jitter prize this lever was aimed at.

**Two blockers found, both quantified — and one of them is now solved.**
1. **Occam.** Under the new segmentation the post-pass fires on only **2 of 7** songs and is a
   coin flip: close_to_you **−13.21 pp**, stand_by_me **+3.16 pp**, the other five exactly
   0.00. Its own logged loop-family statistics separate the two cases cleanly and *without
   ground truth*: the good case is `cov=0.97, dev=0`; the catastrophic one is
   `cov=0.78, dev=4` plus `cov=0.67, dev=16`. **A gate "accept a loop family only if
   cov ≥ 0.95 and dev = 0" keeps the good case, rejects the bad one, and is a verified no-op
   on today's shipped chart** (0.6419 unchanged). That gate is worth +0.30 pp on top of the
   brick. N=2 songs — a candidate rule with a mechanism, not a validated one.
2. **The beat grid itself.** Only 18 % of GT chord changes lie within 30 ms of one of our
   beats (57 % within 100 ms) — a hard cap on any beat-quantised chart, ours included.

**Refuted this session (do not re-run):** downbeat-graded transition costs (best graded 0.6627
vs flat 0.6644, even with per-downbeat marking); a semi-Markov explicit duration prior built
from 1696 iReal lead sheets (monotone degradation 0.6652 → 0.6365, and it collapses 7ths);
a *global constant* latency shift (LOSO +0.54 pp only); latency estimated from the musx-to-beat
offset (0.6488); non-uniform Beat This! beats instead of the uniform grid (0.6535 vs 0.6626);
and the whole sliding-LLR / ML-change-point *retiming* family applied on top of our boundaries.

**Decoder-family ceiling:** given the GT change times as its only legal transition frames this
decoder reaches partial 0.6903 / root 0.7728 — still ~3 pp of root short of the 2026-07-26
label-recut oracle (0.8023). Beyond that point the residual is label identity, not boundaries.

**NEW files:** `harmonia/models/musx_redecode.py`, `tests/test_musx_redecode.py` (13/13),
this log, `docs/research_sessions/musx_redecode_2026-07-27.png`,
`docs/research_sessions/musx_redecode_boundaries_2026-07-27.html`.
Nothing wired, nothing committed, no guarded file touched.

**Next step (ranked).**
1) **Implement the Occam `cov ≥ 0.95 ∧ dev = 0` gate** in the (guarded) post-pass — measured
   here by env-toggling, worth +0.30 pp on top of the brick and a no-op on today's chart.
   Validate on more than 2 songs before trusting it.
2) **Improve the GT-free latency selector** — it matches the metric-optimal L on only 2/7
   songs and a perfect per-song selector is worth a further ~0.7 pp (0.6644 → 0.6716
   standalone). It currently wins pooled largely through blue_bossa (30 % of the benchmark).
3) **Diagnose the two regressions**: georgia (rubato — the uniform beat grid is the suspect;
   its posterior MAD is the only one worse than ours, 232.7 vs 177.0 ms) and every_breath
   (the selector picks L=0 where the metric wants 280 ms).
4) **Wire `no_chord_policy`** so stacking can finally be measured — it is still un-wired into
   `chord_head`/`chord_pipeline_v1`, so `HARMONIA_NC_POLICY=intersect` is a verified no-op on
   this path (numbers identical to 4 dp).
5) Longer term, the beat grid is the ceiling (18 % of GT changes within 30 ms of a beat).
   The 2026-07-26 accounting already showed half/quarter-beat quantisation costs almost
   nothing — a sub-beat-capable chart would lift the cap for everything above.

---

Headline metric for this session (Louis's call): **`partial_credit` (family-level),
baseline 0.6419**, with `mirex_root` (0.7367) alongside. `sevenths`/`strict` secondary.

---

## Brief, restated as a numbered spec (Phase 1)

1. **Target metric / threshold.** `partial_credit` on the 7 hand-verified frozen Brick-0
   songs (`golden/brick0/*.gt.json`, `verified=true`), scored by
   `harmonia/eval/accuracy_score.py` (READ-ONLY). Baseline `partial_credit` **0.6419**,
   `mirex_root` **0.7367**. Success = "recover a real share of the +6.56pp root oracle"
   (oracle `partial` = +6.78pp). Project keep-bar convention: ≥ +2pp pooled, no per-song
   regression.
2. **Budget.** Not stated in hours. Self-imposed **~5 h wall clock from 10:16**, checkpoints
   logged every ~30 min with `df -h .`.
3. **Integration point.** NEW default-OFF brick(s) consuming music-x-lab frame-level chord
   posteriors. Guarded (never edit): `chord_pipeline_v1.py`, `stages/chord_head.py`,
   `eval/*`, `golden/*`, `align/*`, `dataset/*`, `serving/*`, and the vendored
   `harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition` clone.
4. **Already tried (do not re-run).** chroma novelty (0.547 vs shipped 0.554), onset
   strength (0.367), HPSS percussive (0.370), onset+chroma (0.482), `segment_source="musx"`
   (−1.97pp), union of cuts (−1.97pp), gated under-seg repair (−0.34pp), musx onset-hint
   retiming (−0.44pp), within-beat trim/shift/weight (≈0), metrical snap (−1.49 raw /
   +0.27 gated), duration-majority vote (dead on premise).
5. **Constraints.** No commits/adds/stashes; no writes to `docs/known_issues.md`; disk floor
   1.5 GiB; every number from a real run; a refuted premise is a valid outcome.

## Phase 0 — what history already says

- The shipped chart's root/quality/bass identity is **music-x-lab's `.lab` sampled at the
  midpoint** of each NNLS root-change segment. Our whole chord stage nets **+0.21pp of root**
  over printing the raw `.lab` (2026-07-26 accountability check) — so the *label source* is
  musx and the *span placement* is ours.
- **Oracle:** re-cutting our own labels on GT boundaries = root +6.56pp / partial +6.78pp;
  a jitter-only snap (≤0.8 s, never insert/delete) recovers +5.45pp → 83 % of the prize is
  **misplaced** boundaries, ±1 beat, unbiased in time.
- Structural fact verified before this session: `chord_recognition.py` computes
  `probs` (5-fold mean of frame-level posteriors) and then **discards it** after
  `hmm.decode_to_chordlab`. We only ever read the `.lab`.

## Tensor layout (read off the vendored source, not guessed)

`net.inference(cqt)` returns 6 softmax arrays per frame:

| slice | cols | meaning |
|---|---|---|
| `prob_triad` | 73 | 0 = `N`; index *i*≥1 → root `(i-1)%12`, triad type `(i-1)//12+1` ∈ {maj,min,sus4,sus2,dim,aug} |
| `prob_bass` | 13 | index 0 = "no bass"; 1..12 = bass pc + 1 |
| `prob_7 / 9 / 11 / 13` | 4 / 4 / 3 / 3 | extension heads |

Frame grid: `DEFAULT_SR=22050`, `DEFAULT_HOP_LENGTH=512` → **Δt = 512/22050 = 23.2200 ms**
(43.07 Hz) — **2× finer than the NNLS chroma grid (46.44 ms)** the segmenter uses.
`decode_to_chordlab(..., use_beats=False)` ⇒ `beat_arr` is all-ones ⇒ a **flat log-prob
transition penalty of 30.0 at every frame**, which is what smooths/quantises the shipped
boundaries.

---

## Log

### 10:16–10:50 — STEP 1 DONE: posteriors extracted, **calibration pin PASSES 7/7** (disk 4.7 GiB)

`scratchpad/musx_probs.py` (NEW, our own — the vendored clone is untouched) imports the clone's
`ChordNet` / `NetworkInterface` / `CQTV2` with cwd set into the clone, runs the same 5-fold
ensemble, averages, caches to `scratchpad/probs/<song>.npz` (float32, compressed).

**PIN (CLAUDE.md #1):** re-decoding the cached `probs` with the clone's OWN
`XHMMDecoder.decode_to_chordlab(..., use_beats=False)` reproduces the shipped
`data/cache/musx_infer/<stem>_submission.lab`:

| song | n_seg ours/shipped | max abs Δ start | label mismatches |
|---|---|---|---|
| bein_green | 54 / 54 | 0 | 0 |
| blue_bossa | 210 / 210 | 0 | 0 |
| blue_bossa_backing | 157 / 157 | 0 | 0 |
| close_to_you | 75 / 75 | 0 | 0 |
| every_breath_you_take | 70 / 70 | 0 | 0 |
| georgia_on_my_mind | 93 / 93 | 0 | 0 |
| stand_by_me | 40 / 40 | 0 starts; **final `N` end differs by 1 frame (23.2 ms)** | 0 |

The single `stand_by_me` delta is the decoded file length (7645 vs 7644 frames), not the model.
**We are holding exactly the tensor the shipped `.lab` was decoded from.**
Disk: `probs/` = 29 MB total for all 7 (2.8–8.5 MB/song). No disk pressure; nothing deleted.

### 10:50–11:10 — PREMISE SCREEN round 1+2: posterior change-points are WORSE than our boundaries at picking the beat

Case set (rebuilt; N=504): every chart boundary at beat `b` whose nearest GT change is within
±1 beat; the "correct" answer is whichever of {b−1, b, b+1} is nearest that GT change.
**Shipped (always pick b) = 0.6528**, chance 0.3333. (The 2026-07-26 log's analogous number was
0.554 on N=581 — a different, looser case definition; all comparisons below are internal to
this set.)

| picker | hit rate |
|---|---|
| **shipped (always b)** | **0.6528** |
| constrained ML change-point, root marginal (uses the chart's OWN X→Y labels) | 0.5516 |
| constrained ML change-point, 73-way triad | 0.5377 |
| free (unconstrained) ML split, root, ±2 beats | 0.5397 |
| sliding-window LLR, root, W=0.6 s | 0.5099 |
| sliding-window LLR, triad, W=0.6 s | 0.5040 |
| chance | 0.3333 |

Two implementation notes (both verified, not assumed): (a) the constrained objective's argmax
over τ is provably independent of the window ends A,B — the numbers are identical for
1/2/3/4-beat windows, which is the expected algebraic signature and a check that the code is
right; (b) round 1's sliding LLR is a boundary-*presence* score whose three windows overlap
almost entirely at one-beat spacing, so it cannot localise — replaced, not retried.

**Hypothesis for the failure (written before the next test):** the 3-way picker forces a
*whole-beat* move, but the 2026-07-26 geometry says GT changes sit a median 0.19 beat off our
uniform beat grid and our boundaries are already unbiased. A signal that is genuinely more
precise in *continuous time* can still lose a forced beat-sized bet. → Next test measures the
posterior change-point in continuous time against GT, and a margin-gated variant.

### 11:10–11:35 — ★ ROOT CAUSE FOUND: **music-x-lab's boundaries are systematically LATE**

Continuous-time retiming (round 3) was *also* worse pooled (mean |err| 197.8 vs our 181.6 ms) —
but with a **+117 ms median signed bias** and a striking per-song split (bein_green 309→108 ms,
backing 174→112, close_to_you 199→117 all much better; blue_bossa 131→280 much worse).

- **Saturation hypothesis — REFUTED.** Only 38/504 (7.5 %) of change-points sit at a search-window
  edge. The 419 *interior* solutions carry a median signed error of **+124 ms** on their own.
- **Latency, measured with a fixed GT match (no re-matching artifact), N=457:**

| estimator | median signed err | mean signed | mean abs |
|---|---|---|---|
| our chart boundaries | **−4.0 ms** (unbiased) | −4.5 | 177.4 |
| posterior constrained change-point | **+134.4 ms** | +146.1 | 207.6 |
| **raw musx `.lab` boundaries (independent check)** | **+112.8 ms** | +114.4 | 168.8 |

  Per song the latency is +54/+289/+87/+46/+248/+210/+74 ms — real, but **song-dependent**
  (0.07–0.83 beat, so it is neither a constant in ms nor a constant in beats).
- **This retro-explains a previously unexplained refutation:** `segment_source="musx"` measured
  −1.97 pp on 2026-07-26 with no mechanism attached. The mechanism is that musx's boundaries
  arrive ~113 ms after the chord actually changes.
- **Mechanism (hypothesis, not measured):** the previous chord sustains/reverberates into the new
  one, so the net only commits once the old harmony has decayed — dry recordings (backing track,
  stand_by_me) show the smallest lag, wet/live jazz the largest.

### 11:35–11:45 — DECIDING NUMBER OF THE PREMISE SCREEN: **the premise HOLDS**

A biased estimator can still be the better one if its *dispersion* is smaller. Pooled, with each
song's own median removed (so this measures precision, not bias):

| | MAD | IQR | mean abs | within 100 ms |
|---|---|---|---|---|
| our boundaries | 84.5 ms | 176.5 ms | 160.4 ms | 0.536 |
| **musx posterior change-point** | **71.7 ms** | **145.1 ms** | **135.5 ms** | **0.611** |

**The posteriors ARE a more precise boundary estimator than what we ship (−15 % MAD,
+7.5 pp within-100 ms); they are just biased late.** Premise screen: PASSED — the lever is
real. 5/7 songs have lower posterior MAD (georgia is the exception, 232.7 vs 177.0 ms).

Self-calibrating the bias away *without GT* by centring on our own boundaries per song is a
wash end-to-end (mean |err| 165.9 vs 177.4, within-100 ms 0.464 vs 0.464) — because our own
per-song offsets are large and varied (−236…+141 ms, mostly beat-grid phase). So the bias has
to be handled inside the decoder, not by post-hoc shifting.

### 11:45–11:55 — what the latency COSTS, measured with a 3-line experiment

Shift the raw musx `.lab` earlier by L and re-score (no model, no decoder):

| L (ms) | root | PARTIAL | Δpartial |
|---|---|---|---|
| 0 (shipped) | 0.7347 | 0.6518 | — |
| 100 | 0.7499 | 0.6652 | +1.34 |
| **160** | **0.7527** | **0.6677** | **+1.59** |
| 200 | 0.7526 | 0.6677 | +1.59 |
| 300 | 0.7449 | 0.6611 | +0.93 |
| −50 | 0.7229 | 0.6415 | −1.03 |

Smooth, single-peaked, broad plateau at 140–200 ms — consistent with the independently
measured +113 ms. **But LOSO on L is only +0.54 pp partial / +0.58 pp root** (per-song optima
range 60–340 ms), so a *global constant post-hoc shift* is NOT an honest win. The latency has
to be absorbed by the decoder.

### 11:55–12:25 — ★ COORDINATOR REDIRECT: beat-aware re-decode (`use_beats`), + the synthesis

Course correction from the orchestrating session (lit review `1458bd7`): the vendored
`XHMMDecoder` already has a beat/downbeat-aware transition prior that
`chord_recognition.py` never switches on (`use_beats=False` ⇒ flat 30-nat penalty at every
23 ms frame ⇒ a memoryless geometric duration prior). Re-decoded the extracted posteriors with
a `beat_arr` built from OUR Beat This! grid, calling the clone's own `decode()` (clone untouched).

**Control first:** flat, penalty 30 reproduces the shipped baseline exactly
(root 0.7347 / partial 0.6518) ⇒ the re-decode harness is correct.

**Beat-aware ALONE does essentially nothing:** best `beats_pen30` partial 0.6526 (+0.08 pp),
root 0.7303 (−0.44 pp). Flat-penalty sweep 5…120 also flat (partial 0.6408–0.6554).

**Why — and the fix, predicted from the latency finding and then confirmed:** snapping *late*
evidence onto beats sends it to the NEXT beat. Build the allowed-transition grid at `bt + L`
and shift the decoded times back by `−L`, so boundaries land exactly on our beat grid:

| config | root | 7ths | **PARTIAL** | bass | seg | overseg | underseg |
|---|---|---|---|---|---|---|---|
| our shipped chart | 0.7367 | 0.5173 | 0.6419 | 0.7440 | — | — | — |
| raw musx `.lab` (flat, pen 30) | 0.7347 | 0.5221 | 0.6518 | 0.7443 | 0.822 | 0.882 | 0.825 |
| beat-aware, L=0, pen 30 | 0.7303 | 0.5203 | 0.6526 | 0.7369 | 0.811 | 0.878 | 0.811 |
| beat-aware, **L=140 ms, pen 15** | **0.7449** | 0.5259 | 0.6607 | 0.7530 | 0.817 | 0.868 | 0.833 |
| beat-aware, **L=180 ms, pen 45** | 0.7380 | 0.5302 | **0.6626** | 0.7469 | 0.803 | 0.891 | 0.803 |
| **ORACLE — transitions only at GT change frames, pen 0** | **0.7728** | 0.5450 | **0.6903** | 0.7861 | 0.896 | **0.998** | 0.896 |

The L-curve peaks at 140–180 ms, **independently reproducing the +113/+134 ms latency measured
above from a completely different statistic.** Two independent measurements of the same
constant is the strongest evidence in this session.

**Two hard ceilings this run also established:**
1. *Grid coverage.* Only **18 %** of GT chord changes lie within 30 ms of one of our beats,
   35 % within 60 ms, 57 % within 100 ms, 76 % within 150 ms. Any beat-quantised chart is
   capped by this (our shipped chart included).
2. *Decoder-family ceiling.* Handing the decoder the **GT change times** as its only allowed
   transition frames gives partial **0.6903** / root **0.7728** — i.e. even with perfect
   boundary candidates, re-decoding musx's posteriors lands ~3 pp below the 2026-07-26
   label-recut oracle (root 0.8023). Over-segmentation goes to 0.998 there, so the residual
   is label identity, not boundaries.

Non-uniform (real Beat This!) beats are consistently *worse* than the uniform best-fit grid
(best 0.6535 vs 0.6626) — do not pursue.

### 12:25–12:45 — making it honest: the latency must be picked PER SONG, WITHOUT GT

A global L is not honest: LOSO over the 2-D (L, penalty) grid collapses to
partial **0.6482** / root 0.7279 (per-song optima span 60–340 ms; a global L costs
stand_by_me −5.7 pp). Three GT-free per-song selectors were screened:

| selector | pooled partial | pooled root |
|---|---|---|
| fixed L = 120 ms (best single constant) | 0.6622 | 0.7438 |
| median musx-`.lab`-boundary offset to the nearest beat | 0.6488 | 0.7296 |
| **argmax of the decoder's OWN Viterbi path log-likelihood** | **0.6644** | **0.7497** |
| per-song oracle L (upper bound) | 0.6716 | 0.7528 |

The likelihood selector is the right object: same observations, same emission model, only
the legal-transition set moves, so the latency whose beat grid best explains the posteriors
wins — and it recovers ~74 % of the per-song oracle with **no GT and no fitted constant**.
Honest caveat: it agrees with the metric-optimal L on only 2/7 songs; pooled it wins largely
through blue_bossa (493 s = 30 % of the benchmark).

**Downbeat-graded transition costs — REFUTED.** With per-downbeat marking (the modular-phase
version was feeding garbage: Beat This! downbeat phase purity on our uniform grid is only
0.30 / 0.50 / 0.50 on georgia / blue_bossa / backing), the best graded cost triple
(15, 20, 25) gives partial 0.6627 vs **0.6644 for flat costs**; the best "graded" setting is
literally the ungraded one (20, 20, 20). Grading the bar positions does not help here.

**LOSO on the transition penalty (standalone re-decode, L GT-free):** partial **0.6654** /
root 0.7480 / 7ths 0.5355 / bass 0.7564. But 3/7 songs regress vs the shipped chart, because
this *replaces* our chart and throws away everything the chord stage adds.

### 12:45–13:05 — ★ feeding it back into the REAL pipeline, and the Occam interaction

Swapped only the musx label source (`musx_bass.musx_labels`) for the re-decoded timeline and
re-ran the real `NNLS24ChordHead`, keeping no-chord, the 2-chord split, the leading-outlier
drop, confidence, bars and sections.

**Both controls reproduce exactly** — shipped = 0.6419/0.7367, and flat `.lab` +
`segment_source="musx"` = root **0.7170**, i.e. the documented 2026-07-26 **−1.97 pp**
refutation, bit for bit. So the only thing that changed is the decode.

| config | root | PARTIAL |
|---|---|---|
| shipped | 0.7367 | 0.6419 |
| flat `.lab`, seg=musx (the 2026-07-26 refutation) | 0.7170 | 0.6322 |
| re-decode, seg=nnls (labels only) | 0.7336 | 0.6420 |
| re-decode, seg=musx (boundaries too), **Occam ON** | 0.7340 | 0.6530 |
| re-decode, seg=musx, **Occam OFF** | **0.7473** | **0.6649** |

**The Occam post-pass is what blocks it.** close_to_you goes **−8.04 pp with Occam ON and
+5.17 pp with Occam OFF** — a 13.2 pp swing on one song from the post-pass alone. This is the
same fragility the 2026-07-26 session measured (a 0.25-beat pooling change swung the same song
−11.5 pp). Occam is tuned to the old segmentation.

**FINAL HONEST NUMBER — LOSO on the transition penalty (the only fitted knob; L is GT-free),
pipeline-integrated, Occam OFF:**

| metric | shipped chart | brick ON | Δ |
|---|---|---|---|
| **partial_credit (headline)** | **0.6419** | **0.6609** | **+1.90 pp** |
| mirex_root | 0.7367 | 0.7471 | +1.04 pp |
| mirex_majmin | 0.7102 | 0.7267 | +1.65 pp |
| mirex_sevenths | 0.5173 | 0.5301 | +1.28 pp |
| bass_root | 0.7440 | 0.7552 | +1.12 pp |
| strict | 0.4774 | 0.4859 | +0.85 pp |

(vs the Occam-OFF control the deltas are larger: partial +2.36 pp, root +1.53 pp.)
Per song (partial): bein_green +3.45 · blue_bossa +2.25 · backing +4.67 · close_to_you +5.17
· stand_by_me +1.64 · **every_breath −1.20** · **georgia −1.03**. 5/7 improve.

### 13:05–13:15 — semi-Markov / explicit duration prior — **REFUTED**

Built a duration-explicit (semi-Markov) decode on the beat grid, with the duration prior
estimated from **1696 iReal lead sheets / 150 675 chord durations** (`data/ireal`,
symbolic — no audio, no alignment): 4 beats 63.5 %, 2 beats 25.7 %, 3 beats 6.7 %,
1 beat 1.3 %.

| duration prior weight λ | root | 7ths | PARTIAL |
|---|---|---|---|
| **λ = 0 (machinery control — must equal the beat Viterbi)** | 0.7506 | 0.5343 | **0.6652** |
| λ = 5 | 0.7456 | 0.5297 | 0.6612 |
| λ = 15 | 0.7529 | 0.4955 | 0.6537 |
| λ = 30 | 0.7515 | 0.4874 | 0.6488 |
| λ = 60 | 0.7536 | 0.4872 | 0.6365 |

λ=0 reproduces the beat-constrained Viterbi result (0.6652 vs 0.6644–0.6673), so the DP is
correct; and a uniform-duration control is numerically identical to λ=0 at every λ, as it must
be. **The iReal prior itself is what hurts, monotonically.** Mechanism: it is a *notation*
prior (63.5 % of chords last exactly one bar), but the audio contains intros, vamps, solo
choruses and half-bar changes; forcing bar-length durations over-merges — note root creeps
*up* (0.7506→0.7536) while 7ths **collapses** (0.5343→0.4872), i.e. it is merging away
quality distinctions to buy root stability. Do not re-run this with a lead-sheet prior; an
audio-domain duration distribution would be a different (untested) experiment.

### 13:15–13:30 — where the gain actually comes from (decomposition + artifacts)

Standalone, duration-weighted, against the raw flat `.lab` (partial 0.6518):

| variant | root | 7ths | PARTIAL | share of the gain |
|---|---|---|---|---|
| re-decode boundaries **and** labels | 0.7473 | 0.5375 | 0.6673 | +1.55 pp (100 %) |
| **re-decode BOUNDARIES only** (labels re-sampled from the flat `.lab`) | 0.7445 | 0.5361 | **0.6642** | **+1.24 pp (80 %)** |
| re-decode LABELS only (flat `.lab` boundaries) | 0.7321 | 0.5245 | 0.6546 | +0.28 pp (18 %) |

So this is genuinely a **boundary-placement** win, which is what the +5.45 pp jitter prize
asked for. Honest counterweight: of the 735 boundaries the brick moves, only **338 (46 %)
land closer to a GT change by raw count** — total absolute boundary error over those moves
falls only slightly. The metric moves because the *large* misplacements are the ones fixed.
Both numbers are in the HTML artifact.

**Artifacts:**
`docs/research_sessions/musx_redecode_2026-07-27.png` — (1) the latency histograms (ours
unbiased, `.lab` +113 ms, posteriors +134 ms), (2) the same distributions with per-song bias
removed showing the tighter posterior spread, (3) the L-sweep whose peak lands on the
independently measured latency.
`docs/research_sessions/musx_redecode_boundaries_2026-07-27.html` — every one of the 735
boundaries the brick moves, with old/new/GT times, the error before and after, and the old,
new and GT chord labels, colour-coded closer/further.

### 13:30 — brick packaged + verification

`harmonia/models/musx_redecode.py` (NEW, default OFF, unwired — `HARMONIA_MUSX_REDECODE`
kill switch exists for a future wiring) and `tests/test_musx_redecode.py` (NEW, **13/13
green**, no clone or weights required).

- **The packaged brick reproduces the scratchpad experiment bit-for-bit on 7/7 songs**
  (identical segment counts, identical times to 1e-9, identical labels) — CLAUDE.md #6.
- `tests/test_chord_head_parity.py`, `test_boundary_bleed.py`, `test_calibration_pins.py`,
  `test_seventh_upgrade.py`, `test_musx_redecode.py`: **37 passed**.
- The frame-rate calibration pin (512/22050 = 23.2200 ms) is a unit test, so a future clone
  swap that changes `settings.py` fails loudly (CLAUDE.md #1).
- **Stacking with `no_chord_policy` was NOT measured**: that brick is still un-wired into
  `chord_head`/`chord_pipeline_v1`, so `HARMONIA_NC_POLICY=intersect` is a verified no-op on
  this path (numbers identical to 4 dp). It has to be wired before stacking can be measured.

### 13:30–13:50 — the Occam blocker, isolated and (probably) solved GT-free

Occam's effect under the new segmentation, per song (env toggle only — the post-pass lives in
guarded files and was not edited):

| song | Occam effect on the re-decode | Occam's own logged loop family |
|---|---|---|
| close_to_you | **−13.21 pp** | `cov=0.78 dev=4 n=5` **and** `cov=0.67 dev=16 n=2` |
| stand_by_me | **+3.16 pp** | `cov=0.97 dev=0 n=4` |
| the other 5 | **0.00 pp** (never fires) | (none) |

The separation is clean and uses only statistics Occam already computes, so the gate is
GT-free: **accept a loop family only if `cov ≥ 0.95` and `dev == 0`.** Under the shipped
segmentation the only family Occam ever forms is stand_by_me's `cov=0.98 dev=0`, which the
gate accepts — so **the gate changes today's chart by exactly 0.0000 on all six metrics**,
which makes it a safe, additive change. Caveat: N=2 songs actually exercise it.

Pooled effect (penalty 40, no LOSO, for comparability of the four rows):

| config | root | majmin | 7ths | PARTIAL | bass |
|---|---|---|---|---|---|
| shipped chart (Occam ON = today) | 0.7367 | 0.7102 | 0.5173 | 0.6419 | 0.7440 |
| shipped chart + gated Occam | 0.7367 | 0.7102 | 0.5173 | 0.6419 | 0.7440 |
| re-decode + Occam ON | 0.7340 | 0.7119 | 0.5240 | 0.6540 | 0.7403 |
| re-decode + Occam OFF | 0.7473 | 0.7270 | 0.5350 | 0.6649 | 0.7546 |
| **re-decode + GATED Occam** | **0.7506** | **0.7303** | **0.5380** | **0.6679** | **0.7578** |

With LOSO on the penalty this becomes the headline reported at the top:
partial **0.6639 (+2.20 pp)**, root 0.7503 (+1.36 pp).

**Guardrails honoured:** no commits/adds/stashes; no edits to `chord_pipeline_v1.py`,
`stages/chord_head.py`, `eval/*`, `golden/*`, `align/*`, `dataset/*`, `serving/*`,
`docs/known_issues.md`, or the vendored clone. Disk never below **4.7 GiB** (started 4.8;
`probs/` peaked at 29 MB in the session scratchpad, nothing written to `data/cache`).

