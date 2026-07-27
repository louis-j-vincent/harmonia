# Fusion DBN as a SEGMENTER (where), music-x-lab as the NAMER (what) — session log 2026-07-27

# ⚠ READ FIRST — SCOPE DECISION NEEDED (CLAUDE.md #3)

**The frozen Brick-0 GT's chord ONSET TIMES are a chart-derived metronomic lattice, not
acoustically observed events.** Verified directly from the raw JSON, not inferred:

- `georgia_on_my_mind` (a **rubato Ray Charles ballad**) — every one of its 68 chord onsets is
  `15.509 + k × 1.884 s` exactly (lattice fit residual **0.24 ms**, circular concentration
  **R = 1.000**). `stand_by_me`: `16.276 + k × 2.008 s` (0.37 ms). `bein_green`:
  `14.285 + k × 0.8035 s` (0.41 ms). `blue_bossa_backing`: 1.80 ms. **4 of 7 songs are
  lattice-exact to sub-millisecond**; the other 3 are piecewise lattices (per-section).
- This is exactly what `golden/brick0/SCHEMA.md` prescribes (tile the iReal chart from
  `bar1_anchor_time`) — it is not a bug, it is the design. But it means the GT's *onset times*
  carry the tiling's own error, and they **drift away from the audio's real beats by up to
  −2.2 ms/s**, sitting a median **0.076–0.385 beat** off the independent Beat This! grid.

**Consequence, quantified:** of the **+6.48pp root** "re-cut on GT boundaries" oracle that four
sessions have been chasing, only **+2.65pp is reachable by placing chords correctly on the
audio's real beats**. The other **+3.83pp (59%) requires reproducing the annotation lattice's
sub-beat positions** — a metronomic artifact of how the GT was built, not an acoustic event.

**The GT remains sound for chord IDENTITY (hand-verified) and for beat-level timing. It cannot
adjudicate sub-100 ms boundary placement.** Every relative comparison in this session is
unaffected (all systems scored against the same GT); what changes is how much of the oracle is
real. Artifact: `docs/research_sessions/gt_lattice_2026-07-27.png`.

**This is a scope call for Louis, not one I should make.** I have not changed scope; I stopped
at the measurement.

---

**Status: COMPLETE.** 10:19–10:45 CEST (~26 min). Disk 4.7 GiB throughout, never below.
Concurrent session in the repo: `musx_frame_posteriors_2026-07-27.md` (started 10:16) — a
different lever (musx frame posteriors) on the same problem (boundary bleed). No file overlap
intended; I will not touch `harmonia/models/` names it may claim without checking first.

Headline metric: **`partial_credit` (family-level), baseline 0.6419**; `mirex_root` (0.7367)
alongside; `sevenths`/`strict` secondary. 7 hand-verified frozen Brick-0 songs.

---

# FINAL REPORT

## Verdict: the hypothesis is REFUTED. No brick was built (correct outcome — the premise gate said stop).

**Hypothesis:** the fusion DBN is a poor namer but a good *placer*, so take its chord-CHANGE
times and keep music-x-lab's identity. **Answer: no, on every measurement, and the proposed
mechanism runs backwards.**

| measurement (LATENT / no-chart mode, 7 verified songs) | shipped | DBN | verdict |
|---|---|---|---|
| boundary F1 @0.25 s | **0.493** | 0.447 | worse |
| boundary F1 @0.50 s | 0.680 | 0.681 | tie |
| boundary F1 @±1 beat | **0.673** | 0.622 | worse |
| **hold-too-long bias** (median signed offset / % late) | **+0.011 s / 53.2%** | **+0.095 s / 66.5%** | **worse — the opposite of the hypothesis** |
| ±1-beat 3-way picker accuracy (N=480) | **0.698** | 0.642 hard / 0.590 soft | worse |
| **END-TO-END, same musx namer: `mirex_root`** | **0.7360** | **0.6802** | **−5.58pp** |
| **END-TO-END, same musx namer: `partial_credit` (headline)** | **0.6492** | **0.6109** | **−3.83pp** |

**Why it fails (the mechanism, isolated by ablation, not guessed):** the DBN's *evidence* has
no late bias at all — the per-beat **argmax** of the same emission is unbiased (+0.035 s, 54%
late, i.e. the same as ours). The lateness is **created by the persistence prior**, and it is
monotone in it (+0.019 s at `change_penalty`=0.5 → +0.095 s at 5.0). A beat that *straddles* a
chord change carries a chroma mixture; a symmetric self-transition prior resolves every such
beat **in favour of the incumbent**, sliding the boundary to the next beat edge.

> **A persistence/duration prior on a beat grid is a "hold-longer" prior, not a "place-better"
> prior. It buys segmentation RATE, not segmentation PLACEMENT.** There is no setting where it
> gives both: at the rate matching GT (762 vs 729 boundaries) it is 66.5% late; where it is
> unbiased (cp=0.5) it emits 3375 boundaries and F1 collapses to 0.354.

Since the pathology this session was asked to cure *is* "hold the previous chord too long"
(2:1), the idea is **self-defeating as stated**, not merely under-tuned.

## What I tried, in order, and what each cost

1. Baseline reproduced bit-exact (root 0.7367 / partial 0.6419). ✔
2. **Premise screen** — boundary geometry, 5 boundary sets, 3 tolerances. → refuted at pooled
   level, but showed strong per-song complementarity → one refinement round justified.
3. `_CHANGE_PENALTY` sweep (10 values) → the rate/bias trade; no good operating point.
4. **Forward–backward change posterior** (implemented here, not in `align/`) as the principled
   answer to "Viterbi is reluctant to switch" → **0.590, worse than the hard decode (0.642)**.
5. `w_bass` ablation (5 values) → bias flat to 3 dp; **not the bass stream**.
6. argmax vs Viterbi → **the lateness is the decoder, not the evidence** (the key isolation).
7. Beat-chroma pooling shift (5 values) → a trade, not a fix.
8. End-to-end same-namer/different-placer ladder → the −5.58pp verdict.
9. Lag correction of the diagnosed +0.095 s bias → recovers +0.3pp of 5.6pp; **the residual is
   unsystematic**, so the direction is closed on its own diagnosed terms.
10. Snap-our-boundaries-to-DBN (the jitter shape the oracle rewards) → monotonically negative
    (−0.2 to −0.9pp). Oracle control on the same code path reproduces +5.32pp (doc: +5.45pp).
11. **Sub-beat template split at frame resolution** (the fusion asset minus the harmful part,
    aimed at the larger half of the prize) → **premise fails**: mean |offset| 0.184 → 0.196 s,
    only 30.3% improve. End-to-end flat.
12. Beat-grid **ceiling** measurement → the new number below.
13. **GT lattice check** (CLAUDE.md #3) → the scope finding at the top.

## Three findings worth more than the refutation

1. **The beat-grid ceiling.** A *perfect* beat-level segmenter (perfect insert, delete and
   ±1-beat pick) scores root **0.7625** / partial **0.6825** — only **+2.65pp / +3.33pp** over
   today. The rest of the +6.48pp oracle is sub-beat. Combined with the GT-lattice finding at
   the top, that sub-beat remainder is largely **not winnable in principle** on this benchmark.
2. **With a midpoint namer, boundary INSERTION is nearly free and nearly worthless.** Cutting
   at **every single beat** scores root 0.7301 / partial 0.6479 — within 0.6pp / 0.1pp of our
   entire root-change segmenter, because `_coalesce_labeled` erases 82% of the cuts. Only
   *placement* can move the metric. This is the quantitative form of the documented mechanism.
3. **Our boundaries are already at the grid floor.** For the 495 GT changes we find, median
   |ours − GT| = 0.1220 s vs the grid floor |nearest beat − GT| = 0.1240 s — **−2 ms, i.e.
   indistinguishable**. The +35% excess is entirely a tail of wrong-beat picks. There is no
   systematic error left to remove; only the tail.

## Concrete next-step recommendation

- **Do not spend further budget on sub-beat boundary placement against this benchmark.** It is
  59% of the nominal prize and mostly an annotation artifact. If sub-beat placement matters for
  the *product* (it may, for playback feel), it needs a differently-built GT — hand-tapped
  chord onsets on the audio clock, not a chart tiling.
- **The remaining honest boundary target is +2.65pp root / +3.33pp partial**, and it is entirely
  "pick the right beat" — a tail of wrong-beat picks, not a systematic offset. Every acoustic
  signal tried across two sessions now fails it (chroma novelty 0.547, bass novelty 0.520,
  onset 0.367, HPSS 0.370, DBN hard 0.642, DBN posterior 0.590, all vs the shipped 0.698 in
  this harness). **music-x-lab's frame-level posteriors (the concurrent session) are the last
  untried candidate.**
- **Warning for that session, earned here:** do *not* couple musx frame posteriors to the fusion
  DBN's **transition prior** "for smoothing". This session measured what that prior does to
  placement: **+0.095 s of systematic lateness**, exactly the bleed you are trying to remove.
  (Their `use_beats` re-decode with an explicit latency offset is a *different* and better
  shaped mechanism — it moves the allowed-transition grid rather than adding a persistence
  bonus — and it is working: partial 0.6626 vs 0.6419. Nothing here argues against it.)
- **Cross-check with the concurrent session (read both together).** Their +112.8 ms musx `.lab`
  latency and my +0.131 s are independent corroboration. But their two *ceiling* numbers —
  "only 18% of GT changes lie within 30 ms of one of our beats" and the GT-change-frames
  decoder oracle — inherit the GT-lattice problem at the top of this log: a large part of that
  30 ms/100 ms miss is the GT's rigid tiling drifting from the audio's real beats, not real
  chord onsets sitting between beats. Their *latency* result is unaffected (it is a bias, and
  the GT lattice is phase-unbiased relative to the beat grid since it was built from a Beat
  This! pass); their *ceilings* should be read as upper bounds on agreement-with-the-annotation,
  not on agreement-with-the-music.
- FYI for the fusion lane (no action taken, `align/*` untouched): the `_CHANGE_PENALTY = 5.0`
  that gives `inference.py` its good chord *count* is buying it with ~+0.1 s of systematic late
  placement on every chord change.

## Deliverables

**NEW files (repo):** `docs/research_sessions/dbn_segmentation_hybrid_2026-07-27.md` (this log),
`docs/research_sessions/dbn_segmentation_2026-07-27.png` (4-panel refutation diagnostic),
`docs/research_sessions/gt_lattice_2026-07-27.png` (the GT-lattice artifact).
**No brick, no tests** — the premise gate said stop, so there was nothing to wire.
**Guarded files:** none touched (`align/*` imported read-only; `chord_pipeline_v1.py`,
`stages/`, `eval/`, `golden/`, `serving/`, third_party, `docs/known_issues.md` untouched).
No `git commit`/`add`/`stash`. All experiment code lives in the session scratchpad
(`base.py`, `premise.py`, `sweep.py`, `picker.py`, `isolate.py`, `hybrid.py`, `final.py`,
`subbeat.py`, `artifact.py`, `gtcheck.py`, `gtplot.py`).
**Disk trace:** 4.7 GiB at start and at every one of the 11 runs; never below 4.7 GiB;
floor 1.5 GiB never approached. Scratchpad peak 195 MB, cleaned at exit.
**Small-N flag:** N=7 songs, 729 GT boundaries, 1654 s. Every per-song number carries that.
No hyperparameter was fitted to the eval set except the `change_penalty` sweep, which is
reported as a diagnostic, not as a tuned result.

---

## Brief, restated as a numbered spec (Phase 1)

1. **Hypothesis under test.** The fusion DBN (`harmonia/align/inference.py`, chord identity
   LATENT, no chart) is a poor NAMER (root 0.63 / partial 0.44 standalone, vs shipped
   0.737/0.642) but may be a good PLACER, because it has (a) a real duration model (the
   `_CHANGE_PENALTY = 5.0` self-transition prior) and (b) a key-aware transition prior — both
   of which our shipped segmenter (`_root_change_segs`, greedy per-beat root-argmax change
   detector) completely lacks. **Hybrid = DBN change times (where) × musx `.lab` identity
   (what).**
2. **Target / threshold.** `partial_credit` on the 7 verified frozen songs via
   `harmonia/eval/accuracy_score.py` (READ-ONLY). Baseline 0.6419 / root 0.7367. Oracle
   ceiling for pure re-cutting: root +6.56pp, partial +6.78pp. Project keep-bar: ≥ +2pp
   pooled, no per-song regression.
3. **Deciding gate (step 1).** If the DBN's chord-change times are no closer to GT boundaries
   than our current segment edges (distance distribution + boundary P/R/F1 at ±0.25 s / ±0.5 s
   / ±1 beat + the signed hold-too-long bias), **STOP and report**.
4. **Circularity guard (CLAUDE.md #3).** LATENT / no-chart mode ONLY. The chart-OBSERVED
   alignment mode places chords using the iReal chart, and the frozen GT was itself built by
   aligning iReal charts to audio — scoring that against this GT measures an aligner against
   an aligner. Every number below states its mode.
5. **Constraints.** NEW default-OFF brick + tests only. Never edit `harmonia/align/*`,
   `chord_pipeline_v1.py`, `stages/chord_head.py`, `eval/*`, `golden/*`, `serving/*`,
   third_party. No commits/adds/stashes. No writes to `docs/known_issues.md`. Disk floor
   1.5 GiB, `df -h .` before every run.
6. **Budget.** Not stated in hours — self-imposed **~5 h from 10:19**, ~30 min checkpoints
   with disk trace.

## Phase 0 — what history already says (do not re-derive)

- 70.5% of wrong-root duration is an ADJACENT GT chord's root (lift 1.75); 2:1 toward holding
  the PREVIOUS chord; 48–72% localised within 0.5 s of the shared edge.
- Oracle re-cut on GT boundaries: root +6.56pp; jitter-only snap (≤0.8 s) +5.45pp ⇒ **83% of
  the prize is MISPLACED (±1 beat) boundaries**, not missing ones.
- Accuracy by GT span length: 0.192 (<1 beat) → 0.931 (>16 beats).
- REFUTED boundary signals (do NOT re-run): chroma novelty 0.547 / onset strength 0.367 /
  HPSS percussive 0.370 / onset+chroma 0.482 — all below the shipped segmenter's 0.554 on the
  ±1-beat 3-way pick (chance 0.333). `segment_source="musx"` −1.97pp; union of cuts −1.97pp
  (mechanism: `_coalesce_labeled` erases a cut whose two sides share a midpoint label);
  gated under-seg repair −0.34pp; musx onset-hint retiming −0.44pp; within-beat pooling ≈0;
  metrical snap −1.49 raw / +0.27 gated.
- Accountability baseline: raw musx `.lab` scored straight against GT = root 0.7347 /
  partial 0.6518 — our whole chord stage is +0.21pp root and −0.99pp partial over printing
  the `.lab`. **Any boundary win must be measured against BOTH our chart and raw musx.**

## Structural facts read off the source before running anything (CLAUDE.md #1)

- `FusionChordDecoder.decode` builds its own grid: `brick0.beat_this_full(wav)` on a
  **22 050 Hz mono** decode; chroma = librosa CQT, hop 512 @ 22 050 ⇒ **43.07 fps
  (23.22 ms)**. The shipped chord stage uses `beat_backend="beatthis"` too but with
  `beat_period_mode="bestfit"` (a re-fitted constant-tempo lattice), and NNLS chroma at
  46.44 ms. **So the two systems' beat grids are NOT identical** — part of any boundary
  difference is grid, not model. Controls below isolate that.
- The DBN's change times are **beat-quantised by construction** (`coalesce_path` puts every
  boundary on a `_beat_bounds` edge), exactly like ours. So the hypothesis is really "a
  better choice of WHICH beats", not "sub-beat precision".
- `_CHANGE_PENALTY = 5.0` in emission-log units (`_BETA_EM = 10.0` × Pearson agreement) is
  the duration model. It is the knob that directly trades hold-too-long against fragmentation.

---

## Log

### 10:19–10:25 — Phase 2: baseline REPRODUCES bit-exact (disk 4.7 GiB)

Shipped pipeline (`SHIPPED_CONFIG`, `infer_chords_v1`) on all 7 verified songs, pooled
duration-weighted: **root 0.7367 · majmin 0.7102 · 7ths 0.5173 · partial 0.6419 ·
strict 0.4774 · bass 0.7440** — identical to 4 dp to the 2026-07-23 and 2026-07-26 numbers,
and every per-song number matches. CLAUDE.md #1 discharged.

Beat-grid diff (rule #6 — the control I flagged): shipped `beat_times` vs the DBN's Beat This!
grid — 206/207, 1103/**1169**, 786/786, 326/326, 446/446, 281/**272**, 345/347. Identical on
3 songs, ±1–2 on 2, materially different only on blue_bossa (+66) and georgia (−9). So the
comparison below is *nearly* same-grid; blue_bossa carries a grid confound.

### 10:25–10:27 — **PREMISE SCREEN: the hypothesis is REFUTED at pooled level**

**Mode: LATENT / no-chart (`FusionChordDecoder(method="dbn")`, `decode_from_features`), no
chart consulted anywhere. Non-circular.** Boundary geometry only — no labels involved, so the
DBN being a bad *namer* cannot confound this.

Boundary precision / recall / F1, one-to-one greedy matching, pooled over the 7 songs
(GT = 729 interior chord changes):

| set | F1@0.25s | F1@0.50s | F1@1beat | med signed offset @0.5s | % late @0.5s |
|---|---|---|---|---|---|
| **shipped (`_root_change_segs`)** | **0.493** | 0.680 | **0.673** | **+0.011 s** | **53.2%** |
| **DBN latent change times** | 0.447 | 0.681 | 0.622 | **+0.095 s** | **66.5%** |
| all Beat This! beats (control) | 0.299 | 0.340 | 0.340 | +0.011 s | 51.5% |
| Beat This! downbeats (control) | 0.376 | 0.646 | 0.555 | +0.195 s | 85.5% |
| raw musx `.lab` edges (control) | 0.501 | **0.716** | 0.657 | +0.131 s | 81.1% |

**Verdict on the stated hypothesis: NO.** The DBN's change times are *worse* at the tight
tolerance (−4.6pp F1 @0.25 s), a dead heat at 0.5 s (+0.1pp), and worse at ±1 beat (−5.1pp).

**And the specific mechanism the hypothesis was built on fails in the opposite direction:**
the DBN was supposed to *reduce* the hold-too-long bias because it has a duration model. It
**increases** it — 66.5% of its matched boundaries are late vs 53.2% for ours, median offset
+0.095 s vs +0.011 s. A strong self-transition prior (`_CHANGE_PENALTY = 5.0`) does not make
the placement better, it makes the decoder *reluctant to switch*, i.e. it manufactures exactly
the pathology we are trying to remove.

Control reading: the DBN's own beat grid is unbiased vs the GT boundaries (+0.011 s median),
so the lateness is **not** a grid-phase artifact — it is the decoder choosing the beat *after*
the change.

**Two things in the table that are NOT dead** (grounds for one refinement round rather than an
immediate stop):
1. **Per-song, the DBN is strongly complementary, and it wins exactly where we are worst.**
   F1@0.5 s, DBN − shipped: bein_green **+0.162** (shipped 0.431), close_to_you **+0.122**
   (shipped 0.469), backing +0.035 — vs blue_bossa −0.017, georgia −0.031, every_breath
   −0.070, stand_by_me **−0.265** (shipped 0.925). Correlation of the delta with the shipped
   score is strongly negative: the DBN helps the two songs the shipped segmenter fails on and
   wrecks the one it already nails.
2. **`_CHANGE_PENALTY` is untested.** It is the single knob that controls the late bias, and
   the operating point (5.0) was tuned for the DBN's *own* end-to-end labels, not for boundary
   placement. Sweeping it is the principled response to the diagnosed failure (Phase 3 #2),
   not a retry of the same thing.

Also logged for the record (not part of the hypothesis): **raw musx `.lab` edges have the best
boundary F1@0.5 s of anything measured (0.716 vs our 0.680)** but carry a large systematic
**+0.131 s lag** (81% late). `segment_source="musx"` is already refuted end-to-end (−1.97pp,
mechanism: `_coalesce_labeled` erases cuts whose sides share a midpoint label), so this is a
geometry observation, not a revival — but a *lag-corrected* musx boundary set is a different
object from what was refuted and is cheap to test.

Artifact: `scratchpad/premise.py`; numbers above are its stdout.

### 10:27–10:31 — refinement round 1: the ±1-beat picker + the mechanism, ISOLATED

**Sharp diagnostic (`scratchpad/picker.py`).** For every shipped boundary whose GT change is
within ±1 beat, which of {b−1, b, b+1} does each signal pick? (N=480 on the Beat This! grid.
**NOT directly comparable to the documented N=581 / 0.554** — different grid and different
case construction, so I report only the within-harness comparison.)

| picker | pooled |
|---|---|
| **shipped keeps its own beat** | **0.6979** |
| DBN hard Viterbi change indicator | 0.6417 |
| DBN **forward–backward change posterior** (implemented here, not in `align/`) | 0.5896 |
| DBN posterior, only overriding when decisive (≥2× runner-up) | 0.6500 |
| chance | 0.333 |

The soft marginal was the principled response to "Viterbi is reluctant to switch" — it is
*worse* than the hard decode, so the reluctance is not a Viterbi-vs-marginal artifact.
Per song the same complementarity: DBN wins backing (0.460→0.683), close_to_you
(0.784→0.865), bein_green (0.722→0.750); loses blue_bossa (0.846→0.262!), stand_by_me
(1.000→0.838), georgia, every_breath.

**MECHANISM, isolated by ablation (`scratchpad/isolate.py`), pooled, matched @0.5 s:**

| config | n boundaries | F1@0.5s | median signed offset | % late |
|---|---|---|---|---|
| shipped | 735 | 0.680 | **+0.011 s** | 53.2% |
| DBN **argmax** (emission only, NO transition) | 2580 | 0.421 | **+0.035 s** | 54.2% |
| DBN Viterbi (cp=5) | 762 | 0.681 | **+0.095 s** | 66.5% |
| DBN Viterbi, `w_bass` = 0.0 / 0.2 / 0.4 / 0.8 / 1.2 | 726…772 | 0.668…0.681 | **+0.094…+0.095 s (flat)** | ~66% |
| DBN Viterbi cp = 0.5 / 1 / 2 / 3 / 5 / 10 | 3375 / 3075 / 1908 / 1174 / 762 / 365 | 0.354 / 0.376 / 0.497 / 0.617 / 0.681 / 0.517 | **+0.019 / +0.022 / +0.053 / +0.070 / +0.095 / +0.089** | 52.5→67.1% |

**The mechanism is now established and it is the opposite of the hypothesis:**

- **H4 (bass stream is late) — REFUTED.** The offset is *flat* to 3 decimals across a 0→1.2
  sweep of `_W_BASS_EM`. Not the bass.
- **H6 (the evidence is late) — REFUTED.** The per-beat **argmax** of the very same emission is
  essentially unbiased (+0.035 s, 54.2% late — the same as ours). The DBN's *evidence* has no
  late bias at all.
- **The lateness is created by the persistence prior itself, and it is monotone in it**
  (+0.019 s at cp=0.5 → +0.095 s at cp=5). My first read of the cp sweep as "H1 refuted" was
  wrong on this point: cp *does* control the bias. What it does not do is give a setting where
  the DBN is both correctly segmented and unbiased — at the rate that matches GT
  (cp≈5 ⇒ 762 boundaries vs GT's 729) it is 66.5% late; where it is unbiased (cp≈0.5) it emits
  **3375** boundaries, 4.6× GT, and F1 collapses to 0.354.

**Why (the general statement, and the reason this direction is dead):** a beat that *straddles*
a chord change carries a chroma mixture of both chords. A symmetric self-transition prior
resolves every such ambiguous beat **in favour of the incumbent**, so the emitted boundary
slides to the *next* beat edge. **A persistence/duration prior on a beat grid is a
"hold-longer" prior, not a "place-better" prior — it buys segmentation *rate*, not
segmentation *placement*.** That is precisely the pathology (2:1 hold-the-previous-chord) the
hypothesis proposed to cure with it, so the idea is self-defeating as stated, not merely
under-tuned.

Corollary for the fusion lane (FYI, no action taken): the same `_CHANGE_PENALTY` that gives
`inference.py` its good chord *count* is buying that count with a ~+0.1 s systematic late
placement of every chord change.

H5 (pooling-window shift) is a trade, not a fix: shifting the pooled beat chroma +0.5 beat
later flips the bias to −0.121 s but drops F1 to 0.607.

### 10:31–10:32 — **THE DECIDING END-TO-END MEASUREMENT: same NAMER, different PLACER**

Boundary F1 is a proxy; this is the target metric. Every row uses the **identical** label
source and the identical shipped contract (music-x-lab `.lab` sampled at each span's MIDPOINT,
then equal neighbours coalesced) — **only the boundary set changes**, so segmentation is
isolated from identity exactly. `scratchpad/hybrid.py`.

| PLACER (namer = musx midpoint, always) | n spans | root | 7ths | **partial** | strict | bass |
|---|---|---|---|---|---|---|
| shipped segment edges (the core) | 623 | 0.7360 | 0.5225 | **0.6492** | 0.4797 | 0.7454 |
| **DBN latent change times (THE HYPOTHESIS)** | 537 | **0.6802** | 0.4878 | **0.6109** | 0.4440 | 0.6917 |
| union(shipped, DBN) | 634 | 0.7348 | 0.5224 | 0.6506 | 0.4795 | 0.7443 |
| every Beat This! beat | 639 | 0.7301 | 0.5183 | 0.6479 | 0.4730 | 0.7394 |
| downbeats only | 590 | 0.7253 | 0.5165 | 0.6449 | 0.4664 | 0.7359 |
| musx's own edges | 639 | 0.7347 | 0.5221 | 0.6518 | 0.4767 | 0.7443 |
| **GT boundaries (ORACLE)** | 583 | **0.8008** | 0.5784 | **0.7108** | 0.5274 | 0.8136 |

**The DBN as placer is −5.58pp root / −3.83pp partial.** Not a tie — a large, unambiguous
loss. The boundary-F1 "tie at 0.5 s" was a proxy artifact: F1 counts a boundary as matched
anywhere within the tolerance, while the metric pays for every millisecond it is off.

Two further checks, both negative, closing the direction (`scratchpad/hybrid.py`, `final.py`):

- **Snap-to-DBN jitter** (move each shipped boundary onto a nearby DBN boundary — the exact
  shape the +5.45pp jitter oracle rewards): root 0.7344 / 0.7319 / 0.7284 / 0.7270 at
  tol 0.15 / 0.25 / 0.35 / 0.5 s — **monotonically negative**. The oracle control on the same
  code path reproduces the documented prize: snap→GT gives +1.52 / +3.95 / **+5.32pp** root at
  0.25 / 0.5 / 0.8 s (documented +5.45pp) — so the harness is calibrated and it is the DBN's
  boundaries, not the snapping, that fail.
- **Lag-corrected DBN** (the direct fix for the diagnosed +0.095 s bias): 0.6802 → 0.6829
  (−0.05 s) → 0.6826 (−0.095 s) → 0.6831 (−0.15 s) → 0.6806 (−0.20 s). **Flat.** Removing the
  systematic component of the error recovers +0.3pp of the missing 5.6pp; the rest is
  *unsystematic* misplacement. Control: applying the same shift to OUR boundaries is
  −0.2/−0.5pp, confirming ours are already correctly centred.

**Accountability number that fell out of this table (worth its own line):** cutting at
**every single beat** and letting the coalescer do all the work scores root 0.7301 /
partial 0.6479 — within **0.6pp root / 0.1pp partial** of our entire root-change segmenter.
The coalescer erases 82.0% of those cuts (3553 in → 639 spans out; for the shipped set 15.2%,
for union(shipped,DBN) 57.6%). This is the quantitative form of the documented
`_coalesce_labeled` mechanism: **with a midpoint namer, boundary INSERTION is nearly free and
nearly worthless; only boundary PLACEMENT can move the metric.**

**Also measured — and NOTE THE DISTINCTION, this is not the concurrent session's result:**
the raw musx `.lab` edges are **+0.131 s late** here. Shifting the *label-lookup instant* (keep
OUR boundaries, sample musx's label 0.05…0.20 s later at each span midpoint) is **dead flat**:
root 0.7360 → 0.7361 → 0.7360 → 0.7365 → 0.7367, partial 0.6492 → 0.6492 → 0.6481 → 0.6485 →
0.6485. Mechanism: the midpoint label already covers 89% of its span (2026-07-26), so moving
the sample point rarely changes which label is picked.

That is a **different operation** from shifting musx's *own boundary timeline*, which the
concurrent session measured the same day as a real effect (+1.59pp partial at L=160 ms raw,
+0.54pp LOSO). **Both are correct and they do not conflict:** moving where you *sample* a
label does nothing; moving *musx's boundaries* fixes musx's lateness. My +0.131 s and their
independently-derived +112.8 ms / +134.4 ms are three measurements of the same constant from
three different statistics — good mutual corroboration.

### 10:32–10:33 — the number that reframes the whole boundary programme: the BEAT-GRID CEILING

The 2026-07-26 session observed that GT chord changes sit a median 0.19 beat off the pipeline
grid and called it "a structural ceiling for any beat-quantised chart", but never converted it
into a metric. Doing that now (`scratchpad/final.py`): take the **GT boundaries**, snap them to
the nearest beat, and use those as the placer. That is the score of a *perfect* beat-level
segmenter — perfect insert, perfect delete, perfect ±1-beat pick.

| placer | root | **partial** | Δ root vs shipped |
|---|---|---|---|
| shipped segment edges | 0.7360 | 0.6492 | — |
| **GT snapped to the Beat This! grid (BEAT-GRID CEILING)** | **0.7625** | **0.6825** | **+2.65pp** |
| GT snapped to the shipped pipeline grid | 0.7586 | 0.6792 | +2.26pp |
| GT snapped to a 1/2-beat grid | 0.7868 | 0.6973 | +5.08pp |
| GT snapped to a 1/4-beat grid | 0.7920 | 0.7030 | +5.60pp |
| **GT exact (unquantised oracle)** | **0.8008** | **0.7108** | **+6.48pp** |

GT-boundary distance from the grid: median **0.249 beat**; only 33.7% within 0.15 beat,
50.1% within 0.25 beat, 86.3% within 0.5 beat (Beat This! grid; the shipped grid is very
slightly worse at 0.258 / 32.2% / 48.3%).

**Read: of the +6.48pp root oracle prize, only +2.65pp (41%) is reachable by ANY
beat-quantised placer, however perfect. The remaining +3.83pp (59%) is sub-beat and cannot be
reached without leaving the beat grid — and half of it (+2.4pp) is already recovered by a
half-beat grid.** In partial-credit terms the ceiling is 0.6825 vs the oracle's 0.7108.

This does not contradict the 2026-07-26 accounting ("half/quarter-beat quantisers lose almost
nothing": raw musx 0.7347 → per-beat 0.7265, −0.8pp) — that measured quantising an *already
wrong* boundary set, where quantisation error aliases into existing error. When the boundaries
are correct, quantisation is the *entire* remaining error, and it costs 3.83pp.

### 10:33–10:34 — going after the LARGER (sub-beat) half with the fusion asset, minus the harmful part

Having established that (i) the DBN's *transition* prior degrades placement and (ii) 59% of the
prize is sub-beat, the natural next experiment uses the fusion module's **other** half — the
centre-normed CQT chroma vs chord-tone **templates**, available at frame resolution (23.22 ms)
and carrying no persistence prior at all. Keep our boundary set; move each boundary off the
grid to the frame time that best splits the audio between the two adjacent chords:

  τ* = argmax_τ  Σ_{f<τ} ⟨chroma_f, T_A⟩ + Σ_{f≥τ} ⟨chroma_f, T_B⟩

with T_A/T_B the templates of the musx labels either side. Non-circular (audio + musx only).
`scratchpad/subbeat.py`.

**PREMISE SCREEN FAILS.** Over the 511 boundaries with a GT change within ±1 beat, the
template-split time is **further** from GT than the beat edge: median |offset| 0.131 → 0.156 s,
mean **0.1844 → 0.1961 s (+6.3%)**, and only **30.3%** of boundaries improve (a coin flip would
give ~50% if the signal were neutral). Per song it improves only backing (0.159→0.141) and
georgia/every_breath marginally; it is clearly worse on blue_bossa (14.0% improved),
close_to_you and stand_by_me.

Scored end to end anyway for the record (same musx midpoint namer): root
0.7360 → 0.7359 (w=0.5 beat) / 0.7366 (w=0.25) / 0.7358 (w=1.0); partial
0.6492 → 0.6492 / 0.6499 / **0.6530**. The one number above baseline (partial +0.38pp at
w=1 beat) moves root by −0.02pp — inconsistent across metrics, far below the +2pp bar,
**reported as measured-but-below-bar, not a result**.

**So the sub-beat half is not recoverable from chroma-template evidence either.**

### 10:34–10:36 — closing diagnostic: we are already at the beat-grid floor

For the 495 GT changes our segmenter *finds* (a boundary within ±1 beat):

| | median | mean |
|---|---|---|
| \|our boundary − GT\| | **0.1220 s** | 0.1814 s |
| \|nearest BEAT − GT\| (the grid floor) | **0.1240 s** | 0.1343 s |
| excess over the floor | **−0.0020 s** | +0.0471 s (+35%) |

**The median is *at* the floor** (−2 ms, i.e. indistinguishable) — conditional on finding the
change, our typical placement is exactly as good as the beat grid permits. The +35% excess is
entirely in the **tail**: a minority of boundaries put on the wrong beat. That tail is the
+2.65pp beat-grid prize, and it is what the ±1-beat picker (0.698 in-harness) has to improve;
everything else needs a finer grid.

**Inspectable artifact (4-panel):** `docs/research_sessions/dbn_segmentation_2026-07-27.png` —
(a) signed-offset histograms showing the DBN's worse late bias, (b) the persistence-prior
rate-vs-bias trade, (c) the end-to-end placer ladder with the ceiling, (d) our error vs the
grid floor.

### 10:36–10:45 — CLAUDE.md #3 turned on the benchmark itself: the GT onset times are a LATTICE

Having concluded "59% of the prize is sub-beat", the obligatory next question is whether the GT
can *adjudicate* sub-beat placement at all. `golden/brick0/SCHEMA.md` says the chord timeline is
produced by tiling the iReal chart onto the audio from `bar1_anchor_time`. Tested
(`scratchpad/gtcheck.py`, `gtplot.py`): fit each song's GT onsets to an arithmetic lattice
`t = a + k·p`.

| song | fitted p | BPM | mean lattice residual | circular R | median dist from real beats | drift |
|---|---|---|---|---|---|---|
| bein_green | 0.4016 s | 149.4 | **0.41 ms** | **1.000** | 0.076 beat | +0.23 ms/s |
| blue_bossa_backing | 0.8000 s | 75.0 | **1.80 ms** | **1.000** | 0.338 beat | +0.40 ms/s |
| **georgia_on_my_mind** | 1.8842 s | 31.8 | **0.24 ms** | **1.000** | 0.221 beat | **+2.17 ms/s** |
| stand_by_me | 0.6693 s | 89.6 | **0.37 ms** | **1.000** | 0.148 beat | −0.70 ms/s |
| close_to_you | 0.6770 s | 88.6 | 64.7 ms | 0.774 | 0.101 beat | −0.75 ms/s |
| every_breath_you_take | 1.0338 s | 58.0 | 129.6 ms | 0.617 | 0.385 beat | −0.99 ms/s |
| blue_bossa | 0.3481 s | 172.4 | 69.0 ms | 0.259 | 0.306 beat | −0.15 ms/s |

Verified straight off the raw JSON, not through my extraction code:
`georgia_on_my_mind` t0 = `15.509, 19.277, 23.046, 24.930, 26.814, 28.698, 30.583, 32.467, …`
— every gap an exact multiple of **1.884 s**. **Ray Charles does not play a rubato ballad to
0.24 ms.** `stand_by_me` = `16.276 + k·2.008`; `bein_green` = `14.285 + k·0.8035`. GT chord
durations are integer multiples of the fitted beat for **100%** of chords in those 4 songs.
The other 3 fit piecewise (per-section) lattices.

**So the GT's onset times are synthesised, and the audio drifts away from them** (see the
staircase/ramp signature in `gt_lattice_2026-07-27.png`; `blue_bossa_backing` walks
−0.04 → −0.20 s in discrete steps, `georgia` ramps −0.45 → +0.20 s across the song).

**This does not affect any relative comparison in this session** (all placers scored against the
same GT) and does not touch chord IDENTITY, which is hand-verified. What it changes is the
reading of the oracle: the +2.65pp beat-grid part is real and reachable; the +3.83pp sub-beat
part is largely chasing the annotation lattice. Surfaced at the top as a scope call.
