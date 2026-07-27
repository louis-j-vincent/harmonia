# The REPETITION PRIOR for chord segmentation — session log (2026-07-27)

**Status: COMPLETE.** 15:57–16:50 CEST. Disk 3.0 → 1.9 GiB free (floor 1.5; the drop is a
concurrent session, not this one — this session wrote 104 KB of cached charts + 191 KB of
artifacts and deleted its decoded WAVs). N = 7 hand-verified songs for every audio-side
number; the premise screen's second arm is POP909, N = 880. Every number from a real run.

---

## FINAL REPORT (read this first)

**PREMISE: PASSES.** The chord-change rhythm does repeat, measurably, above the metrical base
rate. Conditioned on bar position, the change indicator at a song's own form lag carries
**0.205 bits/slot on the frozen 7 — a 62 % cut in conditional entropy, odds lift 70×**; on
POP909 (N=880, independent annotations) **6–10 % of the entropy, lift ~5×**, best pooled lag
8 bars. **Base rate to hold it against: 87.1 % of frozen-GT changes fall on a downbeat**
(POP909: 57.9 %). The signal is **per-song and lives at the song's own period** — pooled at a
fixed lag it is nearly worthless, so any brick must detect the period per song.

**BUT THE LEVER DOES NOT PAY, AND I CAN SAY EXACTLY WHY.** Purity is monotone in cut count —
**cutting blindly at every beat scores 0.870 nameable, above the GT-change oracle's 0.863** —
so everything must be budget-matched. Budget-matched, the repetition prior is *equal to* the
trivial metrical prior:

| inside `musx_redecode`, matched budget | nameable | swallowed | partial | segments |
|---|---|---|---|---|
| pass 1 flat (the shipped brick) | 0.6112 | 0.1834 | 0.6625 | 663 |
| + repetition prior | 0.6268 | 0.1781 | 0.6647 | 675 |
| **+ metrical (downbeat) prior** | **0.6451** | **0.1710** | 0.6645 | 701 |
| + metre AND repetition | 0.6450 | 0.1699 | 0.6629 | 700 |

**Repetition on top of metre is worth Δ = −0.0001.** Same efficiency per added segment
(0.130 vs 0.124 pp), it just spends less. 87 % of changes are on downbeats, so the profile's
high-probability phases *are* mostly downbeats.

**WHERE IT DOES WIN: as a detector of the changes we MISS** — 2.2× the metrical ranking at a
100-cut budget (37 vs 17 recovered of 438 missed), 1.5× at 500 (141 vs 94). Purity does not
reward that, because splitting a long slice anywhere raises purity whether or not the cut is
real.

**LOUIS'S NEW QUESTION ANSWERED — and it re-ranks the whole effort.**
**63.5 % of merge time is between NEAR chords (≥2 shared tones)**; NEAR merges are 11.4 % of
playing time, DISTANT merges only **6.6 %**. The causal direction holds: a change to a chord
sharing **3** tones is missed **64 %** of the time vs **47 %** for 1 shared tone (odds ratio
1.28 near-vs-distant, 1.96 for sh=3 vs sh=1) — the chroma barely moves, so the acoustics
genuinely cannot see it. And decisively:

| config | plain nameable | **tone-weighted** | distant merges |
|---|---|---|---|
| shipped chart | 0.5887 | **0.9032** | 6.6 % |
| musx_redecode flat | 0.6112 | 0.9014 | 7.3 % |
| + metrical prior | 0.6451 | 0.9101 | 6.5 % |
| + repetition prior | 0.6268 | 0.9048 | 6.9 % |

**Plain nameable time moves +5.6 pp across every lever; harmonically-weighted coverage moves
+0.9 pp.** The chart already covers **90.3 %** of playing time tone-wise. **The real residue is
the 6.6 % of playing time lost to DISTANT merges — i–V and ii°–i confusions on Blue Bossa
(`D:hdim7→C:min7` 19.1 s, `C:min7→G:7` 18.2 s, `G:7→C:min7` 17.9 s) — and not one lever tried
this session moves it.** Segmentation has a low harmonic ceiling; this is a chord-identity /
fifth-confusion problem, which is where the 2026-07-26 diagnosis already pointed.

Does the prior fire on the ambiguous cases specifically? **Yes, weakly**: 72.1 % of what it
recovers is NEAR vs 68.0 % in the pool it draws from.

**THREE MECHANISMS DIAGNOSED (the reusable part of this session).**
1. **Self-confirmation.** A profile estimated from the decode it modifies moves the metric by
   exactly 0.000. Measured: it is the *only* source whose profile is WORSE than metre on the
   beats needing repair (AUC −0.029). The source must be a **more sensitive** detector — a
   deliberately over-segmented decode (penalty 2) is best measured (+0.059 vs metre), which is
   precisely Louis's *"unitairement c'est difficile, mais la répétition aide"*.
2. **Phase slip.** Indexing phase on a uniform-in-time grid loses the form on long takes:
   blue_bossa AUC **0.475 → 0.715** when phase is indexed on real detected beats (pooled
   0.714 → 0.790).
3. **A bad phase estimate refuted the wrong thing.** Downbeat-graded transition costs were
   marked REFUTED on 2026-07-27 (0.6627 vs 0.6644). With a downbeat phase estimated from the
   model's own dense changes (mod 4 on real beats) they are **positive** here: partial 0.6657
   vs 0.6625 flat, nameable +3.4 pp. That is this session's most useful *positive* number and
   it is not the repetition prior.

**⚠ ENVIRONMENT LANDMINE (unrelated, please act on it).** `python <path>/script.py` silently
imports the **stale `~/harmonia` clone** (17 `.py` files vs this repo's 172; 8 of the 17
differ), because the venv's editable install points there and Python 3.11+ omits cwd from
`sys.path` for script runs. Use `PYTHONPATH="$PWD"`, or re-`pip install -e .` from this repo.

**NEW files:** `harmonia/models/repetition_prior.py` (default OFF, unwired),
`tests/test_repetition_prior.py` (18/18; 42 passed with the regression net), this log,
`docs/research_sessions/repetition_prior_2026-07-27.png`, `…_2026-07-27.html`, and
`scratchpad/rep_*.py` + `scratchpad/rep_charts/` (experiment code + cached charts).
Nothing wired, nothing committed, no guarded file touched.

**NEXT STEP (ranked).**
1. **Take the free win first:** wire the *metrical* graded cost into `musx_redecode` using the
   self-estimated downbeat phase (+3.4 pp nameable, +0.32 pp partial, 0 regressions). It is
   the same code path the repetition prior uses; only the level assignment changes.
2. **Retarget on DISTANT merges (6.6 % of playing time).** They are i–V/ii°–i confusions, not
   boundary errors — the fifth-discriminator / bass-informed-root lane (P1a, P4), not this one.
3. **If the repetition prior is revisited**, use it as a *candidate scorer* feeding something
   that can verify a change acoustically (it recovers 2× the real missed changes), never as a
   purity objective — and note that `blue_bossa` and `georgia` fail its gate outright.
4. Report **tone-weighted purity alongside plain purity** from now on; the 0.589-vs-0.903 gap
   is the difference between "how tidy is the chart" and "how wrong is it musically".

---

## Brief, restated as a numbered spec (Phase 1)

1. **Target.** SEGMENTATION ONLY (identity forgotten). Metric = **PURITY**: is the real
   chord the large majority (≥80 %) of the slice we cut? Headline = **nameable time** =
   fraction of GT playing time sitting inside a ≥80 %-pure predicted slice. Baseline
   **62.5 %** (commit `1018ee2`), merges 22 %, straddles 15 %. **Target the MERGE failure.**
2. **Budget.** Not given in hours → self-imposed ~5 h from 15:57, ~30-min checkpoints.
3. **Integration point.** NEW default-OFF brick + tests. Must COMPOSE with
   `harmonia/models/musx_redecode.py` (+2.20 pp partial), not replace it.
4. **Already refuted — do not re-run.** chroma novelty, onset strength, HPSS, majority vote,
   metrical snap, within-beat weighting, the fusion DBN persistence prior, the semi-Markov
   iReal duration prior (63.5 % of chart chords last one bar → over-merges).
5. **Constraints.** No commits/adds/stashes. Never edit `chord_pipeline_v1.py`,
   `stages/chord_head.py`, `eval/*`, `golden/*`, `align/*`, `dataset/*`, `serving/*`, the
   vendored clone, `docs/known_issues.md`. Disk floor 1.5 GiB. N=7 for audio-side claims.

## Phase 2 — the premise screen (cheap, symbolic, no audio)

**Premise:** does the chord-change *rhythm* (binary change/no-change over beat positions)
actually repeat within these recordings, **above the trivial metrical base rate**?

Deciding rule: if the lift over a position-conditional (bar-position) prior is ~0, the lever
is dead and this session stops there.

---

## Log

### 15:57–16:10 — context load + grid calibration pin (CLAUDE.md #1)

Golden GT bar grids are near-uniform; `downbeat_times` interval → implied 4/4 tempo:

| song | bar Δt (s) | implied BPM (4 beats/bar) | external check |
|---|---|---|---|
| bein_green | 3.213 | 74.7 | ballad, plausible |
| blue_bossa | 1.400 | 171.4 | fast bossa, plausible |
| **blue_bossa_backing** | **1.600** | **150.0** | **filename says `150bpm_backing_track` ✔ (hard pin)** |
| close_to_you | 2.714 | 88.4 | ~90, plausible |
| every_breath_you_take | 2.075 | 115.7 | ~117, plausible |
| georgia_on_my_mind | 3.768 | 63.7 | ~64, plausible |
| stand_by_me | 2.008 | 119.5 | ~118, plausible |

The backing track's grid is exact to 0.0 % against its own title → the bar grid is 4 beats,
not 3 or 8, and the downbeat list is bars. Pin passes.

### 15:59 — ⚠ ENVIRONMENT LANDMINE FOUND (error-pattern #1 class) — report to Louis

`python <path>/script.py` **silently imports the STALE `~/harmonia` clone**, not this repo.
Python 3.11+ does not put the cwd on `sys.path` for script execution; `sys.path[0]` is the
*script's* directory, and the venv's editable install
(`.venv/.../__editable___harmonia_0_1_0_finder.py`) points at `/Users/vincente/harmonia/harmonia`.

- The stale clone has **17 `.py` files vs this repo's 172**, and **8 of the 17 differ**.
- Reproduced: `python scratchpad/x.py` → `harmonia.__file__ = /Users/vincente/harmonia/...`;
  `python -c "import harmonia"` from the repo root → the repo copy. Same interpreter.
- It bit this session: `POP909Song.is_downbeat` (added after 2026-07-17) did not exist on the
  stale class, so the screen crashed. A crash is the *lucky* outcome — the dangerous case is a
  module that exists in both and quietly runs July-17 logic.
- **Workaround used everywhere below:** `PYTHONPATH="$PWD" .venv/bin/python …`.
- Suggested permanent fix (not applied — outside brief): `pip install -e .` from THIS repo, or
  delete `~/harmonia/harmonia`.

### 15:58–16:03 — PREMISE SCREEN, arm A: the 7 frozen songs (`scratchpad/rep_premise.py`)

Change vector = binary change/no-change per beat slot, bars from the frozen downbeat grid.
Adjacent identical labels are merged first (a repeated label is not a boundary).

**Base rate (state this next to every lift): 87.1 % of GT chord changes fall on a DOWNBEAT;
100 % fall on an integer beat.** Per-slot: p0 **0.7195**, p1 0.0034, p2 **0.1007**, p3 0.0023.
(Both facts are partly a GT-construction artifact — the frozen GT was built by tiling an iReal
chart onto the bar grid — so this arm cannot answer "do real changes fall on beats". It *can*
answer "does the pattern repeat", and it is the target we are scored against.)

**Repetition, pooled over the 7 (n = 2 640–3 508 slot pairs), conditioned on slot position so
the metrical prior cannot claim the credit:**

| lag (bars) | P(chg \| chg at b−k) | P(chg \| no chg) | lift | H(pos) bits | H(pos,lag) | **MI\|pos** |
|---|---|---|---|---|---|---|
| 1 | 0.621 | 0.100 | 6.2 | 0.3445 | 0.3341 | 0.0105 |
| 2 | 0.698 | 0.082 | 8.6 | 0.3440 | 0.3366 | 0.0074 |
| 4 | 0.750 | 0.068 | 11.0 | 0.3419 | 0.3181 | 0.0238 |
| 8 | 0.776 | 0.057 | 13.7 | 0.3340 | 0.2938 | 0.0402 |
| **16** | **0.937** | **0.014** | **69.6** | 0.3316 | **0.1270** | **0.2045** |
| **32** | **0.938** | **0.011** | **86.0** | 0.3196 | **0.1135** | **0.2061** |

At the form lag the conditional entropy of "is there a change here" falls by **62 %**.
Per song (MI at the song's own form period): bein_green **0.446** · blue_bossa **0.281** ·
backing **0.267** · stand_by_me **0.250** · every_breath 0.148 · close_to_you 0.034 ·
georgia 0.018. **The signal is per-song and concentrated at the song's own period** — short
lags (1–4 bars) are nearly worthless pooled.

**The lever points at the failure.** `bein_green` is the benchmark's disaster (28.5 % nameable,
we UNDER-cut) and it has the *strongest* repetition (0.446 bits/slot at lag 32).

### 16:03–16:05 — PREMISE SCREEN, arm B (non-circular): POP909, N=880 usable of 909

Why: arm A's GT is chart-derived. POP909's chord onsets are independent per-song annotations
and its beat/downbeat grid is real GT (`beat_midi.txt` col 3).
Measurement bug caught first (CLAUDE.md #1): **44.7 % of adjacent POP909 chord events carry the
identical label** (it re-emits the held chord every bar). Merged before counting; the numbers
below are post-merge (the pre-merge version inflated P(change\|downbeat) to 0.99).

- **Base rate: 57.9 % of changes on a downbeat**; P(chg\|beat1) 0.845, beat2 0.107,
  beat3 **0.443**, beat4 0.066. Pop harmonic rhythm is much denser than the jazz benchmark.
- POP909's onsets are *also* exactly beat-quantised (median \|err\| = 0.0000 beat, 100 % within
  0.05 beat) — so **neither corpus can test "do real changes fall on beats"**; both test
  repetition only. Said plainly so it is not over-claimed.

| lag (bars) | P(chg\|chg) | P(chg\|no chg) | lift | H(pos) | MI\|pos | % of H |
|---|---|---|---|---|---|---|
| 1 | 0.753 | 0.146 | 5.15 | 0.6098 | 0.0533 | 8.7 |
| 2 | 0.755 | 0.146 | 5.18 | 0.6097 | 0.0548 | 9.0 |
| 4 | 0.761 | 0.144 | 5.28 | 0.6100 | 0.0584 | 9.6 |
| **8** | 0.767 | 0.144 | **5.34** | 0.6111 | **0.0619** | **10.1** |
| 16 | 0.745 | 0.160 | 4.64 | 0.6148 | 0.0417 | 6.8 |
| 32 | 0.738 | 0.165 | 4.47 | 0.6217 | 0.0386 | 6.2 |

Per-song best lag is spread (1: 25 % · 2: 17 % · 4: 17 % · 8: 16 % · 16: 10 % · 32: 14 %).

**PREMISE VERDICT: PASSES, with a shape.** The repetition is real and ~5–7× in odds on both
corpora, but it is **per-song and lives at the song's own period**, not at a universal lag.
Pooled with a *fixed* lag it is worth 6–10 % of the entropy (POP909); at the *song's own*
period on the jazz benchmark it is worth up to 62 %. ⇒ Any brick must **detect the period per
song from its own output** (GT-free) and gate on periodicity strength, exactly like
`musx_redecode`'s GT-free latency selector.

### 16:05–16:09 — BASELINE: purity harness (`scratchpad/rep_purity.py`), shipped chart

Ran the shipped pipeline on the 7 songs, cached the charts to `scratchpad/rep_charts/*.json`
(chords + real beat_times + sections), deleted the decoded WAVs. **683 predicted slices vs the
682 reported in commit `1018ee2` → the prediction reproduces.**

Metric (mine, stated explicitly because it does not match the commit's split exactly):
purity(slice) = largest single-GT-chord share of the *labelled* GT time inside the slice.
Each GT chord's time is `nameable` (purity ≥ 0.80 and this chord is the slice's dominant one),
`swallowed` (a *different* chord dominates the slice — the merge victim), or `straddle`
(dominates but the slice is mixed).

| song | nameable | swallowed | straddle | slices | mean slice (s) |
|---|---|---|---|---|---|
| bein_green | **0.2776** | 0.2819 | 0.4405 | 46 | 3.35 |
| blue_bossa | 0.4361 | 0.2556 | 0.3083 | 245 | 2.01 |
| blue_bossa_backing | 0.7468 | 0.1166 | 0.1366 | 166 | 1.85 |
| close_to_you | 0.5477 | 0.1752 | 0.2762 | 58 | 3.22 |
| every_breath_you_take | 0.7384 | 0.1445 | 0.1171 | 59 | 3.34 |
| georgia_on_my_mind | 0.6443 | 0.1417 | 0.1907 | 68 | 2.22 |
| stand_by_me | 0.8649 | 0.0601 | 0.0750 | 41 | 3.92 |
| **POOLED (time-weighted)** | **0.5887** | **0.1803** | **0.2287** | 683 | — |

ext-blind (C ≡ C6 ≡ Cmaj7): 0.6201 / 0.1656 / 0.2121 — the commit's 62.5 → 66.9 sensitivity
reproduces as 58.9 → 62.0, and `bein_green` lands at 27.8 % vs their 28.5 %. **My pooled number
is 3.6 pp below theirs because the swallow/straddle taxonomy differs; every delta below is
internal to this harness and directly comparable to this baseline row.**

### 16:09–16:13 — ⚠ PURITY IS GAMEABLE, so every number below is BUDGET-MATCHED

`scratchpad/rep_insert.py`. Inserting boundaries can only raise nameable time, so the
controls are mandatory:

| variant | nameable | swallowed | slices/GT chord | extra cuts |
|---|---|---|---|---|
| BASELINE (shipped) | 0.5895 | 0.1817 | 1.01 | 0 |
| **CONTROL — cut at EVERY beat** | **0.8696** | 0.0542 | 4.46 | 2543 |
| **CONTROL — cut at every downbeat** | **0.7491** | 0.1128 | 1.69 | 503 |
| CONTROL — cut at every GT change (oracle) | 0.8631 | 0.0679 | 1.45 | 328 |

**Cutting blindly on every beat scores 0.870 — above the GT oracle.** So "nameable time went
up" means nothing on its own; from here on everything is reported at a matched cut budget.

### 16:13–16:15 — ANGLE 1 & 2 at matched budget: the naive versions LOSE to metre

Ranking all candidate beats and taking the global top-K (`scratchpad/rep_frontier.py`):

| K | oracle | **downbeat×len** (trivial) | modulo-P profile | modulo×len | block-match(4 bar) |
|---|---|---|---|---|---|
| 100 | 0.6940 | 0.6470 | 0.6522 | 0.6350 | 0.6164 |
| 200 | 0.7655 | **0.6974** | 0.6783 | 0.6770 | 0.6318 |
| 500 | 0.8631 | **0.7491** | 0.7264 | 0.7401 | 0.6361 |
| 1000 | 0.8631 | 0.7491 (saturated) | 0.7729 | 0.7839 | 0.6361 |

Angle 1 (modulo profile from **our own cuts**) and angle 2 (block matching on predicted chord
labels) are a **wash-to-loss** against ranking downbeats by the length of the slice they split.
Not "it doesn't work" — a diagnosis was needed.

### 16:15–16:16 — ★ ROOT CAUSE: the profile SOURCE, not the idea

Hypothesis written before the test: *our own chart is the thing being repaired — it is already
merged, so a profile aggregated from it inherits exactly the misses we want to fix.* Test:
rebuild the same ranking from (a) the GT change vector (upper bound on a perfect profile) and
(b) the **cached raw music-x-lab `.lab`**, which is denser and independent of our segmenter.

| K | downbeat×len | modulo (ours) | **musx profile** | **musx profile ×len** | **GT profile** | GT profile ×len |
|---|---|---|---|---|---|---|
| 100 | 0.6470 | 0.6522 | 0.6594 | 0.6638 | **0.6991** | 0.6840 |
| 200 | 0.6974 | 0.6783 | 0.6747 | 0.7076 | 0.7328 | **0.7371** |
| 500 | 0.7491 | 0.7264 | 0.7416 | 0.7608 | **0.8110** | 0.8019 |
| 1000 | 0.7491 | 0.7729 | 0.8261 | 0.8168 | **0.8593** | 0.8548 |

**With an accurate profile the repetition ranking nearly reaches the per-cut GT oracle
(0.8593 vs 0.8631 at K=1000).** The information is real; our own cuts were too sparse to
express it. The GT-free musx-sourced profile beats the trivial metrical ranking at every
budget ≥ 100 (+1.0 → +6.8 pp).

### 16:16–16:17 — the base-rate control that decides whether this is "repetition" at all

Same change source, **metre-only profile (P = 4) vs form-period profile**, so the only
difference is whether the position index runs over the bar or over the form:

| K | musx metre | musx form | musx metre×len | **musx form×len** | GT metre×len | **GT form×len** |
|---|---|---|---|---|---|---|
| 100 | 0.6738 | 0.6672 | 0.6440 | 0.6636 | 0.6461 | 0.6756 |
| 200 | 0.6873 | 0.6824 | 0.6941 | 0.7087 | 0.7094 | 0.7420 |
| 500 | 0.7407 | 0.7497 | 0.7521 | 0.7643 | 0.7670 | 0.8091 |
| 1000 | 0.7935 | 0.8254 | 0.8149 | 0.8237 | 0.8476 | 0.8693 |

**Isolated repetition effect = +1.0 to +4.5 pp of nameable time over a metre-only profile
built from the identical source**, consistently for K ≥ 150 (below that, metre wins). Real,
modest, and much smaller than the effect of changing the change SOURCE.

### 16:17–16:22 — ANGLE 3 (composed with `musx_redecode`) — a NO-OP, and the reason matters

`scratchpad/rep_redecode.py`. The vendored `XHMMDecoder` already grades change cost per beat
(`beat_arr` 2/3/4 → `beat_trans_penalty[0/1/2]`); `musx_redecode` feeds it a flat grid.
Pass 1 = flat beat-aware re-decode; pass 2 = re-decode with cost LOW where the song's own
repetition profile says changes recur, HIGH where they never do.

Control passes: pass 1 standalone reproduces the previous session (partial **0.6625**,
nameable 0.6112 vs the shipped chart's 0.5895 — the re-decode alone is worth **+2.2 pp of
nameable time**).

**First run had a silent bug** (CLAUDE.md #1): on songs the gate rejected, the profile was
constant, so `f >= hi` and `f <= lo` were both true everywhere and the second assignment made
*every* beat expensive — blue_bossa lost 34 segments and −11.4 pp nameable. Gated-off now
returns pass 1 bit-for-bit.

With the gate correct, over 4 (gate, cost-triple) settings:

| | nameable | swallowed | partial | root | segments |
|---|---|---|---|---|---|
| pass 1 (flat) | 0.6112 | 0.1834 | 0.6625 | 0.7436 | 663 |
| + repetition-graded costs | 0.6095–0.6098 | 0.1839 | 0.6623–0.6629 | 0.7430–0.7436 | 662–666 |

**Δ ≈ 0.000 on every metric.** Only 2 of 7 songs change at all; `blue_bossa_backing`
(gain 0.475 — the strongest repetition in the set) and `stand_by_me` (0.169) decode
*identically*.

**Diagnosed mechanism (written before the next test): the prior is SELF-CONFIRMING.** It is
estimated from pass 1's own boundaries, so making those positions cheaper and the others
dearer cannot move the Viterbi argmax — it re-scores the path it came from. A repetition prior
only carries information if bar *b*'s prior is estimated **without bar *b***.
→ Next test: leave-one-cycle-out profiles, and a cost floor low enough to actually create a
change where the emission is ambiguous.

### 16:22–16:24 — decoder responsiveness pin, and a second silent bug

Before believing "no-op", pinned that the graded cost reaches the decoder at all
(`beat_arr` 2/3/4 → `beat_trans_penalty[0/1/2]`, read off the vendored source, not assumed).
On `blue_bossa_backing`: trip (40,40,40) → 150 segments (== flat, correct control);
(15,40,100) → 165; (5,40,100) → 199; (0,40,100) → 274. **The mechanism works.**
The no-op came from a second bug: the profiles are strongly bimodal, so
`quantile(f,0.33) == quantile(f,0.66)` and my degeneracy guard silently switched the prior
off on exactly the songs with the *strongest* repetition. Replaced by absolute thresholds.

### 16:24–16:26 — leave-one-cycle-out — still ≈ 0

`loco_profile` (beat *i* excluded from its own phase estimate) plus absolute thresholds, over
5 configurations: nameable 0.6089–0.6118 vs flat 0.6112; partial 0.6607–0.6633 vs 0.6625.
Still nothing. LOCO alone does not fix self-confirmation, because with ~10 cycles the other
cycles already agree with pass 1.

### 16:26–16:28 — ★ WHICH SOURCE makes a usable profile (`scratchpad/rep_source.py`)

If the prior must not come from the decode it corrects, where should it come from? Louis's own
framing answers it: *"unitairement c'est difficile"* — take a **deliberately unreliable,
over-segmented** detector and let repetition clean it up. Five cached sources, scored by how
well their LOCO phase profile predicts GT change positions (AUC), with the **metre-only (P=4)
profile from the identical source** as the base-rate control:

| profile source | AUC form | AUC metre | Δ | on beats we did NOT cut: form | metre | Δ |
|---|---|---|---|---|---|---|
| pass40 (the decode itself — self-sourced) | 0.672 | 0.641 | +0.031 | 0.588 | 0.617 | **−0.029** |
| pass5 (over-segmented) | 0.706 | 0.672 | +0.034 | 0.667 | 0.670 | −0.003 |
| **pass2 (very over-segmented, "unreliable")** | **0.714** | 0.646 | **+0.067** | **0.688** | 0.628 | **+0.059** |
| raw musx `.lab` | 0.690 | 0.664 | +0.026 | 0.636 | 0.623 | +0.013 |
| 5-fold `.lab` vote | 0.689 | 0.648 | +0.041 | 0.629 | 0.599 | +0.031 |

**The self-sourced prior is the only one that is WORSE than metre on the beats we need to
repair (−0.029).** The noisiest detector is the best source (+0.059). Louis's mechanism,
measured.

### 16:28–16:30 — ★ second root cause: PHASE SLIP on the uniform grid

`blue_bossa` scored AUC ≈ 0.48 (chance) for *every* source, though its GT has strong 16-bar
structure. Hypothesis: the phase is indexed on a uniform-in-time grid, and over 352 bars a
small tempo error slips the modulo index. Test — index the phase on the chart's **real**
(non-uniform) detected beats instead:

| song | AUC(form), uniform grid | **real beats** |
|---|---|---|
| blue_bossa | 0.475 | **0.715** |
| close_to_you | 0.709 | **0.926** |
| stand_by_me | 0.891 | **1.000** |
| georgia_on_my_mind | 0.512 | **0.665** |
| **pooled** | 0.714 | **0.790** |

Confirmed. But the metrical control improves too (uncut-beat metre AUC 0.628 → 0.731), so the
*isolated* repetition margin shrinks from +0.059 to **+0.031**.

### 16:30–16:33 — the corrected prior in the decoder: a real but small effect…

Sensitive source (penalty 2) + real-beat phase + LOCO, 12 (cost, threshold) settings.
Best joint setting `trip=(15,40,40)` (insert-only: recurring positions cheaper, nothing ever
penalised), `hi_p=0.55`:

| | nameable | swallowed | partial | root | segments | per-song regressions |
|---|---|---|---|---|---|---|
| pass 1 flat | 0.6112 | 0.1834 | 0.6625 | 0.7436 | 663 | — |
| **+ repetition prior** | **0.6267** | **0.1791** | **0.6657** | 0.7450 | 674 | **0** |

+1.55 pp nameable, −0.43 pp merge time, +0.32 pp partial, 11 extra segments, no song worse.

### 16:33–16:35 — …that the BASE RATE matches exactly. The in-decoder lever is REFUTED.

Same decoder, same cost triple, but the cheap beats chosen by the **downbeat phase** instead
(GT-free, estimated from the sensitive detector's own changes mod 4 on real beats):

| config | nameable | swallowed | partial | root | segments | pp nameable per added segment |
|---|---|---|---|---|---|---|
| flat | 0.6112 | 0.1834 | 0.6625 | 0.7436 | 663 | — |
| repetition (15,40,40) | 0.6268 | 0.1781 | 0.6647 | 0.7447 | 675 | **0.130** |
| **metre (15,40,40)** | **0.6434** | 0.1720 | **0.6657** | 0.7473 | 689 | **0.124** |
| metre (10,40,40) | 0.6451 | 0.1710 | 0.6645 | 0.7475 | 701 | 0.089 |
| **metre + repetition, 3-level (10,20,40)** | **0.6450** | 0.1699 | 0.6629 | 0.7465 | **700** | 0.091 |

**At matched budget the composition is worth Δ = −0.0001**: metre alone 0.6451 @ 701 segments
vs metre+repetition 0.6450 @ 700. Repetition and metre have the same efficiency per added
segment (0.130 vs 0.124) and repetition simply spends less. **Inside this decoder the
repetition prior is exactly the downbeat prior wearing a hat.** Mechanically obvious in
hindsight: 87 % of changes are on downbeats, so the profile's high-probability phases *are*
mostly downbeats.

Side finding worth keeping: a downbeat phase estimated from the **model's own dense change
detections** (mod 4 on real beats) turns the previously-refuted downbeat-graded cost
(2026-07-27: 0.6627 vs 0.6644 flat, negative) into a **positive** here
(partial 0.6657 vs 0.6625, nameable +3.4 pp). The earlier refutation was of a *bad phase
estimate* (Beat This! downbeats mapped onto the uniform grid, purity 0.30–0.50 on 3 of 7
songs), not of the idea.

### 16:35–16:37 — where the prior DOES win: as a change DETECTOR

Post-hoc insertion, ranking candidate beats and counting how many **missed GT chord changes**
(438 of 728 total; the shipped chart has no boundary within ½ beat of them) get recovered:

| budget K | **repetition profile** | same profile, bar position only | downbeat × slice length |
|---|---|---|---|
| 50 | **21** | 8 | 4 |
| 100 | **37** | 17 | 15 |
| 200 | **61** | 48 | 29 |
| 300 | **89** | 62 | 51 |
| 500 | **141** | 94 | 87 |

**~2.2× the metrical ranking at K=100, 1.5× at K=500.** The purity metric does not reward
this, because splitting a long slice anywhere raises purity whether or not the cut is at a
real change. As a *detector* the prior is clearly better; as a *purity maximiser* it is not.

### 16:37–16:45 — ★ LOUIS'S NEW QUESTION: how much of the merge loss is musically real?

*"Souvent quand on n'est pas sûr qu'un accord change, c'est sûrement qu'il change à quelque
chose de très proche."* Measured on the shipped chart, common tones between the swallowed GT
chord and the chord actually printed over it (`scratchpad/rep_neardist.py`):

| shared tones | merged time | share |
|---|---|---|
| 0 | 14.0 s | 4.7 % |
| 1 | 94.6 s | 31.8 % |
| 2 | 124.5 s | 41.9 % |
| 3 | 64.3 s | 21.6 % |
| **NEAR (≥2)** | **188.8 s** | **63.5 %** |
| **DISTANT (≤1)** | **108.6 s** | **36.5 %** |

**63.5 % of merge time is between NEAR chords.** In playing-time terms: NEAR merges are
**11.4 %** of playing time, DISTANT merges only **6.6 %**.

**And the claim's causal direction holds** — the changes we MISS are the near ones:

| shared tones between the two GT chords | n | missed | P(miss) |
|---|---|---|---|
| 0 | 32 | 18 | 0.562 |
| 1 | 233 | 110 | **0.472** |
| 2 | 346 | 177 | 0.512 |
| 3 | 117 | 75 | **0.641** |
| NEAR (≥2) | 463 | 252 | **0.544** |
| DISTANT (≤1) | 265 | 128 | **0.483** |

Odds ratio near-vs-distant **1.28**; sh=3 vs sh=1 is **1.96**. A change to a chord sharing 3
tones (Dm7♭5→Fm7, Cmaj7→Am7, C→C6) is missed 64 % of the time — the chroma barely moves, so
the acoustics genuinely cannot see it.

**Plain vs tone-weighted purity — this re-ranks the whole effort.** Tone-weighted credit =
`|tones(true) ∩ tones(printed)| / |tones(true)|` on every second of playing time:

| config | plain nameable | swallowed | straddle | **TONE-WEIGHTED** | distant merges (% playing time) |
|---|---|---|---|---|---|
| shipped chart | 0.5887 | 0.1803 | 0.2287 | **0.9032** | 6.6 % |
| musx_redecode (flat) | 0.6112 | 0.1834 | 0.2054 | **0.9014** | 7.3 % |
| + metrical prior | **0.6451** | 0.1710 | 0.1839 | **0.9101** | 6.5 % |
| + repetition prior | 0.6268 | 0.1781 | 0.1951 | **0.9048** | 6.9 % |

**Plain nameable time moves +5.6 pp across these levers; tone-weighted coverage moves
+0.9 pp.** The printed chart already covers **90.3 %** of playing time harmonically. Nearly
half the merge loss (47.6 % of the swallowed 18.0 %) is recovered by tone weighting alone.
**The genuinely wrong residue is the DISTANT merges — 6.6 % of playing time — and not one of
the levers tried this session moves it (6.5–7.3 %).**

Worst distant merges, all jazz function errors on Blue Bossa/backing:
`D:hdim7 → C:min7` 19.1 s, `C:min7 → G:7` 18.2 s, `G:7 → C:min7` 17.9 s — i–V and ii°–i
confusions, exactly the fifth-related family the 2026-07-26 diagnosis already flagged.

**Does the prior fire on the ambiguous cases?** Yes, but weakly: of the 61 missed changes it
recovers at K=200, **72.1 % are NEAR** vs **68.0 %** in the missed pool it draws from, and
73.0 % at K=500. It is ~4 pp more near-biased than chance — a real but small confirmation of
the mechanism Louis proposed.

### 16:45 — brick packaged + verified

`harmonia/models/repetition_prior.py` (NEW, default OFF via `HARMONIA_REPETITION_PRIOR`,
**unwired**) and `tests/test_repetition_prior.py` (NEW, **18/18 green**, no audio/weights
needed). Regression net: `test_repetition_prior` + `test_musx_redecode` +
`test_chord_head_parity` + `test_calibration_pins` = **42 passed**.

**Packaged-brick reproduction check (CLAUDE.md #6) — 7/7 identical** to the scratchpad
computation (same period, same gain to 1e-9, `max|Δprior| = 0.00e+00` on every gated-on song).
Per-song gate on the raw-`.lab` source: bein_green P=16 (+0.119) ON · blue_bossa P=16 (+0.012)
**OFF** · backing P=128 (+0.199) ON · close_to_you P=8 (+0.036) ON · every_breath P=32
(+0.528) ON · georgia P=8 (+0.004) **OFF** · stand_by_me P=32 (+0.219) ON.

**Artifacts:** `docs/research_sessions/repetition_prior_2026-07-27.png` (4 panels: the premise
vs lag on both corpora; the budget-matched purity frontier with the "cut at every beat" ceiling
drawn on it; the merge-distance histogram; the change-recovery curves) and
`docs/research_sessions/repetition_prior_2026-07-27.html` (phone-first: per-song detected
period + gate, and every one of the 61 recovered changes with its from/to chords and shared-tone
count).
