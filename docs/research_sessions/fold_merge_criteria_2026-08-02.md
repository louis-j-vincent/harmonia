# Merge criteria for section folding — measurement session, 2026-08-02

## SUMMARY (read this, the rest is the audit trail)

**Artifact:** `docs/fold_criteria_billboard.html` (tables + histograms + readable
examples) and `docs/plots/fold_policy_pareto.png`.

**Culprit for Norah:** not the posterior fold. `fold_letter_groups` refused to
fold letter A (stack coherence 0.61 < 0.85 — the shipped acoustic gate was
RIGHT). `minimal_fold` then read `period: 2` out of that *refused* report and
wrote a 2-bar cell twice. Plus the section detector had already put 62 of 66
bars under one letter in six occurrences of 8/6/12/12/16/8 bars.

**The measured cost of that block:** it misrepresents **72.6%** of letter A's
playing time. Writing the whole first occurrence instead (8 bars) would be
37.1%. Merging only the pair that agrees (A3+A4, both 12 bars) is 0.0% harm for
50 written bars instead of 62.

**Criterion menu, best to worst** (Billboard, 889 tracks, target = "does the
written block reproduce what is actually played?", operating point = 2% false
merges):

| # | criterion | what it is | AUC | recall @2% FP |
|---|---|---|---|---|
| 1 | `chroma_align` | 64-slot chord-tone chroma cosine, time-normalised | 0.922 | 63.2% |
| 2 | `chroma_cv_inv` | 1 − CV across the two spans, per slot | 0.932 | 65.3% |
| 3 | `len_agree` (with #1) | occurrence length ratio | 0.782 | +4 pp on top of #1 |
| 4 | `chroma_lag_unstretched` | cosine at equal offset, no stretch (= PERIOD_MIN_SCORE's shape) | **0.947** | 34.5% |
| 5 | `seq_sim` | 1 − normalised Levenshtein on chord labels | 0.892 | 43.0% (letter target) |
| 6 | `tile_cov` | fraction of time the tiled template gets right | 0.897 | 17.7% (letter target) |
| — | `hr_sig` / `chroma_hr` | harmonic-rhythm signature (WHERE changes fall) | 0.728 / 0.787 | **dead — see below** |
| — | `ct_bag_cos`, `chordset_jac`, `bigram_jac` | order-blind chord-content | 0.87–0.89 | **0% — saturates** |

**Recommended pick (with a caveat that changes the order of work):**
`chroma_align >= 0.85 AND len_agree >= 0.95`, applied per letter group by greedy
agglomeration. On Billboard that is **68% of the song written, 4.4% harm** vs the
current all-same-letter policy's **36% written, 20% harm**. On Norah it merges
exactly one pair (A3+A4) and refuses the other 14 — the right answer.

**But applied to our own six charts it barely folds anything** (49 bars written
today -> 328 under the rule), because our detector's same-letter occurrences
disagree in length far more than Billboard's annotators': **61.7% of Billboard's
same-letter pairs agree in length within 5%, vs 26.7% of ours.** The binding
constraint on our data is the section detector, not the merge criterion. See
arm 6 at the bottom.

**Three negative results worth keeping:**
1. The harmonic-rhythm signature is the *worst* criterion measured. Pop sections
   share their change grid whatever the harmony is.
2. Every length-invariant, order-blind chord similarity has no usable threshold:
   1.9% of different-letter pairs are chord-IDENTICAL (Cyndi Lauper's She Bop:
   A and D are both `Am F G Am`, 28 s each).
3. Of the three shipped constants, only **STACK_COHERENCE = 0.85 survives**
   (it lands on the frontier). PERIOD_MIN_SCORE = 0.80 is loose (6.3% harm vs
   4.9% at 0.85); the CV gate at its shipped permissiveness is close to no gate
   at all (16.3% harm). Caveat below — the CV number does not transfer
   literally.


Budget 90 min (start 15:20 CEST). Deliverable = MEASUREMENT + PROPOSAL, no ship.
Brief: Louis wants *several candidate criteria* that separate spans which may be
written once from spans that must be written out, each *measured* on Billboard GT,
so he can pick. Constraint: do not touch `harmonia_min/*.py`.

## Spec restated (from the brief)

1. Target: a ranked comparison of >=4 candidate merge criteria, measured on
   Billboard GT (890 tracks), with the threshold that holds false-merge <= 2%
   and the true-merge recall surviving at that threshold.
2. Budget 90 min, log continuously.
3. Integration point: the section/fold layer (`harmonia_min/sections.py`
   letter grouping + `folding.py` `fold_letter_groups` / `minimal_fold`).
   Analysis only, under `scripts/` + `docs/`.
4. Already tried: acoustic constants PERIOD_MIN_SCORE=0.80, OUTLIER_Z=3.0,
   STACK_COHERENCE=0.85, CV_MAX=0.51 — all calibrated on 2-5 songs.
5. Hard: no edits to harmonia_min, no port 7771, explicit `git add` paths.

---

## 15:25 — CULPRIT DIAGNOSIS (Norah, Don't Know Why). Answer: (a) AND (c), not (b).

The brief offered two candidates. The evidence says the truth is a third thing.

**Facts from `harmonia_min/state/charts/min_norah_jones_don_t_know_why.json`:**

```
nBars 66
sections: LA  label A  reps 6  barRanges [[0,7],[8,13],[14,25],[30,41],[42,57],[58,65]]  bars written: 4
          LB  label B  reps 1  barRanges [[26,29]]                                        bars written: 4
fold report: {"A": {"period": 2, "reason": "stack incoherent (min median pairwise 0.61)"},
              "B": {"period": null, "reason": "no confident loop"}}
```

**(a) The section detector is culprit #1.** It put 62 of 66 bars under ONE letter,
in six occurrences of length 8 / 6 / 12 / 12 / 16 / 8 bars. The letter grouping
(`sections.py:338-368`) merges two segments when their blurred-SSM cross-block
similarity is close to their own internal similarity (`LABEL_COS` ratio). It has
**no length term and no sequence-order term** — nothing in it can notice that a
6-bar span and a 16-bar span are not the same section. Don't Know Why is one
harmonic colour end to end (Bb major, ii-V material), which is the exact failure
the function's own comment says it was written to escape ("every This Love segment
is C-minor material -> one letter for the whole song"). It escaped it for This
Love, not in general.

**(b) `fold_letter_groups` did NOT overwrite anything — it correctly refused.**
The acoustic stack gate fired: median pairwise cosine 0.61 < STACK_COHERENCE 0.85.
Phase 1 wrote zero bars for letter A. **The shipped acoustic criterion got this
song right.**

**(c) `minimal_fold` (display layer) is the proximate culprit.** It reads
`P = fold_report[L]["period"]` — and the *refused* report still carries
`"period": 2`. With P=2 and unequal pass lengths it takes the
`elif P:` branch (`folding.py:504-507`) and writes the 2-bar cell twice.
Reproduced exactly, with bar provenance instrumented:

```
$ minimal_fold(sections, bars, grid, fold_report)   # bars tagged with source index
A reps 6  source bars in block: [0, 1, 0, 1]
B reps 1  source bars in block: [26, 27, 28, 29]
```

So the four printed A bars are source bars 0,1,0,1 — Louis's exact words, "you
grouped them as two-times-two identical bars". Source bars 2..7 (the
`Gm7 C7 | F7 ...` he says is really there) are **never written to the chart at
all**; they only survive in `barRanges`.

**Consequence for the fix:** a *criterion* alone will not fix this song, because
the criterion that should have stopped it (STACK_COHERENCE) already said no and
was ignored. Two separate defects: a gate that is not consulted (c), and a letter
grouping with no length/sequence term (a). The criteria study below addresses (a).

Script: `scripts/fold_criteria_billboard.py`. Repro of (c) is in this log.

---

## 15:33 — Billboard extraction running

890 tracks, pair-level decision unit:
positive = same SALAMI letter, prime = A vs A', negative = different letter,
hard negative = different letter with chord-set Jaccard >= 0.6.
Two arms: chord LABELS (upper bound) and McGill's NNLS `bothchroma.csv`
(real audio features, no decoder in the loop — a better "noisy" arm than our 6
charts, and it is the same substrate the shipped constants live on).

## 15:30 — First corpus result. Two criterion FAMILIES are dead, and I have to reframe the target.

33 954 pairs / 673 tracks (9 075 positive, 4 689 HARD negative, 18 258 easy
negative, 1 932 A~A' prime).

| criterion | AUC vs all neg | AUC vs HARD | median pos | median HARD | τ@2% FP | recall there |
|---|---|---|---|---|---|---|
| `tile_cov` (template tiled, agreed time) | 0.897 | 0.772 | 0.922 | 0.426 | 0.984 | 17.7% |
| `seq_sim` (1 − normalised Levenshtein) | 0.892 | 0.740 | 0.857 | 0.500 | 0.889 | **43.0%** |
| `ct_bag_cos` (duration-weighted chord tones) | 0.893 | 0.748 | 1.000 | 0.968 | 1.000 | **0%** |
| `bigram_jac` | 0.876 | 0.713 | 0.800 | 0.429 | 1.000 | 0% |
| `chordset_jac` | 0.865 | 0.582 | 1.000 | 0.750 | 1.000 | 0% |
| `ct_align` (64-slot chord-tone cos) | 0.848 | 0.755 | 0.974 | 0.565 | 1.000 | 0% |
| `nchord_agree` | 0.779 | 0.693 | 0.909 | 0.667 | 1.000 | 0% |
| `hr_sig` (harmonic-rhythm signature) | 0.728 | 0.686 | 0.952 | 0.882 | 1.000 | 0% |
| `len_agree` | 0.725 | 0.695 | 0.987 | 0.671 | 0.999 | 6.4% |

**Dead end #1 — the harmonic-rhythm signature.** `hr_sig` (where the changes
fall, chord identity discarded) is the *worst* criterion measured: AUC 0.728
overall, 0.686 against hard negatives, median 0.952 on positives vs **0.882 on
pairs that must not merge**. Pop sections share their change grid (one chord per
bar or per two bars) whatever the harmony is — the channel carries almost no
merge information. That kills brief candidate #4 as a gate. Its only defensible
use is as a *tie-break*, and even there it adds nothing: `ct_align AND hr_sig`
scores 46.2% recall vs 53.9% for `len_agree AND ct_align`.

**Dead end #2 — every length-invariant, order-blind similarity SATURATES.**
`ct_bag_cos`, `chordset_jac`, `ct_align`, `bigram_jac`, `nchord_agree`, `hr_sig`
all have τ@2% = 1.000, i.e. **more than 2% of different-letter pairs are exactly
identical on that criterion**. No threshold exists. That is not a defect of the
criteria; it is a fact about pop music: verse and chorus routinely run the same
loop. Only `seq_sim` and `tile_cov` (order- AND length-sensitive) and `len_agree`
survive at all.

**Which means the SALAMI letter is the wrong ground truth for this decision.**
If a verse and a chorus have literally the same chords, writing them once is
*correct on a chord sheet* — the thing that distinguishes them is melody and
lyrics, which the chart does not show. Scoring against letters charges us for
merges that cost nothing. Next experiment (arm 4): keep the same pairs, change
the target to the actual harm — *does the template, tiled, reproduce what is
played?* (label-side `tile_cov` ≥ 0.95 = positive, < 0.80 = negative) and use
only the CHROMA criteria as predictors, since that is all the pipeline sees.


## 15:34 — Arm 4 (reframed target) + the prevalence numbers

Target changed from "same SALAMI letter" to "the block, tiled, reproduces >=95%
of the span's chord-tone TIME" (positive) vs "<80%" (negative). Predictors =
chroma only. 4 751 positive / 26 072 negative pairs.

| criterion | AUC | median pos | median neg | tau@2% FP | recall |
|---|---|---|---|---|---|
| `chroma_cv_inv` | 0.932 | 0.771 | 0.552 | 0.736 | **65.3%** |
| `chroma_align` | 0.922 | 0.897 | 0.639 | 0.863 | **63.2%** |
| `chroma_bag_cos` | 0.927 | 0.994 | 0.937 | 0.996 | 37.9% |
| `chroma_lag_unstretched` | **0.947** | 0.852 | 0.598 | 0.875 | 34.5% |
| `chroma_hr` | 0.787 | 0.717 | 0.556 | 0.808 | 19.3% |
| `len_agree` | 0.782 | 0.995 | 0.668 | 0.999 | 11.2% |

AND-rules: `chroma_cv_inv >= 0.711 AND len_agree >= 0.979` -> 67.6%;
`chroma_align >= 0.834 AND len_agree >= 0.979` -> 67.1%.

Note the inversion at #4: `chroma_lag_unstretched` has the BEST AUC (0.947) and
one of the worst 2%-operating points (34.5%). It ranks well on average and has a
fat right tail of false merges — exactly the wrong shape for an asymmetric loss.
Any criterion chosen on AUC alone would have picked it.

**Prevalence (why this matters at all), same-letter pairs in Billboard GT:**
* 38.3% differ in length by more than 5%; 28.5% by more than 20%.
* **75.9% of tracks (655 of 863) contain at least one same-letter pair of
  unequal length.** "Same letter" does not mean "same length" — confirmed, at
  three-quarters of the corpus.
* 32.7% of same-letter pairs: tiling the first occurrence over the second
  reproduces less than 80% of its time. **One same-letter pair in three cannot
  be written as one block without real damage.**

## 15:36 — Arm 5: the POLICY trade-off (what a criterion actually costs)

AUC does not tell Louis what to pick. Scored per letter group: greedy
agglomeration, decisions from chroma, harm scored on GT chord labels.
`scripts/fold_policy_tradeoff.py`, 889 tracks. Plot:
`docs/plots/fold_policy_pareto.png`.

| policy | chart length written | harm |
|---|---|---|
| `write_out` (every occurrence) | 100.0% | 2.1% (measurement floor) |
| `chroma_cv >= 0.83` | 90.3% | 2.5% |
| `align+len >= 0.90` | 75.0% | 3.5% |
| `align+len >= 0.87` | 69.9% | 4.1% |
| **`align+len >= 0.85`** | **68.1%** | **4.4%** |
| `align >= 0.85`  (= shipped STACK_COHERENCE) | 66.2% | 4.9% |
| `align >= 0.80`  (= shipped PERIOD_MIN_SCORE) | 61.2% | 6.3% |
| `equal_len` alone (Louis's under-fold rule) | 62.7% | 7.3% |
| `cv >= 0.49`  (= shipped CV_MAX 0.51) | 40.4% | 16.3% |
| `letter` (current: all same-letter occurrences share a block) | 36.1% | **20.2%** |

Read the plot down-left. Two things jump out:

1. **The current `letter` policy is nowhere near the frontier.** It writes 36%
   of the song and gets one fifth of the harmony wrong. Every gated policy at
   comparable compression is dramatically better.
2. **`equal_len` alone sits just off the frontier.** `align+len >= 0.83` writes
   66.9% for 4.6% harm; `equal_len` writes slightly less (62.7%) for noticeably
   more harm (7.3%). Length agreement is necessary but not sufficient — two
   same-letter, same-length occurrences still differ often enough to matter.
   Louis's under-fold rule is right in direction and wants a harmony term
   beside it, not instead of it.

**Caveat that must travel with the CV row (rule #4).** My `chroma_cv_inv` is
computed on **L1-normalised** chroma over 64 slots of a whole span; the shipped
`CV_MAX` is on **raw** half-bar chroma across stack members. The 0.51 -> 0.49
mapping is a scale coincidence, not a calibration. The *shape* (variance across
members) is the second-best criterion measured and deserves keeping; the
specific number 0.51 has NOT been shown to be wrong on its own substrate. That
recalibration is a separate measurement I did not run.

## What this session does NOT solve (rule #4)

* **It does not fix the section detector.** The letter grouping
  (`sections.py:338-368`) still has no length term and no order term, and it is
  what put 62 of Norah's 66 bars under one letter. Every criterion here operates
  on the *pairs the detector proposes*; none of them can invent the missing
  boundary at Norah's bar 4. A criterion would have refused the merge — it would
  not have produced the correct 4-bar A.
* **It does not decide the block's PHASE.** All criteria here compare spans that
  are already aligned at their start. `score_periods`'s documented gap (period
  without phase) is untouched.
* **`minimal_fold` reading `period` from a refused report is a code defect, not
  a threshold problem.** No choice of criterion fixes it.
* **No criterion here distinguishes a verse from a chorus when they share the
  same chords** — and on a chord sheet, nothing should. If Louis wants
  verse/chorus split even at identical harmony, that information is not in the
  chord domain and must come from elsewhere (vocal presence, timbre, energy).
* **Not measured: boundary/novelty strength at a proposed SPLIT point** (brief
  candidate #6) in its split form. Its pairwise proxy (`chroma_hr`,
  novelty-curve similarity) was measured and is dominated; the split form is a
  different question and is still open.
* **Billboard is 1950s-1990s pop/rock, not jazz.** Every number here should be
  re-checked before being trusted on a jazz chart with 32-bar AABA forms.

## Scripts

* `scripts/fold_criteria_billboard.py` — pair extraction (label + chroma arms)
* `scripts/fold_criteria_report.py` — ranking, plots, HTML
* `scripts/fold_criteria_norah.py` — the Norah verdict + policy comparison
* `scripts/fold_policy_tradeoff.py` — the compression/harm frontier

## 15:40 — Arm 6: the rule applied to OUR OWN six charts. It inverts the recommendation.

`scripts/fold_criteria_ourcharts.py`, rule = `chroma_align >= 0.85 AND
len_agree >= 0.95`, chroma from each song's own audio on the chart's own bar grid.

| chart | letter | occurrence lengths (bars) | rule's blocks | bars written today | under the rule |
|---|---|---|---|---|---|
| This Love | B | 8 8 8 8 8 | **1 block** (all merge, align 0.908–0.959) | 8 | 8 |
| This Love | A | 16 12 4 | 3 blocks | 4 | 32 |
| Let It Be | A | 36 20 14 | 3 blocks | 4 | 70 |
| Stand By Me | A | 20 8 5 16 | 4 blocks | 8 | 49 |
| Stand By Me | C | 10 7 13 | 3 blocks | 10 | 30 |
| She Will Be Loved | A | 20 16 | 2 blocks | 4 | 36 |
| She Will Be Loved | B | 13 6 27 | 3 blocks | 4 | 46 |
| Norah | A | 8 6 12 12 16 8 | 5 blocks (only 12+12 merge) | 4 | 50 |

Totals across the six charts: **49 bars written today, 328 under the rule, 372
if nothing folds at all.** The rule would fold away 12% of the redundancy where
today's pipeline claims to fold away 87%.

**Why: our own sections are not the same shape as Billboard's.**

| | same-letter pairs with `len_agree` >= 0.95 |
|---|---|
| Billboard GT (9 075 pairs) | **61.7%** |
| our 6 charts (45 pairs) | **26.7%** |

Billboard's annotators cut sections that mostly repeat at the same length. Our
detector does not — `36 20 14`, `20 8 5 16`, `16 12 4`. So on our data the
length term alone refuses almost everything, and the harmony term never gets a
say.

**This changes the recommendation.** Shipping `align+len` today would not make
the charts 4.4%-harmful and 68% long as on Billboard; it would make them ~90%
long, because the spans it is being asked to compare are mis-cut. The binding
constraint on OUR data is the section detector, not the merge criterion. The
one place the rule fires cleanly — This Love's five 8-bar B's, align 0.908-0.959
— is the one letter whose occurrences the detector cut consistently.

Note the harmony term does earn its place even on our data: Norah's occ1 and
occ6 are both 8 bars (`len_agree` 1.000) but `chroma_align` 0.751 — refused for
the right reason, they are genuinely different music.

**Order of work this implies:**
1. Make the section detector produce length-consistent occurrences (or make the
   fold split a letter into length-consistent sub-groups before it decides).
2. Only then does a merge criterion have well-formed pairs to judge, and the
   Billboard-calibrated thresholds transfer.
3. Independently and immediately: stop `minimal_fold` reading `period` from a
   fold report that refused. That is a one-line defect, not a research question,
   and it is what produced Norah's 72.6%-wrong block.

## 15:43 — Arm 7: a PREFIX rule (criterion #10, mine) — wins at the compact end on Billboard, fails on our charts

If a 12-bar A is a 16-bar A minus its tail, the honest chart writes the 16-bar
block once and lets the short pass stop early. Template = the LONGEST
occurrence, written at real time scale; a shorter one joins if the template's
PREFIX (same seconds from the start, unstretched) reproduces it. No tiling.
`scripts/fold_policy_prefix.py`, 889 tracks.

| policy | written | harm |
|---|---|---|
| `prefix >= 0.93` | 82.9% | 3.2% |
| `prefix >= 0.90` | 71.1% | 4.2% |
| `prefix >= 0.85` | 59.6% | 6.0% |
| `prefix >= 0.80` | 53.9% | 7.6% |
| `prefix >= 0.75` | 50.6% | 8.9% |

Against the tiling curve: **prefix dominates below ~64% written** (`prefix 0.85`
= 59.6%/6.0% beats `align 0.80`'s 61.2%/6.3% on both axes; `prefix 0.75` =
50.6%/8.9% beats `align 0.70`'s 49.7%/11.3%), and loses slightly above it
(`align+len 0.85` = 68.1%/4.4% vs `prefix 0.90` = 71.1%/4.2%). So: **if Louis
wants short charts, prefix-merge is the better mechanism; if he wants safe
charts, align+len is.** Both are on the plot.

**On our own six charts, prefix rescues almost nothing** — one extra merge
(She Will Be Loved B occ1+occ2). So the mis-cut spans are not prefixes of each
other either.

## 15:45 — Arm 8: the root cause of arm 6. Our occurrences disagree in PHASE, not just length.

Cheapest possible test: take each same-letter pair on our charts, compare the
first L bars of each (L = the shorter), then slide one occurrence by -4..+4 bars
and re-score `chroma_align`.

**11 of 45 pairs (24%) cross from "refuse" (<0.85) to "merge" (>=0.85) purely by
shifting 1-4 bars.** Examples:

| song | letter | pair | align at shift 0 | best align | shift (bars) |
|---|---|---|---|---|---|
| Stand By Me | C | occ1~occ2 | 0.529 | **0.927** | −2 |
| She Will Be Loved | B | occ2~occ3 | 0.782 | **0.974** | +4 |
| Stand By Me | A | occ3~occ4 | 0.591 | **0.882** | +3 |
| Stand By Me | C | occ2~occ3 | 0.603 | **0.898** | +1 |
| Norah | A | occ1~occ6 | 0.751 | **0.874** | −4 |
| Norah | A | occ1~occ2 | 0.714 | 0.845 | −4 |
| Let It Be | A | occ2~occ3 | 0.846 | 0.888 | −4 |

The detector's cuts land at different points inside the harmonic loop, so two
occurrences of the same music start on different bars of the cycle. This is the
documented unsolved remainder from `score_periods` — **period detected, phase
never** — showing up as a fold failure two stages downstream.

**This is the highest-value next lever, ahead of any threshold tuning:**
phase-align occurrences (cross-correlate the chord-tone chroma over ±P bars,
take the argmax) before comparing or merging them. On our data it would recover
about a quarter of the merges that the length/harmony gates currently refuse,
without touching a single threshold.

## Final ordering of recommendations for Louis

1. **One-line defect, ship whenever:** `minimal_fold` must not use `period` from
   a fold report that carries a `reason` (i.e. that refused). Norah's block is
   72.6% wrong because of this.
2. **Phase-align before merging** (arm 8). Biggest measured lever on our own
   data: 24% of refused pairs are actually the same music, off by 1-4 bars.
3. **Then pick a merge gate.** `chroma_align >= 0.85 AND len_agree >= 0.95` for
   safe charts (Billboard: 68% written / 4.4% harm); `prefix >= 0.85` for short
   charts (59.6% / 6.0%). Both massively beat today's all-same-letter policy
   (36% / 20.2%).
4. **Do not** spend time on harmonic-rhythm signatures or bag-of-chord-tones
   similarity as merge gates. Measured dead (see 15:30).

## 15:50 — Arm 9: is the phase search safe? Yes — and the corpus control confirms the diagnosis.

Arm 8's finding came from 45 pairs on 6 songs — hypothesis grade (rule #5). Two
things had to be checked before recommending it: (i) does giving the gate 9
chances to pass instead of 1 inflate false merges, and (ii) does the contrast
with Billboard hold, or is my shift code a no-op?

**(i) The phase search is free.** `align+len+phase(tau)` on Billboard, shifts
{0, ±1, ±2, ±3, ±4} × the track's median chord duration, with the block written
AND scored at the winning phase:

| tau | with phase search | without |
|---|---|---|
| 0.93 | 0.842 / 0.028 | 0.842 / 0.028 |
| 0.90 | 0.749 / 0.035 | 0.750 / 0.035 |
| 0.87 | 0.699 / 0.041 | 0.699 / 0.041 |
| 0.85 | 0.681 / 0.044 | 0.681 / 0.044 |
| 0.80 | 0.658 / 0.050 | 0.657 / 0.050 |
| 0.75 | 0.648 / 0.054 | 0.647 / 0.054 |

Identical to three decimals everywhere. The extra freedom does **not** buy false
merges — the search simply returns shift 0 when the spans are already aligned.

**(ii) The contrast is real and the code is not a no-op.** Same rescue test as
arm 8, run on all 10 606 Billboard same-letter pairs:

| | a nonzero shift improves align by >0.03 | a shift rescues the pair (<0.85 → ≥0.85) |
|---|---|---|
| Billboard GT (10 606 pairs) | 11.7% | **2.4%** |
| our 6 charts (45 pairs) | — | **24.4%** |

The machinery finds real nonzero shifts on Billboard (11.7%), it just almost
never needs them to cross the threshold. Our detector needs them **10× more
often**. Annotator-cut sections start on the right bar of the loop; ours do not.

With 45 pairs the 24.4% carries about ±6 pp of binomial noise — even the low end
is 5× Billboard's rate. The conclusion holds; the exact number does not.

**Net: phase-alignment is a safe lever with a measured cost of zero on a
890-track corpus, and a measured (small-sample) benefit of ~24% of refused pairs
on our own charts. That is the cheapest real win available here.**
