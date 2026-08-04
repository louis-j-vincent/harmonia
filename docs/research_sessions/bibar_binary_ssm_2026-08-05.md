# Louis's binary bi-bar SSM — implemented, measured, and one metric bug found

**[OURS]** = measured here today. **[PAPER]** = reported by a source.
Harness: `scripts/bibar_prep.py` → 285 McGill Billboard tracks, 2781 annotated
section starts, annotators' own bar grid and own NNLS chroma
(`scripts/billboard_bar_gt.py`, built 2026-08-04). Nothing under `harmonia_min/`
was modified.

---

## ★ READ THIS FIRST — the lit review's 75.2% is a measurement artefact

`scripts/section_start_placement_screen.py` — and therefore
`structure_literature_2026-08-04.md` §1.2 — scored placement as
`argmax(score - 1e-6*|offset|)`: **ties are broken by preferring the smaller
shift.** Any cue whose score is *constant* across the ±4-bar window then scores
**100%**, because the tie-break hands it the right answer.

The chord-string repeat cue is exactly that constant cue. Inside a repeated
section, the 8-bar chord string matches its other occurrence perfectly at
*every* offset, so `best-match fraction` is 1.0 across the whole window and
carries no information about which bar. Same code, same data, only the tie-break
changed **[OURS]**:

| cue | ties → random (honest) | ties → smallest shift (old) |
|---|---|---|
| **trivial: never move at all** | **10.8%** (= chance, 1/9) | **100.0%** |
| chord-string 8-bar repeat | **14.3%** | 78.8% |
| chroma checkerboard novelty (shipped) | 27.6% | 27.6% |

The novelty number is unaffected — it is a continuous float and essentially
never ties. So **34.7% was real; 75.2% was not.** The honest reading is that the
chord-repeat cue is *3.5 pp above chance*, not *2× the acoustic cue*.

This also explains Louis's ear verdict yesterday («en effet notre version est
mieux»): the placer was optimising a quantity that is flat wherever the section
actually repeats, so it moved starts on essentially no evidence.

**Action:** `docs/research_sessions/structure_literature_2026-08-04.md` §1.2,
§1.3, §6 (candidate #1) and §7 rest on this number and should be read as
retracted until re-measured. `scripts/section_start_placement_screen.py` should
not be reused as-is.

---

## 1. What was built

`scripts/bibar_binary_ssm.py`, Louis's four steps, each an explicit knob:

- **(a) unit = the bi-bar**, taken with **stride 1** (bi-bar *i* covers bars
  *i* and *i+1*), not stride 2. Stride 2 halves the resolution and cannot
  answer "which BAR does it start on", which is the metric.
- **(b) binarisation** — swept: top-k chroma bins per half (`topk`), per-bin
  per-song 70th percentile (`quantile`), 50%-of-the-loudest-bin (`relmax`), and
  symbolic chord-tone membership. Bass and treble halves binarised separately.
  Bi-bar vector by `concat` (order-preserving, 48-d) or `union` (order-blind).
- **(c) threshold** — binary dot product normalised by the row norms (= cosine
  on 0/1 vectors), so τ is a clean fraction of shared active bins.
- **(d) reading the sub-diagonals** — `M[i, i-l]` as a function of *i* is the
  lag-*l* sub-diagonal; a run of 1s there means "this passage rejoue what
  happened *l* bars ago"; a section start is the run's **left edge**.

Three readers were tried; the reader mattered as much as the substrate **[OURS]**:

| reader | definition | exact-bar |
|---|---|---|
| `diff` | `max_l min(fwd,8) − min(bwd,8)` | 21.9% |
| `edgelen` | `max_l fwd` where `bwd == 0` | — |
| **`edgesum`** | `Σ_l min(fwd,8)` over lags where `bwd == 0` | **30.8%** |

`edgesum` sums the votes of *every* lag that starts a run at that bar. A chorus
heard four times produces edges at three lags at once, and they agree only at
the true start — this is the cross-occurrence evidence, and it is worth ~9 pp
over the obvious single-lag reading.

---

## 2. The numbers (285 tracks, 2781 starts, ties broken at random)

| cue | exact bar | ±1 bar | median abs err |
|---|---|---|---|
| trivial: never move | 10.8% | 33.8% | 2.0 |
| chord-string 8-bar repeat | 14.3% | 34.8% | 2.0 |
| `diff` reader on the binary matrix | 21.9% | 43.0% | 2.0 |
| **chroma checkerboard novelty — what we ship** | **27.6%** | 49.0% | 2.0 |
| binary bi-bar lag-runs, `edgesum` (Louis) | **30.8%** | 46.4% | 2.0 |
| binary bi-bar, transposition-invariant (12 rot) | 30.6% | 45.6% | 2.0 |
| binary bi-bar on **chord tones** | 33.1% | 47.7% | 2.0 |
| **CONTROL: the same reader on CONTINUOUS cosine** | **36.8%** | 50.3% | 1.0 |
| **FUSION: continuous lag-runs + novelty (z-summed)** | **40.2%** | **54.7%** | **1.0** |

Chosen binary setting: `topk(bass 2, treble 4) / concat / τ=0.8`, off-band SSM
density **9.0%** (printed next to every accuracy — a saturated all-ones matrix
would win the metric on tie-breaks alone and must be visible, not inferred).

### Verdict on Louis's idea, plainly

1. **The sub-diagonal lag-run reading is a real win: +9.2 pp over the shipped
   cue** (36.8% vs 27.6%), and +12.6 pp when summed with novelty (40.2%). It
   also beats the CBM/`as-seg` number from the lit review (35.7%, different
   track set). This is the first thing measured in this repo that clears the
   34.7% ceiling.
2. **The binarisation is a loss: −6.0 pp** (30.8% binary vs 36.8% continuous,
   everything else identical — same bi-bar unit, same threshold shape, same
   reader). Binarising throws away information the reader was using.
3. **It is a placement cue, not a detector.** Unconditioned boundary detection
   from the runs is *worse* than our novelty peaks (below).

### The hypothesis behind the binarisation, tested directly

Hypothesis (mine, offered to Louis as a hypothesis): repeats differ in
*arrangement* — same chords, different density/voicing/vocals — which a
continuous cosine punishes and a binary "are the same notes present" would
forgive. Test: AUC separating GT same-letter section pairs from
different-letter pairs, on the openings of annotated sections **[OURS]**:

| similarity | AUC (n=278 songs) |
|---|---|
| binary bi-bar dot product | 0.839 |
| continuous cosine | **0.874** |

**Not supported.** The continuous feature separates same-section from
different-section pairs *better*, not worse. Whatever arrangement variation
exists, thresholding to 0/1 destroys more signal than it removes.

### Unconditioned boundary detection (bar tolerance, F-measure)

| tolerance | binary bi-bar runs | novelty peaks |
|---|---|---|
| ±0 bars | 0.161 | **0.194** |
| ±1 bar | 0.275 | **0.345** |
| ±2 bars | 0.328 | **0.428** |

The runs are a good answer to "given roughly where the section is, which bar?"
and a poor answer to "where are the sections?". Use it as a **placement /
refinement** layer on top of the existing cut source, not as a replacement.

---

## 3. Transposition invariance (the Sunny finding)

Prompted by Bobby Hebb's *Sunny*: it modulates upward, and the raw-chroma
cross-block ratio makes a transposed repeat look like new material (F vs G:
0.797 raw → 0.973 after an 11-semitone rotation, crossing `LABEL_COS = 0.96`).

Measured the false-merge counterweight on the harness — same-letter decision
over annotated section pairs, threshold 0.80 on the 4-bar opening
similarity **[OURS]**:

| similarity | precision | recall | F |
|---|---|---|---|
| raw chroma | 0.803 | 0.540 | 0.646 |
| **max over 12 rotations** | 0.799 | **0.565** | **0.662** |

**The feared false-merge blow-up does not happen**: precision drops 0.4 pp,
recall gains 2.5 pp, F +1.6 pp. Small but the right sign, on 285 tracks. Note
Billboard is mostly non-modulating pop, so this *understates* the gain on songs
like *Sunny* and overstates the risk elsewhere by roughly the same logic — the
honest summary is "cheap, slightly positive on average, and the mechanism is
sound". Recommend it for the **letter-assignment** step; it changes placement
by −0.2 pp (30.6% vs 30.8%), i.e. not at all.

Precedent: rotation-max chroma similarity is standard in cover-song
identification (Serrà's Qmax / OTI); it is not standard in structure analysis.

---

## 4. Is this RefraiD's time-lag matrix? — PARTIAL

Checked against Goto, *A Chorus-Section Detection Method for Musical Audio
Signals*, IEEE TASLP 2006.

- **Confirmed [PAPER]:** RefraiD works in a time-lag plane and detects **line
  segments (runs) at fixed lags** — geometrically the same object as diagonal
  runs at offset *k* in an SSM. Louis's framing is the right lineage.
- **Refuted [PAPER]:** its base similarity `r(t, l)` is **continuous** in [0,1]
  (normalised Euclidean chroma distance). Thresholds are applied downstream, at
  three separate per-song-adaptive (Otsu-style) stages — never as one global
  binarisation of the matrix. So "binary dot product" is *not* what RefraiD
  does, and our own measurement independently says the continuous version is
  better.
- **Refuted [PAPER]:** handling a section that repeats at several lags is not
  "reading runs" — RefraiD §IV.D is a separate grouping algorithm (a set of
  lags per group, top-down transitive re-detection, three pruning heuristics).
  Our `edgesum` reader is a crude one-line stand-in for it; §IV.D is worth
  reading properly before building the grouping layer.

Full detail in `docs/research_sessions/segment_grouping_2026-08-05.md`.

---

## 5. What I would do next, ranked

1. **Wire the continuous lag-run score into `sections.py` as a placement
   refinement**, summed with the existing novelty (40.2% vs 27.6%). It reuses
   the SSM already computed there; ~2–3 h.
2. **Make the letter-assignment ratio transposition-invariant** (max over 12
   rotations of `M[i,j]`), and recalibrate `LABEL_COS` on the harness rather
   than on 2–3 songs. Fixes *Sunny*'s 8-sections-7-played-once; ~2 h.
3. **Read RefraiD §IV.D and implement its grouping properly** instead of
   `edgesum` — it is the published answer to "the same section repeats at
   several lags", and nothing since has replaced it.
4. **Re-measure everything in `structure_literature_2026-08-04.md` §1** with
   random tie-breaking before any of it is used to rank candidates again.

## Files

- `scripts/bibar_prep.py` — caches the Billboard bar substrate (bar chroma,
  half-bar chroma, per-bar chord symbols, GT starts + letters).
- `scripts/bibar_binary_ssm.py` — the method: binarisation, bi-bar, binary SSM,
  rotation-invariant SSM, the three sub-diagonal readers, scoring.
- `scripts/bibar_sweep.py` — the full binarisation × join × τ × reader sweep.
- `scripts/bibar_final.py` — the headline table, the AUC hypothesis test, the
  transposition-invariance P/R/F, the boundary F.
- `scripts/bibar_report.py` → `harmonia_min/state/reports/sections_v2.html`.
