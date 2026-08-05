# The raw sliding dot product wins — the "must centre" of the morning was judged on the wrong number

2026-08-05. Louis's pattern-dictionary algorithm, implemented literally and
measured. Report: `/reports/pattern_algo.html`. Prototypes:
`scripts/pattern_algo_core.py`, `scripts/pattern_algo_report.py`,
`scripts/pattern_algo_billboard.py`. Nothing under `harmonia_min/` was touched.

Substrate: the live bar-level SSM (`scripts/pattern_dict_core.song_data`, which
spies on `pipeline.analyze`). Songs: only those passing `beats.check_grid`
(metre 4, consistency ≥ 0.80) — This Love, Don't Know Why, Sunny, plus Billie
Jean and Every Breath You Take. **Georgia On My Mind is refused by the guard**
(metre 2, consistency 45%). **Billie Jean now passes** (metre 4, consistency
94%) — the earlier "3.87 beats/downbeat" note in
`pattern_dict_2026-08-05.md` no longer describes the current beats cache.

---

## Headline

| question | answer |
|---|---|
| raw vs cosine vs mean-centred? | **raw**, on the per-song evidence — clearly |
| was "centring is strictly better" (this morning) right? | **no — it was judged on peak contrast, a number that decided nothing** |
| binary diagonal instead of the extracted square? | **nearly as good (F 0.63 vs 0.67), a third of the cost, and it IS the lag-run cue** |
| which diagonal orientation? | **the main one.** Anti-diagonal scores F = 0.08 |
| does the dictionary get past one entry? | **usually not** — one 4-bar cycle already explains the song |
| does any of it beat a trivial periodic baseline on Billboard? | **no** |

---

## The reference, and why the obvious one is unusable

"Do the peaks land on the true other occurrences" needs a reference that is not
the SSM and not the current segmentation. Used here: **the chord string**. The
per-bar chord list is the one `detect_sections` receives — decoded *before* any
section exists, so independent of the segmentation. It is still our own chords,
not ground truth; a disagreement can be the chords' fault.

**Exact chord-string matching is measured too brittle to be the reference.** On
This Love the motif is `G | Cm | Fm | Dø`, and bar 3 is written `D° Fm` while
bar 7 is written `Dø` — the same turnaround heard slightly differently. A strict
test finds **zero** other occurrences of a loop the ear hears eight times. The
reference used is therefore root-only with ≥ 3 of 4 bars matching; the strict
count is reported alongside so the brittleness stays visible.

Sunny gets **no usable reference**: its period is 16 bars and no 16-bar chord
string repeats, because the song modulates upward (the known transposition
issue). Sunny does not vote in the summary table.

## (A) The four sliding statistics — Louis is right about the raw one

Rows frozen at the motif, columns slid: `f(d) = ⟨P, S[b0:b0+L, d:d+L]⟩`.
One peak selector everywhere (topographic prominence ≥ 25% of curve range), so
the statistics are compared at equal arms.

Averaged over the songs with a usable chord reference (±0 bars):

| statistic | peaks kept | precision | recall | F | contrast (σ) |
|---|---|---|---|---|---|
| **raw** | **11.3** | **0.79** | 0.74 | **0.67** | 1.74 |
| cosine | 16.3 | 0.56 | 0.75 | 0.52 | 1.31 |
| mean-centred | 16.7 | 0.55 | 0.75 | 0.53 | **2.15** |
| binary diagonal | 12.0 | 0.74 | 0.74 | 0.63 | 2.66 |
| anti-diagonal | 9.6 | 0.20 | 0.05 | 0.08 | 1.37 |

Per song, This Love / Don't Know Why: raw F **0.83 / 0.80**, cosine 0.56 / 0.63,
centred 0.59 / 0.63, diagonal 0.71 / 0.80.

**All four recover every true occurrence** (recall 1.00 on both songs). What
separates them is how many *false* ones they add: raw keeps 7 peaks where cosine
and centred keep 12–13, and the extra 5–6 are inventions. Louis's stated reason
holds — sliding along x with the rows fixed means the block's overall level is
itself signal, and normalising divides that signal out, which lifts the curve
everywhere there is nothing.

**The morning's "centring is strictly the better reading" is corrected.** It was
established on *peak contrast in background σ* alone
(`pattern_dict_2026-08-05.md` §4), with no check that the peaks landed on
anything. Centred does have the better contrast (2.15 vs 1.74) **and finds
worse**. Error-pattern #1 in the project's own list: a plausible number that
decided nothing.

**Diagonal orientation.** The block is `B[i,k] = sim(bar b0+i, bar d+k)`. A
repeat maps bar `d+k` onto bar `b0+k`, so the mass is on `i == k`: the **main**
diagonal (identity). The anti-diagonal would require the motif to be played
backwards, and measures as such (F 0.08, and 0.00 on the three songs with a
usable reference). Sliding a binary identity is exactly
`mean_i S[b0+i, d+i]` — the mean of the sub-diagonal at lag `d−b0`, i.e. the
**lag-run cue already measured at 36.8% placement** (`tiling_v2_2026-08-05.md`).
Louis's two ideas are the same object up to the choice of L cells vs L².

**What raw does not solve:** it is not comparable across motif lengths (it grows
as L²), so the dictionary's final arbitration cannot use it directly — that step
compares robust z-scores of each pattern's own curve.

## (B) The dictionary, and the two readings of "re-run on what remains"

Step 1 measured, and it is a real finding. **Aggregating the FIRST rows is a
better statistic than aggregating all of them**, not an approximation of it:

| song | first 16 rows | all rows | musically |
|---|---|---|---|
| This Love | **4** | 20 | 4-bar loop |
| Don't Know Why | **4** | 4 | 4-bar loop |
| Sunny | **16** | 4 | 16-bar form |
| Billie Jean | 2 | 4 | 2-bar bass riff |
| Every Breath | **8** | 4 | 8-bar verse |

Aggregating everything collapses Sunny's 16-bar form and Every Breath's 8-bar
verse to the shortest sub-loop, because late-song material drags the profile.

**One length per dictionary.** Re-deriving a length per entry is degenerate:
once the 4-bar cycle is taken, the leftovers' strongest lag is 2, and Billie
Jean fills the dictionary with six 2-bar motifs that match everywhere.

**"Ce qui reste" — two readings, both implemented, both disappointing.**

* *Uncovered bars.* Entry #2 must start on a bar no block covers. Result: **one
  entry** on This Love, Don't Know Why, Sunny and Billie Jean; three on Every
  Breath. One 4-bar cycle recurring 13 times already covers the song.
  A sub-finding worth recording: recomputing the acceptance quantile on the
  *residual* sub-matrix makes this **worse**, not better — This Love's residual
  bars are the outro, which is more self-similar than the song average, so the
  residual 95th percentile goes **up** (0.945 → 0.967).
* *Nothing removed at all* (the most literal reading of the refinement). Every
  bar stays a legal motif start. This produces entries that are the same cycle
  read one bar later — on This Love, motifs at bars 0, 5, 6, 7. A set-overlap
  test does **not** catch them (their occurrence sets are shifted, so Jaccard
  stays low); correlating the two **curves** after aligning on the motif offset
  does. With that guard: 2 / 1 / 1 / 6 / 2 entries.

**The refinement's arbitration, and "clearly above".** Each candidate block is
scored by *every* pattern, converted to a **robust z of that pattern's own
curve** (median/MAD) — without this, a pattern made of homogeneous material
correlates with everything and takes every block. A block is assigned to the
argmax only when `z_best − z_second ≥ τ`; otherwise it becomes its own section.
τ is in σ units, which is what makes it a defensible constant rather than a
tuned one. Default **τ = 1.0 σ**. Sensitivity is plotted per song on the page
(assigned / ambiguous / number of sections, for τ ∈ [0, 3]).

**Honest limitation: on This Love, Don't Know Why and Sunny the arbitration
never fires**, because the dictionary has one entry. It only does real work on
Every Breath (3 entries, 11 ambiguous blocks) and Billie Jean (6 entries, 59
ambiguous). Calibrating τ on two songs, one of which has the weakest grid, is
not a calibration.

## Billboard, used only to refute — and it does damage

278 tracks, peaks vs annotated section starts, ±1 bar:

| cue | precision | recall | F |
|---|---|---|---|
| trivial: every bar | 0.087 | 1.000 | 0.159 |
| trivial: **every L bars** | 0.351 | 0.495 | **0.310** |
| raw | **0.412** | 0.276 | 0.293 |
| cosine | 0.382 | 0.362 | 0.314 |
| centred | 0.376 | 0.371 | 0.315 |
| diagonal | 0.393 | 0.306 | 0.297 |
| anti-diagonal | 0.317 | 0.249 | 0.244 |

1. **No statistic beats "a boundary every L bars".** F 0.29–0.32 against 0.31.
   As a *section-start detector* the whole family sits at the periodic
   baseline. This is not a refutation of the sliding block; it is a refutation
   of "a recurring motif is a section start" — a 4-bar motif recurring 13 times
   in a 9-section song must overshoot by construction. Same caveat as
   `tiling_v2_2026-08-05.md`: this is a placement/candidate layer.
2. **raw − centred = −2.2 pp of F, 95% CI [−4.0, −0.6], zero excluded.** But
   raw has the **highest precision of all seven rows** and the lowest recall —
   exactly what the songs showed. The two measurements do not contradict; they
   punish different errors, and recall against section starts rewards firing
   often.
3. Not made to say more than that: Billboard is pop, its bar grid is
   interpolated inside annotation lines, and the motif length the algorithm
   finds there has median **11 bars** against 4 / 4 / 16 / 2 / 8 on the songs
   above.

## What I would ship, and what I would not

1. **Ship the raw dot product** for finding a motif's other occurrences. Best
   precision on both measurements, same recall on the songs, and the cheapest.
   Do not ship the mean-centred version on the strength of its contrast.
2. **Retract from `pattern_dict_2026-08-05.md` §4** the sentence "this is
   strictly the better reading" about centring, and the `slide_dot` docstring
   in `scripts/pattern_dict_core.py` that repeats it. The contrast numbers there
   are correct; the conclusion drawn from them is not.
3. **The binary main diagonal is the cheap version** and worth preferring if L²
   ever matters — it is the lag-run reading, so adopting it merges this line of
   work with the one already in `sections.py`.
4. **Do not ship the dictionary as a section detector.** It does not beat a
   periodic baseline on Billboard, and on three of five songs it has one entry.
   The arbitration machinery is implemented and calibrated but has been
   exercised on two songs only.
