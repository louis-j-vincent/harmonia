# The peaks sit at the 95th percentile — and that does not buy accuracy

2026-08-05. Louis's two questions on the bar-level SSM, measured.
Report: `/reports/pattern_dict.html`. Prototypes: `scripts/pattern_dict_core.py`,
`scripts/pattern_dict_report.py`, `scripts/pattern_quantile_billboard.py`.
Nothing under `harmonia_min/` was modified.

Bar-level SSM reused from `scripts/ssm_rows_plot.bar_ssm` (spies on the live
pipeline), now cached per song. Billboard harness = the same 285 bar-annotated
McGill tracks as `docs/research_sessions/tiling_v2_2026-08-05.md`, ties broken
**at random**, trivial "never move" baseline printed everywhere.

---

## Headline, in one table

| question | answer |
|---|---|
| do the clear peaks sit at a stable quantile? | **yes — the 95th–96th, on every song with a valid bar grid** |
| does a per-song quantile beat `TILE_MIN = 0.80`? | **no — a wash** (+0.4 pp fused, −1.3 pp standalone, CIs straddle 0) |
| does direct peak detection (no threshold) beat it? | **no — clearly worse**, 29.8% vs 37.6% |
| does summing adjacent half-bars help? | **yes, small and consistent** (+0.15–0.20 σ of peak prominence) |
| does the sliding dot product find the other occurrences? | **yes, and cleanly — once each block is mean-centred** |

---

## (1) The quantile of the visible peaks — Louis's premise is TRUE

« il y a des pics clairs qui doivent correspondre à un quartile des valeurs de
la matrice SSM ».

"Clear peak" is defined with **no absolute threshold**: a strict local maximum
of an SSM row at |lag| ≥ 2, whose topographic prominence is in the top quartile
of that song's peaks. Then: at what quantile of that song's own off-diagonal
distribution does its *value* sit?

| song | quantile of the clear peaks | IQR | all peaks | where `0.80` lands |
|---|---|---|---|---|
| This Love | **95.7** | 90–98 | 85.8 | 81st |
| Don't Know Why | **95.8** | 93–98 | 86.8 | 81st |
| Sunny | **95.8** | 93–98 | 85.5 | **93rd** |
| Georgia ⚠ | 95.9 | 92–98 | 84.5 | **95th** |
| Billie Jean ⚠ | 88.9 | 60–96 | 79.3 | **43rd** |

⚠ = broken bar grid (Georgia 3.06 beats/downbeat and IBI CV 0.35; Billie Jean
3.87) — kept out of every fit. Georgia's *quantile* still lands at 95.9, which
is mild evidence the statistic survives a bad grid; Billie Jean's does not.

**The premise holds and it is remarkably tight**: 95.7 / 95.8 / 95.8 on the
three good songs. What varies wildly is the *other* direction — the same
constant 0.80 is the 43rd percentile on one song and the 95th on another.
Corpus-wide (285 Billboard tracks) the fixed 0.80 passes a median 21.7% of the
matrix with a p10–p90 range of **12.5%–46.0%**; a quantile rule passes
(1−q) of it on every song by construction.

## (2) Half-bar vs bar vs summed adjacent half-bars — Louis right, small

« en cumulant les demi-barres adjacentes on trouve un signal robuste ».

Measured as peak prominence and peak height in units of the background's own σ,
so the three grains are comparable. `sum2[i,j] = ½(Sh[i,j] + Sh[i+1,j+1])` —
pooled over two adjacent half-bars but **still on the half-bar stride**, so it
keeps the resolution the bar grain throws away.

| song | prominence: half / **sum2** / bar | height: half / **sum2** / bar |
|---|---|---|
| This Love | 1.81 / **2.02** / 1.84 | 1.05 / **1.23** / 1.34 |
| Don't Know Why | 2.14 / **2.28** / 2.03 | 1.37 / **1.51** / 1.46 |
| Sunny | 1.60 / **1.71** / 1.68 | 1.03 / **1.20** / 1.19 |
| Georgia ⚠ | 1.64 / **1.81** / 1.75 | 0.91 / **1.13** / 1.09 |
| Billie Jean ⚠ | 1.13 / 1.11 / **1.24** | 0.54 / **0.69** / 0.35 |

Summing beats the plain half-bar on **5/5 songs** for prominence-or-height and
beats the bar grain on 4/5 for prominence, at full half-bar resolution. The
effect is real but modest (+0.15–0.20 σ) — worth taking because it costs
nothing, not worth a redesign.

## (3) Quantile vs direct peaks vs `TILE_MIN` — the generalisation test

Both families have a knob, so best-vs-best is not a test. Protocol: 285 tracks
split **60/40 by song** (seed 0) → tune 171 tracks / 1634 starts, held-out 114
tracks / 1147 starts. Knob picked on tune only; both reported on held-out.

Cue construction is identical for the three rules — binarise the bar SSM, then
read the sub-diagonal run edges (`run_edge_score(..., "edgesum")`, the reading
that produced yesterday's 35.8% / 41.8%):

* **(A) per-song percentile** — `S ≥ quantile_q(offdiag(S))`, tune picks q=0.90
* **(B) direct peaks** — `scipy.signal.find_peaks` per row, prominence floor =
  a fraction of that row's own range; tune picks prominence=0.50
* **(baseline)** `S ≥ 0.80`

**Held-out, standalone:**

| rule | exact bar | ±1 | per-song median | song quartiles | songs ≤ chance | matrix density |
|---|---|---|---|---|---|---|
| trivial: never move | 9.8% | 31.7% | 8.3% | 0–17% | 61.4% | — |
| `TILE_MIN = 0.80` (ships) | **37.6%** | 47.4% | 37.5% | 21–57% | 15.8% | 26.0% ± **17.2** |
| (A) quantile q=0.90 | 37.0% | 45.0% | 33.3% | 22–50% | **14.9%** | 9.9% ± **0.4** |
| (B) direct peaks 0.50 | 29.8% | 46.7% | 26.1% | 11–44% | 24.6% | 24.7% ± 5.9 |

**Held-out, fused with the un-blurred novelty** (yesterday's shipping recipe):

| rule | exact bar | ±1 | per-song median | songs ≤ chance |
|---|---|---|---|---|
| trivial: never move | 26.6% | 41.2% | 20.0% | 25.4% |
| `TILE_MIN = 0.80` (ships) | 42.5% | 52.9% | 43.3% | 16.7% |
| (A) quantile q=0.90 | **42.9%** | 50.8% | 40.0% | **15.8%** |
| (B) direct peaks 0.50 | 35.0% | 51.7% | 31.0% | 22.8% |

Song-level bootstrap (4000 resamples) on held-out, quantile − fixed:
**standalone −1.28 pp, 95% CI [−5.5, +2.8]; fused +0.43 pp, CI [−3.5, +4.4]**.
Zero is inside both.

**Verdict — a first-class negative.** The quantile rule does not buy accuracy.
What it buys is *scale-invariance*: it removes a constant whose meaning varies
by 4× across songs and replaces it with one that passes the same 10% of every
matrix. Its per-song floor is also slightly better (fewer songs at or below
chance in both settings). If the goal is "stop the constant meaning something
different on every song", (A) is free. If the goal is "move the number", it
does not.

**Family (B) is measured false.** Direct prominence peaks with no absolute
threshold keep ~25–41% of the matrix — far more, not less, than a threshold —
because a noisy row has many prominent local maxima. It loses 8 pp standalone
and 7.5 pp fused, and it collapses on a quarter of songs (24.6% at or below
chance vs 15.8%). It is not the family to ship.

## (4) The pattern dictionary and the sliding dot product

Louis's design, implemented literally.

**First entry.** On the *bar* SSM every index is already a downbeat, so
"starts on beat 1 of a bar" costs nothing — what has to be chosen is where the
motif starts and how long it is. Rule: scan bars in order; for each, the
smallest L whose *whole block* returns — mean of `S[b+i, b+L+i]` over i<L above
that song's own 95th percentile (the §1 threshold, not a constant). A block
diagonal, not a single cell: one high value is a coincidence, L in a row is a
repeat.

| song | first motif | occurrences | on a current section start (±1) | current starts recovered | median quantile of the kept `f` peaks |
|---|---|---|---|---|---|
| This Love | bars 0–3 (L=4) | 13 | 5/13 | 5/9 | 90th |
| Don't Know Why | bars 3–6 (L=4) | 14 | 3/14 | 3/10 | 87th |
| Sunny | bars 0–3 (L=4) | 26 | 5/26 | 6/7 | 84th |
| Billie Jean ⚠ | bars 2–3 (L=2) | 37 | 17/37 | 17/20 | 86th |
| Georgia ⚠ | bars 17–48 (L=32) | 7 | 1/7 | 1/6 | 89th |

Read this the right way round. **The occurrences are not section starts and
were never meant to be** — a 4-bar motif recurring 13 times inside a 9-section
song must overshoot, and it does (5/13 land on a start). The useful column is
the *recall*: the occurrence set covers 5/9, 3/10 and 6/7 of the current
section starts. It is a **placement/candidate** layer like the lag-runs, not a
detector — the same caveat that carries over from
`tiling_v2_2026-08-05.md`.

The two suspect songs behave exactly as their grids predict: Billie Jean
degenerates to a 2-bar motif matching almost everywhere (37 occurrences),
Georgia cannot find a short motif at all and falls back to a 32-bar one.

**Sliding along X.** Rows frozen at the motif, columns slid:
`f(d) = ⟨P, S[b0:b0+L, d:d+L]⟩` with `P = S[b0:b0+L, b0:b0+L]`.

**Centre each block before the dot product.** This is the one thing that had to
be added, and it matters more than any threshold. The raw normalised dot
product lives in [0.92, 1.00] — every block of a pop SSM shares the same
overall similarity level, and that common level drowns the signal. Subtracting
each block's mean turns the cosine into a correlation and lifts peak contrast
(peak height − background, in background σ):

| song | raw cosine | **centred** |
|---|---|---|
| This Love | 1.36 | **1.67** |
| Don't Know Why | 1.89 | **2.02** |
| Sunny | 0.97 | **1.16** |
| Billie Jean ⚠ | **−0.09** | **1.16** |
| Georgia ⚠ | 1.34 | 1.32 |

On Billie Jean the un-centred version carries no usable signal at all.

**Does it find the occurrences cleanly?** On This Love, yes and legibly: sharp
peaks at 0, 4, 8, 12, then a flat low zone over bars 16–23, peaks again at 24,
28, 32, low over 36–48. That is exactly its A / B / A / B form, read off one
1-D curve. Same picture on Don't Know Why and Sunny at 4-bar spacing.

**Does it need a threshold, and is that threshold a quantile again?** It needs
a *peak selector*, not a threshold — the same prominence rule as §1 works. The
kept peaks' quantile within `f`'s own distribution is reported per song in the
HTML; it is not as tight as the 95th-percentile result of §1, because `f` is
already a contrast (it has its own zero) rather than a raw similarity.

**Relation to the time-lag matrix (RefraiD, Goto 2006) — mostly no.** Worth
saying plainly. RefraiD reads a *row* of the time-lag matrix `R(t, lag) =
S(t, t−lag)`: one lag at a time, looking for horizontal line segments. Here a
*square* is slid: at each offset the comparison uses the motif's L² internal
relations, not L cells of one diagonal. It is closer to a correlation of Gram
matrices than to a diagonal reading. The practical consequence is that a
uniform shift of key or timbre affecting the whole block cancels in the
centred version, which a diagonal reading does not give you.

---

## What I would (and would not) change in `harmonia_min/sections.py`

1. **Optional, free:** replace `TILE_MIN = 0.80` with `quantile(offdiag(S),
   0.90)`. Same accuracy, removes a constant that means 4× different things
   across songs, and makes the matrix density identical everywhere. Justify it
   as scale-invariance, **not** as an accuracy win.
2. **Do not implement family (B)** — direct prominence peaks as the similarity
   selector lose 8 pp.
3. **Not yet shippable:** the pattern dictionary. It produces a legible,
   musically correct occurrence curve on the three good songs, but it has not
   been measured against Billboard GT as a boundary detector. That is the next
   experiment, and the centred block correlation is the version to measure.
4. **`sum2` (summed adjacent half-bars)** is a cheap +0.15–0.20 σ on peak
   prominence at full half-bar resolution — take it if the half-bar SSM is
   being touched anyway.
