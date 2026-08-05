# Handoff — the harmonic section method (2026-08-05)

Written before a context compaction, at Louis's request: « je vais te compacter
donc log tout ce qui t'est utile ». Everything a fresh session needs to finish
this work. Companion entries live in `docs/known_issues.md` (search
`★★★ THE HARMONIC METHOD IS LOCKED ★★★`).

**Where it is going:** Louis validated the path and the next steps are — one last
fix, then **push to production, replacing the current section pipeline with this
version**. That has not happened yet.

---

## 1. What ships in production TODAY (already pushed)

In `harmonia_min/`, live, affecting every chart:

| change | file | why |
|---|---|---|
| `BLUR_SIGMA` 1.5 → **0.0** | `sections.py` | the blur smears the peak position it is meant to locate; novelty 27.6 → 30.1 % exact-bar, fusion 40.2 → 41.8 % |
| run edges **∪** novelty peaks, no more either/or | `sections.py` | runs alone 35.8 %, peaks alone 30.1 %, union **41.8 %**; the old code also DELETED peaks inside runs, worth 4.9 pp of boundary F |
| `TILE_MIN 0.80` → `TILE_QUANTILE 0.95` | `sections.py` | a fixed 0.80 sits at the 43rd percentile of Billie Jean's matrix and the 93rd of Sunny's — it measured harmonic homogeneity, not repetition |
| grid guard, refuses LOUDLY | `beats.py` | `check_grid()` raises when the metre ≠ 4 or fewer than 80 % of bars hold 4 beats |
| real YouTube search + infinite scroll | `server.py`, `app_shell.html` | search was local-only |
| blob audio fallback on a watchdog | `app_shell.html` | the iOS stall is not "installed vs tab" |
| New-Chord-UX handoff (piano sheet, Learn, coach, compact notation, themes, key lenses) | `app_shell.html` | the 21-07 design drop, never integrated |

## 2. The harmonic method — NOT yet in the pipeline

Canonical: **`scripts/harmonic_method.py`. Import it, never re-implement.** It
had already drifted twice while spread across five scripts.

    bar grid (guarded)
      → musx chord posteriors averaged per bar
      → projected onto the 12 PITCH CLASSES via the chord-tone matrix
         (dot product = harmonic overlap: B♭ {D F B♭} vs Gm {D G B♭} = 0.667,
          vs A♭ {C E♭ A♭} = 0.000)
      → SSM = cosine of those 12-d bar vectors
      → period + phase
      → slide the motif along X — **DIAGONAL** reading, `mean_i S[b0+i, c+i]`
      → keep a peak iff local-baseline margin AND ≥ 90 % of the initial peak

`square_slide()` is **deprecated and raises** unless explicitly acknowledged.

### The dictionary (`scripts/dictionary_harmonic.py`)
1. Round 1 finds the motif by the **mean-based** period+phase; rounds 2+ by the
   **longest strong run** among the bars nobody claimed.
2. A run **starts** above q90 and **continues** above q80 (hysteresis).
3. **motif length = min(run length, lag)** — a 16-bar run at lag 8 is 8 bars
   played twice, not a 16-bar motif.
4. A detected block **leaves the game** (Louis revised his earlier separate-box
   idea). Occurrences of one entry may not overlap each other either.

### Dictionary → sections (`scripts/sections_from_dict.py`)
1. a section is a maximal chain of consecutive occurrences of ONE entry
   (chain when the next starts where the last ended, ±1 bar);
2. what nothing claims becomes its own section;
3. those gaps are matched to each other by the **diagonal** (≥ 0.90);
4. a gap ≤ the surrounding motif is a tail and joins the run before it.
Letters come from the dictionary entry, **never from a similarity threshold**.

### Current output
| | dictionary | sections |
|---|---|---|
| This Love | motifs 4 / 8 / 2 / 2 | A[1-16] B[17-24] A[25-36] B[37-44] A[45-48] C[49-52] C[53-56] B[57-72] D[73-80] |
| Don't Know Why | motifs 4 / 8 / 2 | A[1-12] B[13-14] A[15-22] C[23-30] A[31-38] C[39-46] A[47-62] B[63-66] |

Norah's `C[23-30]` and `C[39-46]` are the two blocks Louis pointed at, finally
sharing a letter. Charts to read: `/?open=min_maroon_5_this_love__dict` and
`/?open=min_norah_jones_don_t_know_why__dict` (built by `scripts/dict_charts.py`).

## 3. Things measured FALSE — do not retry

* **Binarising the chroma** — −6 pp; the continuous feature separates repeats
  *better* (AUC 0.874 vs 0.839).
* **Weighting by musx confidence** — no effect, third decimal.
* **Forcing runs to a multiple of the loop** — premise true (GT sections are a
  multiple 84.6 % of the time, ours 59.6 %) but enforcing it drops boundary F
  0.232 → 0.213, and even an ORACLE phase stays under the unconstrained number.
* **A lag-relaxed similarity floor** — real repeats decay only 0.896 → 0.862
  over a 16× lag range; every β > 0 loses.
* **Direct prominence-peak selection** as the tiling rule — 29.8 % vs 37.6 %;
  noisy rows have many prominent maxima so it keeps MORE of the matrix.
* **Motif search by total evidence above a threshold** — biased toward short
  lags, turned This Love's 8-bar chorus into a 2-bar motif with 15 occurrences.
* **Longest-run search applied to round 1** — drifts to musically arbitrary
  motifs (10, 7, 5, 3, 2 bars starting at 13, 1, 2).
* **The 75.2 % chord-repeat placement figure** — a tie-break artefact, retracted.

## 4. Open problems, in the order they matter

1. **The threshold is fragile by construction.** q90 is 0.986–0.990 on these
   songs, so "same bar" and "different bar" differ by under one percent. The
   hysteresis is a patch. `/reports/threshold_histograms.html` shows the
   distribution the quantiles come from, with the accepted matches in red — that
   page exists so Louis can choose the level by eye. **This is the "one last
   fix" before production.**
2. **No transposition invariance.** Sunny modulates; sections F and G are the
   same music a semitone apart (0.797 as-is, 0.973 rotated) and get different
   letters. A max over 12 rotations was measured safe on letters
   (F 0.646 → 0.662) but is not implemented.
3. **The grid guard has a hole: drift.** Beat It drifts −11.4 % across the song
   and still passes at 96 % consistency. A debug agent has the brief.
4. **Short leftover sections.** Norah keeps `B[13-14]` and `B[63-66]` — real
   gaps, but two bars is not a section on a chart.
5. `sections_from_dict` is not wired into `pipeline.py`. Wiring it is the
   production step Louis has approved but not yet asked to execute.

## 5. Working rules he has stated, that cost me when ignored

* **Matrices first, numbers after.** « Je veux voir visuellement s'il y a une
  matrice qui me départage. Si ça départage visuellement, tout le reste c'est
  donné. » Pages that open on a score table are useless to him.
* **He does not trust the corpus benchmark, with reason** — two headline numbers
  did not survive this week. Use Billboard to REFUTE, his ear to SELECT.
* **Never invent a metric without defining it.** "Salience" was mine and meant
  nothing to him. It is gone.
* Every threshold that has been chosen by eye says so, in the code, next to the
  value.

## 6. Pages, all served from `/reports/` on :7772

`index.html` is the entry point and links them all. The ones that matter here:
`sections_from_dict`, `dictionary`, `dictionary_audio` (playable blocks),
`diag_vs_square`, `criteria_visual`, `threshold_histograms`, `peaks_on_harmonic`,
`ssm_harmonic`, `algo_sections` (the shipped algorithm, constants read live).
