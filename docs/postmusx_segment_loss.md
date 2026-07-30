# Post-musx segment loss on Let It Be — the trace

**Brief (Louis, 2026-07-30):** "14 chord segments that musx correctly decodes are
LOST between the musx frame decode and the baked chart — durations 0.6–1.7 s, and
two of 3.4 s. Find WHICH stage, characterize its rule, propose the minimal fix."

**Answer, in one line:** the "14" is really **3**, and they die at **two** stages —
one at the musx re-decode (a 0.93-beat chord, too short for a beat-quantised
Viterbi) and **two at `_split_collapsed_bars_via_musx`**, the 2-chords-per-bar
brick from `33e8f2e`, which on this song is a net **destroyer of 8 chords, every
one of them exactly 1.00 beat long**.

---

## 0. The bake config — the premise of the brief is void as stated

The brief says the baked chart came from `render_youtube_chart.py` "with the
SHIPPED config (commit `d3f0bae`)". It did not. Verified three independent ways:

| evidence | value |
|---|---|
| last commit touching `docs/plots/inferred_let_it_be_remastered_2009.html` | `09d99d7`, **2026-07-21 13:32** |
| file mtime on disk | **2026-07-21 17:27** (working-tree diff vs `09d99d7` is 2 cosmetic URL lines) |
| `d3f0bae` date, and what it re-baked | 2026-07-30 18:30 — **This Love and Misery only** |
| `git show 09d99d7:scripts/render_youtube_chart.py` | `infer_chords_v1(audio_path, seventh_gate=0.0, cache_dir=…)` — **bare** |
| `infer_chords_v1` defaults at `09d99d7` | `feature_frontend="bp48"`, `bass_frontend="nnls24"`, `quality_frontend="nnls24"`, `segment_source="nnls"` |

So the baked Let It Be chart was produced by a **bp48 + NNLS-24** pipeline in which
**music-x-lab never ran** — the very configuration `d3f0bae` was written to stop
using. The clincher is physical:

```
data/cache/musx_infer/let_it_be_remastered_2009_submission.lab   Jul 22 10:10
docs/plots/inferred_let_it_be_remastered_2009.html               Jul 21 17:27
```

**The chart predates the musx artifact it is being compared against by 17 hours.**
Nothing could have been "lost between the musx decode and the chart", because the
musx decode did not exist when the chart was made. The falsification test in
`docs/ug_score_report.md` compared two different models' outputs, not two stages
of one pipeline.

**Config actually traced below** (established, not assumed): `SHIPPED_CONFIG` from
`harmonia/eval/accuracy_score.py` — `feature_frontend=nnls24`, `bass/quality=musx`,
`segment_source=musx_redecode`, `function_family=True`, `beat_backend=beatthis`,
`beat_period_mode=bestfit`; no `HARMONIA_*` overrides; `HARMONIA_MUSX_DIR` pointed
at the complete third-party clone. Audio re-fetched from the chart's own URL
(`youtu.be/QDYfEBY9NM4`), 243.03 s, matching the `.lab`'s last boundary
(243.043 s) — same recording.

> **Trap for the next session.** With `HARMONIA_MUSX_DIR` unset in a worktree whose
> `third_party/…/data/` is missing, `musx_redecode` fails *silently to a warning*
> and `self._used_redecode` stays False. That flips the Occam gate default
> (`chord_head.py:1066`) from ON to OFF, the Occam post-pass then fires and
> compresses **129 spans → 59**, snapping F→C/G against a 2-chord vocabulary.
> Two runs of "the shipped config" that differ only by an env var produced 114 vs
> 120 chords. Always check for `musx re-decode unavailable` in the log.

## 1. The 14, re-derived

Method per `docs/ug_score_report.md`: `MISSED`-class errors in
`scratchpad/ug_score_let_it_be_remastered_2009.json` whose root is present in the
musx `.lab` with >50 % overlap. Reproduces exactly: **38 MISSED → 14 present in
musx**, durations 0.6–1.7 s plus two of 3.4 s.

**But 8 of the 14 are not losses at all.** Against the baked chart's own chord
list, our chart *does* write the UG root inside the window in 8 cases — the
ordinal diff simply failed to pair them (the same "diff slop" the report already
found in 16/82 elsewhere, not fully removed here). Under the shipped config it is
**11/14 present**.

| | UG root present in our chart |
|---|---|
| baked chart (bp48/nnls24, Jul 21) | **8 / 14** |
| shipped config (musx redecode + fold, today) | **11 / 14** |

Only **3** are genuine: #3 `F @61.6 s`, #9 `F @116.4 s`, #14 `F @235.2 s`.

## 2. Stage × segment table

Anchored on the **musx** span (UG timing is hand-made, so a UG-anchored window
scores a shifted-but-present chord as missing). "Alive" = a chord of that root
covering ≥50 % of the musx segment, allowing ±0.6 beat of slide.

| # | UG | musx span | beats | .lab | redecode+fold | coalesce | occam¹ | finalize | **SPLITTER** | final |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F | 42.10–43.75 | 1.94 | Y | Y | Y | · | Y | Y | Y |
| 2 | G | 47.04–48.72 | 1.97 | Y | Y | Y | Y | Y | Y | Y |
| **3** | **F** | **61.72–62.51** | **0.93** | Y | **·** | · | · | · | · | **·** |
| 4 | F | 81.85–83.55 | 2.00 | Y | Y | Y | · | Y | Y | Y |
| 5 | F | 95.20–96.87 | 1.96 | Y | Y | Y | · | Y | Y | Y |
| 6 | G | 100.24–101.87 | 1.92 | Y | Y | Y | Y | Y | Y | Y |
| 7 | C | 103.54–105.21 | 1.96 | Y | Y | Y | Y | Y | Y | Y |
| 8 | C | 106.18–108.67 | 2.93 | Y | Y | Y | · | Y | Y | Y |
| **9** | **F** | **116.66–117.47** | **0.95** | Y | Y | Y | · | Y | **·** | **·** |
| 10 | C | 117.47–120.93 | 4.07 | Y | Y | Y | Y | Y | Y | Y |
| 11 | F | 157.43–159.27 | 2.16 | Y | Y | Y | · | Y | Y | Y |
| 12 | C | 173.20–176.75 | 4.17 | Y | Y | Y | Y | Y | Y | Y |
| 13 | G | 234.41–235.43 | 1.20 | Y | Y | Y | Y | Y | Y | Y |
| **14** | **F** | **235.43–236.54** | **1.30** | Y | Y | Y | · | Y | **·** | **·** |
| | | **TOTAL** | | **14** | **13** | **13** | 6¹ | **13** | **11** | **11** |
| | | **stage size** | | 131 | 129 | 128 | (64)¹ | 128 | **120** | 120 |

¹ The Occam column is **counterfactual**. `occam_compress_bars` did compute a
64-span compression, but the GT-free gate rejected it
(`OCCAM: GATE REJECTED 1/1 loop family (cov=0.69/dev=15)`) and the chart was left
unchanged. It is shown because it is the single most destructive thing in the
chain when it *does* fire — it would have taken the chart to 6/14 — and because
its gate silently defaults OFF when the re-decode fails (see the trap above).

**The column flips twice.** Everything else — the posterior fold, coalescing,
Occam, finalize — is innocent.

## 3. The guilty rule

### 3a. `_split_collapsed_bars_via_musx` — 2 of the 3, and 8 chords overall

`harmonia/models/chord_pipeline_v1.py:2006-2094` (commit `33e8f2e`, "2-chords-per-bar
collapse on fast harmonic rhythm", ancestor of HEAD; env `HARMONIA_MUSX_2CHORD_BAR`,
**default ON**).

Constants at `:2038` — `fast_beats=2.75, min_run=4, min_half=1.4` (beats).

The function is a two-step *replace*, and the two steps use different filters:

```python
kept = [c for c in chords_out                       # :2062  DELETE everything
        if c.get("label") == NO_CHORD_LABEL         #        whose midpoint is
        or not _in_run(0.5*(c["start_s"]+c["end_s"]))]   #    inside a fast run
...
for (m0, m1, lab) in mx_labels:                     # :2072  RE-EMIT from musx
    a, b = max(m0, lo), min(m1, hi)
    if (b - a) < min_half * period:                 # :2076  ...but SKIP anything
        continue                                    #        under 1.4 beats
```

A chord shorter than 1.4 beats that was **already correctly present** in the
baseline is deleted at `:2062` and never re-added at `:2076`. The function's own
return value is `len(new_run_chords) - n_before` — on Let It Be it returns **−8**,
i.e. the "splitter" removed eight chords. The docstring's premise ("each musx
segment ≥ ~1.5 beats", "returns the number of chords added") does not hold here.

Why it swallows the whole song: `fast_beats=2.75` calls any musx segment under
2.75 beats "sub-bar", and Let It Be is a **2-beat harmonic rhythm throughout**, so
essentially the entire song is one long "fast run" and the chart is rebuilt
wholesale from musx — with the sub-1.4-beat filter applied to all of it.

Matching pre/post within ±0.35 s and identical label (so pure time-jitter is not
counted), the splitter:

* **dropped 9**, of which **8 are exactly 1.00 beat** — `10.29 F`, `23.06 F`,
  `35.83 F`, `105.66 F`, `109.91 F`, `115.87 G`, `116.73 F`, `235.94 F`;
* the 9th is not a drop but a relabel: `204.43 C:maj/G` (2.00 beats) → `C:maj`
  (the bass slash is dropped, since `:2085` re-emits `f"{NOTE[r]}:{sev}"`);
* **added 1** (that same relabelled `C:maj`).

**One rule — "delete-then-refuse-to-re-emit anything under `min_half`=1.4 beats" —
explains 8 of the 8 real drops, and 2 of the 3 genuine missing segments (#9, #14).**

### 3b. `musx_redecode` — the remaining 1

Segment #3 (`F`, 61.72–62.51, **0.93 beats**) never survives the re-decode.
`musx_redecode.make_beat_arr` (`:170-215`) permits a chord change **only on a beat
frame** (penalty 40), and the re-decode emits a clean 2-beat grid there:

```
musx .lab      C[58.44-60.07] G[60.07-61.72] F[61.72-62.51|0.93b] C[62.51-66.69]
redecode+fold  C[58.82-60.53] G[60.53-62.23]        —             C[62.23-66.48]
```

Corpus-wide on this song only **5 of 131** raw musx segments lose their root at the
re-decode, and their median length is 1.91 beats — i.e. the re-decode is *not*
generally lossy; #3 is one of only two sub-beat segments in the whole file. This is
the honest residue of the original "grain" hypothesis, and it is small.

## 4. Proposed minimal fix

**Change one predicate, at `chord_pipeline_v1.py:2076`.** `min_half` exists to stop
the run-boundary clipping (`a, b = max(m0, lo), min(m1, hi)`) from emitting slivers.
It should therefore apply **only to segments the clipping actually shortened**, not
to whole musx segments that are simply short:

```python
clipped = (a > m0 + 1e-6) or (b < m1 - 1e-6)
if (b - a) < (min_half if clipped else 0.75) * period:
    continue
```

A whole 1-beat musx chord is real harmony and is re-emitted; a 0.3-beat fragment
created at a run edge is still discarded.

**What this fix does NOT solve** (project rule #4):

1. **#3, and sub-beat harmony generally.** Anything shorter than a beat is
   destroyed upstream in `musx_redecode`'s beat-quantised Viterbi and never
   reaches the splitter. This fix cannot recover it; that needs a sub-beat
   transition grid, which is a different and much larger change.
2. **The 24 decoder-absorbed MISSED chords** (`docs/ug_score_report.md`): the 13
   `D-7`→`F:maj` shared-tone absorptions are a decode-time root/bass problem,
   untouched here.
3. **The bass slash.** `:2085` re-emits `f"{NOTE[r]}:{sev}"`, so any `/bass` on a
   chord inside a fast run is silently dropped (`C:maj/G` → `C:maj`). That is a
   separate bug in the same function, and it matters because the project's root
   target *is* the sounding bass.
4. **The Occam gate's env-dependent default**, and the silent
   `musx re-decode unavailable` degradation that flips it. Independent, and
   arguably more dangerous than the bug fixed here.
5. **The ug_score ordinal diff's pairing failures** — 8 of the original 14 were
   this, not pipeline losses. The scorer over-reports MISSED.

## 5. Measurements

Brick-0 frozen benchmark, 7 verified songs, duration-weighted, shipped config.

_(baseline recorded; fix measurement in the section below)_

| | mirex_root | partial_credit | majmin | sevenths |
|---|---|---|---|---|
| baseline | 0.7679 | 0.7159 | 0.7537 | 0.5756 |

## 6. Reproduction

Scratchpad drivers (no `harmonia/**` file was edited):

* `derive14.py` — re-derives the 14 from the ug_score json + the musx `.lab`
* `run_shipped.py` — runs `infer_chords_v1(**SHIPPED_CONFIG)` on the song
* `trace_stages.py` — monkeypatches `two_pass_redecode`, `_apply_occam_to_coalesced`,
  `_finalize_chords`, `_split_collapsed_bars_via_musx` and dumps every intermediate
* `stage_table2.py` — the musx-anchored stage × segment table above
* `diagnose.py` — per-loss cause + the jitter-corrected splitter diff
* `brick0.py` — the 7-song frozen benchmark
