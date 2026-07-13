# iReal Pro GT validation — first honest chord accuracy on real audio

**Date:** 2026-07-14 · **Script:** `scripts/validate_against_ireal.py` ·
**Data:** `data/ireal_gt_validation_set.json` · **Plot:**
`docs/plots/ireal_accuracy_comparison.png`

## TL;DR

This **unblocks known_issues.md #35** (GT-eval was blocked by a chart-timeline
mismatch). The fix #35 asked for — inference + iReal GT on a *shared audio clock
via a single DTW pass* — already exists as `harmonia.irealb_aligner`. The stale
`irealb_<slug>.html` artifacts came from an older aligner that under-detected
repeats (autumn_leaves GT span 160 s vs inferred 422 s); the **current** aligner
tiles correctly (span 422 = 422 s, 8 choruses). Re-running it fresh gives valid,
shared-clock pairs.

**Headline (9 gate-passing songs, 2 680 chords, pooled):**

| metric | value |
|---|---|
| root accuracy | **0.47** |
| family accuracy (maj/min/dom/dim/hdim/sus/aug) | 0.40 |
| joint root+family | 0.27 |
| coarse majmin | 0.59 |
| **DTW spurious-alignment floor (mean)** | **0.34** |
| **mean per-song root lift over floor** | **+0.13** |

**The single most important caveat:** `align_irealb_to_inferred` picks the best
of 12 transpositions and time-warps to fit, so it aligns a **wrong** chart to a
baseline ~0.34 root agreement. The pooled 0.47 is therefore only **+0.13 real
recognition** on average, and that lift is **concentrated in two clean-audio
songs**. Quoting "47 % root accuracy" without the floor would be exactly the
fabricated-number trap #35 warned about (CLAUDE.md #2/#3).

## Method

- **Inferred chords** are read from the embedded `const P = {…}` in
  `docs/plots/inferred_<slug>.html` — the *current production model's* output on
  the real audio (root pc, `lv.*` quality dict, `t0/t1`, per-chord confidence
  `lv.exact.c`). No pipeline re-run.
- **iReal GT** is parsed from the corpus tune (`data/ireal/{jazz1460,pop400}.txt`
  → `tune_to_mma`). 14 of 19 inferred charts map to a corpus tune.
- **Alignment:** `align_irealb_to_inferred(mma, p_chords)` transfers timestamps
  onto the GT sequence. Each GT chord is paired with the inferred chord active at
  its midpoint (which lands inside the DTW-matched inferred segment). Root and
  family (`tab_aligner._family`) are scored for both sides consistently.
- **Premise gate (CLAUDE.md #2, #35):** a song counts only if `gt_span/inf_span
  ≥ 0.70` and `aligned_frac ≥ 0.70`. This drops the 5 symbolic charts with no
  audio timing (anthropology[_phone], blue_skies, bye_bye_blackbird, satin_doll —
  none are in `docs/audio/`), which otherwise report degenerate 0 coverage.
- **Spurious floor:** each inferred chart is *also* aligned to 2–3 deliberately
  wrong tunes; the mean root accuracy of those wrong pairings is the floor. Lift
  = correct − floor.

## Per-song (gate-passing, sorted by lift)

| song | n | root | floor | **lift** | family | joint | reps |
|---|---:|---:|---:|---:|---:|---:|---:|
| the_beatles … let_it_be | 304 | **0.66** | 0.28 | **+0.38** | 0.77 | 0.59 | 2 |
| blue_bossa_150bpm_backing_track | 272 | **0.62** | 0.35 | **+0.27** | 0.52 | 0.45 | 12 |
| adele_hello¹ | 325 | 0.41 | 0.30 | +0.12 | 0.32 | 0.19 | 3 |
| my_baby_just_cares_for_me | 273 | 0.48 | 0.38 | +0.10 | 0.37 | 0.26 | 4 |
| autumn_leaves | 264 | 0.43 | 0.33 | +0.10 | 0.35 | 0.22 | 8 |
| blue_bossa | 663 | 0.41 | 0.34 | +0.07 | 0.29 | 0.18 | 22 |
| autumn_leaves_remastered | 132 | 0.41 | 0.35 | +0.06 | 0.29 | 0.17 | 3 |
| muppets_kermit (Bein' Green) | 132 | 0.40 | 0.35 | +0.05 | 0.46 | 0.28 | 2 |
| ray_charles_georgia | 315 | 0.42 | 0.38 | +0.04 | 0.34 | 0.21 | 3 |

¹ `adele_hello` maps to the jazz standard "Hello", **not** Adele's song (title
collision, unverified same tune) — kept as a control; treat as ambiguous.

**Read:** clean, close-mic'd audio (a studio pop master; a metronomic backing
track) is recognised well above floor. On real, full-mix jazz recordings the
per-chord root recognition is only **+0.04 … +0.10** above the DTW floor — i.e.
barely distinguishable from chance under this alignment method.

## Recognition by true chord family (root+family, pooled)

| true family | acc | n |
|---|---:|---:|
| maj | **0.46** | 740 |
| min | 0.22 | 1022 |
| dom | 0.21 | 642 |
| hdim | 0.14 | 276 |

Major chords are recognised ~2× better than everything else; half-diminished
(the ii of a minor ii-V) is nearly lost. This corroborates #35 finding #3
("quality collapses to maj/dom; ø/dim read as their relative maj/dom").

## Confidence correlation (confounded — do not over-read)

Pooled, high-confidence chords (`lv.exact.c ≥ 0.5`) have root acc **0.43** vs
low-confidence **0.49** — *inverted*. This is Simpson's paradox across the
two-domain confidence split (#26/#35 finding #2): `blue_bossa_150bpm` is a
globally-low-confidence song yet one of the most accurate, dragging the low-conf
pooled bucket up. Confidence-vs-accuracy must be read *within* a domain, not
pooled.

## What this does NOT measure (scoped out; harness is ready)

- **Baseline vs retrained-quality-head vs LLM-priors A/B (M2/M5 V2A).** The
  inferred HTML is fixed output of one model configuration. A three-way A/B needs
  the audio pipeline re-run per config (Basic Pitch + model load, ~heavy). No
  time budget was granted for that here. `validate_against_ireal.py` will score
  *any* set of `inferred_<slug>.html` charts, so the A/B is a matter of
  re-rendering the charts under each config and re-pointing the script.
- **≥20 songs.** Only 14 of 19 local inferred charts have a corpus-tune match and
  only 9 pass the gate. Reaching 20 needs more iReal charts fetched (network /
  server), not more local data.

## Bottom line

The GT-eval is unblocked and the numbers are honest. Corpus-wide, the model's
per-chord recognition on real audio is **modest and floor-dominated**: ~+0.13
mean root lift, carried almost entirely by two clean-audio songs. The clear,
actionable failure is **quality**: min/dom/hdim families collapse toward maj/dom
(see the family table), which is the highest-leverage target for M2 and for the
5th-apart / relative-maj confusion already tracked in #5 and #35.
