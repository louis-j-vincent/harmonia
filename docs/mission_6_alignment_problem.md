# Mission 6 · Part 1 — The alignment problem: what misalignment looks like and why it matters

**Date:** 2026-07-13
**Status:** analysis (research/design mission, no code shipped in this part)

## The hidden assumption

Every time we overlay an iReal Pro chart onto a YouTube inference we assert a
map

```
iReal bar/beat  ──►  audio time (s)  ──►  inferred chord slot
```

built by `harmonia/irealb_aligner.py::align_irealb_to_inferred` (DTW on a
family-level chord distance, per-chorus, with an SSM chorus-boundary detector).
The map is *assumed correct* by everything downstream. It is never checked
structurally. When it slips, the failure is silent and it corrupts:

- **Training** — `harmonia/data/yt_chord_corpus.py` reads `result.chords[i].t0/t1`
  and uses the iReal label as the **supervised target** for the audio features at
  those beats. A slipped alignment writes wrong `(features → label)` pairs into
  the corpus. Garbage labels, confidently stored.
- **Evaluation** — a misaligned GT boundary marks a *correct* model prediction as
  a *miss* (and vice-versa). This is exactly the pathology issue #20 warned about:
  a GT off by ~1.5 s is "worse than useless" at chord change-points.
- **The user-facing chart** — `/api/irealb-align` and `/api/irealb-compare` paint
  the GT labels onto the inferred timeline. A slip shows the right chords under
  the wrong bars; the whole chart reads as wrong even when inference was fine.

So alignment quality is a **QA gate that must run before we trust an inferred
chart** — for display, for training admission, and for eval.

## What misalignment actually looks like

Concrete failure shapes, each traceable to a specific mechanism in the current
aligner:

### 1. Whole-chorus slip (repeat miscount)
`_estimate_repeats` / `_find_chorus_boundaries` decide how many times the form
repeats by `round(total_dur / one_chorus_secs)`. On a 17-min recording with
solos (A Foggy Day, issue #20) or a rubato ballad the chorus count is wrong, so
chorus *k*'s iReal form gets laid over chorus *k±1*'s audio. Every label is
shifted by one full form.
- **Signature:** exact/family fractions are *uniformly mediocre* across the whole
  song (not localized); the DTW warp has a big monotone offset.

### 2. Phase / pickup offset (the AABA rotation bug)
`detect_section_boundaries` assumes **phase 0** (section 0 starts at beat 0). A
pickup bar, count-in, or intro vamp shifts the true grid by a few bars. This is
issue #22's documented "cycle-shift bug" (Let It Be's `C-G-Am-F` came out with
the tonic landing *last*). The chart's A section is laid over the audio's turn
of the previous section.
- **Signature:** a *constant small bar offset* everywhere; same-label sections are
  still internally consistent with each other, just shifted vs the audio.

### 3. Slipped repeat / form-order disagreement
iReal says **A-A-B-A**; the alignment maps the audio's second A onto the chart's
B (or maps B twice). One section is placed on content that belongs to a
different section.
- **Signature:** **localized** — one section's family-fraction collapses while the
  rest stay high. This is the case the user described ("if you look at the base
  chart structure you can deduce a misalignment"): the two A's *should* be nearly
  identical harmonically; if the aligned inferred content under A1 and A4 disagree,
  a repeat slipped.

### 4. Section-length mismatch (warp holes)
DTW's free warp lets many iReal bars collapse into one inferred slot (a "warp
hole" — the aligner code explicitly guards against this in `_assign_timestamps`)
when inference is wrong or silent over a span. iReal says a section is 16 bars;
the alignment squeezes it into 8 inferred bars.
- **Signature:** DTW path has a long flat run (steep local slope); the affected
  section's chords get near-zero individual durations.

### 5. Tempo-octave 2× warp
Issue #1 / #20 pattern: librosa locks a 2× tempo octave (song 002 GT 129 vs
detected 63; Ghost-of-a-Chance 117.5 vs true ~58). The beat→time map is off by
2×, so chart-beat *N* lands at half its true time. Global linear stretch of the
whole alignment.
- **Signature:** warp slope ≈ 2.0 (or 0.5) everywhere; boundaries land at
  consistent fractional bar positions.

### 6. Wrong global transpose (key ambiguity)
`_best_transpose_family` picks the semitone offset minimizing DTW cost over all
12. Symmetric qualities (e.g. all-dominant or diminished passages) make several
offsets near-tie; a wrong pick maps the right *shape* a few semitones off.
Airegin (issue #20) was a genuinely different-key recording (+2 vs the F-minor
chart) — no honest alignment exists, and the aligner will still return one.
- **Signature:** family-fraction floor across the whole song; the chosen transpose
  is a near-tie with the runner-up.

## Why they happen — root causes (grouped)

| Root cause | Mechanism in code | Failures it drives |
|---|---|---|
| No phase model | `detect_section_boundaries` assumes phase 0 | #2 |
| Duration-ratio repeat count | `_estimate_repeats` = `round(dur / chorus)` | #1 |
| 1-chorus chart ≠ multi-chorus audio | chart is one head; audio has solos | #1, #4 |
| Beat tracker tempo octave | librosa / madmom 2× lock (issue #1) | #5 |
| Free-warp DTW | `_dtw_family` has no slope/monotonicity constraint | #3, #4 |
| Argmax transpose over 12 | `_best_transpose_family` near-ties | #6 |
| Bad benchmark input | transposed / compilation recordings | #6 |

## The insight that makes Mission 6 tractable

Mission 1 (issue #20) tried to *build* a frame-accurate alignment and failed
twice — beat-grid and chord-template↔chroma DTW both blew the ±150 ms gate
because full-mix chroma carries only ~0.02–0.04 cosine of key-discrimination SNR.
The lesson: **absolute-time GT is expensive and SNR-limited.**

Mission 6 is a *different* question. We do not need absolute time. We need
**relative structural consistency**: given a *candidate* alignment, does the
chart's own form (A-A-B-A, repeats, section lengths) survive the mapping onto the
inferred structure? That is cheap and robust because:

- It reuses signals already validated in `section_structure.py`: the symbolic
  **chord SSM** correctly makes a bridge less similar to A than the two A's are
  to each other (bridge-contrast +0.05–0.11 on 6/8 tunes; odd-one-out 85% across
  371 AABA tunes). That contrast is the exact statistic a slipped repeat destroys.
- It is **non-circular** and needs no hand-annotation: the check compares the
  alignment against the chart's *own* declared structure and against the audio's
  *self*-similarity — never against an external time reference.
- It **localizes** the fault (which section slipped), which absolute-error metrics
  cannot.

So Mission 6 is not another alignment method. It is a **structural validator /
QA gate** layered on the existing aligner: flag the alignment, localize the bad
section, and gate the chart out of training/eval when it fails.

See `mission_6_elastic_matching_design.md` for the algorithm and
`mission_6_implementation.md` for the build.

## What this analysis does NOT solve (CLAUDE.md rule 4)

- It does not *fix* misalignments, only *detects and localizes* them. A detected
  phase slip (#2) could be auto-corrected via `correct_section_phase` (already in
  `section_structure.py`); a repeat miscount (#1) could trigger a re-count. Those
  fixes are follow-on work, gated on the detector proving reliable first.
- The validator cannot distinguish "alignment is wrong" from "inference is wrong"
  when *both* the chart form and the audio structure are ambiguous (e.g. a
  through-composed tune with no repeats gives the validator nothing to check).
  It reports low **confidence**, not a false "misaligned" — see the design doc's
  abstain path.
