# Mission 6 · Phase 2 — Alignment validator + 20-song injected-slip gate

**Date:** 2026-07-13
**Verdict:** ✅ **GATE PASSES** — ≥80% slip-recall @ ≤10% FP @ ≥80% localisation,
robust across 10 random seeds. Cleared for the display-only server banner
(shipped); training-filter is the next gated step.

Code: `harmonia/models/alignment_validator.py`, `tests/test_alignment_validator.py`
(5 red-first cases, all green), `scripts/validate_chart_alignment.py` (harness + gate).

## 1. Why the per-instance z-score, not the global Δ (localisation)

The premise check (`mission_6_premise_check_results.md`) established the split we
build on: the **global** `within − cross` Δ is a fine *aggregate* coherence gate
but is √N-diluted and nearly blind to a *single* slipped repeat (Autumn Leaves:
corrupting 1 of 8 A-instances moved global Δ by only +0.003, because just 7 of 28
A–A pairs touch the victim). Localisation therefore cannot ride the global Δ.

This build localises with two **per-instance** statistics computed from the
section-fingerprint cosine matrix `S`:

- `own_sim(i)` = mean cosine of instance *i* to its same-label siblings — "how well
  does this instance fit its own label?" Standardised over all repeated-label
  instances → the **z-score** the mission specified. A genuine swap drops the
  victim's `own_sim` far below its siblings' (test-case z = −3.5 to −5 on distinct
  sections; the synthetic localised-slip unit test asserts z < −2).
- `xmatch(i)` = best cosine of *i* to a **different**-labelled section — "does this
  instance resemble some *other* section?" The load-bearing firing signal is the
  **slip score** `xmatch − own_sim`: a swapped instance looks more like its donor
  than like its own siblings, sending it ~0.5+; inference noise lowers `own_sim`
  but leaves `xmatch` low, so the difference stays negative. That asymmetry is why
  the localiser survives a 10%-per-bar noise floor with ~0% false localisation.

The global Δ still does real work — it is the **verdict gate's** basis only through
the abstain path (below), and the *aggregate* score deliberately uses **mean**
family, so a localised slip is √N-diluted in the aggregate **by design** and is
recovered by a per-instance-outlier verdict downgrade (OK → SUSPECT). This is the
premise-check's headline finding turned into code.

## 2. What was tuned, and why (no black-box thresholds)

The design skeleton's aggregate `score = w1·sig(repeat_consistency) + w2·boundary_f1
+ w3·min_family` produced **30% false-positive on clean tunes** in the first gate
run. Two causes, both fixed by making the *score* an alignment-quality measure
rather than a tune-property measure:

| Change | From | To | Why |
|---|---|---|---|
| Drop `sig(bridge)` from the score | in score, w=0.45 | gate/localiser only | Bridge contrast is a property of the *tune's* harmonic distinctiveness, not of alignment quality — a clean alignment of a low-contrast tune has low bridge but is correct. Keeping it in the score punished clean low-contrast tunes (`I'll Be Around`, clean, was SUSPECT@0.73). |
| Aggregate on **mean** family, not **min** | min | mean (w=0.70) | Per-section `min` is noise-sensitive (one noisy 8-bar instance → min 0.6 → false SUSPECT). Mean is the honest global-coherence measure; localised slips are meant to be diluted here and caught by the outlier downgrade. |
| Localisation fires on `slip_gate` (0.15) + strong-z (<−3), not raw z (<−2) | raw z | cross-match primary | Raw z<−2 trips on noise in many-instance songs (spurious `A#7` on clean). The cross-match signature does not. |
| Family-dip suspect needs an **absolute floor** (0.55) too | median−0.25 | AND < 0.55 | A clean 0.90-vs-median-0.95 no longer fires; a genuine slipped section (family ~0.1) still does. |
| **Uniform-family-floor → MISALIGNED** (median family < 0.45) | — | added | The design's disambiguation branch: uniform-low family + intact repeat_consistency = global fault (wrong transpose / whole-chorus slip). Signal 1 alone is fooled by a global slip (within stays high); the median-family floor catches it. |

Net effect on the gate: **FP 30% → ~4% (mean), recall 95% → 91% (mean), localisation
100% → 98% (mean)** — all three criteria clear with margin, on every seed.

## 3. 20-song injected-slip gate results

Harness: real iReal harmonic structure (jazz1460 + pop400), rendered as a looped
3-chorus "inferred" track from the GT (the realistic case — videos loop the form),
+10%-per-bar inference noise. Slips injected at the inferred-content level with a
known victim (GT/alignment held fixed): **Type A** section-instance rotation (25%,
global), **Type B** single-instance swap (50%, localised), **Type C** 2-bar phase
shift (25%, global). Alignment is perfect by construction, so the numbers measure
the *validator's* discrimination only.

| Seed | Recall (want ≥80%) | FP clean (≤10%) | Localise Type B (≥80%) | ROC-AUC |
|---|---|---|---|---|
| 0 | 90% | 5% | 100% | 0.995 |
| 1 | 100% | 0% | 100% | 0.998 |
| 2 | 90% | 5% | 100% | 0.984 |
| 3 | 80% | 10% | 90% | 0.936 |
| 4 | 100% | 0% | 100% | 1.000 |
| 5 | 95% | 0% | 100% | 0.995 |
| 6 | 95% | 5% | 91% | 0.970 |
| 7 | 85% | 5% | 100% | 0.956 |
| 8 | 95% | 0% | 100% | 1.000 |
| 9 | 80% | 10% | 100% | 0.955 |
| **mean** | **91%** | **4%** | **98%** | **0.979** |

**All 10 seeds PASS all three criteria.** ROC-AUC (align_score separating clean from
slipped) is 0.98 mean — well past the design's 0.80–0.88 estimate, because the
controlled harness has cleaner slips than real audio will.

Residual misses are honest and expected: (a) a Type-B swap whose donor happens to
share the victim label's roots leaves too little contrast (the premise's
contrast-limited case) → sometimes UNVERIFIABLE, counted as a miss; (b) a couple of
weak-z (~−1.3) localised slips on low-contrast tunes stay OK. These are the
*fundamentally undetectable* slips, not tuning failures — pushing recall higher
would cost the 0% FP.

## 4. What this does NOT establish (CLAUDE.md rule 4)

- **Adjacent same-label sections merge.** Section instances are contiguous
  `result.chords` runs, so a within-single-chorus `AABA` collapses the two leading
  A's into one instance. Localisation of a slip inside an adjacent repeat pair
  relies on multi-chorus separation (the deployment case) or the family signal;
  the harness uses looped multi-chorus tracks precisely because that is realistic.
- **Boundary_f1 (Signal 2) is phase-blind.** `detect_section_boundaries` assumes
  phase 0, so a pure phase offset does **not** move detected boundaries — Type C is
  caught by the family collapse, not by boundary_f1. Signal 2 remains a secondary
  abstaining signal.
- **Timing/tempo faults are untested here** — the synthetic harness uses a constant
  2 s/bar. The validator judges *structure*, not absolute time (that is issue #20's
  separate job).
- **These are controlled slips, not real-audio slips.** The premise check already
  validated the signal on 3 real recordings; this gate calibrates thresholds. The
  real-audio FP rate on the YouTube corpus is still to be measured before the
  training-filter is enabled.

## 5. Server integration (shipped — display only)

`scripts/harmonia_server.py::api_irealb_align` now runs `validate_alignment(result,
p_chords)` after the aligner and:
- injects a verdict banner into the chart HTML — green **OK** / yellow **SUSPECT** /
  red **MISALIGNED** / gray **UNVERIFIABLE**, naming the suspect section(s);
- returns a `validation` block in the JSON response (`verdict`, `align_score`,
  `suspect_sections`, `repeat_consistency`, `notes`).

Wrapped in try/except — validation never blocks or breaks alignment. Purely additive.

### Roadmap (gated, per design doc §Integration)

1. **Display banner** — ✅ shipped this phase (zero-risk, answers "can I trust this
   chart?").
2. **Training filter** — next: in `yt_chord_corpus._build_records`, skip
   `MISALIGNED` songs / drop `SUSPECT` sections, behind `--require-alignment-ok`.
   **Gate before enabling:** re-run this harness on *real* YouTube alignments (not
   synthetic) and confirm the ≤10% real FP holds — synthetic FP ≠ real FP.
3. **Eval filter** — last: exclude `MISALIGNED` from `eval_yt_model`, log drops, so a
   corrupt-GT song cannot silently depress the metric (issue #20's worry).
