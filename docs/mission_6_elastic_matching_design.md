# Mission 6 · Part 2 — Elastic structural-matching design

**Date:** 2026-07-13
**Status:** design (algorithm + confidence metric + estimated accuracy)

## Goal

Given a *candidate* alignment (the output of `align_irealb_to_inferred`), decide
**is this alignment structurally coherent?** — and if not, **which section
slipped?** — without any absolute-time ground truth.

The core object is the chart's own form. iReal gives us, for free, the exact
symbolic structure: section labels, bar counts, and repeats
(`parse_form_compact` → `[A(8,×2), B(8,×1), C(8,×1)]`). We check that this
structure *survives* the mapping onto the inferred chords.

## Three signals, one score

Alignment coherence is judged on three independent signals. Each is cheap, each
targets a different failure shape from Part 1, and each degrades gracefully to
"abstain" when the input is structurally ambiguous.

### Signal 1 — Repeat consistency (the load-bearing check)

**Premise (already validated in `section_structure.py`):** on a symbolic chord
SSM, two instances of the same section label are far more similar to each other
than to a differently-labelled section (bridge-contrast +0.05–0.11; odd-one-out
85% across 371 AABA tunes). A slipped repeat (failure #3) destroys exactly this.

**Method.** From the alignment, collect the **inferred** chord sequence that got
mapped under each iReal section instance. Build a key-relative,
quality-augmented feature per bar (reuse `build_chord_ssm`'s
`[root one-hot | quality one-hot]` representation) and summarize each section
instance as an L2-normalized mean vector (its *fingerprint*, exactly as
`label_sections` does). Then:

```
within_A   = mean cosine( fingerprint(A_i), fingerprint(A_j) )   over same-label pairs
cross      = mean cosine( fingerprint(A),   fingerprint(B)   )   over diff-label pairs
repeat_consistency = within − cross          # want > 0, ideally > ~0.10
```

- `within` high, `cross` low  → repeats are internally consistent, form survived.
- `within` collapses          → a repeat slipped (failure #3), **localizable** to
  the section instance whose fingerprint is the outlier.
- Contrast is *elastic by construction*: it compares mean fingerprints, so a
  section that is 15 bars in the audio vs 16 in the chart still matches — length
  warp does not hurt it.

This is the signal that catches the case the user described ("look at the base
chart structure and you can tell it's misaligned").

### Signal 2 — Section-boundary agreement (elastic IoU)

**Targets:** phase offset (#2), length mismatch (#4), tempo-octave warp (#5).

The iReal chart declares boundary bar positions (cumulative section lengths):
`A|A|B|A` at bars `{8,16,24}`. Independently, run the audio-side structure
detector on the inferred sequence — `detect_section_boundaries(build_chord_ssm(...))`
— to get inferred boundaries. Convert both to the *same axis* (bars, via the
alignment's beat→bar map) and score agreement with an **elastic tolerance**:

```
match a chart boundary to the nearest inferred boundary within ±τ bars (τ ≈ 2)
boundary_recall    = fraction of chart boundaries matched
boundary_precision = fraction of inferred boundaries matched
boundary_f1        = harmonic mean
```

A constant offset in *all* matched boundaries = phase slip (#2). A ~2× spacing
ratio = tempo octave (#5). Missing/extra boundaries in one region = length
mismatch (#4). Abstain when the audio SSM yields no boundaries (through-composed
or too short — `detect_section_boundaries` returns `[]`).

### Signal 3 — Per-section label agreement (localized family-fraction)

**Targets:** localization of #1/#3/#6.

The aligner already computes per-chord `match ∈ {exact, family, mismatch, gap}`
via `_family_dist`. Mission 6 simply **groups these by iReal section** instead of
reporting one global number:

```
for each section instance s:
    section_family_frac[s] = (#exact + #family) / (#chords in s)
```

A healthy alignment has all sections' `family_frac` in a tight high band. A
*localized* dip (one section ≪ the rest) pins the slip. A *uniform* floor points
at a global fault (wrong transpose #6, or whole-chorus slip #1) — which Signal 1
then disambiguates (uniform floor + high repeat_consistency = wrong transpose;
uniform floor + low repeat_consistency = chorus slip).

## The elastic section-matcher (constrained DTW)

Signals 1–3 assume the chart sections and the inferred sections are already put
in correspondence. That correspondence is itself an **elastic match** — the
mission's "one iReal bar might be 0.8–1.2 inferred bars" requirement. We do it
with a **constrained DTW over section fingerprints**, not raw bars:

- **Sequences:** chart section instances `[A,A,B,A]` (fingerprints from iReal
  chord tones via `chord_pc_weights`-style templates) vs inferred sections
  `[X,Y,Z,W]` (fingerprints from `build_chord_ssm`).
- **Local cost:** `1 − cosine(fingerprint_chart_i, fingerprint_inf_j)`, plus a
  **length-warp penalty** `λ·|log(bars_i / bars_j)|` that is ~0 inside the
  elastic band `[0.5, 2.0]` and grows sharply outside it (guards against warp
  holes #4).
- **Constraints:** monotonic, one-to-one at the section level (no section may be
  matched twice — a section matched twice *is* the slip we are detecting), with a
  bounded skip for a genuine extra audio section (solo chorus, intro vamp).
- **Consistency tie-in:** the label structure `A=A≠B` is enforced as a soft
  constraint — two chart sections with the same label should route to inferred
  sections with high mutual fingerprint similarity; a matching that violates this
  is penalized. This is what turns a generic DTW into a *form-aware* matcher.

Section-level DTW (a handful of sections) is O(n²) on tiny n — negligible cost,
and far more robust than bar-level DTW because each fingerprint pools ~8–16 bars
of evidence (√N denoising, cf. issue #28).

## The overall confidence metric

Combine the three signals into one score in `[0,1]` plus a categorical verdict:

```
align_score = w1·sig(repeat_consistency)      # sig = clip((x+0.05)/0.20, 0, 1)
            + w2·boundary_f1
            + w3·min_section_family_frac        # the WORST section, not the mean
            with  w1=0.45, w2=0.20, w3=0.35     # (initial; tune on eval, §accuracy)
```

Use the **worst** section's family-fraction, not the mean, in `w3` — a single
slipped section is the failure we most need to catch, and averaging hides it.

**Verdict thresholds (initial, to be calibrated):**

| `align_score` | verdict | action |
|---|---|---|
| ≥ 0.75 | `OK` | admit to training/eval; show chart normally |
| 0.55–0.75 | `SUSPECT` | show chart with a warning banner; exclude from training |
| < 0.55 | `MISALIGNED` | flag section(s); exclude from training **and** eval |
| any, but Signals 1&2 both abstain | `UNVERIFIABLE` | through-composed / too short; fall back to global `dtw_cost` only, mark low-confidence |

**Localization output:** the section with the largest gap below the song's median
`section_family_frac`, plus its inferred-vs-chart fingerprint distance — this is
the "which section slipped and by how much" report.

## Why this sidesteps the Mission-1 wall

Issue #20 / Mission 1B failed because absolute alignment needs harmonic SNR that
full-mix chroma does not carry (~0.03 cosine key gap). Every Mission-6 signal is
a *relative* comparison of the audio to itself or the chart to itself:

- Signal 1 compares inferred sections to *other inferred sections* — the SNR
  cancels (both sides share the same recording's timbre/noise floor).
- Signal 2 compares two *boundary sets*, not two chroma vectors.
- Signal 3 is already-computed match labels, re-grouped.

None of them re-open the frame-accurate-chroma problem. That is the whole reason
the mission is feasible where Mission 1 was not.

## Estimated accuracy

No labelled misalignment set exists yet, so these are *design estimates* to be
replaced with measured numbers once the eval harness (Part 3) runs:

- **Repeat-consistency separation:** the underlying bridge-contrast is +0.08 mean
  with 85% odd-one-out on GT chords. On *inferred* chords the contrast shrinks
  (noisier fingerprints) — estimate the separation between aligned and
  1-section-slipped fingerprints at ~0.05–0.10 cosine, giving an expected
  **ROC-AUC ~0.80–0.88** for detecting a whole-section slip. Strong for #3,
  weaker for a subtle 2-bar phase offset.
- **Boundary F1** is reliable only when `detect_section_boundaries` fires
  (AABA/verse-chorus with clear repeats); expect it to abstain on ~15–25% of the
  YouTube corpus (through-composed, short clips).
- **Localization:** when a slip *is* detected, pinning the right section via the
  worst-family-fraction should be **>90%** (the signal is sharp once the global
  detection clears).

**Stopping criterion for adoption (CLAUDE.md handoff rule):** on a 20-song set
with hand-injected slips (shift the alignment by one section), the validator
should flag `MISALIGNED`/`SUSPECT` on ≥ 80% of injected-slip cases at ≤ 10% false-
positive on the clean alignments, *and* localize the correct section on ≥ 80% of
true detections. If it clears that, wire it into the training filter; if not,
keep it display-only (a warning banner) and iterate on the signal weights.

## Open design questions (log before building, CLAUDE.md rule 5)

1. **Fingerprint representation for inferred sections** — root-only (robust,
   matches the aligner's dedup-on-root philosophy) vs root+quality (more
   discriminative, noisier). Start root-only for Signal 1's SSM (quality
   inference is the weakest link); A/B on the eval set.
2. **Single-song vs corpus threshold** — thresholds above are single-song
   estimates; per CLAUDE.md rule 5 they must be checked corpus-wide before being
   trusted. The eval harness runs on the full iRealb/YouTube set, not one tune.
3. **Interaction with `correct_section_phase`** — a detected phase slip (#2)
   could be auto-fixed rather than merely flagged, since the fixer already exists.
   Keep detection and correction separate until detection is validated.
