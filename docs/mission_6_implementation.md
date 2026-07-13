# Mission 6 · Part 3 — Implementation roadmap

**Date:** 2026-07-13
**Status:** roadmap + code skeleton (nothing shipped in this mission)

Builds on Parts 1 (`mission_6_alignment_problem.md`) and 2
(`mission_6_elastic_matching_design.md`). The validator is a thin QA layer over
the **existing** aligner — it adds no new alignment method and re-uses
`irealb_aligner` and `section_structure` wholesale.

## New module: `harmonia/models/alignment_validator.py`

Pure, dependency-light (numpy only), unit-testable without audio. All the audio
work already happened upstream in `align_irealb_to_inferred`; this consumes its
result plus the inferred per-beat chord sequence.

```python
"""alignment_validator.py — structural QA gate for iReal↔inferred alignment (Mission 6).

Given a candidate alignment (irealb_aligner.IRealbAlignmentResult) plus the
inferred per-beat chord sequence, judge whether the chart's own form (A-A-B-A,
repeats, section lengths) survives the mapping — WITHOUT any absolute-time GT.
See docs/mission_6_elastic_matching_design.md for the three signals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from harmonia.models.section_structure import (
    build_chord_ssm, detect_section_boundaries,
)

# ── outputs ────────────────────────────────────────────────────────────────────
@dataclass
class SectionReport:
    label: str                 # iReal section label, e.g. "A"
    instance: int              # 0,1,... which repeat of that label
    bar0: int; bar1: int       # chart bar span
    family_frac: float         # (#exact+#family)/#chords  in this section
    n_chords: int
    fingerprint: np.ndarray = field(default=None, repr=False)

@dataclass
class AlignmentValidation:
    align_score: float                 # [0,1] overall coherence
    verdict: str                       # OK | SUSPECT | MISALIGNED | UNVERIFIABLE
    repeat_consistency: float          # within - cross  (Signal 1)
    boundary_f1: float                 # Signal 2  (nan if abstained)
    min_section_family_frac: float     # Signal 3 (worst section)
    suspect_sections: list[str]        # e.g. ["A#2"] localised slips
    sections: list[SectionReport]
    notes: list[str] = field(default_factory=list)

# ── Signal 1: repeat consistency ───────────────────────────────────────────────
def _section_fingerprints(sections, inferred_seq, n_pitches=12):
    """L2-normalised mean [root|quality] vector of the inferred chords mapped
    under each section (same representation as build_chord_ssm)."""
    # inferred_seq: per-beat (root_rel, quality_idx) already sliced per section
    fps = []
    for sec in sections:
        ssm_rows = build_chord_ssm(sec.inferred_beats, n_pitches)  # (k,k)
        fp = ssm_rows.mean(0) if len(ssm_rows) else np.zeros(1)
        n = np.linalg.norm(fp); fps.append(fp / n if n > 1e-9 else fp)
    return fps

def _repeat_consistency(labels, fps):
    """within(same-label) - cross(diff-label) mean cosine. Also returns the
    per-instance outlier score for localisation."""
    S = np.array([[float(a @ b) for b in fps] for a in fps])
    within, cross = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else cross).append(S[i, j])
    w = float(np.mean(within)) if within else np.nan
    c = float(np.mean(cross))  if cross  else np.nan
    # outlier: for each same-label instance, its mean sim to its siblings
    return w, c, S

# ── Signal 2: boundary agreement (elastic IoU) ─────────────────────────────────
def _boundary_f1(chart_boundary_bars, inferred_boundary_bars, tol=2):
    if not chart_boundary_bars or not inferred_boundary_bars:
        return float("nan")
    matched_c = sum(any(abs(cb - ib) <= tol for ib in inferred_boundary_bars)
                    for cb in chart_boundary_bars)
    matched_i = sum(any(abs(ib - cb) <= tol for cb in chart_boundary_bars)
                    for ib in inferred_boundary_bars)
    rec  = matched_c / len(chart_boundary_bars)
    prec = matched_i / len(inferred_boundary_bars)
    return 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)

# ── Signal 3: per-section family fraction (regroup aligner match labels) ────────
def _section_family_fracs(result_chords):
    """Group result.chords by 'section' → (#exact+#family)/#chords."""
    by_sec = {}
    for ch in result_chords:
        key = ch.get("section", "?")
        m = ch.get("match", "")
        by_sec.setdefault(key, []).append(1 if m in ("exact", "family") else 0)
    return {k: float(np.mean(v)) for k, v in by_sec.items() if v}

# ── top-level ──────────────────────────────────────────────────────────────────
def _sig(x, lo=-0.05, hi=0.15):        # squash repeat_consistency → [0,1]
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def validate_alignment(result, inferred_seq, mma_chart,
                       beats_per_bar=4,
                       w=(0.45, 0.20, 0.35),
                       thresholds=(0.75, 0.55)) -> AlignmentValidation:
    """result: IRealbAlignmentResult; inferred_seq: per-beat (root_rel, q_idx)
    of the WHOLE inferred sequence; mma_chart: for chart section structure."""
    sections = _build_section_views(result, inferred_seq, mma_chart, beats_per_bar)
    labels   = [s.label for s in sections]
    fps      = _section_fingerprints(sections, inferred_seq)
    w1, c1, S = _repeat_consistency(labels, fps)
    rep_cons = (w1 - c1) if (not np.isnan(w1) and not np.isnan(c1)) else float("nan")

    chart_bnds = _chart_boundary_bars(mma_chart)
    inf_bnds   = [b // beats_per_bar
                  for b in detect_section_boundaries(build_chord_ssm(inferred_seq))]
    bf1 = _boundary_f1(chart_bnds, inf_bnds)

    fam = _section_family_fracs(result.chords)
    min_fam = min(fam.values()) if fam else 0.0
    median_fam = float(np.median(list(fam.values()))) if fam else 0.0
    suspect = [k for k, v in fam.items() if v < median_fam - 0.25]

    # abstain path
    both_abstain = np.isnan(rep_cons) and np.isnan(bf1)
    if both_abstain:
        return AlignmentValidation(
            align_score=float("nan"), verdict="UNVERIFIABLE",
            repeat_consistency=rep_cons, boundary_f1=bf1,
            min_section_family_frac=min_fam, suspect_sections=suspect,
            sections=sections, notes=["no repeats + no inferred boundaries"])

    parts, weights = [], []
    if not np.isnan(rep_cons): parts.append(w[0] * _sig(rep_cons)); weights.append(w[0])
    if not np.isnan(bf1):      parts.append(w[1] * bf1);            weights.append(w[1])
    parts.append(w[2] * min_fam); weights.append(w[2])
    score = float(sum(parts) / sum(weights))

    hi, lo = thresholds
    verdict = "OK" if score >= hi else "SUSPECT" if score >= lo else "MISALIGNED"
    return AlignmentValidation(score, verdict, rep_cons, bf1, min_fam,
                               suspect, sections)
```

Helpers left as skeletons (`_build_section_views`, `_chart_boundary_bars`,
`_section.inferred_beats`) do the bookkeeping of turning `result.chords`
(which already carry `section` + `bar` + `t0/t1`) into per-section slices of the
inferred beat sequence. They are pure index arithmetic — no audio.

## CLI: `scripts/validate_chart_alignment.py`

```
Usage:
  # one song, from a saved alignment + inference
  python scripts/validate_chart_alignment.py \
      --irealb "irealb://..." --inference chart.json [--audio song.m4a]

  # batch QA over the YouTube corpus (for training-filter decisions)
  python scripts/validate_chart_alignment.py --corpus data/yt_corpus/ --out qa.jsonl
```

Behaviour:
1. Load/compute the inferred `p_chords` (from a saved chart json, or run
   `chord_pipeline_v1` if `--audio` given).
2. Parse the iReal URL → `MMAChart` (`tune_to_mma`); run
   `align_irealb_to_inferred` (the existing aligner).
3. Build the per-beat inferred `(root_rel, q_idx)` sequence (subtract the
   detected key so it is key-invariant, matching `build_chord_ssm`'s contract).
4. Call `validate_alignment(...)`; print the verdict + per-section table.
5. **Diagnostic plot (CLAUDE.md default):** emit an HTML with (a) the section
   fingerprint similarity matrix `S` with same-label pairs boxed, (b) chart vs
   inferred boundaries on a shared bar axis, (c) per-section family-fraction bars
   colour-coded by suspicion. A misalignment should be *visible*, not just a
   number — this is the "inspectable artifact" rule.
6. `--out` appends one JSON line per song for the batch training filter.

## Tests: `tests/test_alignment_validator.py` (red-first, CLAUDE.md rule)

- **Synthetic clean AABA** — inferred = chart's own chords tiled → `repeat_
  consistency > 0`, all `family_frac = 1.0`, verdict `OK`.
- **Injected repeat slip** — rotate the inferred content under A#2 to B's content
  → `repeat_consistency` drops, `suspect_sections == ["A#2"]`, verdict ≤ `SUSPECT`.
- **Injected phase offset** — shift inferred by 2 bars → `boundary_f1` drops with a
  *constant* offset signature; family-fracs stay uniform (distinguishes #2 from #3).
- **Through-composed / no repeats** — verdict `UNVERIFIABLE`, no false MISALIGNED.
- **Wrong transpose** — uniform family floor + high `repeat_consistency` → the
  disambiguation branch reports "global, not localized".

## Integration points

1. **Server (display) — immediate, low-risk.**
   In `scripts/harmonia_server.py::api_irealb_align` (and `_compare`), after the
   `align_irealb_to_inferred` call, run `validate_alignment` and extend the stats
   banner: add `verdict` + colour + `suspect_sections`. A `MISALIGNED` verdict
   turns the banner red and names the bad section. Purely additive; no decode
   change. **Do this first** — it gives the user the "can I trust this chart?"
   signal they asked for, with zero risk.

2. **Training filter — after the eval gate clears.**
   In `harmonia/data/yt_chord_corpus.py::_build_records` (right after the
   `align_irealb_to_inferred` call, line ~247), compute the validation and:
   - skip the song entirely if `verdict == MISALIGNED`;
   - drop only the suspect sections' records if `verdict == SUSPECT`;
   - keep all if `OK` / `UNVERIFIABLE` (unverifiable ≠ known-bad).
   Gate this behind a `--require-alignment-ok` flag until the detector clears the
   Part-2 stopping criterion (≥80% slip-recall @ ≤10% FP on the 20-song injected
   set), per CLAUDE.md "single-song findings are hypotheses" + "use as a filter
   only when validated."

3. **Eval — after training filter proves out.**
   In `scripts/eval_yt_model.py` / Mission-1 benchmark building: exclude
   `MISALIGNED` songs from the eval set and log how many were dropped, so a
   corrupt-GT song cannot silently depress the metric (issue #20's core worry).

## Estimated effort

| Task | Est. |
|---|---|
| `alignment_validator.py` (3 signals + combiner + helpers) | ~0.5 day |
| `tests/test_alignment_validator.py` (5 synthetic cases, red-first) | ~0.25 day |
| `scripts/validate_chart_alignment.py` + HTML diagnostic plot | ~0.5 day |
| Build the 20-song injected-slip eval set + run the stopping-criterion gate | ~0.5 day |
| Server banner integration (display path) | ~0.25 day |
| Training-filter + eval integration (gated on the gate) | ~0.25 day |
| **Total** | **~2.25 days**, front-loaded on the validator + eval gate |

Sequencing (ranked, CLAUDE.md handoff style):
1. Validator + tests + synthetic clean/slip cases (proves the signal exists at all).
2. `--validate-pilots`-style run on the 3 issue-#20 pilot audios (do the signals
   fire on *real* recordings, not just synthetic?). **Stop here and report** if
   `repeat_consistency` doesn't separate — the premise is falsified cheaply
   (CLAUDE.md rule 2) before building the 20-song harness.
3. 20-song injected-slip gate → if it clears, wire training filter + eval; else
   ship display-only banner and iterate weights.

## What this roadmap does NOT deliver (CLAUDE.md rule 4)

- No auto-*correction* of misalignment (phase re-shift via `correct_section_phase`,
  repeat re-count) — detection only. Correction is a separate follow-on, gated on
  detection reliability.
- No help when the chart form is genuinely ambiguous (through-composed) — those
  return `UNVERIFIABLE`, not a guess.
- Does not replace a real absolute-time benchmark (issue #20 / Mission 1's manual
  anchors) — it is orthogonal: this validates *relative* structure, the manual
  anchors give *absolute* time. Both are needed for different jobs.
