# Chord-correction error analysis

_Generated 2026-07-13T15:03:41Z by `scripts/analyze_corrections.py`._

- **Total corrections logged:** 2
- **Songs annotated:** 1 (autumn_leaves (2))
- **Corrections that changed the root:** 2
- **Corrections that changed the quality:** 0
- **Mean model confidence on corrected chords:** 0.394 (min 0.376, max 0.412)

## Top model weaknesses

1. Root errors cluster at **+2 (whole tone)** (2/2 root corrections) — a systematic transposition slip, not random.
2. No false-confidence cases: every correction sat at confidence <= 0.41 (all below the 0.70 band). The model was already uncertain where humans corrected it — a well-calibrated sign.
3. **High-impact errors:** 1 correction(s) each propagated to >= 3 chords — fixing these first yields the largest downstream cleanup.

## Top 10 most-corrected chord types (predicted quality)

| # | Predicted quality | Corrections |
|---|---|---|
| 1 | `maj7` | 1 |
| 2 | `hdim7` | 1 |

## Top 10 most-corrected roots (predicted root)

| # | Predicted root | Corrections |
|---|---|---|
| 1 | G# | 1 |
| 2 | G | 1 |

## Quality confusion (predicted -> corrected)

| Predicted | Corrected | Count |
|---|---|---|
| _(none)_ | | |

## Root confusion (predicted -> corrected)

| Predicted | Corrected | Motion | Count |
|---|---|---|---|
| G# | A# | +2 (whole tone) | 1 |
| G | A | +2 (whole tone) | 1 |

## Confidence bias (false confidence: conf > 0.70 yet corrected)

_No false-confidence cases: no correction had confidence above 0.70._

## High-impact errors (propagation >= 3)

| Song | Bar | Predicted | Corrected | Propagation | Reinfer changed |
|---|---|---|---|---|---|
| autumn_leaves | 2 | `G#:maj7` | `A#^7` | 3 | 3 |

## Local context (chords that changed in reinfer diffs)

| Old label | -> New label | Times |
|---|---|---|
| `G#:maj7` | `A#:maj` | 1 |
| `G:hdim7` | `A#:maj` | 1 |
| `A#:maj7` | `A#:min7` | 1 |
| `G:hdim7` | `C:min7` | 1 |
| `C:7` | `C#:maj7` | 1 |

---
_Feeds Mission 2 (retrain quality head — target the top confused qualities), Mission 3 (calibration — fix false-confidence cases), and future context-failure work._