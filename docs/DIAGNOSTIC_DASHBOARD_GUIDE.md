# Autumn Leaves Diagnostic Dashboard — User Guide

## Overview

The **Autumn Leaves Complete Diagnostics Dashboard** is an interactive web-based tool for investigating three critical issues in the harmonia chord recognition pipeline:

1. **Root Label Mismatches** — Ground Truth vs. inferred root discrepancies
2. **Semi-Markov Configuration** — Chord fragmentation and duration modeling
3. **Chroma-Distance Geometry** — Distance metric bias in chord similarity

## Quick Start

### Generate Diagnostic Data

Run the master diagnostic generator to create all analysis JSON files:

```bash
cd /Users/vincente/Documents/Projets\ Perso/Code/harmonia
python scripts/generate_all_diagnostics.py
```

This will:
- ✓ Analyze root label mismatches across 64 bars
- ✓ Check semi-Markov configuration status
- ✓ Audit chroma-distance metrics in the codebase

Output files:
- `docs/plots/autumn_leaves_root_mismatch.json` (26 KB)
- `docs/plots/autumn_leaves_semi_markov.json` (1 KB)
- `docs/plots/autumn_leaves_chroma_audit.json` (24 KB)

### View the Dashboard

Open in browser:
```
docs/plots/autumn_leaves_complete_diagnostics.html
```

Or open from project root:
```bash
open docs/plots/autumn_leaves_complete_diagnostics.html
```

## Dashboard Tabs

### 1. **Overview**
Central hub showing:
- Investigation status (✓ Complete / ⏳ Pending)
- Key findings summary with metrics
- Ranked recommendations for next steps

**Typical findings:**
- Root Accuracy: X% (Y/Z bars match)
- Semi-Markov Improvement: X%
- Chroma Issues Found: N

### 2. **Root Mismatches**

#### Root Label Comparison Table
Bar-by-bar breakdown with columns:
- **Bar** — Bar number (0–63)
- **Time** — Audio timestamp in seconds
- **Section** — Form section (A, B, C, etc.)
- **Phase 1 GT** — Ground Truth root from Phase 1
- **Phase 2 GT** — Ground Truth root from Phase 2 (if available)
- **Inferred** — Model's inferred root
- **Status** — ✓ Match or ✗ Mismatch
- **Offset** — Semitone difference (if mismatch)

**Filtering:**
- Filter by bar number, time, or section
- Filter by match status (Match / Mismatch / Offset Pattern)

#### Offset Pattern Analysis
Histogram showing distribution of semitone offsets:
- X-axis: Offset range (-6 to +6 semitones)
- Y-axis: Count of bars with that offset
- Green bar: Zero offset (correct)
- Red bars: Systematic error

**Interpretation:**
- Peak at 0 → No systematic offset (good)
- Peak at ±N → Systematic transposition (data problem)

#### Ground Truth Source Verification
Shows which GT was used:
- Phase 1 GT Source (e.g., iRealb)
- Phase 2 GT Source (if different)

⚠️ If sources differ, may explain inconsistencies.

#### Format Consistency Validator
Lists all unique root formats found:
- Example: "C", "C-7", "Cmaj7"
- ⚠️ Mixed formats detected (e.g., "C" vs "Cmaj7")

#### Interactive Transposition Tester
**Slider:** Offset from -6 to +6 semitones

**Metric displayed:** Accuracy % at each offset

**Interpretation:**
- Accuracy 100% at offset=0 → No systematic error ✓
- Accuracy peak at offset≠0 → Systematic transposition ✗

### 3. **Semi-Markov Configuration**

#### Configuration Status Cards
Checklist of semi-Markov setup:
- ✓ or ✗ Semi-Markov enabled?
- ✓ or ✗ Duration prior file exists?
- ✓ or ✗ Duration prior is valid?

**Critical:** If all ✗, chord fragmentation will be high.

#### Fragmentation Comparison Chart
Bar chart comparing:
- **Without Semi-Markov** — Raw inferred chords/bar (red)
- **With Semi-Markov** — Estimated reduction (green)

Breaks down by section (A1, A2, B, C).

**Color zones:**
- GREEN: < 1.1 chords/bar (good)
- YELLOW: 1.1–1.5 chords/bar (moderate)
- RED: > 1.5 chords/bar (bad fragmentation)

#### Duration Prior Distribution
Histogram showing:
- X-axis: Duration in beats (1, 2, 3, 4, ...)
- Y-axis: Count

**Interpretation:**
- ⚠️ Heavy mass at 1-beat → Prior is weak
- ✓ Minimal mass at 1-beat → Prior is strong

#### Before/After Segmentation Comparison
Shows first 16 bars side-by-side:
- Left side: Inferred chord segments without semi-Markov
- Right side: Segments with semi-Markov

Highlights where semi-Markov removed spurious 1-beat segments.

#### Key Metrics
Summary statistics:
- Chords/bar without semi-Markov
- Chords/bar with semi-Markov
- Improvement percentage
- Overall status (CRITICAL / OK)

### 4. **Chroma Geometry**

#### Distance Metrics Audit Table
All distance metric usages in codebase:

Columns:
- **File** — Source file (e.g., `harmonia/models/chord_hmm.py`)
- **Function** — Function name
- **Type** — Metric type (Euclidean, Cosine, etc.)
- **Severity** — Tier 1 (Training) / Tier 2 (Inference) / Tier 3 (Diagnostic)
- **Input Shape** — Tensor shape (e.g., [12, 12])
- **Notes** — Contextual notes

**Filters:**
- By severity tier

**Key issues to watch:**
- ⚠️ Tier 1 (Training) uses Euclidean distance → Potential root cause of systematic errors
- Tier 2 (Inference) uses inconsistent metrics → May cause inference failures

#### Chromatic vs Circle-of-Fifths Geometry
Side-by-side heatmaps (12x12 note-pair distances):

**Left: Chromatic Layout (Euclidean)**
- C-to-C# distance = 1 (adjacent semitone)
- C-to-G distance = 7 (far in chromatic space)

**Right: Circle of Fifths (Harmonic)**
- C-to-C# distance = 11 (far in harmonic space)
- C-to-G distance = 1 (perfect fifth = close)

**Why it matters:**
Model trained with Euclidean distance may confuse C↔C# (semitone neighbors) but correctly handle C↔G (harmonic relationship).

#### High-Confidence Error Overlay
Plot showing:
- Heatmap of observed high-confidence errors
- Color intensity = error frequency/confidence
- Overlay on distance metric space

**Red pattern:** Errors cluster on chromatic neighbors → **Confirms bias**

#### Proposed Fixes Panel
For each Tier 1 finding:

**Before Code:**
```python
# Current (Euclidean distance)
from scipy.spatial.distance import cdist
dist = cdist(chords1, chords2, metric='euclidean')
```

**After Code:**
```python
# Proposed (Circle-of-fifths distance)
from scipy.spatial.distance import cdist
def fifths_distance(u, v):
    # Circle-of-fifths implementation
    ...
dist = cdist(chords1, chords2, metric=fifths_distance)
```

**Impact:**
- Estimated accuracy recovery: +X%
- Retraining required: ✓ Yes / ✗ No

### 5. **Feedback Form**

Interactive form to verify findings and provide ground truth:

#### Question 1: Root Label Systematic Offset
**Options:**
- ✓ No offset — all roots should match at 0
- ✗ Yes, positive offset — specify
- ✗ Yes, negative offset — specify
- ? Unclear — need more information

**If offset selected:** Specify exact semitone offset

#### Question 2: Ground Truth Source
**Options:**
- iRealb annotations
- POP909 annotations
- Guitar tabs
- Other (specify)

**If known:** Provide file path

#### Question 3: Semi-Markov Effectiveness
**Options:**
- ✓ Yes, clear improvement (>20% reduction)
- ~ Some improvement (5–20%)
- ✗ No improvement, still 1.12–2.0 chords/bar
- ✗ Semi-Markov is disabled/broken

**Optional notes:** Describe what you observe

#### Question 4: Chroma Geometry Error Bias
**Options:**
- ✓ Yes, confirmed chromatic bias
- ? Unclear from visualization
- ✗ No, errors are random/distributed differently

**Optional notes:** Describe pattern

#### Additional Notes
Free-form text for any other observations.

**Submit:** Saves feedback to local storage with timestamp
- Persists across browser sessions
- Can be exported for analysis

## Data Files Reference

### autumn_leaves_root_mismatch.json
```json
{
  "song": "autumn_leaves",
  "timestamp": "2026-07-14T...",
  "summary": {
    "total_bars": 64,
    "accuracy_percent": X,
    "offset_distribution": {...}
  },
  "diagnostics": [
    {
      "bar": 0,
      "gt1_root": "C",
      "inferred_root": "C#",
      "offset": 1,
      "match": false
    }
  ]
}
```

### autumn_leaves_semi_markov.json
```json
{
  "enabled": true,
  "duration_prior_file": "...",
  "fragmentation_without": 1.45,
  "fragmentation_with": 1.10,
  "summary": {
    "improvement_percent": 24.1,
    "status": "OK"
  }
}
```

### autumn_leaves_chroma_audit.json
```json
{
  "findings": [
    {
      "file": "harmonia/models/chord_hmm.py",
      "function": "score_chords",
      "type": "Euclidean distance",
      "severity": 1
    }
  ],
  "recommendations": [...]
}
```

## Troubleshooting

### Dashboard shows "Data incoming..."
- Diagnostic data files not generated yet
- **Fix:** Run `python scripts/generate_all_diagnostics.py`

### All inferred roots are null
- No inferred annotations available
- **Fix:** Generate inferred annotations from model output

### Semi-Markov metrics show 0%
- Inferred annotations file missing
- **Fix:** Load or generate inferred chord file

### Offset histogram empty
- All roots match (offset = 0)
- **Status:** ✓ No systematic error (good news)

## Next Steps Based on Findings

### If Root Accuracy < 50%
1. Check GT source discrepancy (Phase 1 vs Phase 2)
2. Look for systematic offset in histogram
3. Use transposition tester to find peak accuracy offset
4. Verify with feedback form

### If Semi-Markov is disabled
1. Check configuration status in tab
2. Verify duration prior file exists
3. Rebuild duration prior if invalid
4. Re-run inference with semi-Markov enabled

### If Chroma Errors cluster on chromatic neighbors
1. Review Tier 1 findings (training-stage distances)
2. Consider switching to circle-of-fifths metric
3. Retrain chord templates with harmonic distance
4. Re-evaluate on test set

## For Developers

### Adding new diagnostic data sources
1. Create Python script in `scripts/generate_*.py`
2. Output JSON to `docs/plots/autumn_leaves_*.json`
3. Dashboard automatically loads via `fetch('./autumn_leaves_*.json')`

### Customizing dashboard UI
Edit: `docs/plots/autumn_leaves_complete_diagnostics.html`

Key sections:
- CSS variables: `--bg`, `--fg`, `--success`, `--error`, etc.
- Tab rendering: `renderOverviewTab()`, etc.
- Data loading: `loadAllData()` fetches JSON files

### Theme support
Dashboard automatically detects light/dark mode from browser:
```css
@media (prefers-color-scheme: dark) { ... }
@media (prefers-color-scheme: light) { ... }
```

## Contact & Issues

For issues or feedback about the dashboard:
- Check `docs/known_issues.md` — authoritative issue tracker
- Run diagnostics and submit feedback form
- Results saved to browser local storage

---

**Last Updated:** 2026-07-14
**Version:** 1.0 (Initial Release)
