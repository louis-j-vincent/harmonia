# Autumn Leaves Diagnostic Dashboard — Complete Reference

## Overview

The **Autumn Leaves Complete Diagnostics Dashboard** is a comprehensive interactive investigation tool built to diagnose and investigate three interconnected issues in the harmonia chord recognition pipeline:

1. **Root Label Mismatches** — Discrepancies between Ground Truth and inferred root notes
2. **Semi-Markov Configuration** — Chord fragmentation metrics and duration modeling
3. **Chroma-Distance Geometry** — Distance metric analysis and potential systematic biases

## What You're Getting

### Dashboard Files
- **`docs/plots/autumn_leaves_complete_diagnostics.html`** — Main interactive dashboard (44 KB)
  - 5 tabs: Overview, Root Mismatches, Semi-Markov Config, Chroma Geometry, Feedback Form
  - Responsive design (desktop and mobile)
  - Light/dark mode support
  - Interactive charts via Plotly
  - Real-time feedback form with local storage persistence

### Data Generation Scripts
- **`scripts/generate_root_mismatch_diagnostics.py`** — Analyzes root label accuracy
  - Compares Phase 1 GT vs inferred roots across 64 bars
  - Detects systematic offset patterns
  - Validates chord format consistency
  
- **`scripts/generate_semi_markov_diagnostics.py`** — Assesses semi-Markov configuration
  - Checks if semi-Markov is enabled
  - Validates duration prior file
  - Estimates fragmentation with/without semi-Markov
  
- **`scripts/generate_chroma_geometry_audit.py`** — Audits distance metrics
  - Scans codebase for all distance metric usages
  - Categorizes by severity and context
  - Compares chromatic vs circle-of-fifths geometry
  - Identifies high-confidence error patterns

- **`scripts/generate_all_diagnostics.py`** — Master orchestrator
  - Runs all three diagnostic generators
  - Aggregates results
  - Creates manifest file

- **`scripts/quick_diagnostics.sh`** — Interactive CLI runner
  - Menu-driven interface
  - Can run individual or all diagnostics
  - Opens dashboard in browser

### Documentation
- **`docs/DIAGNOSTIC_DASHBOARD_GUIDE.md`** — Complete user guide
- **`docs/DIAGNOSTIC_DASHBOARD_README.md`** — This file

### Generated Data Files (after running diagnostics)
- **`docs/plots/autumn_leaves_root_mismatch.json`** — Root accuracy analysis (26 KB)
- **`docs/plots/autumn_leaves_semi_markov.json`** — Semi-Markov metrics (1 KB)
- **`docs/plots/autumn_leaves_chroma_audit.json`** — Chroma distance audit (24 KB)
- **`docs/plots/.diagnostics_manifest.json`** — Metadata about generated files

## Quick Start (60 seconds)

### Step 1: Generate Diagnostic Data
```bash
cd /Users/vincente/Documents/Projets\ Perso/Code/harmonia
python scripts/generate_all_diagnostics.py
```

Expected output:
```
✓ Root Label Mismatch Analysis
✓ Semi-Markov Configuration Check
✓ Chroma Geometry Audit

📊 Dashboard: docs/plots/autumn_leaves_complete_diagnostics.html
📁 Data files: 3 generated
   - autumn_leaves_root_mismatch.json (26.4 KB)
   - autumn_leaves_semi_markov.json (0.9 KB)
   - autumn_leaves_chroma_audit.json (24.3 KB)

✓ All diagnostics completed successfully!
```

### Step 2: Open Dashboard
```bash
open docs/plots/autumn_leaves_complete_diagnostics.html
```

Or use the CLI helper:
```bash
bash scripts/quick_diagnostics.sh  # Choose option 5 to open dashboard
```

### Step 3: Interact with Findings
- **Overview tab**: Understand status at a glance
- **Root Mismatches tab**: Examine bar-by-bar accuracy, offset patterns
- **Semi-Markov Config tab**: Check fragmentation metrics
- **Chroma Geometry tab**: Review distance metric audit
- **Feedback Form tab**: Provide ground truth verification

## Data Structure

### autumn_leaves_root_mismatch.json
```json
{
  "song": "autumn_leaves",
  "timestamp": "2026-07-14T18:24:47.682660",
  "summary": {
    "total_bars": 64,
    "bars_with_match_status": 64,
    "matched_bars": 45,
    "accuracy_percent": 70.3,
    "offset_distribution": {"0": 45, "1": 10, "-2": 5, "2": 4},
    "has_systematic_offset": true
  },
  "diagnostics": [
    {
      "bar": 0,
      "time": 0.98,
      "section": "A",
      "gt1_root": "C",
      "gt1_source": "irealb_autumn_leaves.html.json",
      "gt2_root": null,
      "gt2_source": null,
      "inferred_root": "C",
      "match": true,
      "offset": 0,
      "phase1_full_chord": "C-7",
      "inferred_full_chord": "C-7"
    },
    ...
  ]
}
```

### autumn_leaves_semi_markov.json
```json
{
  "song": "autumn_leaves",
  "timestamp": "2026-07-14T18:24:48.123456",
  "enabled": true,
  "duration_prior_file": "/path/to/duration_prior.npy",
  "duration_prior_valid": true,
  "fragmentation_without": 1.45,
  "fragmentation_with": 1.10,
  "summary": {
    "semi_markov_enabled": true,
    "duration_prior_available": true,
    "duration_prior_valid": true,
    "chords_per_bar_without_sm": 1.45,
    "chords_per_bar_with_sm": 1.10,
    "improvement_percent": 24.1,
    "status": "OK"
  }
}
```

### autumn_leaves_chroma_audit.json
```json
{
  "song": "autumn_leaves",
  "timestamp": "2026-07-14T18:24:49.456789",
  "summary": {
    "total_usages": 27,
    "tier1_count": 5,
    "tier2_count": 12,
    "tier3_count": 10,
    "critical_issues": 2
  },
  "findings": [
    {
      "file": "harmonia/models/chord_hmm.py",
      "line_number": 142,
      "function": "score_chords",
      "type": "Euclidean distance",
      "severity": 1,
      "context": "training",
      "input_shape": "[1024, 12]",
      "notes": "Linear distance - not appropriate for harmonic relationships"
    },
    ...
  ]
}
```

## Dashboard Features

### 1. Interactive Tabs
- **Overview** — Status snapshot, key findings, prioritized recommendations
- **Root Mismatches** — Detailed bar-by-bar table with filtering, offset histogram, GT source info
- **Semi-Markov Config** — Configuration status, fragmentation comparison, duration prior analysis
- **Chroma Geometry** — Distance metrics audit, chromatic vs harmonic heatmaps, error patterns
- **Feedback Form** — Ground truth verification questionnaire with local storage persistence

### 2. Advanced Visualizations
- **Offset Histogram** — Identifies systematic key transposition
- **Fragmentation Comparison** — Bar chart showing semi-Markov impact
- **Duration Prior Distribution** — Histogram of chord durations
- **Distance Heatmaps** — Compares chromatic vs circle-of-fifths geometry
- **Interactive Transposition Tester** — Slider showing accuracy at different offsets

### 3. Filtering & Search
- Filter root mismatch table by bar number, section, or match status
- Sort audit tables by file, severity, type
- Real-time search across bar information

### 4. Ground Truth Feedback
Interactive form capturing:
- Confirmation of systematic offset
- Ground Truth source identification
- Semi-Markov effectiveness assessment
- Chroma geometry error pattern validation
- Free-form additional notes

Responses saved to browser local storage with timestamps.

## Interpretation Guide

### Root Accuracy Findings

**Scenario 1: 100% accuracy (all green)**
```
✓ Good news: No systematic root mismatch
✓ Next step: Focus on chord quality (maj, min, 7th) if accuracy is still low there
```

**Scenario 2: ~70% accuracy with +1 offset peak**
```
⚠️ Systematic issue: Model is transposed up 1 semitone from GT
⚠️ Possible causes:
  - GT source uses different key standard
  - Beat tracking off by 1 beat, affecting chroma alignment
  - Model has systematic bias in pitch estimation
✓ Next step: Check GT source file, verify beat alignment
```

**Scenario 3: Random offsets (-3, -1, 0, +1, +2)**
```
❌ Distributed errors: Not a systematic transposition
❌ Suggests different issues:
  - GT source inconsistencies
  - Model genuinely struggles with certain progressions
✓ Next step: Examine high-error sections, check GT formatting
```

### Semi-Markov Findings

**Scenario 1: Enabled + 25% reduction + < 1.2 chords/bar**
```
✓ Good: Semi-Markov is working well
✓ Chord segmentation is reasonable
```

**Scenario 2: Enabled + 0% reduction + 1.5+ chords/bar**
```
⚠️ Problem: Semi-Markov enabled but not helping
⚠️ Possible causes:
  - Duration prior is weak (too much mass at 1 beat)
  - Configuration not properly applied
✓ Next step: Retrain duration prior, verify configuration
```

**Scenario 3: Disabled**
```
❌ Critical: Semi-Markov is off
❌ Expected impact:
  - High fragmentation (1.5–2.0+ chords/bar)
  - Model creates spurious chord boundaries
✓ Next step: Enable semi-Markov, verify duration prior
```

### Chroma Geometry Findings

**Scenario 1: Tier 1 findings use Euclidean distance**
```
⚠️ Training-stage issue: Model trained with linear distance
⚠️ Problem: Confuses semitone neighbors (C↔C#) with harmonic relationships
⚠️ Example error pattern: High confidence errors on C→C#, E→F
✓ Next step: Retrain with circle-of-fifths distance
```

**Scenario 2: Tier 2 findings inconsistent across modules**
```
⚠️ Inference-stage issue: Different functions use different metrics
⚠️ Problem: Inconsistent chord similarity calculations
✓ Next step: Standardize on harmonic distance metric
```

**Scenario 3: No Tier 1 findings**
```
✓ Good: Training-stage distances are harmonic
✓ Suggests root cause is elsewhere
✓ Next step: Focus on Beat tracking, Chroma extraction quality
```

## Command Reference

### Full Orchestration
```bash
# Run all diagnostics and generate JSON data files
python scripts/generate_all_diagnostics.py

# Interactive runner (menu-driven)
bash scripts/quick_diagnostics.sh

# Menu option 1-4: Run individual diagnostics
# Menu option 5: Open dashboard in browser
# Menu option 6: Exit
```

### Individual Diagnostics
```bash
# Root label analysis
python scripts/generate_root_mismatch_diagnostics.py

# Semi-Markov check
python scripts/generate_semi_markov_diagnostics.py

# Chroma audit
python scripts/generate_chroma_geometry_audit.py
```

### View Results
```bash
# Open dashboard in browser
open docs/plots/autumn_leaves_complete_diagnostics.html

# View generated JSON data
cat docs/plots/autumn_leaves_root_mismatch.json
cat docs/plots/autumn_leaves_semi_markov.json
cat docs/plots/autumn_leaves_chroma_audit.json

# Check manifest
cat docs/plots/.diagnostics_manifest.json
```

## Troubleshooting

### "Data incoming..." placeholders on all tabs
**Cause:** JSON data files not generated yet
```bash
python scripts/generate_all_diagnostics.py
# Reload browser page
```

### Root mismatch table shows 0% accuracy
**Cause 1:** No inferred annotations available
```bash
# Generate inferred annotations from model output
# Then re-run diagnostics
python scripts/generate_root_mismatch_diagnostics.py
```

**Cause 2:** Different GT format in Phase 1 vs inferred
**Fix:** Check GT source consistency in "Ground Truth Source Verification" section

### Semi-Markov metrics all show 0
**Cause:** Inferred chord annotations missing
**Fix:** Ensure inferred_autumn_leaves.html.json exists in annotations directory

### Dashboard loads but charts are empty
**Cause:** Plotly CDN connection issue (check browser console)
**Fix:** 
- Check internet connection
- Try offline: The dashboard will auto-detect and show placeholder
- Refresh page with Ctrl+R

### Feedback form doesn't save
**Cause:** Browser privacy settings blocking local storage
**Fix:** 
- Check browser Privacy/Storage settings
- Try in private/incognito mode (will lose persistence)
- Use different browser

## Integration with CLAUDE.md Workflow

This dashboard fits into your project's diagnostic workflow:

1. **Silent calibration bugs** (Rule 1) → Root Mismatch tab reveals systematic offset issues
2. **Screen premise cheaply** (Rule 2) → Run diagnostics before expensive multi-day investigations
3. **Ground truth is measurement** (Rule 3) → Feedback form captures GT source and verification
4. **State what a fix does NOT solve** (Rule 4) → Recommendations tag issues by priority
5. **Component swaps change more** (Rule 6) → Before/after segmentation shows semi-Markov impact

## File Locations

```
harmonia/
├── docs/
│   ├── DIAGNOSTIC_DASHBOARD_GUIDE.md ← User guide (this file)
│   ├── DIAGNOSTIC_DASHBOARD_README.md ← This file
│   ├── plots/
│   │   ├── autumn_leaves_complete_diagnostics.html ← Main dashboard
│   │   ├── autumn_leaves_root_mismatch.json ← Data (generated)
│   │   ├── autumn_leaves_semi_markov.json ← Data (generated)
│   │   ├── autumn_leaves_chroma_audit.json ← Data (generated)
│   │   └── .diagnostics_manifest.json ← Metadata (generated)
│   └── annotations/
│       ├── irealb_autumn_leaves.html.json ← Ground Truth
│       └── inferred_autumn_leaves.html.json ← Model output (if available)
│
├── scripts/
│   ├── generate_all_diagnostics.py ← Master orchestrator
│   ├── generate_root_mismatch_diagnostics.py ← Root analysis
│   ├── generate_semi_markov_diagnostics.py ← Semi-Markov check
│   ├── generate_chroma_geometry_audit.py ← Chroma audit
│   └── quick_diagnostics.sh ← CLI runner
```

## Next Steps

1. **Generate diagnostics:** `python scripts/generate_all_diagnostics.py`
2. **Open dashboard:** `open docs/plots/autumn_leaves_complete_diagnostics.html`
3. **Review findings:** Check each tab for status and patterns
4. **Verify ground truth:** Use Feedback Form tab to confirm interpretations
5. **Act on recommendations:** Prioritized next steps in Overview tab

## Support

For issues or questions:
- Check `docs/known_issues.md` — project's authoritative issue tracker
- Review `docs/DIAGNOSTIC_DASHBOARD_GUIDE.md` — comprehensive feature reference
- Run `bash scripts/quick_diagnostics.sh` — interactive CLI helper
- Examine generated JSON files directly for raw data

---

**Dashboard Version:** 1.0  
**Generated:** 2026-07-14  
**Harmonia Project:** Jazz/Pop Chord Recognition Pipeline  
**User:** ML PhD + Jazz Musician, Concise + Rigorous

