# Autumn Leaves GT Fix Impact Analysis

**Date:** 2026-07-14  
**Fix:** Added missing G-6 at A2/B boundary (commit 16ac553)  
**Status:** Structural change verified; accuracy impact assessment constrained by inference limitations

---

## Executive Summary

The GT fix correctly addresses a **structural misalignment** where:
- **Before:** A2 ended with a single G-6 (bar 14), B started at bar 15 (off by 1 bar)
- **After:** A2 ends with double G-6 (bars 14-15), B starts at bar 16 (correct alignment)

**Expected impact:** B and C section accuracy should improve significantly (from 12.5% and 20% toward 87.5%+), contingent on inference correctly matching the re-aligned time windows.

---

## 1. Structural Change Analysis

### Before Fix (OLD GT)

| Property | Value |
|----------|-------|
| Total chords | 66 |
| Total bars | 62 |
| A2 chords | 7 (ends with single G-6 at bar 14) |
| B starts at | bar 15 |
| B/C time window | 20.79s–72.51s |
| **Issue** | A2 is incomplete; B/C sections start 1 bar too early |

### After Fix (NEW GT)

| Property | Value |
|----------|-------|
| Total chords | 68 |
| Total bars | 64 |
| A2 chords | 8 (ends with double G-6 at bars 14-15) |
| B starts at | bar 16 |
| B/C time window | 22.12s–75.16s |
| **Fix** | A2 is now complete; B/C sections correctly aligned (+1.33s shift forward) |

### Impact on Each Section

| Section | Before | After | Change | Expected Accuracy Impact |
|---------|--------|-------|--------|--------------------------|
| **A1** | Correct | Unchanged | None | No change (87.5% → 87.5%) |
| **A2** | 7 chords, single G-6 | 8 chords, double G-6 | +1 chord | Possible improvement (50% → ?) due to complete bar sequence |
| **B** | Time-shifted -1 bar | Correct alignment | +1.33s | Major improvement expected (12.5% → 80%+) if inference quality permits |
| **C** | Time-shifted -1 bar | Correct alignment | +1.33s | Major improvement expected (20% → 70%+) if inference quality permits |

---

## 2. Root Cause of Previous Degradation

The previous analysis (Phase 1, session 2026-07-14) identified:

```
A1:  87.5% ┐
     -37.5pp
A2:  50.0% ┤ Progressive degradation
     -37.5pp
B:   12.5% ┤
     -7.5pp
C:   20.0% ┘
```

**Two independent failure modes:**

1. **A2 failure:** 50% accuracy despite identical chord progression to A1
   - Likely caused by: missing G-6 creating phase misalignment mid-section
   - **Should be fixed by:** Adding the missing bar

2. **B/C failure:** 12.5% and 20% accuracy in sections shifted 1 bar early in the recording
   - Caused by: Cascading time misalignment from the missing G-6 in A2
   - **Should be fixed by:** Realigning B/C to correct time windows (+1.33s)

---

## 3. Theoretical Improvement Prediction

With correct alignment, we expect:

| Section | Before | Predicted After | Reasoning |
|---------|--------|-----------------|-----------|
| A1 | 87.5% | 87.5% (baseline) | No change; already correct |
| A2 | 50.0% | 70–85% | Harmonic phase restored; now 8-bar complete progression |
| B | 12.5% | 75–90% | Time window now correct; beat-grid alignment fixed |
| C | 20.0% | 70–85% | Time window now correct; inherits B's fix |

**Key assumption:** Inference quality is adequate. If the inference model itself has fundamental issues (e.g., wrong training data, wrong architecture), alignment alone cannot fix accuracy.

---

## 4. Measurement Challenges & Findings

### Challenge: Inference Quality

Current attempts to measure accuracy post-fix encountered a major blocker:

**Inference Method:** `chord_pipeline_v1` (Gen-2 production pipeline)  
**Audio:** autumn_leaves_remastered.m4a → converted to WAV  
**Result:** 
- Inferred 111 chords (expected ~8 per 10-bar section)
- Inferred roots do NOT match Autumn Leaves progression
- Accuracy by direct matching: **A1: 25%, A2: 0%, B: 0%, C: 20%** ← Much worse than before

**Interpretation:** The v1 pipeline appears incompatible with this task. Possible causes:
- v1 was trained on different genre/style
- Audio preprocessing mismatch
- Model regression between sessions
- Real audio domain calibration issue

### Previous Analysis Method

The 87.5%/50%/12.5%/20% numbers were derived from a different analysis approach (likely inference → beat-grid snapping → bar-level matching), not direct root comparison. That method:
- Allowed time tolerance (within 1-2 beats)
- Matched inferred chords to bars probabilistically
- Did NOT require perfect temporal alignment

**Consequence:** Cannot directly re-run that analysis without the original inference data or inference method.

---

## 5. What the GT Fix Actually Guarantees

✓ **A2 now has 8 chords** (was 7)  
✓ **A2 ends with double G-6** (was single)  
✓ **B section aligned to bar 16** (was bar 15)  
✓ **C section inherited correct alignment**  
✓ **Total song length correct** (64 bars, not 61)  

These are **structural guarantees**. Accuracy improvements depend on:
- Inference method quality
- Beat-grid detection accuracy
- Chroma-based key stability

---

## 6. Validation & Next Steps

### To Verify the Fix Works

1. **Use a known-good inference method**
   - Identify which inference produced the 87.5%/50% numbers
   - Re-run on BOTH old GT and new GT
   - Compare section accuracy before/after

2. **Use iReal Pro as oracle**
   - Load iReal Autumn Leaves (authoritative)
   - Verify it now matches the corrected chart exactly
   - Check timings make sense (no more 1-bar shifts)

3. **Manual spot check**
   - Listen to the audio at B section start (~22s GT time)
   - Verify the inferred chord matches Ah7 (not G-6)
   - If inference is correct, accuracy should be high; if inference is wrong, accuracy will remain low regardless of GT fix

### To Measure Degradation Pattern Impact

```python
# Pseudo-code for proper measurement
old_gt = load("irealb_autumn_leaves_perfectgrid.json.bak_pre_g6fix")
new_gt = load("irealb_autumn_leaves_perfectgrid.json")

inference = run_inference(audio, method="<<KNOWN_GOOD_METHOD>>")

for gt, label in [(old_gt, "OLD"), (new_gt, "NEW")]:
    results = measure_accuracy_by_section(inference, gt)
    print(f"{label}: A1={results['A1']:.1f}%, A2={results['A2']:.1f}%, B={results['B']:.1f}%, C={results['C']:.1f}%")

# Expected:
# OLD: A1=87.5%, A2=50.0%, B=12.5%, C=20.0%
# NEW: A1=87.5%, A2=?%, B=>>50%, C=>>70%  (significant improvement in B/C)
```

---

## 7. Conclusion

**The GT fix is structurally correct and necessary.** However:

- ✅ **Confirms:** A2 was incomplete, B/C were misaligned
- ✅ **Guarantees:** Structural alignment now matches the chart form
- ❓ **Pending:** Whether the fix improves measured accuracy (depends on inference quality)

**Recommendation:** Before declaring success, re-measure with a known-good inference method (the one used in the original Phase 1 analysis). If that inference still produces poor results, the degradation is a model quality issue, not an alignment issue.

---

## References

- **Commit 16ac553:** "fix: Autumn Leaves structure — insert missing G-6 repeat (32-bar AABC form)"
- **Phase 1 Memory:** `project_phase1_findings.md` — degradation pattern analysis
- **Previous Inference:** Method unknown; likely used beat-grid snapping → bar-level matching
- **Current Blocker:** `chord_pipeline_v1` produces incorrect roots for Autumn Leaves
