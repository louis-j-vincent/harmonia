# Phase 2 Algorithm Redesign (Post H-B Evidence)

**Status:** Fundamental reframe based on H-B hypothesis confirmation. The iReal chart is misaligned; we must search for section positions.

---

## Problem Reframe

**Old assumption (WRONG):**
- Song is AABC at constant 181 BPM
- Chart timings are correct (musical form validates them)
- Low root-match at offset 0 = inference quality failure
- Solution: 1D phase search to find where bar 0 starts

**New reality (H-B CONFIRMED):**
- Song structure is AABC (musical form is right) ✓
- Chart **bar timings are shifted** (likely by ~12 bars / ~16s for B/C sections)
- Root-match at offset 0 is 0% (not inference failure, chart mismatch)
- Root-match at offset +12 is 25% (consistent with 43% corpus baseline)
- Solution: 2D search to find where each section actually starts

---

## Algorithm Specification (Revised)

### Problem Definition

**Input:**
- Chart: sections A1, A2, B, C, ... with known chord progressions
- Inferred: per-beat chords from audio (root_pc, confidence, time)
- Goal: find the absolute time where each section starts

**Output:**
- Per-section placement: (section_label, bar_lo, bar_hi, t_start, t_end, root_match_rate, confidence)
- Summary: estimated GT offset error if sections are systematically shifted

### 2D Search Algorithm

**Pseudocode:**

```python
def align_sections_with_search(chart_sections, inferred_chords):
    """
    2D search: for each section, find the time window in which inferred chords
    best match the section's root progression.
    """
    
    results = {}
    
    for section in chart_sections:
        section_roots = [root_pc for root_pc, quality in section.chords]
        
        # Search window: ±20 bars around expected position
        # (since error is ~12 bars, this is a safe range)
        expected_t_start = compute_expected_time(section, bpm_prior=181)
        search_window = (expected_t_start - 30, expected_t_start + 30)  # ±30s
        
        best_offset, best_score = None, -1
        
        # Sample candidate time offsets (coarse: 1-bar steps = 1.33s)
        for t_start_candidate in range(int(search_window[0] * 100),
                                       int(search_window[1] * 100), 
                                       int(1.33 * 100)):  # 1-bar steps
            t_start = t_start_candidate / 100
            
            # Extract inferred chords in this time window
            inferred_in_window = [ic for ic in inferred_chords
                                  if t_start <= ic.t_start < t_start + section_duration]
            
            if not inferred_in_window:
                continue
            
            # Compute root-match (unweighted binary)
            inferred_roots = [ic.root_pc for ic in inferred_in_window]
            match_count = sum(1 for ir, cr in zip(inferred_roots, section_roots)
                              if ir == cr)
            match_rate = match_count / len(section_roots)
            
            # Optional: confidence weighting
            # Use only high-confidence inferred chords (>0.7) as primary signal
            high_conf = [ic for ic in inferred_in_window if ic.confidence > 0.7]
            if high_conf:
                hc_roots = [ic.root_pc for ic in high_conf]
                hc_match = sum(1 for ir, cr in zip(hc_roots, section_roots)
                               if ir == cr)
                hc_rate = hc_match / len(section_roots)
                # Use high-conf rate as primary, with low-conf as fallback
                match_rate = max(hc_rate, 0.5 * match_rate)
            
            if match_rate > best_score:
                best_score, best_offset = match_rate, t_start
        
        if best_offset is not None:
            results[section.label] = {
                't_start': best_offset,
                'root_match_rate': best_score,
                'offset_error': best_offset - compute_expected_time(section),
            }
    
    return results
```

### Confidence Weighting (Revised)

**Key change:** Use confidence as a **refinement**, not the primary metric.

1. **Primary metric:** unweighted root-match (binary per-chord)
   - Does the inferred root match the chart root?
   - No weighting by confidence

2. **Confidence refinement:**
   - High-confidence inferred chords (>0.7) count as strong evidence
   - Low-confidence chords are tentative
   - Upweight placements that have high-confidence root matches

3. **Tie-breaking:**
   - If two offsets have similar root-match rates, prefer the one with higher-confidence chords

### Musicological Constraints

1. **Form constraint:** Sections must appear in order (A1 → A2 → B → C)
   - Enforce: t_start(A1) < t_start(A2) < t_start(B) < t_start(C)
   - But allow systematic offset (e.g., if all B/C sections are +12 bars)

2. **Constant-BPM constraint (soft):**
   - Prefer placements that yield ~181 BPM (±10%) within and between sections
   - If a section needs 200 BPM to fit, flag it as suspicious

3. **No-gap constraint (soft):**
   - Prefer consecutive section placement (A2 ends where B starts)
   - But allow small gaps (< 1 bar = 1.33s) for transition fills

---

## Expected Output for Autumn Leaves

**Hypothesis (from H-B analysis):**

| Section | Expected (offset 0) | Predicted Actual | Root-Match | Offset Error |
|---|---|---|---|---|
| A1 | 0.25s | 0.25s | 87.5% | 0s |
| A2 | 10.86s | 10.86s | 50.0% | 0s |
| B | 21.5s | 37.4s | 25% (growing with refinement) | +15.9s |
| C | 32s | 47.6s | 20–30% | +15.6s |
| A (repeat) | 42.5s | 58.4s | 87.5% | +15.9s |
| B (repeat) | 53s | 68.9s | 25% | +15.9s |
| C (repeat) | 63.5s | 79.5s | 25% | +15.9s |

**Key observation:** If error is systematic (~12 bars / ~16s for B/C onward), the algorithm should detect it and report: "GT offset: +12 bars from bar 8 onward."

---

## Algorithm Justification: Why Not 1D Phase Search?

**Old design (phase search):**
- Assumes chart is correct → find where bar 0 is
- Minimizes time offset globally
- But ignores that *different sections* may have *different errors*
- Result: finds a compromise, misses systematic GT drift

**New design (2D section search):**
- Searches for where each section actually is
- Allows per-section offset discovery
- Can detect systematic errors (all B/C shifted by +12)
- Uses inferred chords as the oracle
- Result: finds true section positions, even if chart is wrong

**Why OT/Hungarian still doesn't fit:**
- OT assumes a cost matrix with symmetric distance
- We have asymmetric data: inferred chords are noisy, chart is fixed
- OT would try to match sections to inferred regions with a balancing constraint
- But we want: find where in the time axis each section actually is
- 2D search (section × time offset) is correct, OT is not

---

## Implementation Strategy (Phase 2 Revised)

1. **Replace** the 1D phase-search algorithm with a 2D section-offset search
2. **For each section:**
   - Search candidate offsets (±20 bars, 1-bar step ≈ 1.33s)
   - Compute root-match rate of inferred chords vs section roots
   - Pick offset with highest match rate
   - Report match rate and offset error
3. **Musicological post-processing:**
   - Check form constraint (sections in order)
   - Detect systematic offset (if B/C all shifted by +X bars, report it)
   - Flag high-BPM or gap violations

4. **Output:**
   - Per-section: (label, t_start, root_match_rate, offset_error)
   - Summary: (estimated GT offset, BPM fit, form validation)

---

## Known Limitations & Future Work

1. **Chord evolution within section:** The algorithm assumes each section has a fixed chord progression. Real music has variations (turnarounds, fills). Refinement: use soft-DTW to allow local compression/expansion.

2. **Per-beat vs per-segment inference:** Current inferred chords are per-beat segments. The 1:1 alignment with chart bars assumes no beat slips. Refinement: use beat-tracking confidence as a weighting factor.

3. **Interactive search space:** Searching ±20 bars × all sections is O(S × 1500 samples) = expensive for large corpora. Refinement: use Phase 1's results as a warm-start (narrow search window around Phase 1's offset).

4. **High-confidence upweighting:** Currently simple (threshold 0.7). Refinement: use isotonic calibration or adaptive thresholding per song.

---

## Summary: Why H-B Changes Everything

**H-B finding:** Inferred chords have 0% match at offset 0, 25% at offset +12. This is not random noise; it's a **clear pattern indicating GT misalignment**.

**Implication:** The algorithm cannot assume the chart is right. It must **search for section positions** using inferred chords as the ground truth, not the chart.

**Design shift:** From 1D phase search (assume chart, find bar 0) to 2D section search (ignore chart times, find sections in audio).

This is a fundamental reversal. Phase 1 accidentally found the right answer (+12 offset) by exhaustively trying all offsets. Phase 2 should formalize this as the core algorithm.

---

## Next: Implementation

Ready to implement the revised 2D section-search algorithm once this design is approved.
