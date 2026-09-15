# Phase 2 Section Alignment — Revised Design (2026-07-14)

**Context:** Phase 1 (naive sliding-window alignment) used **confidence-weighted** match scores, which masked the true alignment problem. User validation revealed: song is AABC (constant ~181 BPM, no vamps), B is at offset 0 (immediately after A2), but has low root-match% due to **inference quality**, not misalignment. Phase 2 must use **unweighted root-match rate** as the metric.

## Critical Findings from Phase 1 / iReal Validation

From `docs/ireal_validation_results.md`:
- **Autumn Leaves corpus root accuracy: 43%** (vs. DTW spurious floor 33% → lift only +10pp)
- **Confidence is inverted:** high-confidence chords 43.2% accurate, low-confidence 49.1% accurate
- **Quality head not integrated** (pending #19) — without it, majority family collapses (min 22%, dom 21%, hdim 14%)

**Phase 1 metrics error:**
- Phase 1 reported B at offset +12 bars (0.96s error)
- Phase 1 used `conf_weighted` (weighted by inferred-chord confidence)
- **True metric:** unweighted root-match = (# bars with matching root) / (total bars)

**Why B has low match% at the correct position:**
Not misalignment — *inference quality*. Autumn Leaves B region (bars 16–23) contains:
- Dense harmonic motion (ii-V-i chaining)
- Possible busy/soloistic arrangement in the audio
- Inference is domain-gapped (model trained on MMA synth, not real audio)
- Quality head v1 (57.5% q5 accuracy on real audio) not wired in yet

## Phase 2 Algorithm Redesign

### Setup
- **Chart:** A1 (bars 0–7), A2 (8–15), B (16–23), C (24–31)
- **Inferred:** per-beat chord stream (root_pc, quality, confidence)
- **BPM:** constant ~181 (60 / (181 / 4) ≈ 1.326 s/bar)
- **Goal:** find absolute time offset (phase) that maximizes **unweighted root-match** across all 4 sections

### Why This Is NOT a Full 2D Assignment Problem

With **constant BPM and rigid sections**, the degrees of freedom collapse:
- Sections are ordered (A1 → A2 → B → C) with ZERO gaps
- Bar durations are fixed: 1.326 s
- Expected times: `t_bar(b) = t_bar0 + b * 1.326` (t_bar0 is the only unknown)

**The problem is 1D phase search, not 2D section assignment:**
- Search space: candidate `t_bar0` values in [0, 10] seconds (coarse: 0.05s steps → 200 candidates)
- For each candidate:
  - Compute root-match rate for A1, A2, B, C
  - Sum across sections (or weight by section importance)
  - Track best `t_bar0`

### Root-Match Metric (Unweighted)

For a chart section (e.g., A1, bars 0–7):
```
root_match_count = 0
for each bar b in section:
  t_bar_start = t_bar0 + b * 1.326
  t_bar_end = t_bar0 + (b+1) * 1.326
  
  # Find all inferred chords in this bar's time window
  inferred_in_bar = [ic for ic in inferred if ic.t_start <= t_bar_end and ic.t_end >= t_bar_start]
  
  # Did ANY of them match the chart root?
  chart_root = chart[section][bar_index].root_pc
  if any(ic.root_pc == chart_root for ic in inferred_in_bar):
    root_match_count += 1

root_match_rate(section, t_bar0) = root_match_count / len(section.bars)
```

Key: **unweighted** (binary per-bar match), **not confidence-weighted**.

### Phase-Search Algorithm

```python
def find_best_phase(chart_sections, inferred_chords, bpm_prior=181.0):
  """Exhaustive phase search with constant BPM."""
  
  bpm = bpm_prior
  bar_dur_s = 60 / (bpm / 4)  # 1.326s for 181 BPM
  
  best_phase, best_score = None, -1.0
  candidates = [i * 0.05 for i in range(200)]  # 0.0 to 9.95s in 0.05s steps
  
  for t_bar0 in candidates:
    # Compute root-match for each section at this phase
    total_score = 0.0
    for section in chart_sections:
      rate = unweighted_root_match(section, inferred_chords, t_bar0, bar_dur_s)
      total_score += rate
    
    if total_score > best_score:
      best_score, best_phase = total_score, t_bar0
  
  return best_phase, best_score / len(chart_sections)  # avg match rate across sections
```

### Refinement: Soft Bar Assignment (Soft-DTW)

For smoother bar↔time coupling (avoiding hard grid snapping), use a soft assignment:
- Each inferred chord contributes to *multiple* bars with a weight that decays over time distance
- Example: an inferred chord at t=12.5s with τ=1.5s "confidence window" contributes:
  - 100% to bar 9 (expected t=12.64s)
  - 50% to bar 8 and bar 10
  - 0% elsewhere
- This gives "smooth bar↔time coupling" without hard snapping

Implementation (simple Gaussian):
```python
def soft_bar_match(inferred_chords, chart_section, t_bar0, bar_dur_s, τ=0.5):
  """Soft root-match with Gaussian weighting over neighboring bars."""
  
  total_match_mass = 0.0
  
  for ic in inferred_chords:
    ic_t = (ic.t_start + ic.t_end) / 2  # midpoint
    ic_conf = ic.confidence  # inferred confidence
    
    # Expected bar positions
    for bar_idx, chart_root in enumerate(chart_section.roots):
      bar_t_expected = t_bar0 + (bar_idx + chart_section.bar_lo) * bar_dur_s
      
      # Gaussian soft assignment
      dist = abs(ic_t - bar_t_expected)
      weight = math.exp(-(dist ** 2) / (2 * τ ** 2))  # Gaussian decay
      
      # Match only if roots agree
      if ic.root_pc == chart_root:
        total_match_mass += weight  # unweighted root, but soft temporal assignment
  
  # Normalize by expected mass (each bar contributes ≈1.0 when properly aligned)
  expected_mass = len(chart_section.bars) * (1.0 if τ > 0 else 1.0)  # TODO: compute exactly
  return total_match_mass / expected_mass
```

Advantage: if an inferred chord is 0.1s late, it still contributes partially to the "correct" bar (not penalised to 0).

### Constant-BPM Constraint Handling

**Hard constraint:** BPM is fixed at prior (181 BPM).
**Soft penalty (optional):** if allowing per-section BPM drift:
- Penalise fitted BPM far from prior: `penalty = λ * (|fitted_bpm - bpm_prior| / bpm_prior)^2`
- Penalise large BPM jumps between sections: `penalty = μ * |(bpm_section_i - bpm_section_i-1) / bpm_prior|`
- But if BPM is truly constant, these penalties are zero and don't matter.

With constant BPM, the tempo dimension is *solved* (parameter is t_bar0 only). There is no OT assignment problem — OT would be overkill (Hungarian would also be overkill; the problem is 1D search).

### Why OT/Hungarian Is Not Useful Here

**Plain OT / Hungarian assumes:**
- A set of "supply" nodes (chart sections) that must be matched to "demand" nodes (audio regions)
- A cost matrix `C[section, region]`
- Goal: find the optimal one-to-one matching

**Our problem:**
- Sections are *ordered and back-to-back* (no choice of which audio region maps to which section)
- Sections are *rigid* (can't split or merge)
- BPM is *constant* (so all sections are already time-locked to each other)
- The only degree of freedom is the global phase offset t_bar0

Result: **optimal assignment is trivial** (A1 → region starting at t_bar0, A2 → t_bar0 + 10.6s, B → t_bar0 + 21.2s, C → t_bar0 + 31.8s). There's nothing to optimize except t_bar0.

**Where OT could theoretically help:**
- If we allowed per-section BPM variation (Soft-DTW / entropic OT with a diagonal regulariser for tempo smoothness)
- Or if sections could overlap / skip (but they don't)
- Or if the metric was distributional (e.g., chord-family histogram match, not per-bar binary root match)

**Our metric (unweighted binary root-match) is not suited for OT** — it's a discrete binary outcome, not a probability distribution. OT shines on distributional matching (Sinkhorn on mass plans, Gromov–Wasserstein on geometric structures). For our binary per-bar metric, exhaustive phase search is **simpler and correct**.

### Recommended Approach: Phase Search + Soft Temporal Assignment

1. **Exhaustive phase search** over [0, 10] seconds (coarse: 0.05s steps; refine to 0.01s if needed)
2. **Soft-DTW temporal assignment** per phase (Gaussian contribution to neighboring bars)
3. **Unweighted root-match metric** (per-bar binary, no confidence weighting)
4. **Output:**
   - Best `t_bar0` (bar 0 start time)
   - Root-match rate per section (A1, A2, B, C)
   - Residual timing per inferred chord (deviation from expected bar grid)
   - Confidence flags: sections with low match rate are marked NEEDS_REVIEW

## Why B Has Low Root-Match at the Correct Position

Given Autumn Leaves corpus data:
- **Root accuracy 43%** overall (not 87% as Phase 1's confidence-weighting suggested)
- **B region in Autumn Leaves:** probably has dense harmony (ii-V-i chains or turnarounds) in a busy/solo arrangement
- **Inference bottleneck:** quality family model not wired (would add +11.7pp majmin on real audio)
- **Conclusion:** B's 37–50% root-match is *expected* at the current inference quality level, not a sign of misalignment

**Test hypothesis:** if we correct the phase and measure unweighted root-match on Autumn Leaves B region:
- Baseline (current misaligned phase): root-match ≈ 20–30%
- After phase correction: root-match ≈ 40–50% (in line with corpus median)
- This would prove the algorithm works, independent of absolute accuracy

## Deliverables (Phase 2)

1. **`scripts/optimal_section_alignment_v2.py`**
   - `find_best_phase()` — exhaustive phase search
   - `soft_bar_root_match()` — Gaussian soft assignment, unweighted metric
   - `align_sections_constant_bpm()` — end-to-end wrapper
   - **NO OT / Hungarian** (problem doesn't need it)

2. **Integration with Phase 1:**
   - Phase 1's `irealb_autumn_leaves_naive_aligned.json` reports B at offset +12
   - Phase 2 will correct to offset 0 by running the phase search
   - Comparison: Phase 1 error ≈ 0.96s, Phase 2 (expected) ≈ ±0.05s residual

3. **Metrics Report**
   - Root-match rates per section (A1, A2, B, C) with corrected phase
   - Comparison to Phase 1's confidence-weighted scores
   - Analysis: how much of B's low match is alignment vs. inference quality?

4. **Research Note** (`docs/phase2_section_alignment_research.md`)
   - Why OT is not needed for constant-BPM rigid sections
   - Literature anchors: soft-DTW, phase-search, tempo tracking in jazz
   - Known limitations & future extensions (per-section BPM, vamp detection)

## Next Step

Implement `optimal_section_alignment_v2.py` with phase search + soft-DTW metric, test on Autumn Leaves, compare Phase 1 vs Phase 2 alignment.
