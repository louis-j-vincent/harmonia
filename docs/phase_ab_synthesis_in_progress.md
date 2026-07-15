# Phase A/B Synthesis (In Progress)

*2026-07-15. Orchestrator analysis of bass-anchor (A) and context-window quality (B) investigations.*

## Current Status

- **Phase A:** COMPLETE (bass-anchor failed; root P4/P5 harmonic, not fixable in isolation)
- **Phase B:** IN PROGRESS (context-window quality model, 3 architectures × 3 normalizations)
  - CNN, LSTM, Transformer being trained
  - ETA: ~30-45 min from 2026-07-15 ~18:00 UTC

## Summary of Findings So Far

### Phase A: Bass-Anchor Investigation (COMPLETED)

**Hypothesis:** Agent 1's bass predictions reduce root P4/P5 errors from 45% → <30%.

**Result:** FAILED due to task mismatch.
- Agent 1 predicts bass/lowest note (0.886 acc on bass task)
- Billboard roots are functional, not physical bass
- Inversions (C/E) break the assumption
- When used as features: P4/P5 actually got worse (40.66% → 41.20%)

**Key insight:** Root P4/P5 confusion is HARMONIC (I/IV/V diatonic overlap), not bass-register. Cannot fix in isolation.

**Architectural implication:** Chord_AI's separate bass/root heads (+joint training) is the right design.

### Phase B: Context-Window Quality Model (IN PROGRESS)

**Hypothesis:** Learned architectures (CNN/LSTM/RNN) can extract chord-progression patterns that agent 2's simple lambdas missed.

**Setup:**
- Input: 7×12 context matrix (3-before + current + 3-after, as P(root) distributions from Agent 1)
- 3 architectures: CNN, LSTM, Transformer
- 3 normalizations: raw, relative-to-key, relative-to-root
- 9 total experiments
- Target: dom recall ≥0.70, balanced acc ≥0.75 (with predicted roots)
- Baseline: Agent 2's oracle-root model (dom recall 0.665)

**Expected outcomes (ranked by likelihood):**
1. **Best case:** One architecture × normalization beats 0.665 (positive result)
2. **Partial case:** Learns some patterns but doesn't exceed oracle (progress but not solution)
3. **Failure case:** All architectures underperform (context not predictive)

## What Phase B Will Tell Us

### Scenario 1: Success (Context Works)
- **Signal:** dom recall 0.665+ → 0.70+ or better balanced acc
- **Implication:** Harmonic context is learnable; context window is the right approach
- **Next step:** Integrate into production, then move to structured-head redesign

### Scenario 2: Partial Success (Some Learning, Not Enough)
- **Signal:** improved over baseline, but still <0.665
- **Implication:** Learned architectures help, but context alone insufficient
- **Next step:** Combine with key/transition priors (revisit Agent 2's approach, but in learned model)

### Scenario 3: Failure (Context Not Predictive)
- **Signal:** all architectures perform worse than baseline
- **Implication:** Quality truly isolated; chord context doesn't help
- **Investigation required:**
  1. Is context actually informative? (correlation analysis)
  2. Alignment correct? (spot-check examples)
  3. Feature representation usable? (Agent 1's probability distributions valid?)
- **Next step:** Move to bass-relative rotation (Agent 2's +26pp trick) or joint root×quality Viterbi

## Consolidated Understanding (Pre-Phase B)

### The Two Bottlenecks

1. **Root P4/P5 (45% of root errors)**
   - Cannot fix via bass (task mismatch) or priors (harmonic problem)
   - Requires joint root×quality decode (#27) to leverage quality context
   - Status: blocked until Phase B results inform architecture redesign

2. **Quality dom (recall 0.15)**
   - Main performance bottleneck (not root)
   - Learnable: 0.15 raw → 0.665 with oracle root (26pp upside)
   - Discriminability issue: b7 low-contrast in chroma
   - Status: Phase B testing whether context helps

### Why Chord_AI Wins

From Agent 3's reverse-engineering:
1. **Structured heads** (root, bass, 7th) trained jointly → rare classes leverage common ones
2. **Separate bass head** (13-class) → handles inversions (C/E, Am/C) explicitly
3. **Separate 7th detection** (bitmap) → b7 as its own feature, not bundled in quality
4. **All on same chroma** → features shared across all heads

Harmonia's current approach:
- Flat 72-class or cascaded root→quality
- No explicit bass
- Quality includes 7th, making it low-contrast

### Path Forward (Post-Phase B)

**If Phase B succeeds:**
1. Use context-window model for quality
2. Integrate Agent 1's bass predictions as context (not direct features)
3. Redesign to structured heads (root + bass + quality/7th)
4. Re-run everything on production BP48 features (not NNLS)

**If Phase B fails:**
1. Context not learnable via simple architectures
2. Fall back to: joint root×quality Viterbi (#27) + bass-relative quality rotation
3. Accept that quality needs architectural changes (separate 7th head)

## Waiting For Phase B

**Expected events (in order):**
1. Phase B agent trains CNN/LSTM/Transformer models
2. Reports results table (balanced acc, per-class recall)
3. Generates diagnostic plots (confusion matrices, ablation heatmap)
4. Failure investigation (if needed) documents why results underperformed
5. Orchestrator receives completion notification
6. Orchestrator synthesizes full Phase A/B findings
7. Final report to coordinator with recommendations

**Monitoring points:**
- Check for Phase B completion (~30-45 min)
- If delayed >1 hour, investigate (common issues: OOM, alignment bugs, class weight misconfiguration)
- Any early results? Monitor progress logs

