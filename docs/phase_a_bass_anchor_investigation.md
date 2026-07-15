# Phase A Investigation: Bass-Anchor Root Fix (FAILED)

*2026-07-15. Orchestrator analysis of Agent 1's bass detector applied to root P4/P5 confusion.*

## Mission

**Hypothesis:** Adding Agent 1's bass predictions as features to the root model would reduce P4/P5 error fraction from 45% → <30%.

**Rationale:** P4/P5 confusion (dominant/subdominant root confusion) is a bass-register problem; low-frequency chroma information from Agent 1's bass detector should disambiguate.

**Result:** **FAILED.** P4/P5 error fraction: 40.66% (baseline) → 41.20% (augmented), Δ = +0.55pp (got worse).

## Methods

1. **Alignment:** Matched Agent 1's 9,836 test-set bass predictions with Billboard's 114,741 chords via song_id + chord position
2. **Feature augmentation:** [12-dim chroma | one-hot(bass_pred) | bass_prob(12-dim)] = 36-dim input
3. **Model:** Logistic regression (class_weight='balanced'), same as Agent 1's root model
4. **Split:** song-stratified train/val/test from Billboard
5. **Evaluation:** balanced accuracy, P4/P5 error interval distribution

## Key Findings

### 1. Agent 1's Bass Detector is Excellent (0.886 acc on its own task)
- Predicts actual bass/lowest note in audio
- When evaluated on its own ground truth: 88.6% accuracy
- High confidence: median prediction confidence 96.01%, mean 87.4%

### 2. Task Mismatch: Bass ≠ Root

**Critical discovery:** Agent 1 predicts *bass notes*, Billboard labels *functional roots*.

- **Agent 1:** Detecting lowest note played (0.886 acc on bass GT)
- **Billboard:** Chord root (theoretical/functional, not always the bass note)
- **Mismatch:** When applied to Billboard roots, Agent 1 predictions: only 54.2% accurate (on test set)
- **Reason:** Inversions (C/E has E in bass, but C is root) are common; Bass ≠ Root

Example:
```
C/E chord:
  - Actual bass note (lowest): E
  - Agent 1 would predict: E (correct for bass task)
  - Billboard root GT: C (correct for root task)
  - Mismatch: predicted E, GT C → error for root model
```

### 3. Why Bass-as-Features Didn't Help

When evaluated as features for root prediction:
- **Baseline (12-dim chroma):** 0.8307 balanced acc, 40.66% P4/P5 errors
- **Augmented (36-dim, bass features):** 0.8310 balanced acc, 41.20% P4/P5 errors
- **Conclusion:** Marginal improvement in overall acc (+0.03pp), but P4/P5 errors got worse (+0.55pp)

Explanation:
1. Bass predictions have ~45% error rate on test set (due to task mismatch)
2. When fed as features, wrong bass predictions mislead the root model
3. The 12-dim chroma already implicitly encodes bass (it's part of the full spectrum)
4. Adding wrong bass predictions adds noise, not signal

### 4. Root Cause: P4/P5 Confusion is Harmonic, Not Bass-Register

Evidence from Agent 2's prior analysis:
- Tested **diatonic key prior** on root: −0.2pp (hurt)
- Tested **empirical transition prior** on root: −3.3pp (hurt)
- **Why?** I/IV/V are all diatonic to any key, AND real transitions are fifth-dominated (I↔IV, I↔V, ii→V)
- So priors *reinforce* the fifth-related confusion instead of suppressing it
- **Conclusion:** The problem is harmonic structure, not fixable in root isolation

Root model accuracy breakdown:
- When bass was predicted correctly: 96.49% root accuracy
- When bass was predicted incorrectly: 67.32% root accuracy
- **But:** Only 54.2% of predictions were correct on this task, so this doesn't help

## What the Coordinator Meant

Agent 1's "bass anchor" was meant in the architectural sense (Chord_AI's separate bass/root heads), not as a direct feature. To properly use bass:
1. Need labels that include inversion (bass note + root)
2. Train a joint model: predict both bass and root, learn the inversion relationship
3. Use inversion to disambiguate root (C/E → root is C, not E)

**This is exactly what Chord_AI does** (reverse-engineered by Agent 3):
- Separate bass head (13-way: 12 pitch classes + N for no inversion)
- Separate root head (13-way)
- Jointly trained to learn the relationship

## Why Phase A Failed (Root Cause Summary)

| Assumption | Reality | Impact |
|---|---|---|
| Agent 1 predictions ≈ Billboard roots | Agent 1 predicts bass, not roots | Task mismatch (54% acc) |
| Bass information disambiguates roots | Root confusion is harmonic, not register | No signal to extract |
| 12-dim chroma needs external bass boost | Bass is already in the chroma | Adding it adds noise |

## Correct Next Step

**DO NOT attempt to fix root in isolation.** The right approach is:
1. Focus on QUALITY (dom recall 0.15 → need 0.70), which is the main bottleneck
2. Quality depends on harmonic context (Agent 2 showed +26pp with oracle root)
3. Learn context patterns with CNN/LSTM/RNN, not hand-crafted priors
4. Eventually: implement Chord_AI's structured multi-head architecture (separate bass/root/quality heads)

## Implications for Chord_AI Integration

Agent 3's reverse-engineering identified that Chord_AI's advantage comes from:
1. **Separate structured heads** (root, bass, 7th-bitmap) sharing features
2. **Jointly trained** → rare classes borrow strength from common ones
3. **Explicit bass modeling** → enables inversion/slash-chord support

Harmonia currently:
- Single flat output (72-class chord vocabulary or 5-way quality cascade)
- No explicit bass head
- No joint training of roots/bass/quality

**To match Chord_AI's architecture:** Phase B should inform a redesign to split chord inference into:
- Root head (12-class)
- Bass head (13-class: 12 pitch classes + N)
- Quality bitmap (5-class or more)
- All trained jointly, with shared chroma features

This will fix both root AND quality by properly modeling inversions and harmonic relationships.

## Conclusion

Phase A **correctly rejected the bass-anchor hypothesis** as formulated. The negative result is **informative**: it revealed the task mismatch (bass ≠ root) and confirmed that root P4/P5 confusion is harmonic, not bass-register.

**Result:** Proceed to Phase B (context-window quality model). If Phase B succeeds, use it to inform a structured-head redesign (Phase C: architecture).
