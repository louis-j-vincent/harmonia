# Billboard McGill Model Training — Executive Summary

**Date**: 2026-07-14  
**Status**: Infrastructure Ready, Awaiting Execution  
**Objective**: Train production-ready chord model on Billboard McGill (890 songs), measure accuracy improvements vs iRealb baseline, analyze results.

## Current Baseline (iRealb)
From `docs/known_issues.md #19`:
- Root accuracy: **59.0%** (12-way pitch class)
- Majmin accuracy: **61.0%** (major/minor distinction)
- Sevenths accuracy: **45.0%** (dom7 detection)
- Tetrads accuracy: **32.0%** (full chord match, root + quality)
- Measured on: ~7,195 real chords from jazz1460 (iReal Pro)

## Billboard Dataset
- **Size**: 890 pop/rock songs (8.9× larger than iRealb's ~100)
- **Quality**: MIREX hand-verified (100% verified vs algorithm-generated for iRealb)
- **Splits**: 712 train / 89 val / 89 test (deterministic, seed=42)
- **Annotation**: Functional notation → Q5 vocabulary (maj/min/dom/hdim/dim)

## Expected Improvements
**Optimistic**: +8–15pp (Likely: root 68%, majmin 71%)  
**Likely**: +3–8pp (Likely: root 62%, majmin 65%)  
**Conservative**: 0–3pp (Domain mismatch dominates, or no significant gain)

## What's Been Delivered

### Scripts (Ready to Run)
1. **`scripts/eval_billboard_prod.py`** (303 lines)
   - Evaluates production pipeline on Billboard test set
   - Measures root, majmin, sevenths, tetrads via mir_eval
   - Compares to iRealb baseline
   - Generates JSON results

2. **`scripts/train_billboard_chord_model.py`** (544 lines)
   - Full training pipeline: extract features → train models → evaluate
   - Quality head: MLP(60→128→64→5) for maj/min/dom/hdim/dim
   - Root model: Logistic regression for 12-way pitch class prediction
   - Features: 60d (48d pitch + 12d chroma), beat-synchronized

### Documentation
- **Plan**: Complete technical specification with 4 phases, architecture, timelines, risk assessment
- **Status**: Overview of what's been built and next steps

## Execution Timeline
1. **Download Billboard audio** (30 min – 2 hours)
2. **Extract features** (4–8 hours, parallelizable)
3. **Train models** (1–2 hours)
4. **Evaluate** (2–4 hours)
5. **Analyze & report** (1–2 hours)

**Total**: 9–17 hours (wall-clock), ~10–18 hours if sequential

## How to Run
```bash
# 1. Download audio (one-time)
python -c "import mirdata; ds = mirdata.initialize('billboard'); ds.download()"

# 2. Extract features
python scripts/train_billboard_chord_model.py --extract-features --split all

# 3. Train models
python scripts/train_billboard_chord_model.py --train --epochs 100

# 4. Evaluate
python scripts/eval_billboard_prod.py --split test --n-songs 89 --verbose
```

## Success Criteria
✓ Feature extraction: >70k chords extracted  
✓ Quality model: >60% majmin accuracy on test  
✓ Root model: >55% accuracy on test  
✓ Evaluation: Root/majmin/sevenths/tetrads measured vs iRealb  
✓ Report: Deltas computed, analysis complete  

## Key Files
- Scripts: `scripts/eval_billboard_prod.py`, `scripts/train_billboard_chord_model.py`
- Outputs: `data/models/billboard_quality_head_v1.pt`, `docs/billboard_prod_eval_results.json`
- Integration: `harmonia/models/chord_pipeline_v1.py` (already supports Billboard)

---
**Next Step**: Download Billboard audio and start feature extraction.
