# Production Deployment Checklist — Real-Audio BP48 Training

*2026-07-15. Checkpoint document for real-audio YouTube+iReal training pipeline.*

**Status: IN PROGRESS** — corpus build running, training and evaluation to follow.

---

## Corpus Build Status

| Stage | Status | Notes |
|---|---|---|
| YouTube+iReal pilot (10 songs) | 🔄 Building | Songs 1-8 complete (~1,450 records so far) |
| BP48 feature extraction | ✓ Done | `data/cache/audio_chord_features.npz` (1.7M) |
| Alignment to iReal GT | 🔄 In progress | Via `harmonia/data/yt_chord_corpus.py` |
| Final corpus.npz packing | ⏳ Pending | Expected ~1,300–1,600 clean records (10 songs) |

---

## Training Plan

Once corpus.npz is ready:

1. **Load and split:**
   - Filter to exact + family matches only (skip mismatches)
   - Song-stratified split: 80% train / 10% val / 10% test
   - Expected: ~1,000 train / 130 val / 130 test records

2. **Train heads (architecture from bridge findings):**
   - **Root head:** MLP(48→128→64→12) on absolute BP48
   - **Quality head:** MLP(48→128→64→7) on root-relative BP48
   - Balanced class weights (inverse frequency)
   - 50 epochs, batch size 32, lr 3e-4, cosine annealing

3. **Evaluate on test set:**
   - Report: root accuracy, per-class quality recall
   - Calculate MIREX metrics (root, majmin, sevenths, tetrads) via `harmonia/eval/mirex_eval.py`

4. **Acceptance criteria (shippable iff ALL met):**
   - Root accuracy ≥ 85% (real audio, harder than synthetic)
   - Quality dom recall ≥ 65% (cascade via predicted root)
   - Quality balanced accuracy ≥ 68%
   - MIREX majmin accuracy ≥ 65% (real audio)

---

## Failure Scenarios & Next Steps

| Scenario | Criterion | Action |
|---|---|---|
| **Root underperforms** | Acc < 85% | Investigate: (a) insufficient training data (10 songs = ~1k records); (b) alignment quality; (c) need bass/register features; (d) consider scaling to 50–200 song corpus |
| **Quality underperforms** | Dom recall < 65% | Likely (a) small corpus size; (b) cascade errors from root; (c) domain gap (real audio has more voicing variation than synthetic) |
| **Both OK, but below 70%+** | Balanced qual < 68% OR MIREX majmin < 65% | Document gap; consider iterative scaling (50→200 songs) or architecture tweaks |
| **All targets hit** | ✓ All pass | Proceed to deployment |

---

## Deployment (if shippable)

Once metrics are solid:

1. **Save production models:**
   ```
   data/models/prod_root_v1.pt       (dated, versioned)
   data/models/prod_quality_v1.pt
   data/models/prod_7th_v1.pt        (if trained separately)
   ```

2. **Inference wrapper:**
   ```
   scripts/inference_production.py
   - Load models + training means/stds
   - Input: audio path, cache_dir
   - Output: root (0-11) + quality + confidence
   - Handles BP48 extraction (root-relative normalization)
   ```

3. **Documentation:**
   - Update `CLAUDE.md` to reflect real-audio training path
   - Log findings in `docs/known_issues.md` #31 addendum
   - Create `docs/production_training_results.md` with:
     - Final metrics (root acc, per-class quality recall, MIREX)
     - Corpus statistics (songs, chords, per-song distribution)
     - Generalization notes (synthetic vs real-audio gap)
     - Deployment assumptions (BP48 input, functional root frame, no bass head yet)

4. **Commit:**
   ```
   git add data/models/prod_*.pt \
           scripts/inference_production.py \
           docs/production_training_results.md \
           docs/production_deployment_checklist.md
   git commit -m "prod: real-audio BP48 training complete (root >85%, quality >0.68)"
   ```

---

## Known Caveats (not blockers, but important)

- **Corpus size:** 10-song pilot is small (blog post from Jul 9 suggests 50–200 songs for stable metrics). Metrics may have high variance.
- **Real-audio domain gap:** Synthetic dom recall 0.776 (#31) vs real-audio baseline ~0.21 (issue #19). Even 0.65 is ambitious; settle for 0.60+ if corpus is too small.
- **Bass/voicing:** Root P4/P5 errors (~44% of root confusions, #31) are not solved by chroma alone — requires bass register or separate bass head. This model is a ceiling without that feature.
- **7th vocabulary:** Q5 model (maj/min/dom/hdim/dim) does not include extensions (7#11, 6/9, etc.). Voicing head is a future addition, not included here.

---

## If Underperforming

Before iterating:

1. **Minimal changes:** Re-run training with different random seeds (5–10 CV) to gauge variance
2. **Diagnostic plots:** Show confusion matrices for root (P4/P5 errors?) and quality (dom→maj confusion?)
3. **Corpus analysis:** How many songs per quality? Are rare classes under-represented? Do certain songs/artists have systematically different chord voicings?
4. **Feature diagnostics:** Does basic 12-dim chroma (melted from 48-dim) perform worse/better on quality? Does absolute vs root-relative help root?
5. **Scaling experiment:** If 10 songs is too small, rebuild with 50–100 songs from the 200-song build (if that ever completed)

Only iterate if investigation finds a clear, fixable issue (not just "need more data").
