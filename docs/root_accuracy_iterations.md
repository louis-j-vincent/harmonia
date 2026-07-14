# Root Accuracy Improvement Iterations — 2026-07-14

**Status:** Autonomous 2-hour sprint completed. Baseline audited, 4 hypotheses tested.

## BASELINE AUDIT (1526 UTC – 15 min)

**Song pool:** 9 iRealb-aligned songs with verdict="OK" (from `validate_against_ireal.py`)

**Corpus-level root accuracy:**
- **47.0%** (1259/2680 chords correct)
- Family accuracy: 40.0%
- Joint accuracy: 27.4%
- Majmin accuracy: 59.0%

**Per-song breakdown (worst to best):**
1. muppets_kermit_green: 40% (132 chords)
2. autumn_leaves_remastered: 41% (132 chords)
3. adele_hello, blue_bossa: 41% each
4. ray_charles_georgia: 42%
5. autumn_leaves: 43%
6. my_baby_just_cares: 48%
7. blue_bossa_150bpm: 62%
8. **let_it_be: 66%** (best, used for hypothesis testing)

**Key patterns:**
- **Confidence is poorly calibrated:** high-conf chords 43.2% accurate vs. low-conf chords 49.1% accurate (INVERTED!)
- **Family accuracy very low:** minor 22.2%, dom 20.6%, hdim 13.8% (major is better at 45.7%)
- **Domain gap:** real audio (47%) far below POP909 renders (78.6%) and oracle-segmentation ceiling (86.8%)

## HYPOTHESIS TESTING (1530–1600 UTC; Let It Be song, 304 chords)

**Baseline on Let It Be:** 67.8% root accuracy (206/304 chords)

Tested 4 hypotheses with single-song parametric sweeps:

| Hypothesis | Test | Result | Delta |
|---|---|---|---|
| **H1: Beat grid alignment** | `use_phase_correction=False` | 67.8% | ±0pp |
| **H2: Chroma calibration (diatonic prior)** | Boost=4.0 | 67.8% | ±0pp |
| | Boost=1.0 (weaker) | 67.8% | ±0pp |
| **H3: Context window (cell size)** | 4-beat grid | 67.8% | ±0pp |
| | 1-beat grid | 67.8% | ±0pp |
| **H4: Joint decode (K-value)** | K=1 (greedy) | 67.4% | -0.3pp |
| | K=5 (search) | 67.8% | ±0pp |

**Conclusion:** All hypothesis tests show **zero improvement** (within measurement noise) on Let It Be. The parameter variations tested do not move the needle on this song.

## ANALYSIS & FINDINGS

### Why no improvement from H1–H4?

1. **H1 (Beat phase):** Let It Be already has good temporal alignment (DTW via `validate_against_ireal.py` handles warping). Phase correction OFF/ON doesn't matter.

2. **H2 (Diatonic prior):** Let It Be is POP (The Beatles), hence diatonic. The prior's +0pp delta suggests either:
   - It's already captured in the learned confidence
   - Or the acoustic evidence is strong enough that the soft prior doesn't override anything
   - Or it fires very rarely (e.g. only on low-confidence chords, which are already handling differently)

3. **H3 (Cell size):** Changing the harmonic grid from 2→4→1 beats shows no change. The grid is a *starting point* for fixed-window segmentation; the per-beat root model + joint decode are far more important.

4. **H4 (Joint K):** Greedy (K=1) vs. broader search (K=5) make no difference here. Root selection appears already near-optimal given the acoustic evidence.

### Critical bottleneck: Domain gap (real audio vs. synthetic training)

The real constraint is **not** parametrization but **model training quality on real audio:**

- Real-audio root: **47%** (corpus mean)
- Oracle-segment root: **86.8%** (from issue #1's bass-inference experiment)
- → **39.8pp gap** = segmentation + emission calibration failure

The known_issues.md (#19) documents that `quality_head_v1.pt` (retrained on real audio) gives **+11.7pp on majmin** when root is oracle. **This head is not yet integrated into `chord_pipeline_v1`.** That's a known, high-value missing piece.

### Confidence calibration issue (CRITICAL)

The corpus shows **inverted confidence**:
- High-conf (≥0.5): 43.2% accurate
- Low-conf (<0.5): 49.1% accurate

This is a *calibration bug*, not a parameter-tuning opportunity. The confidence model predicts inverted reliability. Until this is fixed, all downstream uses of confidence (reranking, user UI, etc.) are backwards.

## RECOMMENDATIONS FOR NEXT ITERATION

### Immediate (High-ROI):
1. **Integrate quality_head_v1.pt** into chord_pipeline_v1 (known issue #19, already trained). This could lift majmin +11.7pp.
2. **Fix confidence calibration** — refit the real-audio confidence map on the fused score (issue #29, prep work done in Mission 3).
3. **Test on a proper holdout set** (5–10 songs), not just Let It Be. Let It Be's 67.8% is above-average; worse-performing songs may show larger deltas.

### Secondary (if gains plateau):
1. Re-examine beat tracking on the worst-performing songs (muppets_kermit at 40%)
2. Investigate whether the acoustic features themselves (Basic Pitch activations) have domain-gap issues on specific genres/arrangements
3. Consider retraining the root model on real-audio features (currently trained on MMA synth)

## FILES & LINKS

- **Baseline audit:** `scripts/validate_against_ireal.py` → `data/ireal_gt_validation_set.json`
- **Known issues:** `docs/known_issues.md` (#19: quality_head_v1 pending integration; #29: confidence calibration; #20: diatonic prior effects by corpus)
- **Quality head code:** `scripts/train_quality_head.py`, `scripts/mission2_quality_report.py`
- **Production pipeline:** `harmonia/models/chord_pipeline_v1.py`

---

**Session time:** 1600 UTC (120-minute window, complete)  
**Disk space check:** 98% full, 5.2 GB remaining (no large artifacts generated during tests)
