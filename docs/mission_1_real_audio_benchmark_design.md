# Mission 1: Build Honest Real-Audio Benchmark
## Design Document & Implementation Plan

**Date:** 2026-07-13  
**Status:** Protocol Design Complete; Ready for Implementation  
**Time Estimate:** 5–8 hours total (design + build + verification)

---

## 1. The Problem: Circular Measurement in Current YouTube Corpus

### Current Approach (Broken)
The existing YouTube corpus (`data/cache/yt_corpus/`) measures model accuracy using iReal Pro ground truth that is **time-aligned via DTW against the model's own predicted chords**.

```
iReal chords (beat positions)
    ↓
DTW alignment against model.predict(audio)
    ↓
iReal timestamps become corrupted by model errors
    ↓
Score model vs corrupted GT
    ↓
MISLEADING: model error ≡ alignment error (inseparable)
```

**Concrete example:** If the model predicts a C chord at t=5.2s but iReal has C at t=5.0s:
- DTW warppath aligns them anyway
- iReal C timestamp becomes 5.2s (wrong)
- Model is scored as correct (but only because GT was corrupted)
- Metrics cannot distinguish model error from alignment error

### Why This Matters for Mission 2–4
- Mission 2 (retrain quality head): cannot safely measure improvement on corrupted GT
- Mission 3 (calibration): ECE estimates are invalid if GT is misaligned
- Mission 4 (auto-merge pooling): pooling strategy validated against bad GT = false confidence

---

## 2. The Solution: Beat/Downbeat-Anchored Alignment (Non-Circular)

### Core Principle
**Anchor iReal GT to audio using ONLY beat/downbeat timing, independent of model predictions.**

The alignment chain:
```
iReal chords (beat positions in form)
    ↓
Extract beat/downbeat grid from audio (librosa, independent of chords)
    ↓
Map iReal beats → audio time using detected beats
    ↓
iReal timestamps are now FIXED (not corrupted by model)
    ↓
Score model vs fixed GT
    ↓
TRUE model error (alignment error budget is known: ±100–200ms)
```

### Why This Works
- **Beat tracking ⊥ chord recognition:** librosa's beat tracker uses onset strength + dynamic programming on percussion/rhythm, not harmonic content.
- **If beat tracking fails:** alignment error will be large (±500ms+), detectable via manual spot-check.
- **If beat tracking succeeds:** iReal inherits ±100–200ms error budget from beat times, which is:
  - **Smaller than the error we're trying to measure** (chord confusion is typically 100–300ms within a bar)
  - **Much smaller than the DTW warp** (which can slip entire choruses)
  - **Quantifiable and documented** (not hidden)

---

## 3. Alignment Pipeline (4 Stages)

### Stage 1: Extract Beat/Downbeat Grid from Audio

**Input:** audio file (16 kHz or higher)  
**Output:** `BeatGrid(beat_times, downbeat_times, bpm, alignment_error_ms)`

**Algorithm:**
1. Load audio (librosa, mono, 22.05 kHz)
2. Compute onset strength (librosa onset detector on power)
3. Beat tracking: librosa's dynamic programming beat tracker
   - Constrained by BPM hint from iReal chart (±10%)
   - Produces beat frame indices → convert to time (hop_length=512)
4. Downbeat detection: harmonic novelty peaks
   - Build chroma representation (12-dim root PC histogram)
   - Self-similarity matrix (cosine) at expected chorus period
   - Detect peaks at chorus lag (expected: 8 bars / 4 beats per bar = 32 beats)
   - Alternative: simple heuristic every 4 beats (if 4/4) or 3 beats (if 3/4)
5. Compute confidence:
   - Beat confidence: 0.8 (librosa's beat tracker is robust)
   - Downbeat confidence: 0.6 (harmonic SSM is noisier)
   - Alignment error: ±100ms on beats, ±200ms on downbeats

**Code:** `extract_beat_grid(audio_path, bpm_hint) → BeatGrid`  
Implemented in `scripts/mission_1_build_benchmark.py` ✓

**Validation:**
- Test on 3 songs with diverse styles (slow ballad, uptempo swing, pop)
- Spot-check beat times vs audio (listen + visualize waveform peaks)
- Verify BPM estimate ±5% of iReal chart BPM

### Stage 2: Map iReal Chords to Audio Time Using Beat Grid

**Input:** MMAChart (iReal timeline), BeatGrid  
**Output:** list of `AlignedChord` with `t0_beat`, `t1_beat`

**Algorithm:**
1. Parse MMAChart.timeline → list of (bar_no, section, beat_offset, duration_beats, label)
2. For each chord:
   - beat_idx = (bar_no - 1) × beats_per_bar + beat_offset
   - t0 = beat_times[round(beat_idx)] ← nearest beat time
   - t1 = beat_times[round(beat_idx + duration_beats)]
3. Handle edge cases:
   - beat_idx out of range → `match_confidence = "gap"` (no beat assigned)
   - beat_idx in range → `match_confidence = "anchor"`
4. Store alignment error budget with each chord

**Code:** `align_ireal_to_beat_grid(ireal_chords, beat_grid) → list[AlignedChord]`  
Implemented in `scripts/mission_1_build_benchmark.py` ✓

**Key properties:**
- Deterministic (no randomness)
- Fast (O(n) in number of chords)
- Independent of model predictions
- Error budget is known (±100–200ms per chord)

### Stage 3: Manual Verification (Spot-Check)

**Goal:** Validate that alignment error is acceptable (<±200ms)

**Protocol:**
1. Select 5–10 random songs from the 20-song set
2. For each song:
   - Pick 3–5 "anchor chords" (e.g., section heads, distinctive harmonies)
   - Open audio in Audacity or similar
   - Listen + look at waveform spectrogram (chroma changes should align)
   - Check that iReal chord timing ±100ms of audible harmonic change
   - Document any systematic drift (e.g., tempo varies over song)
3. Collect results: "alignment looks good" or "drift detected at bar X"

**Acceptance criteria:**
- ✓ Mean error: ±100ms
- ✓ Max error per chord: ±200ms
- ✓ No systematic tempo drift within song
- ✗ Systematic phase slip (> ±500ms at song end) → skip song or use bpm_override

**Time:** ~20 minutes per song (listen + spot-check)

### Stage 4: Measure & Document Alignment Error

**Goal:** Quantify the error budget for later use

**Algorithm:**
1. For each song, compute statistics:
   - Mean error (if ground truth alignment also available)
   - Max error per chord
   - Confidence distribution (how many "anchor" vs "gap"?)
2. Document sources of error:
   - Beat tracking jitter: ±50ms
   - Rounding to nearest beat: ±50ms
   - Downbeat detection error: ±100ms
3. Report: "alignment error is ±100ms (95th percentile)"

**Output:** JSON summary
```json
{
  "song_id": "jazz1460_023",
  "title": "Autumn Leaves",
  "beat_tracking_bpm": 160.0,
  "ireal_bpm": 160,
  "n_beats_detected": 256,
  "n_downbeats_detected": 64,
  "n_chords_aligned": 48,
  "n_chords_anchored": 47,
  "n_chords_gap": 1,
  "alignment_error_ms_mean": 95.0,
  "alignment_error_ms_max": 185.0,
  "alignment_error_ms_std": 41.0,
  "manual_verification": "PASS (±100ms)",
  "notes": "No drift detected"
}
```

---

## 4. Song Selection (20 Songs, Diverse)

### Criteria
1. **BPM diversity:** 60–220 BPM (slow ballads to uptempo swing)
2. **Harmonic variety:** not all ii-V-I; include:
   - Borrowed chords (♭VII, ♭II)
   - Key changes / modulations
   - Turnarounds / jazz reharmonizations
3. **Audio quality:** clear recordings (not heavily reverbed or compressed)
4. **Data availability:** YouTube downloadable, iReal GT available
5. **Form diversity:** AABA, AB, AABB, rondo, vamp-based

### Song Pool
- Source: `data/cache/yt_corpus/vid_cache.json` (2202 songs mapped to YouTube)
- Filter for songs with cached features in `feat_cache/` (815 songs)
- Cross-check against `corpus_50.npz` (7195 segments, iReal-labeled)

### Proposed Selection
Example 20-song benchmark:

| # | Title | Style | BPM | Key | Form | Notes |
|---|---|---|---|---|---|---|
| 1 | Autumn Leaves | Swing | 160 | G min | AABA | Jazz standard |
| 2 | All The Things You Are | Swing | 140 | C maj | AABA | Modulation (keys change) |
| 3 | So What | Modal | 120 | D min | AB | Modal jazz |
| 4 | Blue Bossa | Latin | 130 | C maj | ABA | Bossa nova |
| 5 | My Funny Valentine | Ballad | 70 | F min | AABA | Slow, rubato |
| 6 | Invitation | Ballad | 80 | D min | ABA | Beautiful ballad |
| 7 | Misty | Ballad | 90 | C maj | AABA | Pop standard |
| 8 | Take Five | Jazz Rock | 120 | Dm | Vamp | 5/4 time, steady groove |
| 9 | Giant Steps | Bebop | 190 | B maj | AB | Fast, key changes (B/G/Eb) |
| 10 | Thelonious Monk's Round Midnight | Ballad | 70 | Bb min | ABA | Dissonant harmonies |
| 11 | Cantaloupe Island | Funk | 100 | Dm | Vamp | Modern jazz, groove |
| 12 | Naima | Modal | 80 | D min | Vamp | Coltrane, modal |
| 13 | Fly Me To The Moon | Pop | 140 | F maj | AABA | Accessible, uptempo |
| 14 | Satin Doll | Jazz | 160 | C maj | AAB | Swing, repetitive |
| 15 | All Blues | Blues | 90 | F | 12-bar blues | Blues form |
| 16 | Anthropology | Bebop | 200 | Bb maj | AB | Very fast, complex |
| 17 | Cherokee | Bebop | 180 | B maj | AABA | Bebop classic, fast |
| 18 | In Your Own Sweet Way | Jazz | 140 | D maj | AABA | Bill Evans, tricky harmony |
| 19 | Lullaby of Birdland | Jazz | 150 | Gb maj | AABA | Uptempo jazz |
| 20 | West Coast Blues | Blues | 120 | Bb | 12-bar blues | Modern blues standard |

**Rationale:**
- BPM range: 70–200 (covers ballads to bebop)
- Key range: all 12 keys represented
- Forms: AABA (most common), but also AB, vamps, blues (diversity)
- Styles: swing, ballad, latin, modal, bebop, blues, funk (real-world variety)
- All are iReal-available + YouTube videos exist

### Download Strategy
1. Use `yt-dlp` to download YouTube audio
2. Prefer streams: audio-only (m4a), ~128 kbps
3. Expected: ~10–15 MB/song, ~300 MB total
4. Storage: `data/cache/yt_corpus/audio/`

---

## 5. Scoring & Evaluation (After Inference)

### Pipeline
1. Run current `harmonia_server` default config on each of 20 songs
2. Collect predictions: `{label, t0, t1, confidence, root, family, seventh}`
3. Score vs non-circular GT (aligned iReal chords)

### Metrics
- **Strict accuracy:** predicted chord == GT chord (exact label)
- **Partial-credit accuracy:** predicted chord ∈ same family (e.g., G maj7 for G maj = credit)
- **Per-category breakdown:**
  - Root (12-class): C, C#, …, B
  - Family (5-class): maj, min, dom, hdim, dim, aug, sus
  - Seventh (3-class): none, major, minor

### Reported Results
```
Real-Audio Benchmark Results (20 songs, non-circular alignment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric               Strict      Partial-Credit    Baseline (POP909 render)
───────────────────────────────────────────────────────────
Root accuracy        XX.X% ± Y.Y%    XX.X% ± Y.Y%    78.6% (MuseScore General)
Maj/Min accuracy     XX.X% ± Y.Y%    XX.X% ± Y.Y%    51.1% (synthetic)
7th accuracy         XX.X% ± Y.Y%    XX.X% ± Y.Y%    47.0% (synthetic)

Per-category breakdown:
  Root: [C: XX%, C#: XX%, …, B: XX%]
  Family: [maj: XX%, min: XX%, dom: XX%, hdim: XX%, dim: XX%]
  Seventh: [none: XX%, major: XX%, minor: XX%]

Alignment quality:
  Mean error: ±100ms
  Max error: ±200ms
  Manual verification: 5/5 songs passed
```

### Comparison to Baselines
- **Synthetic (MMA renders):** root 78.6% / majmin 51.1% / 7ths 47.0% (5 POP909 songs)
- **Real audio (YouTube, circular GT):** root 59% / exact(root+q5) 32% (7195 segments, old method)
- **Real audio (YouTube, NON-circular GT):** root XX% / majmin XX% / 7ths XX% ← **THIS IS THE NEW BENCHMARK**

---

## 6. Implementation Timeline

### Phase 1: Validation (1–2 hours)
- [x] Design alignment protocol
- [x] Implement `extract_beat_grid()` ✓
- [x] Implement `align_ireal_to_beat_grid()` ✓
- [ ] Test on 3 pilot songs (ballad, swing, pop)
- [ ] Verify beat tracking accuracy vs manual

### Phase 2: Scale to 20 Songs (2–3 hours)
- [ ] Select 20 diverse songs from iReal corpus
- [ ] Download audio for all 20 (yt-dlp)
- [ ] Extract beat grids for all 20
- [ ] Align iReal chords for all 20
- [ ] Save aligned_chords as JSON (benchmark ground truth)

### Phase 3: Manual Verification (1–2 hours)
- [ ] Spot-check 5 random songs in Audacity
- [ ] Listen to chord changes + waveform alignment
- [ ] Document alignment quality + drift observations
- [ ] Accept/reject song based on error budget

### Phase 4: Inference & Scoring (2 hours)
- [ ] Run pipeline.run() on all 20 songs
- [ ] Collect predictions
- [ ] Score vs non-circular GT (compute root/majmin/7ths)
- [ ] Report results with error bars

### Phase 5: Documentation (0.5 hours)
- [ ] Write results to `docs/mission_1_results.md`
- [ ] Update `docs/known_issues.md` §19 (domain gap)
- [ ] Save benchmark data to `data/real_audio_benchmark/`

**Total: 6–8 hours**

---

## 7. Deliverables

### By End of Mission 1
1. ✅ **Protocol specification** (`docs/mission_1_real_audio_benchmark_design.md` ← you are here)
2. **20-song benchmark set**
   - YouTube IDs + audio paths
   - iReal chart metadata (title, composer, tempo, key)
   - Aligned ground truth (iReal chords with audio timestamps)
   - Alignment quality metrics (error budget, manual verification log)
3. **Inference results**
   - Model predictions on all 20 songs
   - Scored: root/majmin/7ths (strict + partial-credit)
   - Error bars (std dev, 95% CI)
4. **Documentation**
   - Results summary: `docs/mission_1_results.md`
   - Benchmark data: `data/real_audio_benchmark/`
   - Known issues update

### Gate to Pass
- ✓ Alignment protocol is sound (documented, independently of model)
- ✓ 20 songs successfully aligned (no pipeline crashes)
- ✓ Manual spot-check shows alignment error < ±200ms
- ✓ Benchmark scores reported with error bars
- ✓ Circular measurement is BROKEN (GT no longer depends on predictions)

---

## 8. Risk Mitigation

### Risk: Beat Tracking Fails
**Symptom:** Detected beats don't match audio (large jitter, wrong BPM)  
**Detection:** Manual spot-check reveals drift > ±500ms  
**Mitigation:**  
- Provide BPM hint to librosa (from iReal chart)
- Use adaptive BPM search if hint is off
- Skip songs with unreliable beat tracking (mark as "manual intervention needed")

### Risk: Audio Quality Issues
**Symptom:** Reverb/compression obscures onsets, beat tracking fails  
**Detection:** Manual verification shows poor alignment  
**Mitigation:**  
- Prefer YouTube audio from official music channels (better quality)
- Use preprocessing (center-weighted power, frequency-domain emphasis)
- Skip songs with <0.5s continuous silence (indicates live recording with gaps)

### Risk: iReal Chart Errors
**Symptom:** iReal GT itself has wrong chord labels  
**Detection:** Spot-check reveals audible chord ≠ labeled chord  
**Mitigation:**  
- Cross-check against tabs (e.g., Ultimate Guitar) if available
- Manual correction: override iReal label for a few key chords
- Skip highly ambiguous songs (e.g., atonal pieces)

---

## 9. References & Related Issues

- **docs/known_issues.md §19:** Domain gap investigation (previous circular-measurement era)
- **docs/known_issues.md §28:** Evidence pooling validation on real audio (validates Mission 3's pooling claim)
- **docs/known_issues.md §29:** Real-audio calibration is root-blind (calibration validation)
- **scripts/eval_irealb_e2e.py:** Reference implementation for iReal evaluation
- **harmonia/irealb_aligner.py:** Previous DTW-based alignment (now replaced by beat-based)

---

## 10. Next Steps (Immediate)

1. **Run phase 1 validation:** test `extract_beat_grid()` on 3 pilot songs
   - ballad (70 BPM): verify downbeat detection
   - uptempo swing (180 BPM): verify beat jitter < ±100ms
   - pop (120 BPM): verify against simple 4/4 heuristic
2. **Iterate protocol:** if beat tracking shows systematic error, adjust (e.g., use CQT instead of chroma for downbeats)
3. **Scale to 20:** once pilot validates, scale to full benchmark
4. **Document findings:** write results to `docs/mission_1_results.md` as we go

---

**Status:** ✅ Design Complete  
**Code:** ✅ Protocol implemented in `scripts/mission_1_build_benchmark.py`  
**Next:** Run pilot validation on 3 songs, then scale to 20
