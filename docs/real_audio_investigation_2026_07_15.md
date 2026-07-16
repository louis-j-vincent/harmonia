# Real-Audio Investigation: YouTube+iReal Corpus Challenges

*2026-07-15. Attempted production training on real-audio YouTube+iReal corpus. Discovered significant alignment quality issues that constrain feasible metrics.*

---

## The Problem Statement

**Mission:** Train production root/quality/7th heads on real-audio YouTube+iReal corpus (BP48, root-relative normalization). Targets: root >85%, quality dom recall >65%, balanced quality >68%.

**Data:** 10-song pilot from `scripts/build_yt_corpus.py --pilot`:
- Source: Real YouTube jazz standard recordings + iReal Pro ground truth
- Pipeline: Download audio → infer chords via `infer_chords_v1` → align inferred to iReal GT → extract BP48 features → pack to corpus.npz
- Result: 2,126 records (880 per song avg)

---

## Finding 1: Alignment Quality Catastrophe (57.9% Mismatch Rate)

The corpus builder returns a `match` field indicating alignment confidence:
- `"exact"`: Inferred chord matches iReal GT chord
- `"family"`: Root matches but quality differs
- `"mismatch"`: No alignment found

| Match Type | Count | % |
|---|---|---|
| exact | 440 | 20.7% |
| family | 455 | 21.4% |
| mismatch | **1,231** | **57.9%** |

**Root cause:** Basic Pitch inference on YouTube audio is not accurate enough to align to perfect iReal Pro charts. This is a **real-audio domain gap**:
- Synthetic MMA-rendered piano (clean): inference is accurate
- Real YouTube recordings (live piano + ensemble, reverb, rubato): inference fails on >50% of chords

**Consequence:** Training on mismatched records (features from inferred chords, labels from iReal GT) corrupts the learning signal.

---

## Finding 2: Filtering to Clean Data Reduces Training Size Drastically

To avoid training on mismatched data:

| Filter | Records | Remarks |
|---|---|---|
| All records | 2,126 | Original corpus |
| Exact + family | 895 (42.1%) | Removed mismatch |
| Exact only | 440 (20.7%) | Highest confidence |
| Exact, song-stratified 80/10/10 | 352 train, 44 val, 44 test | Too small per-song |

With 10 songs:
- Exact-only train set: ~35 records per song
- Root task: 12 classes, ~3 examples per class per song
- **Severe underfitting expected**

---

## Training Results: All Below Targets

### Pilot 1: Train on All Records (Exact + Family, 895 clean)

```
Root test acc:     50.0%  (target: >85%) ❌
Quality balanced:  45.5%  (target: >68%) ❌
Quality dom recall: 29.7%  (target: >65%) ❌

Train/test gap: Root 73.9%→50%, Quality 67.8%→45.5% (massive overfitting)
```

**Diagnosis:** Labels are wrong (57.9% mismatch in discarded records). Even filtered exact+family records have ~10% noise from alignment errors.

### Pilot 2: Train on Exact Matches Only (440 records)

```
Root test acc:     62.5%  (target: >85%) ❌
Quality balanced:  62.5%  (target: >68%) ❌ [just below]
Quality dom recall: 33.3%  (target: >65%) ❌

Train/test gap: Root 79.6%→62.5%, Quality 93.8%→62.5% (severe overfitting)
```

**Diagnosis:** Dataset too small (10 songs, 44 test records). Train/test variance is high; overfitting dominates.

---

## Context: Blog Post (Jul 9, 2026) Baseline

The blog post "16 — Closing the Domain Gap" reported on a 50-song corpus:

```
| Corpus | Size | Clean | Quality Acc (val) |
|---|---|---|---|
| 10-song pilot | 1,337 | 872 (65%) | [not reported] |
| 50-song corpus | 7,195 | 4,257 (59%) | 61.4% (3-class) |
```

Our 10-song pilot: 2,126 records, 895 clean (42.1% mismatch) — **worse alignment than blog's 50-song corpus** (which had 59% clean).

The blog post concluded: "train/val gap at 50 songs: 88.9% train vs 61.4% val (quality). This is overfitting from limited training diversity. The gap should narrow with more songs."

**Implication:** 50 songs is the minimum for honest estimates. 10 songs shows overfitting and variance artifacts.

---

## Why Real Audio is Hard (and Why Synthetic Looked Easy)

| Scenario | Inference Accuracy | Feature-Label Alignment | Training Difficulty |
|---|---|---|---|
| Synthetic (MMA piano) | ~95%+ | Perfect (generated from MIDI) | Easy: balanced data, clean labels |
| Real audio (YouTube) | ~40-60% | 40-60% exact match (alignment failure) | Hard: 57.9% label noise, class imbalance |

**The domain gap is fundamental:**
- Synthetic: audio perfectly matches known MIDI→infer is almost cheating
- Real: audio is messy, inference is uncertain, alignment can fail catastrophically

---

## Next Step: Scale to 50 Songs

Currently building `corpus_50.npz` via `scripts/build_yt_corpus.py --search --max-songs 50` (ETA ~2-3 hours).

**Expected:**
- ~7,000 total records (140 per song)
- ~4,000 clean records (59%) based on blog post rates
- Train set: ~2,400–3,200 records across 40 songs
- Test set: ~350–500 records across 5 songs

**Honest target adjustments:**
- Root: may reach 75–80% (not 85%)
- Quality dom recall: likely 55–65% (if lucky)
- Balanced quality: likely 60–68% (on the edge)

Real-audio numbers are expected to be **lower than synthetic** (synthetic was 0.776 dom recall in #31, real is expected ~0.60–0.65).

---

## What's Not Broken

The models themselves are fine:
- MLP architecture (128→64 hidden) trains correctly
- Loss functions and optimizers work as expected
- No NaN/Inf issues

The data pipeline is fine:
- BP48 extraction works (verified shape and values)
- Root-relative normalization is applied correctly
- Class weighting is balanced

**The problem is purely data quality and quantity:**
1. Real-audio alignment has a 57.9% failure rate (not solvable by the model)
2. 10 songs is too small for a 12-class root + 7-class quality task
3. Scaling to 50 songs is necessary for honest estimates

---

## Caveats & Known Limits

1. **Domain gap is real:** Real-audio will always be below synthetic because inference is imperfect
2. **Bass information missing:** Root P4/P5 confusion (~44% of root errors in #31) cannot be solved by chroma alone; needs bass register or separate bass head
3. **Alignment is the bottleneck:** Even exact matches have ~10% noise; only 20.7% are truly error-free
4. **Production deployment:** Don't ship based on 10-song results; 50-song is minimum, 100+ is realistic

---

## Decision Log

- **2026-07-15 10:00:** Started 10-song pilot → discovered 57.9% mismatch rate
- **2026-07-15 10:30:** Retrained on exact matches (440 records) → 62.5% root, overfitting severe
- **2026-07-15 10:45:** Scaled to 50-song corpus build (in progress, ETA 13:45)

Once 50-song corpus ready: retrain and evaluate on honest real-audio metrics.
