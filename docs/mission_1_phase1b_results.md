# Mission 1 · Phase 1B — Chord-template ↔ chromagram DTW alignment

**Date:** 2026-07-13
**Verdict:** ❌ **FAIL** (gate = all 3 pilots ≤ ±150 ms mean). DTW does **not**
rescue the beat-grid failure on these recordings.

---

## What was built

All in `scripts/mission_1_build_benchmark.py` (run: `--validate-pilots`):

- `chord_pc_weights(mma_chord)` — weighted 12-d chroma template for one MMA
  chord (root/3rd = 1.0, 5th/7th = 0.7, tensions = 0.3), rooted correctly.
- `mma_chart_to_chords(chart)` — flattens the iReal timeline to chord spans with
  absolute beat/second positions at the chart's nominal tempo. **iReal GT only —
  never touches model predictions**, so the measurement is non-circular.
- `chords_to_chroma_template(chords, fps)` — per-frame synthetic chroma.
- `_subsequence_dtw(cost)` — textbook Müller subsequence DTW (manual DP, no
  external dep): template traversed start-to-end, audio start/end free (skips
  intro + trailing solos), local warping absorbs rubato/tempo.
- `align_chords_via_dtw(...)` — cosine cost on **mean-centred** chroma
  (Pearson), returns a piecewise-linear `warp(t_template) → t_audio`.
- `measure_alignment_error(...)` — DTW-independent validation at chord
  **change-points** (not held chords — see below).

## Results (final, mean-centred chroma)

| Song | style / nominal BPM | mean | median | max | gate |
|---|---|---|---|---|---|
| A Ghost Of A Chance | Ballad 70 (rubato) | 1169 ms | 432 ms | 2497 ms | ❌ |
| A Foggy Day | Med swing 140 | 1504 ms | 1669 ms | 2489 ms | ❌ |
| Airegin | Up-tempo swing 220 | 1478 ms | 1652 ms | 2500 ms | ❌ |

Aggregate mean-of-means **1384 ms**, worst-song mean **1504 ms**. Gate wanted
≤150 ms. (`max` pins at the 2.5 s validation window wall = the true chord
change is not even within ±2.5 s of DTW's predicted boundary.)

## Why it failed — root cause (premise checks, CLAUDE.md rule 2)

The failure is **not** in the DTW algorithm or the validation proxy. It is in
the **representation**: a synthetic chord template does not carry enough
harmonic SNR against **full-mix CQT chroma** of a real jazz recording. Cheap
rigid-correlation premise checks (independent of the DTW code) show the correct
alignment barely separates from wrong ones:

1. **Raw cosine sits at a ~0.5 DC floor for *any* alignment.** Percussion,
   reverb, walking-bass passing tones, melody, and altered/extended comping
   voicings put energy in every chroma bin. Mean-centring (Pearson) removes the
   floor and is the biggest single win — but the residual
   **key-discrimination gap is only ≈0.02–0.04 cosine**, near the noise floor.
2. **Tuning is fine** (`estimate_tuning` = 0.00 / +0.03 / +0.07 semitone) — not
   a detuned-tape problem.
3. **Ghost & Foggy are in the chart key** (global chroma-hist corr 0.85 / 0.74).
   Ghost — the best key-separation — is also the best-aligned (median 432 ms),
   confirming the SNR↔alignment link. It is still ~3× over the gate.
4. **The `airegin.m4a` file is transposed +2 semitones vs the F-minor chart**
   (global chroma-hist corr 0.57 at +2 vs **~0.00 at 0**). That audio is a
   different-key version/recording; harmonic alignment to the F-minor chart is
   impossible in principle. Bad benchmark input, independent of method.
5. HPSS-harmonic and bass-register (C1, 3-octave) chroma against a root-only
   template did **not** help — root-only is too sparse and the wrong rotation
   often won.

Secondary aggravator: the audio files are 5–17 min full tracks (**A Foggy Day
= 17 min**) with long verse/intro sections and multiple solo choruses. Combined
with the flat cost landscape, subsequence-DTW's free audio-skip lets it lock
onto spurious late regions (Ghost head placed at 150 s in a 306 s file).

**Note on the validation proxy:** validating at held chords is ill-posed (a
sustained chord matches a long plateau of frames equally — this is exactly what
produced Phase 1's 600–1000 ms "errors"). Phase 1B validates only at
**chord change-points** with contrasting pitch-class sets, finding the audio
change-point that best separates chord\_before from chord\_after in a local
window — a genuinely DTW-independent reference. That fix is real; the alignment
under test is what fails.

## Go / No-Go

**No-Go on frame-level chord-template↔full-mix-chroma DTW.** The premise (a
synthetic template discriminates alignment on full-mix chroma) is falsified.
Scaling this to 20 songs would produce a benchmark whose "GT" is off by ~1.5 s —
worse than useless (it would mislabel model hits as misses at chord boundaries).

## Fallback proposal (ranked)

1. **Manual downbeat anchors + piecewise-linear (highest confidence, ~15
   min/song).** Hand-mark 3–4 section boundaries per song (head-in, A/B, top of
   solo) in Audacity; interpolate chord times linearly between anchors. The iReal
   chart already gives bar structure; only the anchors are needed. For a 20-song
   benchmark this is ~5 h of annotation and gives ±100–200 ms with **no**
   representation-SNR dependence. This is the safe path to a real benchmark.
2. **Chord-recognition-posterior DTW.** Replace the binary template's audio side
   with a *trained* per-frame chord/PCP posterior (e.g. a chroma→chord logistic
   model) instead of raw CQT chroma. This is what MIREX sync tools do; it lifts
   SNR far above the ~0.03 gap seen here. But it needs a trained front-end and is
   arguably circular unless the recogniser is fully independent of the pipeline
   under test.
3. **Beat-synchronous chroma DTW.** Median-pool chroma within detected beats
   before DTW (Ellis/Müller cover-song style). Raises SNR by averaging out
   transients; local beat errors don't need a correct global tempo. Reintroduces
   beat tracking but only *locally*, which is more forgiving than the global
   beat-grid that failed in Phase 1.
4. **Curate the benchmark inputs first, regardless of method:** re-fetch clean
   single-take recordings, trim to the head, and **verify each audio's key
   matches its chart** (the Airegin +2 check above) before any alignment. A
   transposed or 17-min compilation input defeats every method.

**Recommendation:** go with fallback (1) manual anchors for the 20-song
benchmark now (unblocks Missions 2–4 with trustworthy GT), and keep (2)
posterior-DTW as the eventual automation once an independent recogniser exists.

## Artifacts

- `scripts/mission_1_build_benchmark.py` — DTW functions + `--validate-pilots`.
- `docs/mission_1_phase1b_dtw_results.json` — per-boundary error detail.
