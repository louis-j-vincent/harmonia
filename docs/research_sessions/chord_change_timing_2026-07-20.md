# Chord-CHANGE-timing detection precision — 2026-07-20

Budget 3h. User diagnosis (confirmed by ear + grid-align signed-offset histograms):
grid/tempo detection is solid (std 173ms on bar-first chord onsets vs raw beats);
boundary PLACEMENT is the noisy part. Goal: sharpen chord-change detection.

## Literature (orchestrator lit review)
- Harte & Sandler 2006 HCDF: 6-D tonal-centroid projection before frame-to-frame
  L2. Their own change-detection F ≈ 64.9% — known-hard, not an impl bug.
- Korzeniowski & Widmer 2016 DeepChroma (in madmom): learned, less noisy chroma.
- HPSS (librosa) — remove percussive broadband before chroma.
- APSIPA 2025 source-sep preprocessing — DEFERRED (licensing-flagged, next round).

## Build order (premise-check cheap first, CLAUDE.md rule #2)
1. HCDF TCS projection vs current raw-chroma L2 flux (`_chroma_flux`).
2. HPSS pre-filter before chroma.
3. madmom DeepChromaProcessor as chroma source.
4. Combine winners; re-measure change-F1 vs music-x-lab change times.

## Harness
Reference = music-x-lab chord-change times (`musx_labels` segment boundaries), the
same reference used in the Mayer/Henny work. Predicted = adaptive peak-pick on the
novelty curve. F1 by greedy one-to-one match within ±150ms (Harte-comparable) and a
looser tolerance. SAME peak-picker for every curve — only the novelty differs.

## Results — change-F1 vs music-x-lab change times (matched set, 9 songs, threshold-swept best-F1)

### Lever 1 — HCDF tonal-centroid vs raw-chroma L2 (on our NNLS-24 chroma)
Best single config = `hcdf-treble` + Gaussian smoothing σ=4. Mean best-F1@150ms:
**raw-L2 0.122 → hcdf-treb-sm4 0.186 (+52% rel)**; wins 7/9 songs. Real but modest —
the TCS projection helps because within-chord voicing moves little in tonal-centroid
space, but our NNLS chroma is still frame-noisy so precision stays ~0.1–0.27.

### Lever 2 — HPSS-harmonic pre-filter (librosa) before NNLS chroma
MIXED: Let It Be 0.317→0.390, Billie Jean 0.066→0.096, but Henny 0.178→0.136. Not a
clean universal win, and dominated by Lever 3. Not pursued.

### Lever 3 — madmom DeepChromaProcessor (Korzeniowski & Widmer 2016) — BIG WIN
Learned deep-chroma extractor as the novelty SOURCE (HCDF-TCS on top). Mean best-F1:
**@150ms 0.122 → 0.442 (3.6×); @250ms 0.275 → 0.580 (2.1×)**. Wins on ALL 9 songs.
Per-song @150ms: justaint 0.73, abba 0.57, Let It Be 0.55, commodores 0.55, Henny 0.51,
Stand By Me 0.43, Bein' Green 0.44, Billie Jean 0.17, aretha 0.02 (only 5 changes).
Precision jumps to 0.16–0.71. **On clean DeepChroma, HCDF-TCS ≈ raw-L2 (0.552 vs 0.538
Let It Be) — the win is chroma QUALITY, not the projection.** Roughly reproduces the
Harte-Sandler ~65% regime on the easy songs; the sparse vamps (Billie Jean 6s gaps,
aretha 1-chord) stay hard for everyone, as the literature predicts.

**madmom caveat**: broken on Py3.12 by default (`from collections import MutableSequence`,
`np.int`) — needs a small compat shim (collections.abc + numpy alias) at import. Adds a NN
forward pass (~seconds) per analyze. This is a real productionization cost → any wiring is
opt-in pending a madmom-py312 fix.

Harness: `scratchpad/change_f1.py`, `deepchroma_corpus.py`, `lever23.py`.

## Downstream-impact finding (the honest wiring result)
The pipeline's ONLY flux consumer is `_flux_downbeat_phase` → a single downbeat phase
φ∈{0,1,2,3} (+ bar pooling on that grid). Measured φ across novelties (matched set):
- **DeepChroma agrees with the current raw flux on 5/6 songs** (letitbe/henny/billiejean/
  commodores/justaint), differing only on abba (2→3), with UNIFORMLY SHARPER combs
  (commodores 1.19→1.55, abba 1.07→1.22). → the current raw flux already recovers the
  correct phase; DeepChroma is more confident but yields the SAME grid.
- HCDF changes φ on 3/6 (letitbe/billiejean/justaint) — LESS stable → unsafe to default.

**Implication**: the 3.6× change-F1 win does NOT change the current pipeline output,
because the pipeline consumes only the downbeat PHASE (already correct), not the change
TIMES. The user's 173ms boundary-PLACEMENT noise originates in the SEGMENTATION stage
(`_root_change_segs` / musx segment times), which the flux never touches. The real
payoff of the sharp DeepChroma novelty is a boundary-placement REFINER that snaps chord
onsets to DeepChroma change peaks (±1 beat) — a new segmentation consumer, next round,
gated on a madmom-py312 fix. This round ships the measurement + a kill-switched,
default-OFF DeepChroma/HCDF flux selector so that consumer can be built on a validated
front-end. Wiring to the flux alone is (correctly) a near-no-op and left default-raw.
