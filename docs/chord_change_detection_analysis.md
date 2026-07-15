# Chord-change / boundary detection — analysis + benchmark, 2026-07-15

Mission: research how Chord ai detects chord transitions, audit Harmonia's own
boundary detection, quantify the gap, and try to beat it. This doc reports a
fresh, disk-safe benchmark (no extraction) plus the synthesis. It reconciles
with the two prior investigations that already touched this problem:
`docs/chord_change_engine_2026-07-06.md` and `docs/chord_ai_reverse_engineering.md`
(both authoritative; do not re-derive without reading them).

## TL;DR

1. **Chord ai does not use a dedicated "boundary detector."** The literature it
   belongs to (McFee/Bello → BTC → ChordFormer) does per-frame chord
   classification and lets a CRF / Neural-Semi-CRF / self-attention layer place
   the change *implicitly* where the frame label flips. There is no separate
   "is_boundary?" head. The mission's Approach-A premise ("Chord ai has a learned
   boundary head") is not supported by the source class.
2. **Harmonia's SSM novelty (`structure.py`) is near-useless as a *chord*-change
   detector** — AUC 0.557, exact-beat F1 0.45 on jazz1460. But that is a
   category error: `structure.py` is a *section* segmenter (kernel_size=8, feeds
   key inference), not a chord-change detector. It was never meant to place chord
   boundaries, and it shouldn't be judged as if it were.
3. **A learned per-beat detector nearly doubles exact-beat F1 (0.45 → 0.78) and
   reaches ±1-beat F1 0.86** — confirming "learned > hand-built" for the boundary
   task itself. **But none reach the F1>0.90 target**, and the project's own
   oracle-boundary experiment (2026-07-06 §4) already showed that *perfect*
   boundaries do **not** improve end-to-end chord accuracy on this data —
   **labeling, not boundary placement, is the bottleneck.** So the learned
   detector is a real win on the *sub-task* with a predicted ~0 end-to-end
   payoff. Integrate only if a future experiment overturns the oracle result.

## Part 1 — How Chord ai detects transitions (reconstruction)

Chord ai is closed-source; this is the system *class* (see
`chord_ai_reverse_engineering.md` for the full trust-order caveat). Regarding
transitions specifically, across the open literature it is built on:

- **No "chord boundary" head exists.** McFee & Bello (ISMIR 2017), BTC (2019),
  ChordFormer (2025) all emit *per-frame* chord components (root / bass /
  quality bitmap / 7-9-11-13). A boundary is wherever consecutive frame labels
  differ — an *output* of frame classification, not a separately-detected event.
- **Temporal smoothing does the boundary work.** autochord uses a Bi-LSTM-**CRF**;
  ChordFormer uses a **Neural Semi-CRF** whose explicit segment durations give
  "precise chord-interval boundaries"; BTC/ChordFormer use self-attention for
  long-range context. These suppress 1-frame flicker (the false-positive
  problem) as a side effect of decoding the most likely *label path*, not by
  thresholding a novelty curve.
- **No spectral-novelty / chroma-energy-drop / silence detector** is used for
  boundaries in this class of model. That family (Foote novelty, SSM
  checkerboard) is the *structure-segmentation* lineage (sections), which is a
  different task — the same distinction Harmonia already draws internally.
- **False-positive control** is therefore the CRF/Semi-CRF transition cost +
  duration model, i.e. learned, not a hand-tuned min-gap rule.

**Implication for Harmonia:** the "right" architecture for chord-change timing
is *not* a bolted-on boundary detector; it is a strong per-frame/per-beat
emission + a duration-aware decoder. Harmonia already shipped exactly that
shape: the per-beat semi-Markov (explicit-duration) decoder (known_issues #27,
Mission 2). That is the correct home for boundary placement, and it matches the
Semi-CRF the SOTA uses.

## Part 2 — Audit of Harmonia's current boundary handling

Two distinct mechanisms exist; the mission brief conflated them:

| mechanism | file | job | detects chord changes? |
|-----------|------|-----|------------------------|
| SSM checkerboard novelty | `harmonia/models/structure.py` | **section** boundaries → key inference | no (not its job) |
| block cosine-distance cut | `scripts/chord_change_engine.py` | chord changes (2-beat grid) | yes, ±1 F1 ≈ 0.89 |
| semi-Markov explicit-duration | `harmonia/models/semi_markov_decode.py` | chord segmentation in production | yes (shipped, #27) |

Prior findings that already stand (do not re-litigate):
- `structure_repetition_ssm.py`, 40 songs: SSM section-boundary F vs GT = **0.29**
  (raw), 0.25 (diagonally enhanced) — barely above chance. Jazz AABA sections
  share key/vocabulary, so harmony-only SSM can't separate them. Section
  detection was **closed** as neither necessary (dropping it costs 0.5 majmin)
  nor achievable from harmony (2026-07-06 §7).
- Chord-change engine: fixed 2-beat merge + cosine cut → **change-F ±1 ≈ 0.89**
  (P 0.91 / R 0.88); exact-beat ~0.50 is a hard ceiling set by Basic-Pitch onset
  smear (2026-07-06 §3–4, §10).
- **Oracle-boundary diagnostic (the decider):** feeding *GT* change beats
  (F=1.00) raised chord accuracy by ~0 (root 55.2 vs 60.5 coarse; oracle 89.1 vs
  coarse 85.6 after the harness fix). Perfect segmentation does not fix labeling.

## Part 3–5 — New benchmark: learned vs hand-built, quantified

Repro: `scratchpad/train_boundary_detector.py` (uses the cached
`data/cache/bakeoff_jazz_perbeat_s1.pkl`; no audio extraction). 70 jazz1460
songs, 10,412 beats, **2,970 GT chord changes (28.5% base rate)**. GT boundary =
per-beat GT root changes. Evaluated at **beat resolution** (GT is per-beat, so
exact-beat is the honest hard number; ±1 beat is the MIREX-style tolerant
number). Learned models are out-of-fold (GroupKFold-5 by song, no song leak).

| detector | type | P | R | **F1 (exact)** | AUC | **F1 (±1)** |
|----------|------|---|---|----------------|-----|-------------|
| ssm-novelty (`structure.py`) | hand-built | 0.29 | 0.99 | **0.450** | 0.557 | 0.453 |
| adj-cosine (engine signal) | hand-built | 0.32 | 0.85 | **0.459** | 0.584 | 0.536 |
| fixed-2-beat grid | hand-built | 0.57 | 0.98 | **0.717** | — | 0.718 |
| logreg [f_t, f_{t-1}, |Δ|] | learned | 0.70 | 0.77 | **0.736** | 0.897 | 0.812 |
| **mlp (64,32)** | **learned** | 0.76 | 0.81 | **0.783** | **0.921** | **0.858** |

Plot: `docs/plots/boundary_detection_evaluation.png` (F1 bars + PR curves).

Findings:
- **Learned nearly doubles the SSM's exact-beat F1** (0.45 → 0.78) and lifts AUC
  0.56 → 0.92. Mission hypothesis "learned > hand-built" **confirmed** on the
  boundary sub-task.
- **The naive "cut every 2 beats" grid (F1 0.717) beats both hand-built novelty
  detectors.** jazz1460's harmonic rhythm has a strong 2-beat mode, so a fixed
  grid out-predicts a chroma-novelty curve. This corroborates the 2026-07-06
  decision to use a fixed 2-beat grid rather than novelty peak-picking, and is a
  caution against ever wiring `structure.py` novelty into chord-change placement.
- **`structure.py` novelty as a chord-change detector = recall 0.99 / precision
  0.29**: it fires almost everywhere (AUC 0.557 ≈ chance). Again, expected — it's
  a coarse *section* kernel, not a chord kernel. Do not repurpose it for chord
  changes.
- **F1>0.90 is not reached** (best ±1 = 0.858). The residual is the
  Basic-Pitch onset-smear / walking-bass ceiling the engine investigation
  already characterized; exact-beat placement is evidence-limited, not
  tuning-limited.

Caveat (rule #5): this is jazz1460 (MMA renders, walking bass, dense ~2-beat
changes). POP909 (real piano, sparse changes, timing-limited not
labeling-limited) would give different absolute numbers — the handoff already
found POP909's limiter is timing while jazz1460's is labeling. The *ranking*
(learned ≫ SSM ≫ chance) is expected to hold; the absolute F1 will not transfer.

## Part 6 — Integration recommendation

**Do NOT replace `structure.py`'s SSM.** It does section segmentation for key
inference, which is a separate, already-closed problem; it was never the
chord-change detector and swapping it would not touch chord timing.

**Do NOT wire the learned beat-boundary detector into the chord pipeline yet.**
The oracle-boundary experiment (2026-07-06 §4/§10) is the governing evidence:
even F1=1.0 boundaries do not raise end-to-end chord accuracy on this data,
because labeling (root under walking bass; dom-vs-maj/min quality) is the
bottleneck — and that is exactly what the parallel Billboard/bass work
(known_issues #31) is attacking. The learned detector's +0.14 F1 over the fixed
grid is real on the sub-task but has a predicted ~0 downstream payoff.

**The one place it could matter** is the per-beat semi-Markov decoder (#27):
its `build_log_duration` / emission interface is the correct Semi-CRF-shaped
home, matching the SOTA. If a future experiment shows the semi-Markov decode is
*boundary-limited* (not emission-limited) on some corpus, feed this detector's
per-beat P(change) as a boundary prior there — but gate that on an oracle test
first (rule #2: screen the premise cheaply). Absent that, this is a documented
negative-integration result, not a shipped feature.

## Sources / repro
- Benchmark + models: `scratchpad/train_boundary_detector.py` (retrains in <10 s;
  model not persisted — disk at 99%, and integration is not recommended).
- Plot: `docs/plots/boundary_detection_evaluation.png`; raw scores pickle
  `docs/plots/boundary_detection_results.pkl`.
- Prior art (authoritative): `docs/chord_change_engine_2026-07-06.md`,
  `docs/chord_ai_reverse_engineering.md`, known_issues #22 (sections), #27
  (semi-Markov), #31 (labeling bottleneck).
