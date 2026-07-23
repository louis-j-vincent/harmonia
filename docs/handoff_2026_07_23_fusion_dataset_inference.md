# Handoff — Fusion Aligner + Dataset + Inference lane (2026-07-23)

**For the next agent picking up this lane.** This lane ran concurrently with the
serving-refactor lane on `main` and never collided (disjoint files, never `git add -A`).
Full design + blow-by-blow reasoning: `docs/fusion_aligner_design.md` (checkpoint log at the
bottom, morning summary at the top of its AUTONOMOUS RUN section). This is the orientation map.

---

## TL;DR

We turned the pile-of-thresholds aligner into a principled **Bayesian bar-pointer fusion DBN**,
built a **high-precision chord-training dataset** on top of it, **productionized** the DBN as a
refactor-compatible brick, and started the **chord-inference** model (the DBN with chords latent).

- **Benchmark**: 7/8 songs frozen (`golden/brick0/*.gt.json`, `verified=true`). Only **Autumn**
  unfrozen (its solos are acoustically unresolvable — see below).
- **Fusion DBN aligner** (`harmonia/align/fusion.py`): reproduces the frozen benchmark
  byte-identically, plus reliability-weighted fusion + forward-backward **posterior confidence**
  + form-anticipation downbeat + drift/vamp states.
- **Dataset** (`harmonia/dataset/`): 3-way gate (clean GT / substitution-review / drop),
  **~95.5% strict precision** on frozen, **461 clean (segment→chord) pairs** + substitution queue.
- **Productionized** (`harmonia/align/chart_aligner.py`): `FusionChartAligner` — the refactor's
  ABC+dataclass convention; lossless adapter to the serving payload.
- **Inference** (`harmonia/align/inference.py`, `c569906`): the DBN with chords LATENT
  (`FusionChordDecoder`). First baseline vs the frozen GT (chords latent, NO chart, pooled over the
  7 songs): **root 0.629 · maj-min 0.571 · partial 0.437 · sounding-bass 0.632 · 7ths 0.212 ·
  strict 0.174**. The DBN prior beats per-beat argmax (+0.09 root, +0.13 maj-min, +0.11 partial) and
  is ~2× the always-tonic floor. Honest first version — an architecture + a number to improve from.

---

## The arc — what was built and why (the reasoning trail)

1. **The wall.** The old aligner stacked single instruments (harmonic agreement, offset-ramp,
   windowed drift, form-vamp) behind hand-tuned thresholds. It broke on solos (Autumn "cacahuète":
   comping drops → chroma agreement is noise) and dense mixes (Let It Be phantom gap) — a
   metric-up/ear-down fragility.
2. **The fix (Louis): fuse complementary signals, each weighted by local reliability**, in one
   Bayesian model. Premise-checked every stream cheaply BEFORE building (rule #2) — two dead ends
   (drum-timbre downbeat, spectral-signature downbeat) were killed by cheap checks, saving builds.
   Findings (all in the design doc): **drums carry the beat through solos but NOT the downbeat**
   (backbeat 2-beat symmetry); the **downbeat is a global-phase argmax** over the beat grid
   (Louis's simplification), form-refined for anticipation; **bass** recovers the sounding-bass
   root on pop/soul, self-downweights on walking jazz bass.
3. **Fusion DBN** (Stage 2 + 2b): bar-pointer state-space; observation = reliability-weighted
   {harmony, drum-beat, bass, harmonic-rhythm}; Viterbi MAP + forward-backward confidence; drift-τ
   + vamp states. Reproduces the benchmark; **Autumn 0.35→1.00** (vamps) while honestly flagging
   its solos; **let_it_be NOT forced** (bad golden — the v2→v3 overfit lesson respected).
4. **Dataset**: the alignment confidence gates a training corpus. Substitutions (chart≠recording,
   e.g. Georgia's F#dim played as B7) are a FEATURE — routed to a review queue for a future
   ear-training active-learning loop, not mislabeled.
5. **Productionization**: `FusionChartAligner` slots into the refactor via a thin adapter; the
   dataset harvest now runs on the productionized DBN (precision held, +12 pairs).
6. **Inference**: the same DBN with chord identity latent = the chord-recognition product.

**Deep reason it all connects:** alignment = the DBN with chords OBSERVED; inference = the SAME
DBN with chords LATENT. The alignment instruments ARE the inference pipeline.

---

## Architecture + how it inserts into the refactor (the hybrid)

The refactor lane (`harmonia/{core,stages,pipeline}`) is a typed-contract skeleton: one stage =
one module, pluggable via ABC + factory, dataclass I/O at every seam, `PipelineConfig`. Its
`stages/*` are mostly `NotImplementedError` stubs (greenfield). **The chart-to-audio aligner has
no stage yet** — our `FusionChartAligner` is its drop-in upgrade.

Insertion is done ENTIRELY in our lane (zero edits to theirs): `harmonia/align/chart_aligner.py`
exposes a `ChartAligner` ABC + `FusionChartAligner` + `ChartAlignment`/`SectionMarker` dataclasses
matching their convention, with a lossless adapter to the section-marker payload their
`serving/loaders._load_ireal_alignment` + server route already consume.

---

## What's next (ranked; try-order + stop criteria)

1. **[COORDINATE] Serving route swap** — the ONE change in THEIR files: swap the single
   `align_tune_sections_to_audio` call (~`scripts/harmonia_server.py:2925`) for
   `FusionChartAligner().align(...)`, behind a kill-switch (A/B both aligners' markers). Deferred to
   lane-convergence; do NOT do it unattended — it's a `❓ QUESTION FOR LOUIS`.
2. **Inference improvements** — the decoder is a first baseline (root 0.63 / partial 0.44). Levers,
   in order: (a) **a no-chord (N) state** — the stated top project priority (`feedback_simplicity_
   principle`); costs nothing on the fully-chorded benchmark but is needed in the wild; (b)
   **modulation-aware key** — it's currently a single global Krumhansl-Schmuckler estimate, the
   weakest structural link; (c) a **learned emission** — 7ths/strict are low because a centre-normed
   chroma template can't separate maj/maj7/6 (Bayesian-first, ML-later: the fused streams become its
   features once the frozen benchmark supplies training data).
3. **Frontend decoupling** — `fusion.py` imports `scripts/brick0_propose.py` via importlib (25
   entangled symbols). Promote brick0's non-circular front-end into a shared module both import —
   only once brick0 is unfrozen (rule #6: it's the golden byte-reference; don't fork it live).
4. **Autumn** — the honest exception: the DBN aligns its structure + flags its swing solos as
   low-confidence (can't be resolved acoustically). Louis's call to freeze "aligned+flagged" or
   leave it out. Needs the fusion/ear, not more code.
5. **Grow the dataset** — `harmonia.dataset.ingest.add_song(chart, youtube)` (yt-dlp + irealb).
   Disk-gated (see below).

---

## Concurrency boundary — DO NOT BREAK

- **This lane owns**: `harmonia/align/*`, `harmonia/dataset/*`, `scripts/brick0_propose.py`,
  `golden/brick0/*`, `docs/fusion_aligner_design.md`, `docs/dataset_harvest_design.md`,
  `docs/brick0_review/*`, and its own tests.
- **Never touch (refactor/grid lanes own)**: `harmonia/stages/*`, `harmonia/serving/*`,
  `harmonia/core/*`, `harmonia_server.py`, `harmonia/output/*`, `harmonia/models/*`,
  `harmonia/theory/*`, `harmonia/eval/parity.py`, `harmonia/irealb_fetcher.py`,
  `golden/frozen_parity/*`. **`docs/known_issues.md` is the refactor lane's checkpoint — stay off
  it; log here instead.**
- **NEVER `git add -A`** — stage explicit files only. This is what lets the lanes share `main`.
  Transient `index.lock` on concurrent commits → wait 1–2 s, retry.

---

## Ear-review status (nothing frozen without Louis's ear)

7/8 frozen: Stand By Me, Bein' Green, Blue Bossa backing, Every Breath (downbeat phase 1
confirmed), Blue Bossa, Georgia (out-head B-A + B7/split overrides + truncation), Close To You
(mirror head-fix). **Autumn unfrozen** — the acoustic exception.

---

## Disk (a recurring hazard on this machine — real disk-full incident in project history)

System volume runs ~98–100% full. It dropped to **1.4 GiB** under two concurrent lanes; Louis
freed dataset space back to ~4 GiB. **Guard every heavy run** (`df`; stop < 1.5 GiB), reuse warm
caches, never re-download in bulk. Our lane's safe regenerable footprint is small (~0.5 GiB:
system scratchpad + `docs/brick0_review/*` embedded-audio pages + `scratchpad/_drift_work` decoded
WAVs). The real space is datasets (`data/accomp_db` 2.2 GiB) — Louis's call, not ours.
`data/cache` is NOT a pure cache — it holds trained weights (`local_key_*.pt`) + the refactor
lane's live parity corpus (`aligned_corpus`, `_parity_wav_tmp`). Only the 16-hex-char `*.npz`
(PitchExtractor cache) is safely clearable.
