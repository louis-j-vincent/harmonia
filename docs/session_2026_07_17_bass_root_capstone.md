# Session capstone — NNLS + pYIN bass/root system (2026-07-17)

*One agent's contribution to the multi-agent bass/root push. Every number here comes
from a completed run with the cited repro script; no expected/placeholder values.
Companion detailed entries live in `docs/known_issues.md` (dated 2026-07-17).*

---

## 0. Phase-0 audit — the 0.890 vs 0.379 discrepancy (resolved)

The user caught a prior agent's claim *"NNLS root stable across corpora, Billboard
0.379 → JAAH 0.378"* against the documented **0.890** NNLS-Billboard root. Verdict:
**the two are not comparable measurements** — they differ on all three axes at once.

| number | audio source | feature source | recipe | protocol |
|---|---|---|---|---|
| **0.890** | McGill's OWN | McGill `bothchroma.csv` | full: nonlinear MLP + root-rel rotation + trigram | oracle bnd, 97.7k/884 |
| **0.379** | OUR YouTube re-source | our real-VAMP extraction | bare MLP (no rotation/trigram) | oracle bnd, GroupKFold, 20 songs |

Further, **"JAAH 0.378" is not even NNLS** — the only JAAH corpus in the repo is
`jaah_bp48.npz` (Basic Pitch, root 33.7%), and the agent's cited scripts
(`jaah_nnls_bp48_*.py`) do not exist. Full table + resolution logged as the
"PHASE-0 AUDIT" entry in `docs/known_issues.md`.

---

## 1. Error-structure analysis (what actually fails on RWC BP48)

`scratchpad/bass_error_analysis.py` (read-only on `rwc_bp48.npz`, pooled test preds).
The bass-pc / root / quality errors decompose into concrete, actionable patterns:

- **Bass-pc head, three failure modes:** (1) **short spans** — err 66% on Q1
  duration [0.32–0.98 s] → 35% on Q3, a pooling failure; (2) **fourth/fifth slips** —
  32% of errors are a P4/P5 interval; (3) **concentrated bad songs** — a few songs
  76–95% err (BP48 bass block unusable on specific mixes), median 56%.
- **Root head:** on inversions, errors land on the sounding bass 58.8%; even on
  root-position chords the top errors are P5/P4 (pervasive fifth ambiguity).
- **Quality head:** dom recall 39% → maj (the maj↔dom 3rd-vs-7th confusion).

All three bass modes point to the same lever: **read the lowest sounding pitch
directly** (monophonic tracker / sharp low-register extractor), not a folded chroma.

---

## 2. Literature / pretrained-tool scan (actionable subset)

`docs/literature_review_nnls_bass.md`. Key points for our bottleneck:
- The go-to chroma tools (**autochord, madmom CRF**) are **maj/min 25-class only** —
  don't address bass/inversion. (autochord wraps the real NNLS-Chroma VAMP.)
- Tools that DO output bass/inversion **and ship weights**: **BTC-ISMIR19**
  (voca=True, ~170-class) and **music-x-lab Chord Structure Decomposition**
  (factored root+bass+quality). **ChordFormer** (2025) is the architectural north
  star (explicit bass slot, CQT 36 bins/oct) but no released weights.
- **Source separation is cautionary** (Ko: Demucs stems → full-mix ACR got *worse*,
  artifacts) BUT the narrow **bass-stem → monophonic tracker (pYIN/CREPE)** path is
  strong (CREPE 72% F on FiloBass, +10% over Basic Pitch).

---

## 3. maj/min cascade + a real pretrained tool run

`scratchpad/cascade_analysis.py`, `scratchpad/madmom_cascade.py` (madmom imported via
a py3.12/numpy compat shim; deep-chroma maj/min CRF run on 5 RWC songs, audio
streamed+deleted).

- **RWC is 87.2% maj/min family** (60.1% pure triad); the hard residual
  (dom/sus/dim/aug/hdim) is only **12.8%**.
- **Our model on the maj/min majority:** joint root&quality only **0.390** — the
  bottleneck is ROOT (0.615), not the maj/min distinction.
- **madmom pretrained on the maj/min subset:** root 0.707, maj/min 0.848, **JOINT
  root&majmin 0.677** — substantially beats our 0.39. On the residual madmom is
  structurally 0 (maj/min-only). → **Cascade is justified:** offload the easy 87%
  to a pretrained maj/min tool, let our specialized stack carry the 13% residual.
- **End-to-end cascade, built + evaluated** by the parallel NNLS-recipe agent
  (known_issues.md "Addendum 2"): a soft-hierarchical two-stage combine **beats flat
  NNLS by +8.1pp raw accuracy (0.830 vs 0.749)** but **loses ~7pp balanced accuracy
  (0.587 vs 0.657)** — the flat classifier already implicitly protects rare-class
  recall; the explicit cascade trades that for a raw-accuracy gain on the common-chord
  majority. **Verdict:** cascade wins for common-chord-dominated deployment (the
  play-along chart); flat NNLS stays better for rare jazz-quality coverage.

---

## 4. The pYIN bass lever — and its scale-up revision

`scratchpad/phase2_pyin_bass.py` (5 songs) → `scratchpad/pyin_extract_cache.py`
(robust, incremental, stream-one-song-delete; octave-safe pc, voiced-flag gate,
confidence + fallback flag; 36 RWC songs cached).

- **5-song first look:** pYIN (low-pass 400 Hz + tracker) bass-acc 0.810 all / 0.708
  inv, beating BP48 argmax by +20–35pp and the trained BP48 bass-pc head (0.664).
- **28-song scale-up REVISED the winner** (CLAUDE.md #5): compared against **NNLS
  bass-half argmax**, pYIN is *second*. NNLS-bass is the strongest single estimator
  and it is untrained + free on all 100 songs.

---

## 5. ★ Combined system — end-to-end (the deliverable)

`scratchpad/combined_system_cv.py`, reusing the verified NNLS harness
(`rwc_nnls_multihead_cv.py` / `multihead_training.py`). **5-seed song-grouped CV,
38 pYIN-covered songs, 5669 pooled test chords (460 inversions), deployable
predicted-root setting.** (Reproduces the 28-song run: numbers stable within ±0.02.)

**Front-end = NNLS-24 for root, quality AND bass:**

| output | metric | value | BP48 baseline |
|---|---|---|---|
| Root (NNLS MLP) | acc | 0.763 | 0.616 |
| Quality (NNLS cascade) | raw acc | 0.589 | — |
| Bass (NNLS bass-argmax) | acc all / inv | **0.776 / 0.743** | 0.564 / 0.485 |

**Bass estimator ranking (sounding-bass pc):** NNLS-argmax **0.776** > pYIN 0.751 >
BP48 0.564. Where NNLS-bass & pYIN **agree** (74% of chords) bass-acc = **0.906**
(disagree: 0.407) — agreement is a per-chord confidence gate.

**End-to-end full-chord (root & quality & sounding-bass all correct):**

| system | full-chord acc |
|---|---|
| root & quality only | 0.551 |
| **+ NNLS-bass (BEST)** | **0.497** |
| + BP48-bass (baseline) | 0.354 |

→ **+14.3pp end-to-end** from the NNLS front-end over the BP48 baseline.

**Inversion detection** (inversions 8.1% → precision-hard): NNLS-bass≠root prec 0.160
(BP48-bass 0.109); **ensemble (NNLS & pYIN agree, both ≠ root) prec 0.220 — best to
date**, recall 0.285. Ship bass/inversion as a NEW rendered output ("C/E"), not yet a
root-corrector.

**pYIN robustness:** octave errors moot for pc (mod-12); voiced-flag flags only 3.6%
of spans unreliable, and bass-acc there drops 0.76→0.43 (the flag correctly isolates
hard spans rather than guessing); `voiced_prob` retained as a soft signal.

---

## 6. What to ship / open items

1. **NNLS-24 is the front-end for root + quality + bass** — one extractor, three
   outputs, each beating BP48 (+16pp root, +bass on inversions 0.38→0.77).
2. **pYIN = corroboration + confidence layer** (agreement gate 0.907/0.467; voiced-
   flag fallback), not the primary bass source at corpus scale.
3. **Ensemble inversion detector** (NNLS∩pYIN agreement) is the best inversion
   precision yet (0.249) but still short of a net-positive root-redirect gate.
4. **maj/min cascade** with a pretrained tool (madmom 0.677 joint on the easy 87%)
   is a validated fast-path; our stack specializes on the 13% residual.
5. **Open:** scale pYIN to all 100 songs; run BTC/music-x-lab large-voca models for a
   direct bass comparison.

## 7. Cross-corpus check — GuitarSet (out-of-domain generalization)

`scratchpad/gset/gset_bass_check.py`. GuitarSet (Zenodo 3371780) — **guitar-only,
comp chords, zero alignment risk** (bundled audio+JAMS). **Explicitly an out-of-domain
generalization probe, NOT an RWC replacement.** 12 comp clips / 144 chords, real NNLS
VAMP extraction (same `roll(·,9)`→C-frame, L2-per-half pooling as RWC). Two important
domain caveats: (a) GuitarSet chord labels are coarse comp/lead-sheet chords with **no
inversions**, so this tests only the bass→**root** anchor, not the sounding-bass/
inversion headline; (b) guitar comping voicings frequently do NOT put the root in the
lowest string.

| finding | GuitarSet (guitar) | RWC (pop) |
|---|---|---|
| UNTRAINED NNLS bass-argmax → root | **0.583** | ~0.78 |
| UNTRAINED NNLS treble-argmax → root | 0.347 | — |
| TRAINED NNLS-24 root head (1 split, 2 held clips) | **0.955** | 0.763–0.789 |

**Interpretation (and the session's recurring lesson, again):** the *untrained*
bass-argmax anchor is **domain-sensitive** — it drops 0.78→0.58 on guitar because
comping voicings don't foreground the root in the bass, so the "bass IS a root anchor"
premise partially breaks out-of-domain. **BUT a trained NNLS-24 root head still
decodes root strongly (0.955 on held-out clips)** — the root information is present in
the full 24-dim bass⊕treble vector even where the raw argmax doesn't surface it,
mirroring RWC (trained root ≫ argmax-implied). So: the **NNLS-24 *feature* generalizes;
the untrained-argmax *shortcut* does not.** Ship the trained head, not the argmax
heuristic, for cross-domain root. Small-sample caveat: 12 clips / single split =
high-variance; the 0.955 also benefits from limited per-clip chord vocab — read it as
"root is linearly decodable from NNLS-24 out-of-domain," not a headline accuracy.
Next: scale to more clips + the sounding-bass test needs an inversion-labeled
out-of-domain corpus (GuitarSet can't provide it).

### Repro / artifact index
| script (scratchpad/) | produces |
|---|---|
| `bass_error_analysis.py` | error-structure breakdown |
| `cascade_analysis.py`, `madmom_cascade.py` | maj/min cascade numbers |
| `phase2_pyin_bass.py` | 5-song pYIN first look |
| `pyin_extract_cache.py` → `pyin_bass_cache.npz` | cached pYIN bass (36 songs) |
| `combined_system_cv.py` | ★ combined end-to-end eval |

Docs: `docs/known_issues.md` (PHASE-0 AUDIT, ERROR-STRUCTURE, PHASE 2 — pYIN,
COMBINED SYSTEM — CAPSTONE), `docs/literature_review_nnls_bass.md`.
