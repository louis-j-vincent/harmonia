# Mission 4 — Auto section-merge pooling: READY TO WIRE

**Status (2026-07-13): prep complete, gated on Mission 1 benchmark.**
Detection + eval scripts built and smoke-tested; the in-pipeline auto-fire is
deliberately NOT switched on until Mission 1's real-audio benchmark validates
the lift. Blind averaging was Gen-1 Candidate C and it hurt — this fires only on
a confidently-detected repeat.

## What issue #28 established (the premise, already measured)
Pooling Basic-Pitch evidence (feat48) across repeats of the same chord within a
song lifts the quality head 43.8 → 53.8% (+10pp) on **real** audio, growing with
rep count (≥5 reps: +9.8pp). The pooling mechanism exists
(`user_constraints.SectionMerge` → `pool_beat_evidence`, wired into
`infer_chords_v1(user_constraints={"merges": ...})`) but only fires on a **user**
assertion. Mission 4 fires it automatically where a repeat is confidently
detected.

## Deliverables
- **`scripts/detect_auto_merges.py`** — repeat detection + gating. Pure over a
  `ChordChart` (+ optional audio), so the scoring/gating logic unit-tests in
  isolation (`--self-test` passes: fires an A-A repeat; blocks on structural
  mismatch; fail-safe to no-fire without acoustic evidence; yields to a
  user-asserted merge).
- **`scripts/eval_auto_merge.py`** — before/after majmin + 7ths measurement.
  Default path loads the Mission-1 benchmark and **exits 2 with guidance if it
  isn't built yet** (the gate). `--synth-fallback` runs the identical
  detect→merge→score loop on MMA-rendered jazz1460 to exercise the plumbing now.

## Gating logic (a merge fires only if ALL hold)
1. Two sections share a label (same-section by the SSM labeller,
   `section_structure.label_sections`).
2. Equal musical length (equal `n_bars`) — `pool_beat_evidence` rejects unequal
   beat counts, so this is required for the merge to take effect at all.
3. **Structural confidence** > threshold — the two spans' decoded (root, q5)
   sequences agree slot-for-slot (relative-offset sampling, tempo-robust).
4. **Acoustic agreement** > threshold — the two spans' per-slot mean chroma is
   highly correlated (Pearson-style cosine, the DTW aligner's geometry, which
   cancels the full-mix DC floor).
5. Neither span overlaps a **user-asserted** merge (user assertion is stronger).
Default threshold 0.75 for both. No acoustic evidence ⇒ acoustic conf 0 ⇒ never
fires (fail-safe).

## Smoke test (synth, plumbing only — NOT the headline)
`--synth-fallback --start 20 --n 8 --limit 3`: 1/3 songs fired a merge; that song
gained 7ths +4.0% (aggregate +1.7%), majmin/root flat, **0 regressions**. Small,
as expected — synth base is already near-clean; issue #28's +10pp is a
real-audio, noisy-evidence effect. This run only proves the detect→pool→score
wiring is correct.

## To run once Mission 1 is ready (try-order + stopping criterion)
1. `.venv/bin/python scripts/eval_auto_merge.py` (default 0.75/0.75). Read the
   per-song table: `n_fired`, Δmajmin, Δ7ths, and the regressions list.
2. **Stopping criterion — ship the in-pipeline auto-fire iff**: aggregate Δ7ths
   ≥ +5pp AND zero (or explainable) per-song regressions on the 20 songs. That
   matches issue #28's floor and the CLAUDE.md "+5..+10pp" claim.
3. If Δ is positive but < +5pp, sweep thresholds
   (`--struct-threshold`/`--acoustic-threshold` at 0.70 / 0.80) — looser fires
   more merges (more denoising, more risk), tighter fires fewer (safer). Pick the
   knee where regressions stay at 0.
4. If any song regresses, inspect its fired candidates
   (`detect_auto_merges.py CHART.json --audio SONG.wav` prints struct/acoustic
   per pair) — a regression means a false-positive repeat cleared both thresholds;
   raise the offending modality's threshold rather than disabling the feature.

## The remaining integration step (deferred by design)
The scripts detect merges from a first (unconstrained) `infer_chords_v1` pass and
feed them back via `user_constraints` on a second pass — a clean, two-pass
external wiring that needs no pipeline surgery. Moving auto-detection **inside**
`infer_chords_v1` (so a single call auto-pools) is the last step; hold it until
step 2's criterion passes, then place it right after section detection
(`chord_pipeline_v1.py` §10b) feeding the existing `pool_beat_evidence` at §
beat-evidence pooling — note that requires a preliminary decode for the SSM, i.e.
the same two-pass structure, just internalised.
