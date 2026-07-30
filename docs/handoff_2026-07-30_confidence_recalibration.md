# Confidence audit — CLOSED as "do not recalibrate" (2026-07-30)

This file used to be an open task ("re-fit the displayed-confidence map").
The audit ran and the answer is **don't**. Kept as the record of why, and of
two wrong code readings that cost real time.

## What was measured

7 verified Brick-0 songs, shipped config, fold ON, 603 predicted chords,
duration-weighted. Target = P(root + parent family right) — Louis's (b),
partial credit. Full numbers + plot: `docs/known_issues.md` (2026-07-30,
"displayed per-chord confidence"), `scratchpad/confidence_audit.png`.

| | mean displayed | actual | ECE |
|---|---|---|---|
| app today | 0.465 | 0.827 | 0.379 |
| raw, unmapped | 0.625 | 0.827 | 0.265 |
| isotonic refit (LOSO) | 0.829 | 0.827 | 0.095 |
| constant base rate | 0.827 | 0.827 | 0.000 |

## Why "recalibrate" is the wrong fix

**AUC(raw score) = 0.480.** The score does not rank correct chords above wrong
ones — it is flat and slightly inverted (raw 0.2–0.4 → 87% right; raw 0.8–0.9 →
76% right; stand_by_me's AUC is 0.044). A monotone map cannot manufacture
discrimination. The LOSO isotonic fit proves it by collapsing to a constant
~0.836 for every raw value ≥ 0.2: honest, and still meaningless, because a
constant rendered per chord reads as per-chord information.

The target was never wrong. The *score* is the problem.

## What to do instead, in order

1. **Show it at chart level, not per chord** — "≈83% of chords right" is a true
   statement the current score can support. Cheap, honest, ships today.
2. **Find a score that discriminates**, then calibrate that. Untried and
   obvious: music-x-lab's own frame-posterior margin (top − runner-up) or
   entropy over the folded span. Note the earlier objection to margin
   ("uncalibrated") is now void — it would replace something uncalibrated AND
   non-discriminative, and calibration is the easy half.
3. **Drop the per-chord percentage** until 2 exists.

## DEPLOYMENT TRAP (read before touching the npz)

`harmonia/models/nnls24_conf_calibration.npz` has **two** consumers:
- `stages/chord_head.py::_finalize_chords` → the displayed number
- `chord_pipeline_v1.py` L3448 → `bar_conf` for the Occam bar-compression's
  Bayes arbitration, which **changes chord labels**

Overwriting it in place moves live chord decisions. A display-only map must be
a separate file applied at the display emission.

## Two wrong code readings, recorded so they are not re-derived

- `_get_conf_calibrator` / `data/cache/confidence_calibration{,_real}.npz` is the
  BILLBOARD/legacy path. Neither file exists on disk. Harmless — the live nnls24
  path never calls it. (I briefly concluded "there is no map at all". False.)
- `chord_pipeline_v1._finalize_chords` (L3613) and the chord emission at L4724
  are both DEAD under the shipped config: `infer_chords_v1` returns from
  `_infer_nnls24` at L4059 before reaching them. Live chain:
  `infer_chords_v1 → _infer_nnls24 → NNLS24ChordHead.run_full → chord_head._finalize_chords`.
- `root_conf` is `None` for every chord on the live path, so the "fused"
  (conf × root posterior) variant does not exist there.

Repro: `.venv/bin/python scratchpad/refit_conf_shipped.py`
