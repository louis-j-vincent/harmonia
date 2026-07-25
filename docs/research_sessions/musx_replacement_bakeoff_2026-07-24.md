# music-x-lab replacement bake-off — front-end labeler swap (2026-07-24)

**Goal.** Find/test a replacement for music-x-lab (musx) as the chord root/quality
front-end, since the clean isolation showed **81% of the ±P4/P5 fifth-bleed is
INSIDE musx** (25% boundary + 56% genuine label, dominated by min→dom). A
genuinely different labeler is the real lever.

**Method.** Oracle-boundary LABELER comparison: for each hand-verified GT chord
span in `golden/brick0/<song>.gt.json`, sample the model's label at the span
CENTER, extract root pc + maj/min, duration-weight vs GT. Boundaries are GT's
(identical for every model) so this isolates the labeler from segmentation.
Frozen-7 = {blue_bossa, blue_bossa_backing, bein_green, every_breath_you_take,
georgia_on_my_mind, close_to_you, stand_by_me}, 1649 s pooled.

## musx reference (DISK-FREE, from cached `.lab`) — the number to beat

| metric | at GT centers (labeler) | end-to-end (pipeline seg) |
|---|---|---|
| root | **0.806** | 0.7367 |
| maj/min | **0.784** | 0.7102 |

Fifth-bleed AT GT centers (pure musx label error, seg removed):
**55 spans / 90.7 s** of ±P4/P5 root error (corroborates the isolation doc's
"56% genuine musx label = 90.0 s" exactly). Of that, **min→dom (GT minor →
musx dom7 a 4th/5th away) = 22 spans / 32.8 s** — the target sub-error.
Almost all bleed is in the two hardest jazz tunes: blue_bossa (36 spans/48.3 s,
20 min→dom) and bein_green (8/20.1 s). backing-track/pop tunes are near-clean.

## RESULTS (disk freed to 6.3 GiB — all 4 variants ran, one song at a time, ~0 temp)

**Primary metric = boundary ALIGNMENT** (recall of chord-change times, ±τ s,
mir_eval optimal 1-1 match). Harness validated: musx pooled recall@0.5 = **0.697**
== the given 0.698. musx UNDER-segments (over-seg 0.87, 632 boundaries vs 726 GT).

| model | recall@.25 | recall@.5 | Δrec@.5 vs musx | prec@.5 | F1@.5 | over-seg | root@ctr | min→dom fixed |
|---|---|---|---|---|---|---|---|---|
| **music-x-lab (ref)** | 0.489 | **0.697** | — | 0.797 | 0.738 | 0.87 | **0.806** | — |
| madmom DeepChroma+CRF | 0.375 | 0.640 | **−5.7** | 0.695 | 0.664 | 0.92 | 0.689 | 4/22 |
| madmom CNN+CRF | 0.428 | 0.723 | **+2.6** | 0.616 | 0.663 | 1.18 | 0.712 | 2/22 |
| **BTC maj/min (25)** | 0.627 | **0.850** | **+15.3** | 0.568 | 0.678 | 1.50 | 0.783 | 8/22 |
| **BTC large-voca (170)** | 0.636 | **0.871** | **+17.4** | 0.457 | 0.596 | 1.92 | 0.768 | **9/22** |

**Winner on alignment: BTC (both heads).** +15–17 pp recall@0.5 over musx, and
still +13.8 pp at the *tight* ±0.25 s (0.627 vs 0.489) — so the gain is genuine
alignment, not just boundary density. **Mirage check: recall gain is UNIFORM —
positive on ALL 7 songs** for both BTC heads (Δ range +0.02..+0.33), largest on
the jazz tunes blue_bossa (+0.31/+0.33) and georgia (+0.13/+0.19) — exactly
where musx's fifth-bleed lives. madmom CNN's +2.6 pp is a near-mirage (mixed:
4 songs up, 3 down). madmom DeepChroma loses outright.

**The tradeoff:** every recall-winner OVER-segments (musx under-segments). No
replacement beats musx on F1@0.5 (musx's precision). At ±1.0 s, BTC maj/min
F1=0.739 ties musx's ±0.5 s F1. For an ALIGNMENT-FIRST pipeline (fusion-aligner:
boundaries as candidates, prune later) high recall is the right side — a missed
boundary is unrecoverable, a spurious one is prunable (Occam post-pass exists).

**Label (secondary):** no replacement beats musx root 0.806 (BTC maj/min closest,
−2.3 pp) — the old "BTC lost to musx on labels" (2026-07-17) still holds on the
LABEL axis. But BTC repairs ~40 % of musx's min→dom substitutions (9/22 voca,
8/22 mm) — all concentrated in blue_bossa (20 of the 22 spans live there; BTC
fixes 8–9 of those 20; bein_green's 2 unfixed by anyone).

## Candidate status

| candidate | availability | run status | note |
|---|---|---|---|
| **madmom** DeepChroma+CRF / CNN+CRF | installed (py3.12 shim needed) | **READY, disk-blocked** | different arch (learned DeepChroma/CNN + CRF); may not share musx's min→dom |
| **BTC-ISMIR19** (voca 170 + majmin 25) | cloned+patched (68 MB) | **READY, disk-blocked** | bi-transformer; lost to musx on RWC 2026-07-17, never tested on frozen-7 |
| autochord | pip, but needs system **Vamp/Chordino** plugins + gdown model | not feasible now | Vamp is a system .so dep, not pip; skip |
| essentia | pip base (~40 MB) | low value + install-gated | base `ChordsDetection` is HPCP-template maj/min (≈nnls24); deep models need essentia-tensorflow (large) |
| ChordFormer 2025 (arXiv 2502.11840) | **no official weights** | not feasible now | only unrelated `cameron-cs/chordformer` exists; don't chase |

**madmom py3.12 shim** (in `scratchpad/repl_madmom_shim.py`): restore
`collections.MutableSequence` etc. from `collections.abc`, and
`np.float/int/bool/object` → builtins. Both chord chains then import and run;
smoke-tested end-to-end on synthetic C:maj/A:min (correct).

**BTC patches:** numpy-attr shim (as above) covers `np.float/int/bool`;
`yaml.load`→`yaml.safe_load`; `torch.load(..., weights_only=False)`. Both
weight files load; forward pass verified (no audio).

## DISK (resolved)

Disk was 1.0–1.1 GiB (below the 1.5 floor) for the first pass; Louis freed it to
6.3 GiB and all 4 variants then ran, one song at a time, df ≥ 6.2 GiB throughout.
madmom/BTC stream-decode via ffmpeg/librosa (no temp WAV — df was flat across all
runs), footprint ~0. The self-gate held per song.

### Re-run commands
```
cd <repo>
python scratchpad/repl_run_model.py madmom_deepchroma   # 24-class maj/min root
python scratchpad/repl_run_model.py madmom_cnncrf        # CNN feature + CRF
python scratchpad/repl_run_model.py btc_voca             # BTC 170-class large-voca
python scratchpad/repl_run_model.py btc_majmin           # BTC 25-class maj/min
```
Each self-gates on df ≥ 1.5 per song, scores root/maj-min at GT centers, and
reports `fifth_fixed`/`mindom_fixed` vs the 55/22 musx bleed spans. Writes
`scratchpad/repl_result_<model>.json`. Expected footprint ~0 disk; runtime a few
min/song on CPU.

## Integration hook (if a replacement wins)
`harmonia/models/chord_pipeline_v1.py::_label_segments`, line ~3426–3428:
```python
mx_root, mx_sev = musx_seg_rq[i]
if mx_root >= 0 and mx_sev is not None:
    root, sev_h = mx_root, mx_sev          # <- swap source here, or ensemble
```
Swap: replace `musx_seg_rq` with the winner's per-segment (root, sev). Ensemble:
keep where musx==repl agree; arbitrate disagreements by the stronger per-family
(musx wins maj/dom, repl wins min if it fixes min→dom).

## New files (nothing staged, no commit)
- `scratchpad/repl_madmom_shim.py` — madmom py3.12 compat shim
- `scratchpad/repl_scorelib.py` — GT loader + GT-center scorer + fifth-bleed span finder
- `scratchpad/repl_musx_reference.json` — musx reference numbers (this doc)
- `scratchpad/repl_madmom_smoke.py` — madmom end-to-end smoke (synthetic)
- `scratchpad/repl_btc_infer.py` — BTC wrapper (+ smoke)
- `scratchpad/repl_btc/` — BTC-ISMIR19 clone (weights in-repo)
- `scratchpad/repl_run_model.py` — unified decode-gated head-to-head runner
