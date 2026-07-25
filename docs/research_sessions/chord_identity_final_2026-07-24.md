# Chord-identity investigation — FINAL verdict + wiring handoff (2026-07-24)

Single entry point for a multi-wave autonomous+interactive session on "make the chord
detector run as well as possible." Baseline shipped nnls24 = **root 0.7367** on the 7 frozen
Brick-0 songs (determinism-confirmed, bit-identical across runs).

## ★★ ADDENDUM (later same session — SUPERSEDES the "musx is the ceiling" framing below on the ALIGNMENT axis)
Louis reframed the metric: for a usable chart, the target is **boundary ALIGNMENT with GT (do the chord
changes land at the right time), NOT MIREX label accuracy.** Under that metric the shipped chart is
mediocre — **final-chart boundary F1 ~0.679 / recall ~0.682 (misses ~32% of changes; under-segments/
merges — which IS the fifth-bleed mechanism).** Result of the boundary work:
- **Boundary DETECTION is solvable.** A gated fusion (add BTC/DeepChroma high-recall boundaries ONLY
  where musx under-segments — `scratchpad/align_gated_local.py`) BEATS musx on change-times (F1 0.720).
  BTC-ISMIR19 is a real recall lever: +15pp change-recall vs musx on ALL 7 songs (non-mirage).
- **But a boundary-only swap is NULL end-to-end** — `_coalesce_labeled` erases a recovered boundary
  when musx labels both sides the same. A recovered boundary survives only if the two sides carry
  DIFFERENT labels.
- **THE ALIGNMENT WIN (2nd shippable deliverable): the HYBRID** — put BTC's identity (maj/min BOUNDARIES
  + voca LABELS) inside the gated under-segmented regions, keep musx everywhere else: final-chart
  alignment **F1 0.658→0.706 (+0.048), recall +0.060 over shipped**, BOTH clean songs improved
  (backing +0.100), global label preserved (Δroot −0.000). Control proves the mechanism (differing BTC
  label, not the boundary alone). min→dom label errors are NOT gate-addressable (only 8/30 in-region) —
  a separate label-targeted pass. **Hook (the lane's coordinated edit — chord_pipeline_v1 READ-ONLY
  here):** in `_infer_nnls24`, after `segs`/before `_label_segments`, swap in-region segs+labels for
  BTC's (maj/min boundaries, voca labels), region-scoped so out-of-region is byte-identical. Ties into
  the fusion-aligner lane. Caveat: +0.048 measured in the parity chord-stage harness (nnls-seg 0.658
  there vs full-pipeline 0.679); clearing 0.679 needs the full-pipeline injection. Full detail:
  `boundary_alignment_fusion_2026-07-24.html` §8.
So the honest bottom line has TWO shippable deliverables: (a) no-chord intersect **+2.10pp root** (label
axis, below); (b) the HYBRID BTC-in-gated-regions ensemble **+0.048 F1 / +0.060 recall** (Louis's
alignment axis). Neither is wired (both are the lane's coordinated step). The "musx is the ceiling"
verdict below holds for GLOBAL LABEL accuracy but is FALSE for boundary RECALL.

## DECISION (assumed, per Louis "fais ta reco, je veux que ça marche le mieux possible")
**Best-working config today = shipped musx + the no-chord `intersect` brick (+2.10pp root).**
No model swap: BTC already lost to musx and doesn't emit bass; the in-house ChordFormer showed
no decisive win. musx (~0.735 root) is the identity CEILING among tools on hand. Lock the real
gain; do not chase a speculative deep-model integration unattended.

## THE ONE SHIPPABLE WIN — no-chord `intersect` brick (+2.10pp root)
`harmonia/models/no_chord_policy.py`, mode="intersect" (emit N only where musx-N ∧ NNLS-energy-N
agree). On the 7 frozen songs: **root 0.7367 → 0.7577 (+2.10pp)**, bass +2.05, partial +1.67,
7ths/strict +1.35, **zero per-song regressions**. Production-safe: on 15 s synthetic silence it
keeps 92% as N (blanket suppress keeps 0% and hallucinates chords). Why it works: musx over-fires
N — on the frozen chord-continuous material it marks **112 s as N, all false positives**; the
intersect gate removes them without sacrificing genuine-silence recall.
- Status: DEFAULT-OFF no-op (`HARMONIA_NC_POLICY` default "off" → passthrough); unit tests
  `tests/test_no_chord_policy.py` **8/8 green**; verified this session.

### WIRING HOOK (the refactor lane's coordinated edit — NOT done here)
In `chord_pipeline_v1._infer_nnls24`, immediately before the final
`labeled = _label_segments(..., seg_no_chord=seg_no_chord)` call (grep the symbol — the file is
live-edited, line numbers drift):
```python
from harmonia.models.no_chord_policy import gated_no_chord_mask
_nnls_nc = _nnls_no_chord_segs(arr, times, bt, segs)          # raw-energy silence gate (exists, line ~2823)
seg_no_chord = gated_no_chord_mask(seg_no_chord, _nnls_nc)    # ship HARMONIA_NC_POLICY=intersect
```
`_nnls_no_chord_segs` is confirmed present in the current live file. Default-OFF returns the mask
unchanged, so wiring it in is a behavioural no-op until `HARMONIA_NC_POLICY=intersect` is set.

## THE COMPLETE NEGATIVE MAP (do NOT re-run — every cheap identity lever is closed)
The dominant loss is chord IDENTITY on correctly-TIMED spans (boundaries verified unbiased: median
per-boundary |Δ| ~250 ms of *jitter* around 0, no systematic offset; ~37% of root loss is the
±P4/P5 fifth confusion). Root cause of every failure below is the **overtone-fifth acoustic fact**
(a played root has real energy at its fifth = 3rd harmonic; Billboard heads reproduced 45% P4/P5
on a 2nd corpus → fundamental to chroma, not a musx quirk):

| lever tried | result |
|---|---|
| 5 post-hoc adjudicators (root_resolve, fifth-resolver, bass→root, 7th-disc, 3rd-disc) | all negative |
| per-frame directional delta-chroma (Louis's idea), standalone | NULL on real audio (0.42 vs 0.87 mean-pool) |
| delta-chroma as auxiliary channel in a learned model | NULL (downweighted to noise, −0.97pp) |
| top-K root×quality COUPLING (revive dormant joint_decode) | −11pp; in-house head root **13.4pp worse than musx** on real audio |
| learned root+quality head on real-audio-derived POP909 features | in-domain 0.90 → **−25pp domain collapse** to 0.64; loses 6/7 |
| BTC-ISMIR19 deep front-end | already run live vs musx on RWC (2026-07-17) → **lost**; not on disk; no bass output |

Conclusion: no feature-engineering on pooled NNLS chroma, no adjudication between the two existing
sources, and no available deep model beats musx on identity. `_clip_pool` (bleed-free framing) is
the only leakage-robust feature that survives, but it edges mean-pool only in a micro-window regime.

## THE ONLY REMAINING CEILING-RAISE (a deliberate future project, NOT a quick experiment)
A genuinely newer deep real-audio chord model (ChordFormer-2025 lineage or a fine-tuned encoder)
that has LEARNED to discount the overtone fifth — obtain/train weights, integrate as a swappable
emission in `FusionChordDecoder` (`harmonia/align/inference.py::viterbi_decode`, swappable
emission/transition, leakage-free per-beat) behind a kill-switch. High ceiling, real cost,
uncertain (BTC lost; in-house ChordFormer no decisive win). Attack the fifth in the EMISSION/
front-end, never the transition prior (the ready fifth-motion + trigram priors REINFORCE the P5
error). Scope deliberately with a budget; do not run unattended.

## Data caveat (carried throughout)
The "461-pair" aligned set is on disk **139 pairs / 8 songs = the SAME audio as the frozen set**
(not new songs). The 7 hand-verified frozen Brick-0 songs are the trustworthy target.

## Artifacts produced this session (docs/research_sessions/)
brick0_baseline_erroranalysis_2026-07-24.html · brick_wave2_bricks_2026-07-24.md ·
brick_wave2_A_suppressed_spans.html · brick_boundary_vs_label_overlay_2026-07-24.html ·
brick_wave3_third_discriminator_screen_2026-07-24.md · leakage_robust_identity.html ·
coupling_root_quality_screen_2026-07-24.html · learned_engine_transition_2026-07-24.html · this file.

## Files to STAGE (nothing staged/committed by this session; refactor lane owns the wiring)
Modified (docstring/test only, default-OFF no-op preserved): `harmonia/models/no_chord_policy.py`,
`tests/test_no_chord_policy.py`. New: the docs/research_sessions/* artifacts above + the exp scripts
(`scripts/exp_leakage_robust_*.py`, `scripts/exp_learned_engine_*.py`, and screen scripts in scratchpad).
No edits to chord_pipeline_v1.py / eval / align / stages / golden.
