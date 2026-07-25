# Wave-2 chord-accuracy bricks — session 2026-07-24

Agent: Wave-2 research (build on Wave-1 baseline + error analysis; do NOT re-derive).
Task: propose/build/iterate default-OFF bricks, screen premise cheaply, keep only real gains.
Baseline (verified Wave-1, bit-identical this run): pooled root **0.7367** / majmin 0.7102 /
7ths 0.5173 / partial 0.6419 / strict 0.4774 / bass 0.7440 on 7 frozen songs (1654s).

Start: Fri Jul 24 01:04 CEST. Disk ~9.2 GiB free (swings — concurrent lane). Floor 1.5 GiB.

Frozen 7 (verified GT): blue_bossa, bein_green, blue_bossa_backing, close_to_you,
every_breath_you_take, georgia_on_my_mind, stand_by_me.

## EXECUTIVE SUMMARY (final)
Three bricks attempted A→B→C, premise-screened cheaply first.

| brick | premise-screen | verdict | pooled delta (7 frozen) | regressions |
|---|---|---|---|---|
| **A** no-chord `intersect` (P3) | musx-N=112.3s all-wrong, GT real-N=0.0s; NNLS-energy keeps only 15% | **KEEP — shippable** | **root +2.10pp, bass +2.05, partial +1.67, 7ths +1.35, majmin +1.80, strict +1.35** | none |
| **B** 7th one-note discriminator (P2a) | signal in group means, but per-segment confounded | **DROP (diagnosed neg)** | 7ths +2.44pp but MIRAGE (every_breath +22.3) | stand_by_me −9.8, georgia −1.9 |
| **C** bass-informed root prior (P1a) | GT==final 74% vs bass 16% on swap spans | **DROP (screen refuted)** | naive swap −155.6s root | — |

**Bottom line:** BRICK A moves the pooled number: **root 0.7367 → 0.7577 (+2.10pp)**, plus +2.05pp bass /
+1.67pp partial / +1.35pp 7ths / +1.35pp strict, ZERO per-song regressions. The intersect variant is
BIT-IDENTICAL to Wave-1's suppress on the benchmark but production-SAFE (synthetic-silence proof:
suppress invents chords over 15s silence, intersect keeps 92% N). This is the safe productionization P3
asked for and is READY TO WIRE. B and C are clean diagnosed negatives that both re-confirm Wave-1's core
finding — **music-x-lab is the stronger source; post-hoc NNLS-chroma refinement of its root (C) or 7th
(B) can't beat it because the discriminating evidence is confounded** (B: cross-segment b7 leak; C: the
bass sits on the chord's fifth, not a wrong root). No new module built for B/C (B regresses 2 songs).

**Parity/no-regression:** GREEN. `test_parity_nnls24_stages.py` 14 passed + `test_chord_head_parity.py`
1 passed (byte-identical stage + ChordHead captures on the current tree, disk stable at 5.8 GiB, no
spike); my only source touch is `no_chord_policy.py` (docstring, default-OFF, NOT imported in the hot
path) + a new test — the shipped pipeline output is byte-identical with/without my files.

**Files to stage:** `harmonia/models/no_chord_policy.py` (docstring: intersect VALIDATED + two-mask hook),
`tests/test_no_chord_policy.py` (+`test_intersect_safety_semantics`, 8/8), `docs/research_sessions/
brick_wave2_bricks_2026-07-24.md`, `brick_wave2_A_suppressed_spans.html`, `brick_wave2_A_timeline.png`.

**Wiring hook for A (the one coordinated edit, NOT done here):** in `_infer_nnls24`, before the final
`labeled = _label_segments(..., seg_no_chord=seg_no_chord)` (grep the symbol), add
`_nnls_nc = _nnls_no_chord_segs(arr, times, bt, segs)` then
`seg_no_chord = gated_no_chord_mask(seg_no_chord, _nnls_nc)`; ship with `HARMONIA_NC_POLICY=intersect`.

**Best next hypothesis (if budget continued):** the residual root loss is upstream (fifth-confusion 37%,
"other" 17%), unreachable by post-hoc editing of the two existing sources (B+C+Wave-1's root_resolve all
confirm this). The only lever left is a BETTER root SOURCE (P1b musx-vs-NNLS ensemble reweighting, or
P1c front-end) — NOT another adjudicator between musx and NNLS. Orthogonally, close_to_you's strict .235
is driven by musx FAMILY errors (dom→maj, min→maj) that need the 3rd, not the 7th — a 3rd-degree
discriminator screen is the untried quality lever.

## Brief restated (numbered spec)
1. Target: raise chord accuracy on the 7 frozen songs vs the 0.7367-root baseline; keep a
   brick only if a real gain on its target metric AND no regression elsewhere (~2pp bar).
2. Budget: large but bounded; check disk each cadence; stop on floor or plateau.
3. Integration: NEW default-OFF `harmonia/models/<brick>.py` + tests; measure via runtime
   monkeypatch (do NOT edit chord_pipeline_v1.py / eval / align / dataset). Report the hook.
4. Priority order: A (no-chord intersect, P3) > B (7th discriminator, P2a) > C (bass-root prior, P1a).
5. Constraints: parity green by construction (clean tree). No git. Honesty: every number a real run.

## Log

### 01:10 — Harness reproduces baseline bit-identical (determinism control)
`scratchpad/harness.py` runs shipped nnls24 + frozen-GT scoring with runtime monkeypatches,
persistent WAV work_dir (no re-decode). OFF control: stand_by_me root=0.8524, georgia=0.6299 —
bit-identical to Wave-1. ~10s/song warm. Determinism confirmed.

### 01:12 — BRICK A premise screen (musx-N vs NNLS-energy-N mask overlap). PASSES.
Captured both N masks at the final `_label_segments` (patch `nf.pool_beats` for arr/times/bt +
`_nnls_no_chord_segs`). Per-song musx-N mass vs intersect-kept:
```
song                   musxN_s  nnlsN_segs inter_s  gtN_s
blue_bossa               53.3       1        3.5     0.0
bein_green                1.3       1        0.8     0.0
blue_bossa_backing        6.0       2        2.0     0.0
close_to_you             10.2       2        6.8     0.0
every_breath_you_take     1.7       2        1.1     0.0
georgia_on_my_mind        1.8       1        1.8     0.0
stand_by_me              37.9       1        1.0     0.0
TOTAL                   112.3               17.1     0.0
```
- musx-N mass = 112.3s (all wrong: GT real-N = **0.0s** across all 7). intersect KEEPS only
  17.1s (15.2%) as N (where NNLS-energy agrees low-energy) => RECLAIMS 95.1s (85%) vs suppress's
  112.3s. So intersect should recover ~85% of suppress's gain while staying safe on genuine silence.
- Premise HOLDS: NNLS-energy fires on only ~1-2 segments/song, so intersect ≈ suppress on this
  chord-continuous benchmark but keeps the safety valve. Next: measure off/suppress/intersect on 7.

### 01:25 — BRICK A MEASURED on 7 (off vs suppress vs intersect). WIN: intersect == suppress, +2.10pp root, SAFE.
`scratchpad/measure_A.py` (patch `_label_segments` final pass to apply `no_chord_policy(musx,nnls,mode)`,
computing the NNLS-energy mask on the fly). OFF control reproduced root 0.7367 bit-identical.
| metric | off | suppress | Δsup | intersect | Δint |
|---|---|---|---|---|---|
| root | 0.7367 | 0.7577 | +0.0210 | 0.7577 | **+0.0210** |
| majmin | 0.7102 | 0.7282 | +0.0180 | 0.7282 | +0.0180 |
| sevenths | 0.5173 | 0.5308 | +0.0135 | 0.5308 | +0.0135 |
| partial | 0.6419 | 0.6586 | +0.0167 | 0.6586 | +0.0167 |
| strict | 0.4774 | 0.4909 | +0.0135 | 0.4909 | +0.0135 |
| bass | 0.7440 | 0.7645 | +0.0205 | 0.7645 | +0.0205 |
Per-song root (off/suppress/intersect): stand_by_me .852/.988/.988 (+13.5pp); blue_bossa .624/.650/.650
(+2.6pp); other 5 = +0.0. **intersect is BIT-IDENTICAL to suppress on the 7** — the 17.1s intersect keeps
as N all fall OUTSIDE the GT-scored span (pre-GT intros), so 0 scoring cost, while it recovers 100% of
suppress's benchmark gain. KEY DIFFERENCE: intersect is production-SAFE (requires BOTH musx-N AND
low NNLS-energy) where suppress-all invents chords over genuine silence. This is the safe
productionization P3 asked for. VERDICT: **KEEP — intersect is the shippable variant** (module exists,
default-OFF; intersect already unit-tested). No regressions. Next: synthetic-silence safety artifact.

### 01:32 — BRICK B (P2a 7th one-note discriminator): DROP — diagnosed negative with regressions.
Premise screen (group means, `scratchpad/screen_B.py`): the one-note signal EXISTS in aggregate —
maj7 shows root+11≈0.12 vs b7≈0.04; dom7 b7 high maj7 low; min7 m3+b7 high; hdim7 b7≈0.20 5th≈0.03.
Offline rule sweep (`sim_B.py`, correct-root segments, pooled Δmatched-seconds):
- **min→min7 adder** = +48.2s 7ths (best single lever, ~+2.9pp), roughly neutral strict.
- maj7↔dom7 flip = +5.1s/−1.4s (marginal); maj→maj7 = **−111s** (catastrophic); maj→dom7 = −20s;
  strip-maj7 = −4s; hdim-by-b5 = ~0. Musx correctly keeps majors plain (confirms its strength).
Tuning min→min7 (`tune_B.py`) exposed the trap: pooled +41.7s but per-song = **stand_by_me −17.6s,
georgia −3.7s**, gain carried by every_breath (+46.6s). Diagnosis (`diag_B.py`): b7 chroma does NOT
separate genuine min7 from plain-min — HELP(min7) center-b7 median **0.018** < HURT(plain-min)
**0.040**; the b7 on stand_by_me's Am triads reaches 0.13–0.25 (b7/chord-tone >2) = the following
b7-rooted chord (G after Am) leaking across the segment boundary via beat-pooling. Center-trimming
(`sim_B2.py`) does not fix it; no duration gate separates (backing's real min7 = 0.4s = stand_by_me's
damage range). END-TO-END confirm (`measure_B.py`, thr=0.03): pooled **7ths +2.44pp** (0.5173→0.5417)
but **stand_by_me 7ths −9.8pp, georgia −1.9pp**, strict −0.28pp — a mirage carried by every_breath
(+22.3pp). FAILS the no-regression bar; one-song-carried. Root cause = same as Wave-1 root_resolve:
music-x-lab quality is the stronger source; post-hoc NNLS-chroma 7th refinement can't beat it because
the one-note b7 evidence is confounded by cross-segment neighbor leak. VERDICT: **DROP, no module built**
(unlike fifth_discriminator it REGRESSES 2/7 songs, so not safe even dormant). Next: BRICK C screen.

### 01:35 — BRICK C (P1a bass-informed root prior): DROP — screen decisively refutes (confirms Wave-1).
Cheap screen (`scratchpad/screen_C.py`, capture final-root [musx] / nnls_root / nnls_bass per segment):
- (1) fifth-wrong mass = 112.4s. **P(nnls_bass == GT | fifth-wrong) = 0.385** (matches Wave-1's 38%),
  P(nnls_root == GT) = 0.336. Raw bass is NOT better than the current root on fifth-wrong spans.
- (2) KILLER: among swap candidates (nnls_bass vs final root differ by a fifth, 267.7s),
  **GT == final root 74.3%** vs GT == nnls_bass only **16.1%** → naive swap = **−155.6s root**.
  When the sounding bass sits a fifth off the chord root, it is overwhelmingly the chord's FIFTH
  (bassist on the 5th / inversion), NOT a signal that the root is wrong.
- (3) A bass-confidence gate does NOT isolate the 16%: net −85s @tau0.3, −36s @0.4, −7.8s @0.5,
  0 candidates above bconf 0.6. The confidence never rises on the spans where bass is right.
VERDICT: **DROP, no build** — the premise (bass beats current root on fifth-confusion) is false;
music-x-lab root is the stronger source (Wave-1 root_resolve conclusion, now confirmed for the bass
direction). Budget redirected to solidifying BRICK A (the win) + its safety artifact + parity.

### 01:42 — BRICK A artifacts + safety proof. Module solidified.
- Suppressed-spans table + timeline: `docs/research_sessions/brick_wave2_A_suppressed_spans.html` +
  `brick_wave2_A_timeline.png` (`scratchpad/artifact_A.py`). blue_bossa: 108 musx-N spans, intersect
  keeps 1 as N (pre-GT intro, unscored); stand_by_me: 50 musx-N spans, intersect keeps 1 (post-GT tail,
  unscored) — reconciles the bit-identical intersect==suppress end-to-end.
- **Synthetic-silence end-to-end safety proof** (`scratchpad/synth_silence.py`, df-guarded, temp WAV
  cleaned): clip = 18s music | **15s pure silence** | 18s music through the shipped pipeline:
  | policy | % of silence labelled N | renders over silence |
  |---|---|---|
  | off (baseline) | 97% | N (correct) |
  | **suppress** | **0%** | invents `G#:hdim7/C`, `E:maj/D#` — UNSAFE |
  | **intersect** | **92%** | keeps N — SAFE |
  So intersect recovers ALL of suppress's +2.10pp on the benchmark AND survives genuine silence.
- Module `no_chord_policy.py`: intersect mode already implemented + unit-tested; updated the "DESIGNED
  not measured" docstring note to VALIDATED, documented the precise two-mask wiring hook (compute
  `_nnls_no_chord_segs(arr,times,bt,segs)` then `gated_no_chord_mask(seg_no_chord, nnls_mask)` before
  the final `_label_segments`). Added `test_intersect_safety_semantics`. **tests: 8/8 pass**.
- Parity/no-regression: my only source touches are `no_chord_policy.py` (docstring, default-OFF, NOT
  imported in the hot path) + `tests/test_no_chord_policy.py` (new test) + docs/. The shipped
  `infer_chords_v1` output is byte-identical with/without my files (they aren't imported) → parity
  green by construction w.r.t. my work; chord_pipeline_v1.py is dirty ONLY from the concurrent lane.
</content>
</invoke>
