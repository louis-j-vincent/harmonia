# Brick 0 real-audio chord-accuracy — session 2026-07-23

Autonomous budget-driven session. Target: measure the shipped detector on the 7
FROZEN Brick-0 songs (`harmonia/eval/accuracy_score.py`, READ-ONLY) and improve
it via default-OFF bricks. ~90 min budget. Start 14:06 CEST.

## EXECUTIVE SUMMARY (final)
- **First trustworthy real-target chord number** (shipped nnls24, 4 cache-warm frozen songs,
  948s): pooled **root 0.664 / majmin 0.636 / 7ths 0.527 / partial 0.580 / strict 0.509 /
  bass 0.670**. 3/7 songs disk-blocked (fresh music-x-lab decode spikes ~2 GiB; disk was at
  the 1.4 GiB floor). Numbers on a tree dirty in local_key.py+chart_model.py (concurrent lane).
- **Bottleneck = ROOT LABELING (upstream), confirming STEP 9/10/11.** Root loss dominates every
  family; bass is gated by root (fix root -> bass follows). 42% of chord-root errors are
  fifth-related (P5+P4).
- **Bricks tried (each brick-ON minus brick-OFF, delta valid on dirty tree):**
  1. flip-margin gate — NEGATIVE (+0.17pp root best, < 2pp bar; DROP). Confirms STEP-10 mechanism.
  2. global time-shift — NEGATIVE (optimum at 0s; grid already aligned; DROP).
  3. **no-chord suppression — WIN: +3.61pp root** (0.664->0.700), +2.9pp partial, +3.5pp bass.
     New default-OFF module `harmonia/models/no_chord_policy.py` + tests. KEEP (with caveats).
- **Parity:** zero pipeline/eval edits -> gate green by construction.
- **Next hypothesis (biggest remaining gap):** target the 42% fifth/neighbor root confusion with
  a key-aware transition-prior / Viterbi root-smoothing brick, and validate no_chord_policy on the
  full 7 (needs disk) + a chord-continuous vs silence-bearing split for the repertoire guard.

## Reproducibility caveat (recorded up front)
- HEAD `3c13a2a`.
- Working tree DIRTY in core inference (concurrent session): `harmonia/theory/local_key.py`,
  `harmonia/output/chart_model.py` (both `M`). `harmonia/models/chord_pipeline_v1.py` is CLEAN.
- Baseline numbers below are on this dirty tree; on/off brick DELTAS are valid regardless
  (same base, measured brick-ON minus brick-OFF).
- Disk at start: 3.4 GiB free (99% full). Floor 1.5 GiB — checked before every run.

## Levers triage (from Phase-0 code read)
- **flip-margin gate** (`segmentation_gate.py`): hooks `_root_change_segs`, which IS on the
  shipped `segment_source="nnls"` hot path. Testable via runtime monkeypatch (no file edit).
  STEP 10: shrank to +1.33pp root on POP909-musx; UNTESTED on real audio. → primary lever.
- **emission_scoring dot/cosine**: `grep ChordInferrer|emission_scoring harmonia/models/chord_pipeline_v1.py`
  → ZERO hits. The nnls24 shipped path uses music-x-lab per-segment labels, not ChordInferrer
  template scoring. Lever is a NO-OP on the shipped config (issue #5 SUPERSEDED). → dropped, not run.
- HMM/transition/quality-bass mapping tweaks → new default-OFF bricks if baseline diagnosis warrants.

## Log

### 14:10 — BASELINE (shipped nnls24, live_defaults). DISK-BLOCKED to 4/7 songs.
First full-7 attempt hit the 1.5 GiB disk floor after ONE song: fresh music-x-lab
5-fold decode of the 8-min blue_bossa spiked disk 3.4→1.43 GiB (persisted, did not
recover — a concurrent live session shares this disk). Cache-hit scoring is cheap and
disk-safe (held 1.43 GiB across the 4 warm songs). Cache-COLD songs (every_breath,
close_to_you, blue_bossa_backing) need a fresh ~2 GiB-spiking decode → CANNOT run below
the floor without risking the known disk-full incident → reported disk-blocked.

BASELINE (4 cache-warm frozen songs, HEAD 3c13a2a, tree dirty in local_key.py+chart_model.py):

| song | root | majmin | 7ths | partial | strict | bass | dur | n |
|---|---|---|---|---|---|---|---|---|
| blue_bossa | 0.624 | 0.587 | 0.513 | 0.535 | 0.513 | 0.622 | 493s | 286 |
| bein_green | 0.630 | 0.625 | 0.535 | 0.621 | 0.450 | 0.672 | 154s | 64 |
| georgia_on_my_mind | 0.630 | 0.575 | 0.347 | 0.528 | 0.322 | 0.631 | 151s | 68 |
| stand_by_me | 0.852 | 0.852 | 0.731 | 0.731 | 0.731 | 0.852 | 161s | 41 |
| **POOLED (958s)** | **0.664** | **0.636** | **0.527** | **0.580** | **0.509** | **0.670** | | |

mir_eval cross-check agreed within 0.02 on all (no warnings logged). blue_bossa is the
8-min jazz outlier (lowest root); stand_by_me (3-chord pop) is the ceiling. This is the
first trustworthy real-target chord number on the frozen benchmark.
DISK-BLOCKED remainder: every_breath_you_take, close_to_you, blue_bossa_backing (3/7).

### 14:15 — DIAGNOSIS (artifact: docs/research_sessions/brick0_diagnosis_2026-07-23.png, panel A)
Duration-weighted loss attribution on the 4 warm songs:
- **ROOT is the bottleneck.** root_lost per family: hdim 70% of its 93s, dom 36% of 250s,
  min 31% of 372s, maj 21% of 243s. Quality-family confusion (dom↔maj 39s, min→dom 16s,
  hdim→min 11s) is small by comparison.
- **Bass is gated by root.** bass_lost tracks root_lost near-exactly; bass-miss breakdown
  = 308.5s root-wrong vs 7.6s root-ok. When root is right, sounding-bass is essentially
  free (only 7.6s of true slash/inversion misses). The whole game is ROOT.
- Root-error structure (251s of chord-vs-chord root errors, N excluded): **fifth-related
  42%** (P5 23% + P4 19%), then m3 12% / TT 12% / m7 10% (relative-minor / tritone-sub /
  dominant-neighbor). Classic root-vs-fifth + functional-neighbor confusion (ii-V-I chords
  are a 4th/5th apart). This is the dominant STRUCTURED mode and the biggest future target.
- Per-song signatures: blue_bossa = Ab/G# attractor in its Db-major bridge (82 intervals);
  georgia = local neighbor-chord slips (NOT a global time offset, see shift test).

### 14:18 — BRICK 1: flip-margin gate (segmentation_gate.py, HARMONIA_FLIP_MARGIN). NEGATIVE.
Monkeypatched `_root_change_segs` -> `confidence_gated_segs` at each T (0.0=exact baseline
control; reproduced baseline root 0.6642 bit-identical -> determinism confirmed). On/off Δroot:

| T | Δroot | Δpartial | Δ7ths | Δbass | segs/song |
|---|---|---|---|---|---|
| 0.1 | +0.0017 | +0.0042 | +0.0042 | +0.0008 | 300 |
| 0.2 | +0.0015 | +0.0042 | +0.0031 | +0.0002 | 286 |
| 0.3 | −0.0037 | −0.0002 | −0.0010 | −0.0045 | 263 |
| 0.5 | −0.0097 | −0.0038 | −0.0026 | −0.0071 | 203 |

Best +0.17pp root @T=0.1 — far below the ≥2pp bar, and SMALLER than STEP 10's +1.33pp on
POP909-musx. Confirms the STEP-10 mechanism: musx labels + `_coalesce_labeled` already merge
spurious NNLS flips, so the gate has nothing to do. The error is root LABELING, not segmentation
precision. VERDICT: DROP (keep dormant; do not wire). Artifact panel B.

### 14:22 — BRICK 2: global time-shift post-pass (disk-free arithmetic). NEGATIVE.
Hypothesis (from georgia's neighbor slips): prediction systematically lagged vs frozen grid.
Got each pred once (cache), shifted chord times by delta, re-scored vs frozen GT.
Pooled-root optimum is EXACTLY delta=0.00s (smooth peak at 0). Per-song: blue_bossa/georgia
want −0.15s (+1.2/+0.8pp), bein/stand want 0.0 — inconsistent boundary jitter, no systematic
latency. The model grid and frozen GT grid already agree to <0.15s (well under a beat).
VERDICT: refuted — georgia's "lag" is genuine local labeling error, not a phase offset.
Artifact panel C.

### 14:26 — BRICK 3: no-chord suppression (NEW module harmonia/models/no_chord_policy.py). WIN.
Diagnosis lead: spurious N = 61.2s = 6.45pp of the pooled score, and the frozen GT for these
chord-continuous standards has ZERO N spans -> EVERY predicted N is wrong. music-x-lab marks
108/900 segments N on blue_bossa alone. Suppressed both N-mask sources (all-False) and re-scored.

| metric | baseline | brick | Δ |
|---|---|---|---|
| root | 0.6642 | 0.7003 | **+0.0361** |
| majmin | 0.6360 | 0.6670 | +0.0310 |
| sevenths | 0.5269 | 0.5502 | +0.0233 |
| partial | 0.5804 | 0.6092 | +0.0288 |
| strict | 0.5092 | 0.5325 | +0.0233 |
| bass | 0.6702 | 0.7055 | +0.0353 |

Per-song root: stand_by_me +13.5pp (its doo-wop bass loop read as no-chord), blue_bossa +2.6pp,
bein_green/georgia +0.0. Realized +3.61pp vs 6.45pp ceiling -> ~56% of un-N'd spans still land a
wrong root (partial, not free). Clears the ≥2pp bar. Built as default-OFF module
`harmonia/models/no_chord_policy.py` (env HARMONIA_NC_POLICY, off=exact passthrough), unit-tested
(tests/test_no_chord_policy.py, 7/7), and END-TO-END validated via the real module hook
(stand_by_me 0.852->0.988 with HARMONIA_NC_POLICY=suppress; off-mode identical). Artifact panel D.
VERDICT: KEEP as default-OFF brick. Caveats (in the module docstring): suppress-all is UNSAFE
where real silence exists (pop intros/breakdowns) -> needs a repertoire guard before production;
validated on N=4 only (gain concentrated in stand_by_me); mode="intersect" designed-not-measured.

### Parity
ZERO edits to any pipeline/eval source: `chord_pipeline_v1.py` and `segmentation_gate.py` are
CLEAN; only `chart_model.py`+`local_key.py` are dirty (the concurrent session's, untouched by me).
All three bricks were runtime monkeypatches in /private/tmp throwaway scripts. The parity gate
(`harmonia/eval/parity.py`) cannot regress — no hot-path code changed. (Did NOT run the gate
capture: it needs fresh disk-spiking inference below the floor; parity is green by construction.)

</content>
