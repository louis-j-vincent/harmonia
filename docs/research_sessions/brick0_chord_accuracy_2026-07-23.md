# Brick 0 real-audio chord-accuracy — session 2026-07-23

Autonomous budget-driven session. Target: measure the shipped detector on the 7
FROZEN Brick-0 songs (`harmonia/eval/accuracy_score.py`, READ-ONLY) and improve
it via default-OFF bricks. ~90 min budget. Start 14:06 CEST.

## EXECUTIVE SUMMARY (final — updated after disk unblock)
- **FIRST COMPLETE real-target chord number** (shipped nnls24, ALL 7 frozen songs, 1654s):
  pooled **root 0.737 / majmin 0.710 / 7ths 0.517 / partial 0.642 / strict 0.477 / bass 0.744**.
  Determinism confirmed (margin=0.0 control reproduced root 0.7367 bit-identical). Numbers on a
  tree dirty in local_key.py+chart_model.py (concurrent lane); on/off brick DELTAS unaffected.
- **Bottleneck = ROOT LABELING (upstream), confirming STEP 9/10/11.** Root loss dominates every
  family; bass is gated by root (fix root -> bass follows: bass-miss 308.5s root-wrong vs 7.6s
  root-ok). 42% of chord-root errors are fifth-related (P5 23% + P4 19%).
- **Four bricks tried (each brick-ON minus brick-OFF):**
  1. flip-margin gate — NEGATIVE (+0.17pp root, < 2pp bar; DROP). Confirms STEP-10 mechanism.
  2. global time-shift — NEGATIVE (pooled optimum at 0s; grid already aligned; DROP).
  3. **no-chord suppression — WIN: +2.10pp root** on the full 7 (0.737->0.758), +1.7pp partial,
     +2.1pp bass, ZERO per-song regressions. New default-OFF module `no_chord_policy.py` + tests.
     KEEP. (4-warm subset gave +3.61pp; gain concentrated in stand_by_me +13.5, blue_bossa +2.6.)
  4. key-aware fifth-resolver (TASK 2) — NEGATIVE (−4.30pp root at best tau; DROP). Premise
     refuted: fifth-confusion needs a BETTER root source, not re-adjudication between musx & NNLS.
     New default-OFF module `root_resolve.py` + tests (kept dormant).
- **Parity:** zero edits to any pipeline/eval source -> gate green by construction (all bricks
  are runtime monkeypatches in throwaway scripts; the two new modules are default-OFF no-ops).
- **Next hypothesis:** root is upstream — the +2pp no-chord reclaim and the refuted fifth-resolver
  both point to the NNLS/musx ROOT SOURCE itself. Highest-value next step: (a) an N-precision brick
  (intersect musx-N ∧ energy-N, or a repertoire/duration guard) to make no_chord safe for pop
  intros, and (b) improve the root source (musx ensemble weighting / a bass-informed root prior)
  rather than post-hoc root editing.

## ★ FULL 7-SONG BASELINE (disk unblocked to 4.1 GiB by Louis; 2nd checkpoint)
Ran all 7 frozen songs, shipped nnls24. Disk held 4.05 GiB throughout (cold decodes
18-27s each, no spike — the earlier 2 GiB drop was transient/concurrent-lane, not the decode).

| song | root | majmin | 7ths | partial | strict | bass | dur | n |
|---|---|---|---|---|---|---|---|---|
| blue_bossa | 0.624 | 0.587 | 0.513 | 0.535 | 0.513 | 0.622 | 493s | 286 |
| bein_green | 0.630 | 0.625 | 0.535 | 0.621 | 0.450 | 0.672 | 154s | 64 |
| blue_bossa_backing | 0.879 | 0.860 | 0.498 | 0.733 | 0.498 | 0.879 | 307s | 156 |
| every_breath_you_take | 0.747 | 0.747 | 0.519 | 0.747 | 0.519 | 0.764 | 202s | 58 |
| georgia_on_my_mind | 0.630 | 0.575 | 0.347 | 0.528 | 0.322 | 0.631 | 151s | 68 |
| close_to_you | 0.864 | 0.806 | 0.497 | 0.695 | 0.235 | 0.879 | 187s | 63 |
| stand_by_me | 0.852 | 0.852 | 0.731 | 0.731 | 0.731 | 0.852 | 161s | 41 |
| **POOLED (1654s)** | **0.737** | **0.710** | **0.517** | **0.642** | **0.477** | **0.744** | | |

FIRST COMPLETE real-target number. Hardest: blue_bossa/bein_green/georgia (jazz, ~0.63 root);
easiest: backing/close_to_you/stand_by_me (~0.85-0.88). strict is low on close_to_you (0.235:
right root+family, wrong exact 7th token) — a partial-vs-strict gap, not a root problem.

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

### ★ BRICK 3 re-measured on FULL 7 (disk unblocked): +2.10pp root, ZERO regressions. KEEP.
| metric | baseline(7) | brick(7) | Δ |
|---|---|---|---|
| root | 0.7367 | 0.7577 | **+0.0210** |
| majmin | 0.7102 | 0.7282 | +0.0180 |
| sevenths | 0.5173 | 0.5308 | +0.0135 |
| partial | 0.6419 | 0.6586 | +0.0167 |
| strict | 0.4774 | 0.4909 | +0.0135 |
| bass | 0.7440 | 0.7645 | +0.0205 |
Per-song root: stand_by_me +13.5pp, blue_bossa +2.6pp, other 5 = +0.0 (no spurious N on them).
EVERY song delta ≥ 0 — suppress-N never hurt any of the 7 (none has a true no-chord span). Real,
safe-on-this-benchmark +2.10pp root, concentrated in 2/7 songs. (4-warm subset gave +3.61pp; the
3 added songs dilute with zero gain.) The production caveat is unchanged: unsafe where real silence
exists — needs a repertoire guard. VERDICT: KEEP default-OFF.

### 14:26 — BRICK 3 (first measurement, 4 warm songs): no-chord suppression. WIN.
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

### 15:xx — BRICK 4: key-aware fifth-disagreement root resolver (NEW module root_resolve.py). NEGATIVE.
TASK-2 hypothesis (attack the 42% fifth/neighbour root confusion). PREMISE SCREEN first
(georgia + blue_bossa): where the FINAL root is wrong by a P4/P5, does the NNLS argmax
posterior already hold the GT root? blue_bossa 44/122 spans (23.8s) recoverable, georgia 3/8
(2.8s) → ceiling ~1.6pp pooled, one-song-dominated. KEY POINT: under quality_frontend='musx'
the final root is music-x-lab's; a pure key/transition prior CANNOT break a fifth-confusion
(V and I are both diatonic, and the fifth is the commonest diatonic motion) — only the NNLS
posterior can adjudicate. So the brick resolves fifth-disagreements toward the NNLS root when
it is confident (mass≥tau, margin≥0.10) AND diatonic to the inferred key.

MEASURED on/off, full 7 songs, tau sweep (baseline pooled root 0.7367):
| tau | Δroot | Δpartial | swaps |
|---|---|---|---|
| 0.30 | −0.0474 | −0.0511 | 328 |
| 0.35 | −0.0463 | −0.0500 | 321 |
| 0.40 | −0.0452 | −0.0489 | 313 |
| 0.45 (least-bad) | −0.0430 | −0.0468 | 299 |

Monotonically NEGATIVE. Damage concentrates where musx is RELIABLE: blue_bossa_backing −11.8pp,
close_to_you −8.4pp, blue_bossa −3.9pp; the jazz songs barely move (georgia +0.5pp). Mechanism:
even though NNLS holds the truth in 42% of fifth-wrong spans, the confidence+diatonic gate cannot
ISOLATE them from the 58% where NNLS is also wrong OR where musx was already correct — swapping
breaks the larger correct set. Music-x-lab's root is simply the stronger source (that is why it's
the shipped default). VERDICT: DROP (module kept dormant default-OFF; do NOT wire). Bass unchanged
(root-only edit). Premise refuted end-to-end — the fifth-confusion is NOT fixable by re-adjudicating
between the two existing root sources; it needs a BETTER root source (upstream), per STEP 9/10/11.

### Parity
ZERO edits to any pipeline/eval source: `chord_pipeline_v1.py` and `segmentation_gate.py` are
CLEAN; only `chart_model.py`+`local_key.py` are dirty (the concurrent session's, untouched by me).
All three bricks were runtime monkeypatches in /private/tmp throwaway scripts. The parity gate
(`harmonia/eval/parity.py`) cannot regress — no hot-path code changed. (Did NOT run the gate
capture: it needs fresh disk-spiking inference below the floor; parity is green by construction.)

</content>
