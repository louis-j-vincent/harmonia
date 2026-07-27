# Handoff — 2026-07-27, after the synthetic-clock discovery

Read this whole file before touching anything. Then read, in order:
`CLAUDE.md` (the six error patterns are load-bearing), the top ~200 lines of
`docs/known_issues.md`, and `git log --oneline -12`. Everything below is already
committed; nothing here is speculation unless labelled as such.

---

## 1. The one-paragraph state of the world

The chord pipeline's segmentation just moved to music-x-lab's own chord-change
times (`segsource=musx_redecode`, shipped, +2.6 pp family-level). Underneath
that, a **calibration bug was found that reframes the whole boundary effort**:
the model lays its chords on a *synthetic constant-tempo lattice* it builds
itself, not on the beats it actually detected — and the frozen benchmark's
reference is a synthetic lattice too. Both clocks were wrong in the same way, so
they agreed by accident. Fixing either one alone *loses* points. Fixing both
gains only +0.43 pp, but the boundary-collar diagnostic halves, which says
roughly half the boundary error was the fake clock and half is genuine label
error. Meanwhile the benchmark's reference is confirmed ~12% wrong by the
owner's ear, and two previously-celebrated wins have been retracted as
artefacts. The remaining error is now known to be **62% wrong-root, and a long
tail** — 163 distinct confusions on 7 songs — which closes the "find the
systematic defect and patch it" family of solutions.

---

## 2. What is live right now

Server: `scripts/harmonia_server.py --port 7771`, **restarted 2026-07-27 22:27**
(it had been up since Jul 23 13:51, which is why four days of commits had never
reached the owner's phone — treat "did you restart the server?" as the first
question whenever live behaviour disagrees with a benchmark).

Reachable over real HTTPS at `https://louiss-macbook-air.tail87ced3.ts.net`
via `tailscale serve --bg --https=443 7771`. The mic works there (Safari needs a
secure context); the record→analyze round trip was verified end-to-end with both
iOS-style AAC/mp4 and Chrome-style Opus/webm uploads. Disable with
`tailscale serve --https=443 off`.

Shipped defaults changed today, all reversible by env var:

| knob | env var | value | rollback |
|---|---|---|---|
| segmentation source | `HARMONIA_ANALYZE_SEGSOURCE` | `musx_redecode` | `=nnls` |
| musx re-decode | `HARMONIA_MUSX_REDECODE` | `1` | `=0` |
| Occam gate (scoped to redecode) | `HARMONIA_OCCAM_GATE` | on | — |
| musx latency compensation | `HARMONIA_MUSX_LATENCY` | `auto` | `=0` |

Benchmark effect, pooled over the 7 frozen songs: partial 0.6419→**0.6679**,
root 0.7367→**0.7510**, majmin 0.7102→**0.7318**, 7ths 0.5173→**0.5365**,
strict 0.4774→**0.4924**, bass 0.7440→**0.7582**. No cell regresses. Holds
against the repaired reference too (L4 0.6199→0.6481). Two songs regress
individually (`every_breath` −1.20, `georgia` −1.03) — both are the songs with
the most confirmed-wrong reference labels, so read that as a reference problem.

---

## 3. Bricks that exist but are NOT wired

Every one of these is default-OFF with an env kill-switch, and has **no importer
outside its own module and tests**. Wiring them is a real decision, not a
formality — read the verdict column first.

| brick | measured | verdict |
|---|---|---|
| `function_family` | **+2.69 pp raw / +3.12 pp repaired** | **WIRE THIS.** 21 chords fixed, 0 broken, zero per-song regressions in 4 config×reference combos, LOSO-validated. Hook documented in its docstring; not wired only because `chord_pipeline_v1.py` was concurrently owned. |
| `real_beat_grid` | +0.43 pp | Correctness fix, not a lever. **Do not flip** until the second lattice (§5) is fixed — the Occam post-pass silently undoes it. |
| `harmonic_texture` | n/a (detector) | Ships as a capability. Reproduces the reference overlay's IMPLIED spans to 0.000000000 s on all 7 songs. |
| `no_chord_policy` | **RETRACTED** | Do NOT ship. See §4. |
| `repetition_prior` | doesn't pay | Premise passes (70× odds lift) but 87% of chord changes already fall on downbeats, so it buys nothing budget-matched. Survives only as a *detector* (2.2× odds). |
| `fifth_discriminator`, `root_resolve`, `harmonic_downbeat` | dormant | Unevaluated against current defaults. |
| `seventh_upgrade` | **exactly 0.0000** | Under the headline metric it does nothing. Leave it. |

---

## 4. Retractions — do not re-derive these

**`no_chord_policy`'s +2.10 pp was an artefact, twice over.** It does not
reproduce against a bit-identical OFF control (+0.10 pp partial, not +1.67), and
the residue is **−0.0044** on the repaired reference. Mechanism: suppressing `N`
over Stand By Me's bass-only intro turns **41 chart cells into 87** — twelve
chords in a 10 s intro, one per beat, because the argmax root flaps when nothing
sounds above the bass. Occam loop-family coverage drops 0.98→0.97 and the
simplicity post-pass **rejects its own gate**, so the flicker survives. The brick
fights the Occam pass (error-pattern #6).

**"Hold the previous chord through an implied passage" is refuted.** The
reference *is* the chart, so this is directly testable: over the 65.8 s of
implied time the chart **changes chord 88.9% of the time**. Blue Bossa's bass
solo runs the full 16-bar form. Only 7.3 s is a genuine hold. The obvious rescue
(hold only when the span is shorter than the song's median stated chord)
qualifies **zero spans**.

**Filling implied spans acoustically is refuted.** It prints **76 cells where 11
belong** (2.0 chords/s against a real harmonic rhythm of 0.8/s) for zero
family-level gain, and on Blue Bossa scores *worse* than holding (0.15 vs 0.23).

**The DBN/persistence prior does not place boundaries better.** It is a
"hold-longer" prior, not a "place-better" one, and boundary error already leans
2:1 toward holding.

**Also already refuted, do not retry:** chroma novelty, onset, and HPSS as
per-instant change evidence; post-hoc chroma adjudication of music-x-lab output
(5 attempts, cross-boundary leakage).

---

## 5. Ranked priorities, with try-order and stopping criteria

### P1 — Wire `function_family` (hours, high confidence)
The largest validated unwired gain. Hook is in the module docstring; the blocker
was file ownership, not evidence.
**Stop/continue:** ship if pooled partial gains ≥ +2.0 pp on **both** the raw and
repaired references with **zero per-song regressions** — it already met this
offline. Abandon if wiring it regresses any song.

### P2 — Fix the SECOND lattice in the Occam post-pass (hours, blocks P3)
`chord_pipeline_v1.py:2602` (`np.arange(phi*beat, …, bar_period)`) and
`chord_head.py:604` (`anchor_t + b*bar_len`) re-lay spans onto their own
synthetic grid, silently undoing `real_beat_grid`. Proof it is real: with
`HARMONIA_OCCAM_POSTPASS=0`, the same run lands boundaries **0.0 ms** from
detected beats. Stand By Me is the visible symptom — its collar gain *rises*
under the grid brick (+2.25 → +3.53).
**Stop/continue:** the grid brick's collar-gain reduction must hold on Stand By
Me too. If fixing this does not change Stand By Me's collar behaviour, the
diagnosis is wrong — stop and re-diagnose rather than patching further.

### P3 — Then decide `real_beat_grid` (after P2)
Only after P2 is the measurement trustworthy. Note the trade-off explicitly:
the uniform lattice exists because librosa's tempo scalar is a *local* median
(0.5–2.3% systematic, ~4 bars of drift over a song). `bestfit` is **not** the
culprit and undoing it would regress; the defect is the `arange` reconstruction.
**Stop/continue:** flip the default only if matched-clock partial gains ≥ +1.0 pp
**and** the ±0.25 s boundary-collar gain drops on ≥ 5 of 7 songs. The collar is
the primary evidence — it cannot be gamed by two clocks agreeing. Current
figures: +0.43 pp, collar +4.04 → +2.13 on the 5 songs the brick reaches, strict
collar +1.62 → **+0.31**.

### P4 — Build a FORM model (days; this is the real remaining lever)
Measured, not speculative: reading the model's **own** chart one form-period
back — period from its own label self-similarity, no reference consulted —
scores root **0.49** against 0.25 for holding and 0.30 for acoustic filling.
Roughly double, and right for the right reason. This is also what the owner
asked for directly: *"c'est la logique de la répétition qui nous aide beaucoup
et qu'il faut qu'on exploite au maximum"*.
**Stop/continue:** it must beat 0.30 (acoustic filling) on held-out songs, and
must not print more cells inside implied spans than the median stated harmonic
rhythm (0.8 chords/s) — that is the failure mode that killed the acoustic
variant.

### P5 — Re-capture the parity goldens (owed debt)
`LIVE_ORACLE_KWARGS` is now pinned to the old config, so the byte-identity net
**no longer covers the shipped segmentation**. Separately,
`tests/test_chord_head_parity.py::test_chord_head_byte_identical` fails on
`let_it_be` against a golden that is internally impossible for current code
(`chords=112, segments=45`). Re-capture needs cold posteriors for 5 songs.
The net earned its keep today — it caught the Occam gate silently changing
`let_it_be` from 120 to 112 chords. Do not let it rot.

### Not a priority: root errors
62.8% of lost time is wrong-root, and it is a **long tail** — 163 distinct
(GT→predicted) pairs on 7 songs; 21 cover 50%, **46–56 cover 80%**, 82 cover
90%. No targeted discriminator has the right shape. If you attack this, it must
be by improving recognition upstream, not by post-hoc correction. Say so rather
than writing the fiftieth rule.

---

## 6. The benchmark's reference is a measurement too

The 7-song frozen benchmark's reference is a **chart-derived metronomic
lattice**, not a transcription of the recordings. Residual of a single fitted
constant tempo: **0.00–0.28 ms on 4 of 7 songs**. Georgia — a rubato Ray Charles
ballad — at 0.28 ms is the decisive proof. Independent check: detected beats
score 1.3–3.4× chance on spectral-flux onset strength and beat a random-phase
null at p<0.005 on 6/7 songs; **the reference lattice sits inside the null on the
same 6**.

`every_breath_you_take` is the honest exception in both columns — that recording
is machine-steady, so the lattice *is* the real grid there. Any grid work should
expect it to be the null case and should not be alarmed when it regresses.

A repaired overlay exists: `golden/frozen_parity/gt_repair_2026-07-27/*.overlay.json`
(read its README). Ladder, family-level partial:

| step | partial | scoreable | % kept |
|---|---|---|---|
| L0 baseline | 0.6419 | 1654 s | 100% |
| L1 + retime | 0.6361 | 1654 s | 100% |
| L2 + excise bass solo | 0.6498 | 1619 s | 97.9% |
| L3 + drop IMPLIED | 0.6528 | 1589 s | 96.1% |
| **L4 + drop WRONG** | **0.6792** | 1414 s | **85.5%** |

**Score every lever against BOTH the raw reference and the overlay, and report
scoreable time on every row.** A gain that exists only against the raw reference
is an artefact — that is exactly how `no_chord_policy` fooled us. Conversely,
**L4's gain is mostly from deleting regions selected *because* the model
disagreed there**, so it is an upper bound, not a score. The true value is
0.64–0.68. Selection bias is the dominant residual risk and is not fully
removable.

The `golden/brick0/*` originals are owned by another lane and were not edited.
What that lane still needs to change is listed at the end of the "GT repair"
entry in `known_issues.md`.

---

## 7. Landmines — each of these has already cost real time

- **`data/cache/raw_beat_times_v2/` is NOT Beat This! output.** Its values are
  off the 50 Hz grid and it reports 80 BPM on a 118 BPM song. Use the
  pipeline-captured `beat_times_real`. An agent that trusted it silently
  accumulated a **37 s** error on Blue Bossa.
- **Blue Bossa's detected beat grid is missing 25.2% of its beats.** Any re-lay
  needs a coverage guard that *refuses* rather than interpolates.
- **GT-boundary leakage.** Sampling a posterior at sub-interval midpoints reads
  at times cut on *reference* boundaries. This inflated one screen from a true
  +1.04 pp to a reported +4.23 pp. Read only at your own segment bounds.
- **Purity/coverage metrics are gameable.** Cutting at every beat scores 0.870,
  beating the oracle's 0.863. Always compare at a matched segment budget.
- **`PitchExtractor` caches to `data/cache/*.npz` and the cache key does NOT
  cover module-level constants.** Clear the cache after changing one.
- **Disk has hit 234 MiB free during this session** and halted work twice.
  Check `df -h /` periodically; stop below ~2 GiB. Harmonia is not the cause
  (~2.5 GiB of a container that was 99.5% full).
- Song 002's tempo has a 2×-octave trap via the librosa fallback (see
  `CLAUDE.md`).

---

## 8. Working rules for this repo

- **A second Claude session runs on `main` concurrently.** NEVER touch
  `harmonia/align/*`, `harmonia/dataset/*`, `scripts/brick0_propose.py`,
  `golden/brick0/*`, `docs/fusion_aligner_design.md`,
  `docs/dataset_harvest_design.md`, `docs/brick0_review/*`, or the tests
  `test_brick0_drift.py`, `test_drum_pattern.py`, `test_downbeat.py`,
  `test_bass_salience.py`, `test_dataset_gate.py`, `test_fusion.py`.
  Run `git status` first and leave unfamiliar modified files alone.
- **Never `git add -A` / `git add .` / `git commit -a` / `--no-verify`.** Stage
  explicit paths.
- **Screen the premise cheaply before implementing** (error-pattern #2). Two
  candidates in this project's history were fully built before their premise was
  checked; both failed. Today, three separate ideas were killed by a cheap check
  *before* implementation — that is the pattern working.
- **Never claim success on a metric alone.** Produce something inspectable — the
  owner reviews on an iPhone and listens. Match the style of the existing
  phone-first pages in `docs/research_sessions/` (self-contained HTML, short
  embedded audio as data: URIs, no external references).
- **The headline metric is family-level `partial_credit`.** Report `mirex_root`
  and strict as secondary. Report **per-song**, never only the mean — 7 songs is
  small, read the shape not the third decimal.
- **A clean negative result is a valid deliverable.** The owner explicitly values
  being told when a number does not support a claim. Three of today's most useful
  outputs were refutations.
- The owner is an ML PhD and jazz musician: fluent in each field separately, so
  **explain jargon at the intersection**, not within either domain. Be brief. He
  will say so, bluntly, if you are not.
