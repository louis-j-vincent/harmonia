# Handoff — rewrite orchestration + grid-unification thread (2026-07-22)

_You are the next orchestrator. Prior session took the staged rewrite through
Phase 1 (green) and Phase 2's beat_grid/alignment redesign (resolved), then went
deep on a user-driven investigation of chord/beat/bar grid unification. This doc
carries the full reasoning trail; the blow-by-blow is in `docs/known_issues.md`
"PHASE 2 STEP 1..8" (read those — they are authoritative and detailed)._

## Read first, in order
1. `CLAUDE.md` — 6 error patterns + env gotchas. Especially: **stale-clone trap**
   (run with cwd=canonical repo AND `PYTHONPATH=$PWD`; verify `harmonia.__file__`
   is under the canonical repo, NOT `~/harmonia` which lacks recent files).
2. `docs/known_issues.md`, sections **"PHASE 1.2/1.3 GATE"** and **"PHASE 2 STEP
   1..8"** (the tail) — the complete trail of everything below.
3. `docs/rewrite_execution_plan_2026_07_21.md` §3 — the 8-phase plan. Phase 2
   (beat_grid+alignment) is DONE/resolved; Phases 3–8 are the un-started port.
4. `docs/handoff_2026_07_21_phase1_gate.md` — the prior orchestration handoff.
5. Memories (auto-loaded): `feedback_rwc_for_tests`, `project_key_representation`,
   `project_harmonic_downbeat_brick`, `project_shippable_decisions`.

## YOUR ROLE (unchanged): orchestrator, delegate by default
Hold only: gate decisions, the commit-at-green-gate call, plan order, and the
one-paragraph state you write to `known_issues.md`. Delegate reading/coding/
capture to Opus subagents (each researches history first, returns a tight result).
Commit at each green gate (specific files, never `git add -A`, no `--no-verify`).
Autonomy: questions go to `known_issues.md` as `❓ QUESTION FOR LOUIS` and you keep
advancing. Every number from a real run. **Other human sessions are HALTED** (Louis
2026-07-21) — you may edit the production UI files now, BUT their uncommitted WIP is
on disk in `chart_model.py`, `app_shell.html`, `local_key.py`, `local_key_context.py`
— do NOT revert/clobber it; build with it.

## DONE + committed this session (verify `git log --oneline -20`)
- Phase 1.2 gate GREEN (2a9ca16, BP48 wrapper byte-identical), 1.3 GREEN (b17236a,
  corpus_schema enum).
- Repo hygiene: tracked 4 `harmonia/data/*.py` source modules that a bare-`data`
  gitignore had hidden (e5963b5, fresh clones were ImportError-ing on the core
  pipeline); anchored `/data`//`.venv` in the tracked `.gitignore` (bb279ba).
- Fixed a BACKWARDS calibration note in CLAUDE.md (song 002 GT is ~64 BPM, not 129).
- **Phase 2 native per-bar-downbeat bar grid** (146d46f, kill-switch
  `HARMONIA_NATIVE_BARGRID`, default OFF): pipeline_downbeat_F 0.292→0.647 on POP909.
  Gate green (9d610e2). `test_native_bargrid.py` 9/9.
- **KEY DECISION — the grid is DISPLAY-only, NOT chords** (a311704, STEP 7): a
  paired OFF-vs-ON chord audit shows the +0.355 does NOT reach chord accuracy (net
  −0.015 on grid-changing POP909; byte-identical on most; RWC-P001 flat). Root cause
  (STEP 8): the chord decode is beat/segment-level and **grid-independent**; the bar
  grid touches chords ONLY via the Occam loop-compression post-pass (where it's
  net-negative, 624 −0.19). **Chord bottleneck is UPSTREAM — features / musx labels /
  recording selection.**
- Key-viz: causal continuity track as single Python source — but committed to the
  WRONG surface (211a89f, `chart_interactive.py` = dev render). Production is
  `app_shell.html` + `chart_model.py` served by `harmonia_server.py`.

## Locked directives (do NOT re-litigate)
- **`HARMONIA_NATIVE_BARGRID`: keep OFF now** (comparison baseline) but **flip to
  default-ON when everything ships** (Louis) — after the render-contract + Occam×grid
  follow-ups. Flagged in STEP 7.
- **RWC is the REFERENCE benchmark; POP909 is complementary only.** ⚠ **RWC full-song
  AUDIO is GONE** (empty dirs; licensed/not re-downloadable; only
  `docs/audio/rwc_rwc_p001.m4a` survives). RWC benchmarks CHORD accuracy from cached
  features (`rwc_nnls24.npz`, 100 songs) but CANNOT run audio-dependent tests
  (grid/beat) → use `docs/audio` + aligned_corpus real audio for those.
- **Key representation:** causal hold-until-forced continuity, NO lookahead, NEVER
  per-chord; `min`=triad {0,3,7} everywhere.
- Louis wants **clear modular bricks + a UNIFICATION of chord-change vs beat/bar grids.**

## IN FLIGHT — background agents (you'll get completion notifications)
1. **Grid-unification experiment** (harmonia-researcher, agent id `a871106a40a760db3`):
   "Does aligning chord segmentation/decode to the Beat This! beat+downbeat grid
   improve root/quality/family accuracy?" Premise-screens FIRST (do GT chord changes
   concentrate on Beat This! downbeats?), then tests downbeat-informed segmentation +
   per-bar pooling, on docs/audio+aligned_corpus (sounding-bass target). **The
   premise-screen number is the key gate** — if GT changes don't align with downbeats,
   the whole idea is likely dead and that's a valid result. When it returns: log the
   result as STEP 9, and if a variant wins, plan its clean kill-switched integration.
2. **Downbeat-phase brick** (`harmonia/models/harmonic_downbeat.py`) — Louis launched
   this in a PARALLEL session from a prompt this session wrote. It owns downbeat PHASE
   (harmonic-rhythm "which beat is the 1"); premise screened (ramp scorer, top-decile
   precision 0.73). Do NOT touch that file or the phase-selection — it's a separate
   lane. When it ships, YOU wire its brick behind a kill-switch into `_infer_nnls24`.

## Ranked open work (try-order + stop criteria)
1. **Await the grid-unification result** → decide (STEP 9). Stop criterion: a variant
   beats baseline ≥ ~2pp root or family with no material regression → build it as a
   kill-switched brick; else log the negative and keep grids separate with a documented
   reason.
2. **Key-viz production correctness (small, high-value):** `app_shell.html`'s
   `continuityScaleTrackV2` (L176) is hold-until-forced (good) but defaults
   `lookahead=2` (L177) — Louis wants **causal (0)**. Fix: set lookahead=0, ideally
   feed from the Python single source (retire the JS copy). Verify the `fitsCollection`
   there is harmonic-minor-aware (no #23 Autumn-Leaves oscillation). Render a chart to
   confirm hold-until-forced holds and no per-chord flicker.
3. **Native-grid ship-ON prerequisites** (so the flip is safe when Louis ships):
   (a) render-contract follow-up — make the display actually consume `bar_times`
   (currently redraws a uniform grid at the single-int `grid_anchor_beats`); (b) audit
   the Occam×grid interaction or make Occam grid-invariant (else ON degrades songs like
   624). Only then is default-ON safe.
4. **RWC audio gap:** ❓ QUESTION FOR LOUIS pending — re-source RWC audio, or accept
   RWC = chord-accuracy-from-features only. Until resolved, audio-dependent "reference"
   tests use docs/audio+aligned_corpus.
5. **Resume the systematic rewrite (Phases 3–8)** per plan §3 — this session did NOT
   advance the port beyond Phase 2; pick it up if the investigation threads settle.

## Gotchas
- The `_infer_nnls24` decode: chord-change boundaries = NNLS root-argmax flips snapped
  to Beat This! BEATS (`seg_bounds=[(bt[s],bt[e])]`, chord_pipeline_v1.py:3612); labels
  = musx per-segment (`_label_segments`→`_coalesce_labeled`); downbeats/bars are a
  separate overlay (sections/Occam/display only). This decoupling is THE thing Louis
  wants closed.
- Disk has run tight (~1.9 GiB); render one wav at a time + delete each; `df -h .`
  before big runs; a real disk-full has happened before.
- Scratchpad reusables from this session: `bargrid_pop909_test.py`,
  `bargrid_rwc_test.py`, `bargrid_chord_audit.py` (+ rows), `parity_lkc.py`, and the
  grid A/B pages copied to `docs/plots/grid_ab/` (gitignored).
