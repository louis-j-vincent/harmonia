# Handoff — RESUME the staged rewrite orchestration (Phases 3–8) (2026-07-22)

_You are the orchestrator for the **staged Harmonia rewrite** (the systematic port
of the monolith into modules). This is a DELEGATED thread: a concurrent session is
keeping the **grid-unification** thread and owns part of the codebase (see the hard
boundary below). Your job is the rewrite plan's remaining phases._

## ⚠ CONCURRENT-SESSION FILE-OWNERSHIP BOUNDARY (read this FIRST — collision safety)
Another live session (the one that wrote this handoff) is actively working the
**grid-unification** thread and **OWNS these — do NOT edit them**:
- `harmonia/models/chord_pipeline_v1.py` — specifically the **segmentation / chord-
  change / decode region**: `_infer_nnls24` (segmentation + native-bargrid block),
  `_label_segments`, `_coalesce_labeled`, `segment_source`, `_flux_anchored_bar_root`,
  the Occam pass, and the `HARMONIA_NATIVE_BARGRID` code.
- `harmonia/models/beat_grid.py`, `harmonia/models/downbeat_anchor.py`.
- `harmonia/output/app_shell.html` (the key-viz work) + `harmonia/output/chart_interactive.py`.
- Also OFF-LIMITS (halted human sessions' uncommitted WIP): `harmonia/output/chart_model.py`,
  `harmonia/theory/local_key.py`, `harmonia/models/local_key_context.py`.
- Also OFF-LIMITS: `harmonia/models/harmonic_downbeat.py` (a parallel agent's downbeat brick).

**Because `chord_pipeline_v1.py` is a 10k-line monolith shared by both threads,
prioritize rewrite phases that DON'T require the segmentation region.** Good
low-collision starting points: **Phase 6 `serving/` PORT** (`scripts/harmonia_server.py`
→ `serving/api.py`+`render.py`+`cache.py` — mostly disjoint from the decode region),
and any corpus/data/alignment phases. If a phase genuinely needs the owned region,
STOP and coordinate (log a `❓ QUESTION FOR LOUIS` / note for the grid session) rather
than editing it. Never `git add -A` — only your specific files.

## Read first, in order
1. `CLAUDE.md` — 6 error patterns + env gotchas (esp. the **stale-clone trap**:
   cwd=canonical repo AND `PYTHONPATH=$PWD`; verify `harmonia.__file__` is under the
   canonical repo, not `~/harmonia`).
2. `docs/rewrite_execution_plan_2026_07_21.md` — **§3 is your plan** (the 8 phases).
   §0 two-net principle, §1 GT-provenance trap, §2 parity-mismatch decision tree.
3. `docs/handoff_2026_07_22_grid_unification.md` — full state of everything this
   session did (both threads); background for you.
4. `docs/known_issues.md` "PHASE 1.2/1.3 GATE" + "PHASE 2 STEP 1..8" — the detailed
   trail. `docs/handoff_2026_07_21_phase1_gate.md` — the original rewrite handoff.

## What's DONE + committed (verify `git log --oneline -25`)
- Phase 0 (harness+skeleton+calibration), Phase 1.1 (audio loader PORT), **1.2 GREEN**
  (2a9ca16, BP48 wrapper byte-identical), **1.3 GREEN** (b17236a, corpus_schema enum).
- **Phase 2 (beat_grid+alignment) is RESOLVED**: native bar grid built + gated
  (146d46f, kill-switch `HARMONIA_NATIVE_BARGRID` default OFF); the chord audit proved
  it is **DISPLAY-only, not a chord-accuracy win** (STEP 7). Decision: keep OFF now,
  **flip ON when everything ships**. This phase is the grid session's territory —
  don't reopen it.
- Repo hygiene fixed (untracked `harmonia/data/*.py` now tracked e5963b5; anchored
  gitignore bb279ba); a backwards CLAUDE.md calibration note corrected (song 002 ~64 BPM).

## Your job: Phases 3–8 (the un-started port)
Work them per plan §3, in an order that respects the ownership boundary above (start
with the least-overlapping — `serving/` PORT is a strong first pick). Each phase is a
PORT unless the plan says REDESIGN → the gate is **byte-identical old-vs-new on the
frozen benchmark** (two-net principle); commit at each green gate. Delegate the
reading/porting/parity-capture to Opus subagents (each researches history first,
returns a tight diff table + verdict). You hold only the gate decision, the commit,
plan order, and the one-paragraph state you log to `known_issues.md`.

## Locked directives (do NOT re-litigate)
- **RWC is the REFERENCE benchmark; POP909 is complementary.** ⚠ RWC full-song AUDIO
  is GONE (empty dirs; licensed; only `docs/audio/rwc_rwc_p001.m4a` survives) → RWC =
  chord-accuracy-from-cached-features only; audio-dependent tests use
  `docs/audio`+aligned_corpus.
- Scope = everything (pipeline, serving, scripts, scratchpad, docs). musx vendored as-is.
- Out of scope: symbolic learned-similarity thread; Buisson audio-native segmentation;
  growing aligned_corpus past 148.

## ⚠ OVERNIGHT UNATTENDED OPERATION (Louis is asleep — no human until morning)
Optimize for going as far as possible WITHOUT messing up, and for surviving a long run.

- **Never block on Louis.** Any decision you cannot make from the repo/plan → write a
  `❓ QUESTION FOR LOUIS` in `known_issues.md` and CONTINUE with the next in-scope
  phase. A question is never a stop. Do not wait.
- **Context discipline is survival — delegate EVERYTHING.** All file reading, porting,
  parity captures, investigations → Opus subagents (each researches history first,
  returns a TIGHT result). Keep in YOUR own context ONLY: the current gate decision,
  the commit call, plan order, and the one-paragraph state you write to
  `known_issues.md`. Never read a large file yourself; never write a module by hand.
  If you catch yourself doing either, stop — it was a subagent's job. This is the one
  lever that lets you run many phases before compacting.
- **Checkpoint continuously so compaction is a non-event.** After every green gate:
  write state to `known_issues.md` + commit specific files. If a compaction fires you
  must be able to resume from `known_issues.md` + git ALONE. Compact/checkpoint ONLY at
  gate boundaries, never mid-port.
- **Go as far as possible.** After each green gate, immediately start the next phase —
  no confirmation pauses. Prefer several small PORT phases to one big one.
- **Do NOT mess up (hard safety bar, unattended):**
  - Gate every PORT on **byte-identical old-vs-new** on the frozen benchmark before
    advancing (two-net). NEVER advance a red gate — log it and move to an independent phase.
  - Commit specific files only; **never `git add -A`, never `--no-verify`**. Respect the
    file-ownership boundary above (the grid session's files are OFF-LIMITS).
  - Any behavior change goes behind a **kill-switch, default OFF**.
  - **Honesty bar:** every number from a real run. A fabrication was caught by audit
    before — if you cannot verify a gate, say so and do NOT claim success.
  - **Nothing destructive or irreversible unattended:** no force-push, no history
    rewrite, no mass/`rm -rf` deletes, no dependency upgrades, no touching another
    session's uncommitted WIP. When unsure whether an action is safe to do unattended,
    DON'T — log it as a question and continue elsewhere.
- Disk has run tight (~1.9 GiB) — `df -h .` before big runs, render one wav at a time +
  delete each, and include a disk check in your cadence (a real disk-full has happened).

## Doctrine
Orchestrator delegates by default; commit specific files at green gates (no
`--no-verify`, never `git add -A`); every number from a real run; questions →
`known_issues.md` as `❓ QUESTION FOR LOUIS` and keep advancing; compact at gate
boundaries.
