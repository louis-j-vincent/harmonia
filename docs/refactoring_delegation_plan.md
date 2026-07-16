# Refactoring delegation plan — Harmonia

_Written 2026-07-15 (Opus). This is an execution plan for a LATER session to run.
Nothing here was executed. Read `docs/refactoring_suggestions.md` first — this doc
assumes its findings._

**Guiding principle (from CLAUDE.md errors #1 and #6):** every change in this repo
that "looked cleaner" and silently changed a downstream number has cost real time.
So this plan is not organized around aesthetics — it is organized around
**behavior-preservation with a verification gate between every phase**. No phase
starts until the previous phase's gate is green.

---

## Non-negotiable rules for every delegated agent

1. **Research first.** Before touching anything, read `docs/known_issues.md` (the
   relevant tail), recent `git log`, and `docs/blog/` — same discipline the user
   uses. The user runs concurrent Claude sessions on this repo; check for
   unfamiliar recent changes before overwriting anything (esp. anything under
   `harmonia/output/chart_interactive.py` and the UI/`*.dc.html` surface).
2. **Do NOT touch actively-iterated surfaces.** Per CLAUDE.md commit-timing: if the
   user is iterating on it in chat, hands off. Treat as high-care / do-not-move
   without explicit go: `harmonia/output/chart_interactive.py`, `harmonia_server.py`,
   anything under the UI/`*.dc.html` / `handoff*/` surface, and the current
   in-flight `train_*` / `bass_root` / `bridge` work (recent uncommitted files:
   `scripts/train_real_audio_final.py`, `scratchpad/bridge_*`, `scratchpad/extract_bp48_absolute.py`).
3. **Never move + edit in the same diff.** A file move is one reviewable diff; a
   behavior change is another. Mixing them makes "did behavior change?" unanswerable.
4. **Small, independently verifiable diffs.** One phase = one agent = one reviewable
   PR-sized change with its own green gate. No "rewrite everything" agents.
5. **Commit timing:** commit at a phase gate only when the gate is green AND the
   user has signaled the round is settled (CLAUDE.md). Writing files is always fine;
   `git commit` waits.

---

## Phase 0 — Regression harness FIRST (before moving anything)

**Goal:** capture the CURRENT behavior of the load-bearing numeric paths so every
later refactor can be proven behavior-preserving, not just "looks cleaner." This is
the phase that makes the rest safe; do not skip it or reorder it.

**Work:**
- Add a `tests/conftest.py` fixture set if needed and write **characterization
  tests** (golden-value, pin current output) for:
  1. `harmonia/models/stage1_pitch.py` feature extraction on one committed short
     audio fixture — pin frame rate, output shape, and a checksum/first-few-values
     of the chroma. (error pattern #1)
  2. `harmonia/eval/mirex_eval.py` — pin strict + partial-credit scores on a small
     fixed (pred, GT) pair. (pattern #6)
  3. One end-to-end `HarmoniaPipeline.run()` on a short fixture — pin the chord
     labels + count + tempo. This is the "the shippable pipeline still runs" gate.
  4. `harmonia/data/billboard_translator.py` + `pop909_parser.py` translation on a
     handful of known-tricky labels (`/bass`, colon-quality, maj7 family). (pattern #3)
- Confirm the existing 380 tests still pass on an untouched tree (baseline).

**Gate:** new characterization tests + all 380 existing tests green on a tree with
**no source changes**. If a characterization test can't be written cheaply for a
path, that path is too tangled to refactor safely yet — note it and skip refactoring
it, don't refactor it blind.

**Scope:** ~1 focused session (1 agent).

---

## Phase 1 — Single source of truth for the corpus `match` schema (fixes the trigger bug)

**Goal:** make the §2a bug structurally impossible. Highest value, well-contained.

**Work:**
- New `harmonia/data/corpus_schema.py`: a `MatchQuality` enum (`EXACT`, `FAMILY`,
  `MISMATCH`, `NONE`, and a decision on `billboard_gt` — either alias it to `EXACT`
  or promote it to a real level, user's call), plus `filter_by_match(records, minimum)`
  and `save_corpus`/`load_corpus` with a fixed documented key set.
- **`load_corpus` raises on an unknown `match` value** rather than silently
  filtering it out — the specific defense against the reported bug.
- Add a test: unknown match value → error; round-trip write/read preserves keys.
- Then, one trainer at a time (separate diffs), replace `match == "exact"` literals
  with `filter_by_match(...)`: `train_real_audio_final.py:179`,
  `train_yt_exact_matches.py:121`, `train_yt_real_audio.py:201`, and the billboard
  builders in `scratchpad/`. Each swap is verified by re-running that trainer's
  data-load step and asserting the **row count is unchanged** vs the literal filter
  (behavior-preserving) — except where the old count was the bug, which is then
  documented as an intended change.

**Gate:** new schema tests green; each edited trainer loads the same #rows as before
(or a documented, intended increase). Phase-0 tests still green.

**Scope:** 1 agent for the module + tests; then 1 small diff per trainer (~4–5
trivially reviewable diffs). Document in the schema docstring what it does NOT solve
(CLAUDE.md rule #4) — e.g. it does not validate feature dtypes/shape (that's Phase 2).

---

## Phase 2 — Centralize feature extraction

**Goal:** collapse the 23 inline-chroma + 32 inline-basic_pitch call sites onto one
entry point, closing the pattern-#1 calibration-drift surface.

**Work:**
- New `harmonia/data/features.py` — a thin, documented wrapper over the canonical
  `stage1_pitch` path exposing the exact operation corpora/trainers need.
- Migrate inline call sites **in the current/active scripts only** (leave
  archived/dead scripts alone — Phase 3 removes them). Each migration is verified by
  the Phase-0 feature-extraction characterization test: the wrapper must reproduce
  the pinned chroma bit-for-bit.

**Gate:** feature characterization test green; migrated scripts produce identical
corpora (checksum a rebuilt corpus vs a pre-change one). This is a CLAUDE.md rule #6
"diff ALL intermediate outputs" checkpoint, not just the headline metric.

**Scope:** 1 agent for the wrapper, then batched small migration diffs. Medium
effort — this touches many files but each edit is mechanical and test-guarded.

---

## Phase 3 — Archive dead code + fix the broken bits (pure moves, no logic)

**Goal:** cut the `scripts/`/`docs/` namespace down to what's current. Zero behavior
change.

**Work (each a separate move-only diff):**
- Identify the current-best trainer/eval path **in writing** (in the new
  `docs/STATE.md`) before archiving its predecessors — otherwise you can't tell
  which `train_*_v3` is the keeper.
- `git mv` superseded `train_*_v2/v3/v4`, one-shot `migrate_*`, and dead experiments
  into `scripts/archive/`. Verify: nothing in the kept set imports the moved set
  (`grep` gate).
- Remove/relocate root clutter: `Taking over app UIUX.zip`, `alignment_comparison.html`,
  `harmonia.html`, `.coverage`; decide on `handoff 2/` and the `*.dc.html`.
- Add `scratchpad/*.npz`, `*.log` to `.gitignore`; untrack `root_posteriors.npz`.
- Fix or remove the broken `[project.scripts] harmonia = "harmonia.cli:main"`
  (no `harmonia/cli.py` exists).
- Remove `harmonia/models/block_fold.py` if git history confirms it's dead (0
  importers measured).

**Gate:** full test suite green (moves shouldn't affect it); `pip install -e .`
succeeds; a `grep` proves no kept file imports an archived one.

**Scope:** 1 agent. Low risk, high readability payoff. Pure moves — easiest to review.

---

## Phase 4 — Docs: split known_issues + build STATE.md

**Goal:** fix the "what's true right now?" problem (§5 of the suggestions doc).

**Work:**
- Create `docs/STATE.md` (short): current best model + weights path, current best
  dataset + path, compact open-issues table (id/line/status/next). Hand-curated from
  the tail of `known_issues.md`; resolve the Billboard blocked-vs-unblocked
  contradiction explicitly.
- Rename `docs/known_issues.md` → `docs/known_issues_log.md`; add a header pointing
  readers to STATE.md for current status. Keep it append-only.
- `docs/archive/`: move `MISSION_*`, `PHASE_2_*`, per-experiment result dumps, and
  the 4-line `billboard_training_results_v2.md`. Leave `architecture_extensions.md`,
  `suggestions.md`, `blog/` in place.
- Update CLAUDE.md's "Where things live" pointer (known_issues.md → STATE.md +
  known_issues_log.md). **This is a CLAUDE.md edit — get explicit user sign-off.**

**Gate:** user reviews STATE.md for correctness (only the user knows the current
truth). No code touched, so no test gate — but this phase must not run unattended;
it needs the user's read.

**Scope:** 1 agent to draft, but user-in-the-loop. Cheap mechanically, needs human
judgment on "what's current."

---

## What must NOT be touched (or handled with extra care)

- `harmonia/output/chart_interactive.py` and the UI/`*.dc.html` surface — has
  regressed silently before with no recovery path (CLAUDE.md). Do not move/edit as
  part of this refactor without explicit user direction.
- `scripts/harmonia_server.py` (7 kLOC) — its own subsystem; out of scope here.
- Any file the user is actively iterating on in chat, and the current in-flight
  bridge/bass-root/real-audio work (recent uncommitted files listed in rule 2).
- The `harmonia/` package's directory structure — it's already the right shape; only
  ADD `corpus_schema.py`/`features.py`, don't reorganize existing modules.

---

## Rough total scope

| Phase | Effort | Risk | Can run unattended? |
|---|---|---|---|
| 0 Regression harness | ~1 session | low | yes (gate is green tests) |
| 1 Corpus schema SoT | ~1 session + 4–5 tiny diffs | low–med | yes |
| 2 Feature extraction | ~1–2 sessions | **med** (touches many files) | yes, test-guarded |
| 3 Archive/dead-code | ~1 session | low | yes |
| 4 Docs split | ~half session | low | **no** — needs user review |

**Total: roughly 5 delegated agents / 4–6 focused sessions.** Phases 0→1→3→4 are the
high-value, low-risk spine and could be greenlit as a batch; Phase 2 is the biggest
single chunk and the user may want to see Phase 1 land first before committing to it.
None of this is a rewrite — the package stays as-is; the work is one SoT module, one
feature wrapper, a lot of `git mv`, and a docs split. The user can greenlight the
spine (0,1,3,4) and defer 2, or take it one phase at a time — each phase is
independently valuable and leaves the repo in a working state.
