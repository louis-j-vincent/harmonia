# Handoff — ground-up rewrite of the Harmonia pipeline (2026-07-21)

_You are the fresh orchestrator for a multi-week, staged rewrite of this project.
The prior session's context got too loaded to keep owning it, so this doc IS the
handoff — read it plus the three docs it points to, and you have essentially
everything the prior session held. Written at the project's handoff standard:
ranked priorities, explicit try-order, per-phase start/stop criteria._

## Read these first, in this order (do not skip — the reasoning matters, not just the task list)
1. `CLAUDE.md` (repo root) — 6 hard-won error patterns + environment gotchas.
   Non-negotiable. The whole rewrite plan is shaped around error patterns #1
   (silent calibration bugs) and #6 (component swaps change more than the target).
2. `docs/rewrite_execution_plan_2026_07_21.md` — **the plan you are executing.**
   The two-net principle (§0), the GT-provenance trap (§1, critical), the
   parity-mismatch decision tree (§2), and the 8 phases (§3). Everything below is
   a pointer into this.
3. `docs/production_model_audit_2026_07_21.md` — where the current live model is
   weak, with real numbers. This is why the phase order is what it is.
4. `docs/architecture_proposal_2026_07_21.md` — the target architecture (the
   what/why behind the plan).

## How Louis wants you to work (explicit, 2026-07-21)
- **Work autonomously.** You may ask him questions and he may steer you, but **if
  he is not available to answer, do NOT wait — keep advancing.** Surface questions
  as `❓ QUESTION FOR LOUIS:` entries in `docs/known_issues.md` (he watches it
  live) and continue other in-scope work in parallel. A question is never a gate.
- **Log continuously** to `docs/known_issues.md` — after every gate result, every
  bug found, every rejected hypothesis — not batched at the end.
- **Commit at each green gate** (specific files, never `git add -A`, no
  `--no-verify`, clear messages). Push when a phase's gate is green.
- **Run everything synchronously/foreground.** This project has a repeated, costly
  failure mode of agents launching a background job and ending their turn "waiting
  for a monitor." Block on your own tool calls until you have a real result.
- **Delegate phase-by-phase** to subagents if useful, but each must research
  history first and stay inside one phase with its own green gate. You (the
  orchestrator) hold the plan and check the gates.
- **Honesty bar**: every number traces to a real run; multi-seed for any headline
  claim. This project had a real fabrication incident caught by audit — that
  standard holds.

## The three things the prior session held that are NOT fully in the docs
1. **DISK IS THE FIRST BLOCKER.** Disk is at ~200 MiB / 99% full — **external
   cause** (other concurrent sessions), NOT this pipeline (its temp files are
   `rmtree`'d per-tune, verified). Phase 0 cannot safely capture golden outputs or
   stream RWC until this is cleared. **Bulk-deletion discipline is mandatory**
   (see memory `feedback_bulk_deletion_discipline.md` / CLAUDE.md): a real incident
   destroyed 10 legitimately-referenced `*.npz` cache files after sampling only 3
   filenames. Before any glob-delete, check EVERY matched filename individually
   against live references. Louis may clear disk himself — confirm current `df -h`
   state before assuming.
2. **An open judgment question for Louis is already logged** (audit §1 /
   `known_issues.md`): the jazz-vs-pop accuracy split (jazz 0.0-0.4 vs pop 0.6-0.8
   root acc). His ear on whether that's mis-ALIGNMENT vs mis-RECOGNITION would
   pre-load Phase 2 — but Phase 2 settles it empirically either way, so don't wait
   on it.
3. **A background audit agent exists and is resumable** (prior session's, agent id
   `a12aaa66bc65d18ff`) — it built `scratchpad/production_audit_root_quality_bass.py`
   (reusable for a larger real-audio run) and the two-net-relevant verifications.
   You don't need to resume it; noted so you don't rebuild what it made.

## Ranked priorities / try-order
**P0 — clear/confirm disk, then Phase 0 (foundation & safety net).** Nothing can
be gated without: frozen benchmark + per-stage golden outputs + the golden-diff
harness (`harmonia/eval/parity.py`) + the calibration unit-test layer. Start here.

**P1 — Phase 1 (`core/audio`, `core/features`, `data/corpus_schema`).** PORT.
Everything downstream depends on canonical feature extraction + corpus schema.

**P2 — Phase 2 (`beat_grid` + `alignment`). REDESIGN, deliberately before chords.**
This is the highest-value phase and the reason the order is non-obvious. The
audit's chord numbers are contaminated by alignment (scoring happens at aligned
timestamps), so alignment is fixed and re-measured BEFORE chord heads are touched.
Gate on INDEPENDENT GT only — RWC-AIST beats + POP909 downbeats, **never
`aligned_corpus`** (its timing is the alignment output; circular). Investigate the
6/14 octave-lock cases and jazz first-beat failures explicitly. This is where
Louis expects real bugs — treat mismatches as leads (decision tree, plan §2).

**P3 → P8** in the plan's order (chords → key/structure → pipeline → serving →
scripts/scratchpad → docs). Each with its PORT/REDESIGN tag and gate as written.

## Per-phase start/stop criteria (the short version — full detail in the plan)
- **Start a phase** only when the previous phase's gate is green and committed.
- **A PORT phase's gate is green** when new output == old output on the frozen
  benchmark (exact for labels, epsilon for float vectors) AND calibration pins
  stay green. A diff on a PORT stage is a bug — investigate which side (plan §2),
  don't advance red.
- **A REDESIGN phase's gate is green** when accuracy vs independent GT is ≥ the
  old behavior (or the regression is understood and accepted), multi-seed, with
  the finding written up.
- **Stop / escalate to a `❓ QUESTION FOR LOUIS`** when a mismatch needs his
  musical/theoretical judgment to classify (real GT error vs legitimate reading),
  and keep working other items while it's open.

## The four decisions Louis already locked (do not re-litigate)
- Scope = **everything** (pipeline, serving, all of `scripts/` + `scratchpad/`, docs).
- Parity benchmark = **RWC-Popular + `aligned_corpus` (148 songs)** — with the
  GT-provenance rule (plan §1): alignment measured only vs RWC-AIST/POP909.
- A parity mismatch is a **lead, not a verdict** (three outcomes, plan §2); he
  expects real bugs around alignment/first-beat.
- **musx stays vendored as-is.**

## Explicitly OUT of scope for this rewrite (flagged, not to be built now)
- The symbolic learned-similarity structure thread — its "+0.010" does not
  reproduce (9/9 fresh seeds negative); the adaptive agglomerative extension was
  re-falsified. Port the working block8 + `chordtone` path; don't chase this.
- Buisson et al. (audio-native self-supervised multi-level segmentation) is the
  one genuinely-new research lead — a future dedicated phase, weeks of work, NOT
  part of the port.
- Growing `aligned_corpus` further — Louis stopped it at 148 songs; enough.
