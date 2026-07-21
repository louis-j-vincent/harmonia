# Handoff — Rewrite orchestration, Phase 1.2 gate → Phase 1.3 → Phase 2 (2026-07-21)

_You are the next orchestrator for the staged Harmonia rewrite. The prior
session (Phase 0 + 1.1 + 1.2 code) is done and committed; this doc carries the
full reasoning trail, not just a task list. Written at the project's handoff
standard: ranked priorities, explicit try-order, per-phase start/stop criteria._

## Read these first, in order (do not skip)
1. `CLAUDE.md` (repo root) — 6 hard-won error patterns + environment gotchas.
   The whole rewrite is shaped around #1 (silent calibration bugs) and #6
   (component swaps change more than the target metric).
2. `docs/rewrite_execution_plan_2026_07_21.md` — **the plan you are executing.**
   Two-net principle (§0), GT-provenance trap (§1, critical), parity-mismatch
   decision tree (§2), the 8 phases (§3).
3. `docs/handoff_2026_07_21_rewrite.md` — the original rewrite handoff (Phase 0
   entry). Still the authority on the four locked decisions + out-of-scope list.
4. `docs/known_issues.md` — search for "PHASE 0 COMPLETION", "PHASE 1.2 STATUS"
   (near the tail) for the exact state left by the prior session.

## What is DONE and committed (verify with `git log --oneline -6`)
```
22be68a  Phase 1.2 status update
035d380  Phase 1.2: Feature extraction abstraction + integration
daaad67  Phase 1.1: Audio loader PORT
209de77  Phase 0: Foundation & safety net (harness + skeleton + calibration)
```
- **Phase 0 GREEN**: `harmonia/eval/parity.py` (golden-diff harness),
  `tests/test_calibration_pins.py` (7/8 pins pass, 1 skipped — needs POP909
  audio), empty skeleton (`harmonia/core/`, `stages/`, `serving/`),
  `PipelineConfig` dataclass in `harmonia/pipeline.py` with `live_defaults()`.
- **Phase 1.1 DONE**: `harmonia/core/audio.py` (load_audio/get_duration_s),
  ported verbatim from `chord_pipeline_v1.py:2155`.
- **Phase 1.2 code DONE, GATE NOT YET RUN**: `harmonia/core/features.py`
  (FeatureExtractor ABC + factory + BasicPitch48/NNLS24/Musx impls + result
  types with `pool_to_beats`). Integrated into `chord_pipeline_v1.py:4304-4317`
  (replaced `PitchExtractor()` with `FeatureExtractor.create("bp48")`).
  **Only an import-level smoke test has run.** The real parity gate has not.

## ⚠️ YOUR FIRST JOB — close the Phase 1.2 parity gate (do NOT skip to 1.3)
The prior session changed the live BP48 feature path but never proved it
byte-identical. Per the two-net principle, a PORT stage is not green until
new == old on the frozen benchmark. Concretely:

1. Pick a **fixed 5-song subset** of `aligned_corpus` (data/cache/aligned_corpus/
   aligned_corpus.npz — 148 songs; `np.unique(corpus["song_id"])`). Write the
   chosen song_ids into `docs/known_issues.md` so the subset is frozen, not
   re-randomized each run.
2. Capture OLD output: `git stash` any dirt, checkout `daaad67` (pre-1.2),
   run `infer_chords_v1(audio, feature_frontend="bp48", ...live defaults...)`
   on each of the 5 songs, save each ChordChart JSON as golden.
3. Capture NEW output: checkout `035d380` (current), rerun the same 5.
4. Diff with `harmonia/eval/parity.py::diff_charts`. **Exact match on chord
   labels, epsilon on floats.**
   - **(a) match** → Phase 1.2 gate GREEN. Commit the golden JSONs + a
     `known_issues.md` entry. Advance to 1.3.
   - **(b) new differs, new is right** → a latent bug was surfaced; log it,
     update golden, keep new. (Unlikely on a pure BP48 wrapper — investigate hard.)
   - **(c) new differs, old is right** → regression in the wrapper. Most likely
     suspect: `ActivationResult.onsets` vs the old `acts.onset_probs`, or the
     `pool_to_beats` sum-pooling vs the old `_pool_beats` (they should be
     identical — diff them line by line). Fix before advancing. Do NOT advance red.
5. **Audio**: the 5 songs need real audio. Reuse cached `data/cache/nnls_infer/`
   or `data/cache/musx_infer/` stems if present; otherwise yt-dlp ~50 MB.
   **Check `df -h` first** — disk has been at ~200 MiB–2.3 Gi (external cause,
   other sessions). If <1 GB free, run the 5-song set one at a time, deleting
   each wav immediately (bulk-deletion discipline: verify each filename before rm).

## Then Phase 1.3 — `data/corpus_schema.py` (PORT, high value)
- Add `save_corpus`/`load_corpus` + a **match-value enum** replacing the
  free-string `"exact"/"family"/"billboard_gt"` (this silently dropped corpora
  to zero rows — CLAUDE.md #1 class of bug) + `filter_by_match`.
- Gate: every corpus builder/trainer rerouted through the new entry reproduces
  its corpus identically; calibration pins still green.
- `harmonia/data/corpus_schema.py` already exists (77 stmts) — read it first;
  this may be an extend-not-create.

## Then Phase 2 — `beat_grid` + `alignment` (REDESIGN, the high-value phase)
Only after Phase 1 is fully green. This is where Louis expects real bugs.
- Gate on **INDEPENDENT GT only**: RWC-AIST beats + POP909 downbeats. **Never
  `aligned_corpus`** for beat/alignment (its timing IS the alignment output —
  circular, plan §1).
- Explicitly investigate the audit's **6/14 octave-lock cases** and **jazz
  first-beat failures**. Each confirmed bug → fix + log (outcome (b)).
- Then re-run the chord audit with clean alignment and report the
  decomposition: how much of `root=0.525` was alignment contamination.

## How Louis wants you to work (unchanged from the original handoff)
- **Work autonomously.** Questions go to `docs/known_issues.md` as
  `❓ QUESTION FOR LOUIS:` and you keep advancing other in-scope work. A
  question is never a gate.
- **Log continuously** to `known_issues.md` after every gate/bug/rejected idea.
- **Commit at each green gate**, specific files, never `git add -A`, no
  `--no-verify`.
- **Run everything foreground/synchronous.** No "waiting on a background
  monitor" turn-ends.
- **Honesty bar**: every number from a real run; multi-seed for headline claims.
  A fabrication incident was caught by audit before — that standard holds.
- **musx stays vendored as-is.** Scope = everything (pipeline, serving, scripts,
  scratchpad, docs).

## ⚡ DELEGATION IS YOUR DEFAULT REFLEX — you are an orchestrator, not an implementer
This is the single most important instruction in this doc. **Your job is to
delegate.** You hold the plan and a high-level understanding of the state; you
do NOT read large files, write code, run captures, or grind through refactors
yourself. Every one of those is a subagent task. The prior session's best move —
by far — was handing the feature-extraction design to an Opus subagent that
returned a production-ready 360-line module in one shot, at near-zero cost to the
orchestrator's context. That is the template for essentially everything.

**Default to spawning an Opus subagent. Doing the work yourself is the exception
that needs a reason** (e.g. a one-line edit too small to be worth a spawn). When
in doubt, delegate.

- **Delegate (almost everything)**: "read `corpus_schema.py` + all 30 `np.savez`
  call sites and report the schema variants", "design the match-value enum +
  migration", "run the 5-song parity capture and report the diff table", "port
  the audio loader and prove it byte-identical". Each subagent MUST research
  history first (known_issues.md, git log, docs/blog) and return a tight result
  — a decision, a diff table, a committed file — not a transcript.
- **Keep in your OWN context (only this)**: the gate decision (a/b/c), the
  commit call at a green gate, the plan ordering, and the one-paragraph state
  summary you write to known_issues.md. That is the entire orchestrator job.
  If you find yourself reading a 2,000-line file or writing a module by hand,
  stop — that was supposed to be a subagent.
- **Why**: a subagent starts cold, burns ITS context on the bulky work, and
  hands you back only the conclusion. Your window barely grows, so you stay far
  under the compaction threshold and keep the high-level thread across many
  phases. Delegation is not just for hard tasks — it is how you preserve context.
- Subagents run in the background by default — you're notified on completion.
  Pass `run_in_background: false` when the result blocks your next decision
  (e.g. a parity capture that gates 1.2). Spawn several in parallel when tasks
  are independent (e.g. capture-old and read-corpus-schema at once).

## Context hygiene — compact BEFORE the threshold, never mid-gate
Louis wants you to stay well clear of the auto-compact threshold, not ride up
against it. You cannot trigger your own auto-compact (that is harness-driven),
but you can make it a non-event:
- **Checkpoint all state to disk continuously.** Frozen 5-song subset, gate
  decision (a/b/c), phase status, every rejected idea → `docs/known_issues.md`,
  committed. Never hold a fact only in conversation. If a compaction fired right
  now, the next window must be able to resume from git + known_issues.md alone.
- **Delegate anything read-heavy or design-heavy to Opus subagents.** They start
  cold, do the bulky work in THEIR context, and return a tight result — your own
  window barely grows. This is the single biggest lever for staying under the
  threshold.
- **End your turn cleanly at each green gate** with a one-paragraph "state now +
  next action" summary written to known_issues.md, and tell Louis a fresh window
  (or `/compact`) is a safe place to continue. Do not push a second phase in the
  same window if you are already past ~half your context — checkpoint and stop
  the turn instead. A gate boundary is the only safe compaction point; a
  half-ported stage is not.
- **If you notice your context filling mid-phase**, do not power through — write
  the partial state + exactly where you stopped to known_issues.md, commit, and
  end the turn recommending continuation in a fresh window.

## The four decisions Louis already locked (do NOT re-litigate)
- Scope = everything. Parity benchmark = RWC-Popular + aligned_corpus (148),
  with the GT-provenance rule (alignment measured only vs RWC-AIST/POP909).
- A parity mismatch is a lead, not a verdict (three outcomes). musx vendored as-is.

## Explicitly OUT of scope
- Symbolic learned-similarity structure thread (+0.010 does not reproduce, 9/9
  fresh seeds negative). Port working block8 + chordtone; don't chase this.
- Buisson et al. audio-native self-supervised segmentation — a future dedicated
  phase, NOT part of the port.
- Growing aligned_corpus past 148 songs — Louis stopped it there.
