# Nightly agent runbook

This is the operating spec for the unattended nightly session on harmonia
(cron-triggered, ~22:00, ~4h budget). Read this file fully before doing
anything else. If anything here conflicts with a live instruction from the
user in the current conversation, the live instruction wins.

## Golden rules (non-negotiable, from the user directly)

- Never modify system/OS settings, never delete existing files.
- Never `git push`.
- Every change must be **traceable and recoverable**: the human must always
  be able to say "revert to the version that worked best" or "that used
  mechanism X" or "that displayed things this way" and get there in one
  command. See "Versioning & recoverability" below.
- Evaluation metrics used to justify any claim must be **clear, unbiased,
  and deliberately reasoned about** — a metric picked carelessly can burn
  hours "fixing" something that was never broken. See "Metrics discipline."
- Work must be decomposed into **nuclear tasks**: small, independently
  attackable and independently verifiable. Do not open-endedly attack a
  whole known_issues.md entry in one sitting.
- Before picking a task, **cross-check known_issues.md and suggestions.md
  against the actual current state of the model/code** — some entries may
  already be resolved or made irrelevant by recent commits and just not
  marked as such yet.
- If context gets saturated mid-run, compact or hand off to a fresh
  subagent per best practice (see "Context management") rather than
  degrading quality silently.

## Mission (three-tier priority — do not drift without a written reason)

### Tier 1 — Real-world chord inference quality (top priority, observed failures 2026-07-12)

Tested on iPhone with "Georgia On My Mind" (real YouTube audio, live production pipeline). Three
concrete failure modes identified, logged as **known_issues.md #20, #21, #22**. These are the
highest priority items for unattended sessions. **Attack in this order** (each unlocks the next):

1. **#20 — Diatonic scale prior per section (cheapest, clearest gain).**
   Root is often correct; family is wrong. The fix is a diatonic quality prior conditioned on
   the section key — a log-weight that makes maj/min/dom7/min7 prefer the diatonic quality for
   their scale degree, overridden only when acoustic confidence is high. See issue #20 for the
   full spec including the cheap premise check to run first.

2. **#22 — Section structure inference (AABA / A-B-bridge).**
   Section boundaries are not correctly inferred. A chord-level SSM (not audio SSM) + form-length
   prior (standard jazz forms: 8/16/32/64-bar) is the proposed approach. Literature survey is
   required before implementation — see issue #22.

3. **#21 — Chord bigram/trigram progression model.**
   After individual chords and sections are better anchored (#20, #22), a bigram coherence prior
   (from the iReal 2229-song corpus) + a small attention-based sequence model can catch remaining
   progression-level errors. This is the hardest of the three. See issue #21.

**These three are large tasks — each is assigned to a dedicated Opus subagent. See §Multi-agent
strategy below.**

### Tier 2 — End-to-end real-audio pipeline quality (ongoing)

Take a YouTube-style real-world recording and extract a clean, correct chord chart. Scope:
- **Audio→beat/chord detection accuracy** on real (non-synthetic) recordings — timbre, mixing,
  live performance, tempo/rubato (known_issues.md #19).
- **GT-to-video alignment quality** — a misaligned iReal Pro GT looks identical to a model error;
  verify alignment (DTW drift, wrong tune version, wrong key) before attributing disagreement to
  the model (CLAUDE.md rule #3).
- **Evaluation methodology** — bias in metric definition, eval-set composition, label format
  limitations (CLAUDE.md rule #3).
- **Proactive error-mode hunting** — failure patterns by chord family/key/tempo/recording style.
Relevant code: `harmonia/irealb_aligner.py`, `harmonia/irealb_fetcher.py`,
`scripts/build_yt_corpus.py`, `scripts/eval_yt_model.py`, `harmonia/data/yt_chord_corpus.py`,
plus any pipeline stage that affects real-audio extraction quality.

### Tier 3 — UX of the app (background)

`harmonia/output/chart_interactive.py`, `scripts/harmonia_server.py`, `docs/pwa/`. Should feel
fun/interactive (see CLAUDE.md collaboration conventions). Any visual state change must be logged
before it's changed (Neon Lights regression is the canonical failure — see [[feedback_ui_state_logging]]).

Only pick Tier 3 work when no Tier 1 or Tier 2 task is currently eligible — and say so explicitly.

---

## Multi-agent strategy (Tier 1 tasks)

Issues #20, #21, #22 each require substantial research + implementation. Each is assigned to a
**dedicated Claude Opus subagent** (use `model="opus"` in Agent spawn, or equivalent). Guidelines:

### Agent A — Diatonic prior (#20)
**Scope:** implement + evaluate the diatonic quality prior per section in `chord_pipeline_v1`.
**Deliverable:** metric delta on jazz1460 held-out 25 + "Georgia On My Mind" manual listen.
**Entry condition:** run the premise-check script first (fraction of GT chords that are diatonic).
If <60% diatonic, stop and report — the prior may not be worth implementing.
**Files:** `harmonia/models/chord_pipeline_v1.py`, `scripts/eval_irealb_e2e.py`.

### Agent B — Section structure (#22)
**Scope:** literature survey (3–5 papers, 30 min cap) + chord-SSM section detector + form-length
prior.
**Deliverable:** ARI / boundary-F on jazz1460 vs gmerge baseline; commit `harmonia/models/section_structure.py`.
**Entry condition:** confirm chord-SSM is cleaner than audio-SSM on ≥3 songs before building the
full detector.
**Files:** `harmonia/models/periodicity.py` (reference), `harmonia/models/chord_pipeline_v1.py`.

### Agent C — Bigram/attention progression model (#21)
**Scope:** bigram premise check → bigram matrix → (if premise passes) small attention encoder.
**Deliverable:** per-chord accuracy delta on jazz1460; `data/cache/chord_bigrams.npz`.
**Entry condition:** ≥70% chord pairs in top-50 bigrams (transpose-invariant). If premise fails,
stop and report — this task is premature.
**Files:** `scripts/train_online.py`, `harmonia/models/chord_pipeline_v1.py`.
**Prerequisite:** #20 and #22 should be evaluated before Agent C runs, to reduce error rate in
the chord stream that the bigram model ingests.

### Agent D — Cleanup & refacto (runs after A/B/C are verified)
**Scope:** dead code removal, naming consistency, docstring gaps, test coverage for new code.
**Entry condition:** A, B, C have all committed and their test suites pass. Agent D does NOT
introduce new functionality — strictly cleanup.

### Agent E — Token consumption audit (runs periodically, independently)
**Scope:** profile where token budget is spent in unattended runs. Identify:
- Redundant file reads (same file read multiple times per session without change).
- Overly long context in subagent prompts (missing content filters).
- Expensive tool calls with cheap alternatives (Bash `cat` vs Read, full file vs offset).
- Spots where compaction should fire earlier.
**Deliverable:** a short annotated report in `docs/nightly_runs.md` as a separate entry, listing
the top-3 token sinks and a concrete fix for each.
**Entry condition:** can run any night independently of other agents.

### Spawning protocol
1. Each agent **reads `docs/known_issues.md`, `docs/blog/`, and `git log --oneline -20` first**
   before starting (CLAUDE.md delegation convention).
2. Each agent writes its result to `docs/nightly_runs.md` + `docs/known_issues.md` in the
   standard format before exiting.
3. Agents may NOT start if pre-flight (§Pre-flight checklist) detects a concurrent session or
   disk < 10GB.
4. Agent D (cleanup) may only start once A, B, and C have each appended a verified-checkpoint
   entry to `docs/nightly_runs.md`.

## Pre-flight checklist (every run, before touching anything)

1. `df -h` the data volume — if free space < 10GB, stop immediately and
   write a report explaining that (see "Stop conditions").
2. `git log --oneline -40` and `git status` — look for commits or
   uncommitted changes that don't match the last known state (a sign
   another session is active or just finished). If detected, **do not
   touch anything** — write a short report noting what was found and stop.
   (This project runs multiple concurrent Claude sessions; silently
   clobbering another session's work has happened before — the "Neon
   Lights UI" incident.)
3. Read `docs/known_issues.md` and `docs/suggestions.md` in full. For each
   OPEN entry relevant to the two focus areas, check whether recent commits
   (`git log`) or recent `docs/blog/` entries have already resolved or
   invalidated it. Update the entry's status inline if so — this is itself
   useful work if nothing else fits in the remaining budget.
4. Pick the highest-priority still-open, still-relevant entry in a focus
   area. State the one nuclear subtask you will attempt tonight, in one
   sentence, before writing any code.

## Metrics discipline

- Before trusting a metric to justify a change, ask: what does it actually
  measure, what does it discard, and is there a known bias (cf. CLAUDE.md
  rule #3 on POP909's functional-root GT, and iReal Pro > guitar tabs >
  model trust hierarchy). If the metric itself looks suspect, that becomes
  the task for tonight rather than something to work around.
- Reuse the **exact same eval invocation** (same script, same flags, same
  eval set) across nights for a given focus area so numbers are directly
  comparable run-to-run. If the invocation must change, say so explicitly
  in the report as a metric-definition change — never let it change
  silently.
- Report both strict and parent-family (hierarchical) accuracy where
  applicable (predicting maj7 when GT is maj should show as partial
  credit, not just a miss).
- State what the metric does NOT capture alongside the number, if relevant
  (cf. CLAUDE.md rule #4, state what a fix does not solve).

## Versioning & recoverability

- At the point a nuclear subtask is complete **and verified** (tests pass,
  or a diagnostic plot/listen confirms the intended effect):
  1. `git add` only the files that belong to that subtask.
  2. Commit with a message that states the mechanism in plain English, not
     just "fix X" — someone should be able to `git log` and know what
     mechanism was used without reading the diff.
  3. Create an annotated tag: `git tag -a nightly/YYYY-MM-DD-HHMM-<slug>
     -m "<one-line summary + key metric>"`. This is what makes "go back to
     the version that did X" a one-command operation regardless of what
     happens on `main` afterward.
- If a subtask is abandoned or makes things worse, do not force a commit to
  "show progress" — an honest null result belongs in the report, not in a
  misleading commit.
- Never rewrite history (`git commit --amend` on already-tagged commits,
  no rebase of shared history).

## Context management

If context usage is getting high mid-run and there's meaningful budget
left: write a full handoff (current state, what's been tried, what's next,
same standard as CLAUDE.md's "Handoffs to fresh sessions" convention — the
full reasoning trail, not a task list) into the in-progress
`docs/nightly_runs.md` entry, then compact or spawn a fresh subagent with
that handoff as its prompt. Don't silently keep going in a degraded state.

## Stop conditions (check on a ~30 min cadence)

Stop and write the mandatory report if any of:
- Time budget (~4h) reached.
- Free disk < 10GB.
- A concurrent session was detected during pre-flight (stop before
  starting, not mid-way).
- The nuclear subtask is done and verified, and there isn't a similarly
  cheap next subtask clearly worth starting with remaining budget.
- Genuinely blocked (missing dependency, ambiguous spec that needs the
  user) — better to stop and ask in the report than guess destructively.

## Reporting (mandatory every run, even on a null result)

1. Append one entry to `docs/nightly_runs.md` following the schema defined
   at the top of that file.
2. If the result is significant (a real metric move, a real UX
   improvement, a real bug root-caused), also add a narrative entry to
   `docs/blog/` in the existing devlog style.
3. Update `docs/known_issues.md` / `docs/suggestions.md` status for
   anything resolved, advanced, or found obsolete during pre-flight or
   execution.
