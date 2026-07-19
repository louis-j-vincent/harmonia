---
name: research-loop
description: >
  Hypothesis-driven, budget-persistent research doctrine for open-ended
  harmonia questions worked interactively (foreground, no subagent) — e.g.
  "push metric X toward target Y", "figure out why Z plateaus", "find a
  workaround". Screens the premise cheaply before committing budget, treats
  every negative result as a diagnosis to act on rather than a dead end, and
  keeps going until the target is hit or the stated budget runs out — never
  stops silently on "I tried the obvious thing." Use when the user asks for
  sustained iteration toward a number, not a single well-specified change.
  Companion to the `.claude/agents/harmonia-researcher.md` subagent, which
  runs the same doctrine unattended/backgrounded — use this skill instead
  when the user wants to stay interactive and steer live.
user-invocable: true
---

# research-loop — sustained hypothesis-driven iteration

This is not a one-shot fix procedure. It's for sessions where the user has
given a target and a budget and wants genuine iteration until one of those
runs out — not a single attempt followed by a status report.

## 0. Premise check FIRST (mandatory, before any real work)

Per CLAUDE.md rule #2: write and run the cheapest possible check that could
falsify the current plan's core assumption, before investing real budget.
This includes checking whether the *target itself* is well-posed:

- Does prior project history (`docs/known_issues.md`, `docs/blog/`, recent
  handoffs) already establish a ceiling that conflicts with the target? If
  so, say that out loud to the user rather than quietly attempting past it
  or quietly lowering the bar.
- Is the metric/eval setup itself trustworthy, or could a "ceiling" actually
  be a metric artifact (wrong granularity, mislabeled GT, an oracle computed
  under an assumption that no longer holds)? When in doubt, audit the eval
  code and one worked example by hand before trusting any existing number,
  including numbers from a previous session's own findings.
- Confirm any reused baseline still reproduces (rule #1 — environment drift
  is real) before treating it as a fixed reference point.

If the premise check itself falsifies or reframes the plan, that IS the
session's first real result — log it (step 3) and adjust before continuing,
don't discard it as a false start.

## 1. Confirm the brief

Before iterating, state back explicitly: target metric + threshold, budget
(time or attempt count), integration point / constraints, and what's already
been tried. If any of this is missing, ask — don't assume.

## 2. Hypothesis-driven loop (the core discipline)

**A negative result is data about the mechanism, not a verdict on the
direction.** For every experiment that fails to move the metric:

1. Root-cause it — inspect intermediate outputs, not just the final number
   (CLAUDE.md rule #6: a component swap or config change can move more than
   the target metric; diff everything).
2. Write an explicit hypothesis for *why* it failed, before designing the
   next test.
3. Design the next experiment to isolate or address that specific
   hypothesized cause — not a repeat of the same idea, and not a pivot to
   something unrelated.
4. Only abandon a direction after a genuine, principled attempt to address
   the diagnosed failure mode — and when you do, write down specifically
   what is now known not to work and why (rule #4), so it isn't re-tried.
5. Treat single-example findings as hypothesis sources only; validate
   anything real at corpus/dataset scale before trusting it (rule #5).
6. Produce an inspectable artifact (plot, rendered output, table with real
   numbers) for every claimed improvement before asserting it worked —
   metric deltas alone are not evidence.
7. External research (web search, literature) is fair game when a failure
   points at a domain-knowledge gap — verify anything pulled in against this
   project's actual data before trusting it.

Loop this — hypothesize, test, diagnose, refine — until the target is hit or
budget is exhausted. The loop is the deliverable.

## 3. Log continuously, not at the end

Write findings to `docs/known_issues.md` (search tag pattern already used in
this project, e.g. `★ STRUCTURE / SEGMENTATION`) immediately after each
nontrivial result — bug root-caused, sweep finished, hypothesis confirmed or
falsified — not batched to session end. For a multi-step investigation,
consider a running scratch log too (`docs/research_sessions/` or similar) so
mid-session state is auditable without waiting for a final summary.

## 4. Stopping conditions

Stop when EITHER the target is met with real (not single-example) evidence,
or the stated budget is consumed. Budget exhaustion without hitting target
is a normal, reportable outcome — not a failure to hide and not a reason to
keep going unbounded. Report exactly where things landed, what's now known
not to work and why, and a concrete next-step recommendation.

Escalate to the user immediately (don't keep spending budget) if the premise
check or a later finding reveals the target is unreachable given a hard
structural constraint (e.g. the data source doesn't carry the needed
signal) — that's a scope decision, not something to resolve unilaterally.
