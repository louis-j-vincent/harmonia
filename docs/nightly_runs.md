# Nightly runs log

Append-only. One entry per unattended nightly session (see
`docs/nightly_agent_runbook.md` for the operating rules that produce these
entries). Do not edit past entries except to fix a typo — this file is the
source of truth for "what changed, when, and how to get back to it."

## Entry schema

```
## YYYY-MM-DD HH:MM — <one-line task title>

- **Git tag:** `nightly/YYYY-MM-DD-HHMM-slug` (commit `<sha>`, or "none — no
  verified checkpoint this run")
- **Focus area:** UX | YouTube-GT-alignment | other (justify if other)
- **Source issue:** known_issues.md #N / suggestions.md §X — relevance
  re-checked at pre-flight: <still valid | updated | resolved>
- **Nuclear subtask attempted:** <one sentence, decided before starting>
- **Mechanism / what changed:** <plain-English summary, not just a diff pointer>
- **Metrics (same schema every run for this focus area):**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|

- **What this does NOT solve / known caveats:**
- **Verification performed:** tests / plots / listening check / visual diff
- **Stop reason:** time budget | disk low | concurrent session detected | subtask done | blocked
- **Revert command:** `git checkout nightly/...` (or "n/a")
- **Next suggested step:**
```

---

## 2026-07-12 — diatonic-prior implementation (STOPPED: disk full)

- **Git tag:** none — no verified checkpoint this run
- **Focus area:** other — issue #20 diatonic quality prior
- **Source issue:** known_issues.md #20 — diatonic prior for chord family inference
- **Nuclear subtask attempted:** Implement diatonic log-prior on chord family prediction in chord_pipeline_v1.py
- **Mechanism / what changed:** nothing — stopped at pre-flight
- **Metrics (same schema every run for this focus area):**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|
  | — | — | — | — | stopped at pre-flight |

- **What this does NOT solve / known caveats:** n/a
- **Verification performed:** none — stopped at pre-flight
- **Stop reason:** disk low — only 2.4 GB free on /dev/disk3s5 (threshold: 10 GB). Pre-flight rule: abort if < 10 GB free. Free up disk space before re-running (check ~/harmonia/ stale clone, data/cache/*.npz, .venv, pip cache).
- **Revert command:** n/a
- **Next suggested step:** `du -sh ~/harmonia/ data/ .venv/ && pip cache purge` — clear stale clone and caches, then re-run the diatonic-prior nightly task.

---

## 2026-07-12 — chord-SSM section boundary detector (STOPPED: disk full)

- **Git tag:** none — no verified checkpoint this run
- **Focus area:** other — issue #22 section structure detection (AABA / A-B-Bridge)
- **Source issue:** known_issues.md #22 — global song structure inference is poor; gmerge detects ≤2-beat chord changes, not 8-16 bar section boundaries
- **Nuclear subtask attempted:** Implement chord-SSM-based section boundary detector in `harmonia/models/section_structure.py` and integrate into `infer_chords_v1`
- **Mechanism / what changed:** nothing — stopped at pre-flight
- **Metrics (same schema every run for this focus area):**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|
  | — | — | — | — | stopped at pre-flight |

- **What this does NOT solve / known caveats:** n/a
- **Verification performed:** none — stopped at pre-flight
- **Stop reason:** disk low — only 2.4 GB free on /dev/disk3s5 (threshold: 10 GB). Pre-flight rule: abort if < 10 GB free. There is also a prior stopped session from this same date (diatonic-prior) with the same root cause. Free up disk space before re-running.
- **Revert command:** n/a
- **Next suggested step:** Free disk space first (`du -sh ~/harmonia/ data/ data/cache/ .venv/`; consider clearing `~/harmonia/` stale clone, `data/cache/*.npz`, pip cache), then re-run this section-structure task. Implementation plan is ready: (1) lit review on chord-SSM / MSAF / Foote novelty, (2) premise check on 3–5 jazz1460 songs (chord-SSM vs audio-SSM diagonal clarity at section boundaries), (3) if premise passes → `harmonia/models/section_structure.py` with `build_chord_ssm` + `detect_section_boundaries`, wired into `infer_chords_v1` as post-processing, (4) eval boundary-F vs gmerge baseline on jazz1460 held-out songs ≥70.

---

## 2026-07-12 — Token consumption audit (Agent E)

- **Git tag:** none — no code changes
- **Focus area:** other — token optimization (Agent E per runbook)
- **Source issue:** `docs/nightly_agent_runbook.md` §Agent E — "profile where token budget is spent in unattended runs"
- **Nuclear subtask attempted:** Profile top-3 token sinks in nightly agent sessions and propose concrete fixes for each.
- **Mechanism / what changed:** Read-only analysis of file sizes, line counts, and runbook access patterns. No code or model files modified.
- **Metrics (same schema every run for this focus area):**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|
  | — | — | — | n/a — read-only audit | n/a |

- **Token audit findings:**

  Token estimates use chars/4 ≈ tokens (GPT-style tiktoken approximation). "Frequency" = how many times the file is read per nightly session across all spawned agents (main + A + B + C + D or E).

  | sink | size | tokens/read | frequency (reads/session) | estimated tokens/session |
  |---|---|---|---|---|
  | `docs/known_issues.md` | 1,632 lines / 96 KB | ~24,000 | 5+ (main pre-flight + each subagent reads it per runbook §Spawning protocol §1) | **~120,000** |
  | `harmonia/models/chord_pipeline_v1.py` | 1,329 lines / 58 KB | ~14,500 | 3–4 (Agent A modifies it, Agent B references it, Agent D cleans it, orchestrator may read it) | **~50,000** |
  | `data/cache/yt_corpus/vid_cache.json` | 109 KB | ~27,900 | 1–2 (Tier 2 eval agents; risk is Bash `cat` or Read of the whole file for a single lookup) | **~28,000–56,000** |

  Secondary sinks (not top-3 but worth noting):
  - `docs/nightly_agent_runbook.md`: 222 lines / 12.5 KB, ~3,100 tokens × 5 reads = ~15,500 tokens/session — cheap individually but mandatory for every spawn.
  - `scripts/harmonia_server.py`: 2,098 lines / 91 KB, ~22,700 tokens — only read on Tier 3 nights but large; no module docstring to short-circuit.
  - `harmonia/models/chord_hmm.py`: 923 lines / 42 KB, ~10,600 tokens — frequently in agent context due to known_issues.md #0–#8 historical references, even though the module is FROZEN.

  **Root cause of `known_issues.md` bloat:** issues #0–#14 are all resolved/fixed (clearly marked), but the full investigation trails (code diffs, measurement tables, root-cause analysis) are preserved inline. An agent following the runbook reads all 96 KB even to check the three OPEN Tier-1 issues (#20, #21, #22) it actually needs.

- **Recommended fixes (ranked by impact):**

  1. **Add an `## ACTIVE ISSUES — QUICK REFERENCE` section to `docs/known_issues.md`** (~50 lines, immediately after the header preamble). List each open issue as a single line: `#N — title — status — next action`. Closed/resolved issues: one-liner "resolved, see §N". Update the runbook pre-flight instruction from "Read `docs/known_issues.md` ... in full" to "Read the ACTIVE ISSUES quick reference section; read a specific issue's full §N only when working on it." **Estimated saving: ~100,000 tokens/session (5 full reads → 5 short reads of ~2K tokens each).**

  2. **Constrain subagent prompts to use `offset`/`limit` on `chord_pipeline_v1.py`** rather than reading the full 1,329-line file. The module already has a good 30-line docstring (lines 1–30) that names all 10 pipeline stages. Agents should read lines 1–30 first to orient, then read only the function/section they're modifying. Add this as a standing instruction in each Agent A/B/C spawn prompt in the runbook. **Estimated saving: ~30,000–40,000 tokens/session.**

  3. **Prohibit full reads of `data/cache/yt_corpus/vid_cache.json` in agent prompts.** The file is 109 KB (27,900 tokens). For point lookups, use `jq '.[] | select(.video_id == "XYZ")'`; for listing all IDs, use `jq 'keys'`; for summary stats, use `jq 'length'`. Add a one-line warning to the runbook Tier-2 section and to Tier-2 subagent spawn prompts: "Never Read or cat vid_cache.json in full — use jq for point lookups." **Estimated saving: ~28,000–56,000 tokens per Tier-2 session.**

- **What this does NOT solve / known caveats:**
  - These are estimates from static analysis; actual token counts depend on which subagents are spawned on a given night and how their prompts are structured. Disk-full nights (all three prior runs were stopped at pre-flight) have near-zero token cost regardless.
  - Fix #1 requires a human or targeted agent to write and maintain the quick-reference section; it will drift if not updated alongside the main entries.
  - Fix #2 only helps if the orchestrating agent explicitly crafts targeted subagent prompts — a generic "read chord_pipeline_v1.py and fix X" prompt will still read the whole file.
  - The `docs/nightly_agent_runbook.md` itself (3,100 tokens × 5 reads = ~15,500 tokens) is not in the top-3 but could be compressed if the multi-agent strategy section were moved to a separate file.

- **Verification performed:** line counts via `wc -l`, byte sizes via `wc -c`, token estimates via chars/4, access pattern from runbook §Pre-flight and §Spawning protocol cross-referenced against file sizes.
- **Stop reason:** subtask done
- **Revert command:** n/a — docs-only append, no git tag needed
- **Next suggested step:** Implement fix #1 (add ACTIVE ISSUES quick reference to `docs/known_issues.md`) — a targeted 30-min agent task that does not touch any code and has immediate effect on every subsequent session.
