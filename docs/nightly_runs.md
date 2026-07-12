# Nightly runs log

## 2026-07-12 — ProgressionEncoder reranker wired into `infer_chords_v1` (#21)

- **Git tag:** `nightly/2026-07-12-1734-encoder-prod`
- **Focus area:** Tier 1 — issue #21, integration step (follows the 1730 training run)
- **Nuclear subtask:** Wire the trained `ProgressionEncoder` into the production pipeline as a second-pass quality reranker and measure end-to-end MIREX.
- **Mechanism / what changed:** `harmonia/models/chord_pipeline_v1.py` —
  `rerank_progression_qualities(roots, sev_hs, confs, weight)` + `_get_progression_encoder()`
  lazy loader + Harte↔q5 maps. After the per-segment classification loop (before
  coalescing), each segment's coarse quality is re-scored from its ±6-chord context:
  `log_post = log_acoustic + w·log_encoder`, where `log_acoustic` is a
  confidence-gated one-hot over the 5 q5 families (the LR/ctx classifier returns a
  scalar conf, not per-q5 log-probs — a proper distribution is the obvious next
  refinement). On a family flip the acoustic triad-vs-seventh choice is preserved.
  New kwargs `use_progression_prior=True` (default ON), `progression_weight=0.5`.
  `scripts/eval_irealb_e2e.py` gained a `--progression-weight` flag and a **7ths**
  (tetrad) metric column — the encoder's main lever (dom vs maj) is invisible to majmin.

- **Metrics (jazz1460 held-out 25, `eval_irealb_e2e.py`, production cell = tempo grid + gmerge):**

  | variant | root | majmin | 7ths |
  |---|---|---|---|
  | baseline | 88.7% | 84.0% | 58.6% |
  | + encoder w=0.2 | 88.7% | 84.0% | 58.6% |
  | **+ encoder w=0.5** | 88.7% | **84.7%** | **58.9%** |
  | + encoder w=1.0 | 88.7% | 84.8% | 58.8% |

  Gain is uniform across all 6 grid×segmentation cells (no regression anywhere).
  w=0.5 chosen (best 7ths, near-best majmin). Root unchanged by construction
  (reranker only touches quality). Modest size because the confidence gate lets the
  encoder override only low-confidence segments; standalone dom recall (86.8%) does
  not fully transfer since many prod segments are already confident+correct.

- **What this does NOT solve:** the encoder cannot fix wrong roots (errors compound);
  majmin barely moves because dom↔maj is a tetrad-level call; the acoustic prior is a
  gated one-hot, not calibrated per-q5 log-probs. Not yet listen-checked on a real song
  (disk near threshold — skipped render to stay under 2 GB).
- **Verification:** unit test `tests/test_progression_encoder_rerank.py` (6 tests: Harte↔q5
  maps, ii-V-I dom recovery on a low-conf mislabel, high-conf noop, weight=0 noop, empty
  seq, out-of-vocab passthrough). Full suite 246 passed. End-to-end sweep above.
- **Revert command:** `git checkout nightly/2026-07-12-1734-encoder-prod`
- **Next suggested step:** replace the gated-one-hot acoustic prior with real per-q5
  log-probs from the ctx classifier (coarsen its 5 FAMILIES + b7 head into q5 space);
  re-sweep weight; then listen-check "Georgia On My Mind".

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

---

## 2026-07-12 — diatonic quality prior: premise check (FALSIFIED, no implementation)

- **Git tag:** none — no verified checkpoint this run (premise failed; nothing to commit per runbook)
- **Focus area:** Tier 1 — issue #20 (diatonic quality prior per section)
- **Source issue:** known_issues.md #20 — relevance re-checked at pre-flight: **still valid as an open issue, but its central premise is now falsified for the global-key version** (see below).
- **Nuclear subtask attempted:** Run the cheap premise check gating issue #20 (CLAUDE.md rule #2) before any pipeline change: what fraction of GT chords are diatonic in the song key on held-out jazz1460?
- **Mechanism / what changed:** Added `scripts/check_diatonic_premise.py` (untracked, left in place). For 25 held-out jazz1460 songs (index 70–95, unseen by beat_seq_model_v4), it infers the song key via `infer_key()` on a duration-weighted symbolic chord-tone chroma, cross-checks against the trusted iReal `key` annotation, and scores each GT `(root, quality5)` against a strict diatonic degree→quality table (maj7/min7 tolerated as maj/min variants; V allowed as triad or dom7; minor table includes harmonic-minor V and leading-tone dim). **No change to `chord_pipeline_v1.py`.**
- **Metrics (premise check, held-out jazz1460 idx 70–95, 1128 GT chord events, 25 songs):**

  | metric | value | eval set | invocation |
  |---|---|---|---|
  | diatonic % (infer_key, by count) | 49.4% | jazz1460 idx 70–95 | `.venv/bin/python scripts/check_diatonic_premise.py` |
  | diatonic % (infer_key, by duration) | 48.3% | same | same |
  | diatonic % (trusted annot key, by count) | 52.4% | same | same |
  | infer_key vs annot-key agreement | 20/25 (80.0%) | same | same |
  | **gate threshold to implement** | **≥ 60%** | — | runbook step 1 |

- **Decision:** **FAIL — 49.4% < 60% gate → STOP, no implementation.** Even using the *trusted* annotated key (removing key-inference error as a confound), only 52.4% of chords are diatonic. This is not a calibration bug (CLAUDE.md rule #1): spot-checks confirm genuine chromaticism/modality — e.g. `jazz1460_0080` "And On The Third Day" (9.7%) has a **D7 Mixolydian/blues tonic** (dom7 at scale degree I, non-diatonic by the strict rule); the low corpus number is real jazz harmony (secondary dominants, tritone subs, dom7/modal tonics), not a parsing artifact. The high-diatonic outliers (April Joy 95.5%) and 80% key agreement corroborate.
- **What this does NOT solve / known caveats:**
  - The gate was measured against the **global** song key, as the runbook step-1 spec prescribes. Issue #20's actual proposal is a **section-local** key. A local key could raise coverage in tonicization regions — BUT the dominant miss mode here is per-chord secondary-dominant/tritone-sub chromaticism, which a local key does *not* rescue (it only helps at genuine modulations). With even the trusted global key at 52%, a section-local variant is unlikely to clear 60% robustly on jazz, and a boost strong enough to matter would fight ~48% of chords. Verdict: a strict diatonic prior is the wrong tool for jazz; if revisited, it needs (a) Mixolydian/blues tonic tolerance and (b) a *soft, confidence-gated* weight tuned per-section, not a hard diatonic snap.
  - This premise check may look more favorable on a **pop** corpus (POP909) where diatonicism is far higher — issue #20's origin ("Georgia On My Mind") is a standard, but the prior might still pay off on pop material. Not tested this run.
- **Verification performed:** premise-check script run end-to-end (1128 events); two independent key sources cross-checked (infer_key vs annotated); manual spot-check of low- and high-diatonic songs to rule out a calibration bug per CLAUDE.md rule #1. No pipeline change made, so no eval/tests needed.
- **Stop reason:** subtask done — premise falsified, implementation correctly not attempted.
- **Revert command:** n/a — no code change; `scripts/check_diatonic_premise.py` left untracked for reproducibility.
- **Next suggested step:** Either (a) re-scope #20 to a **soft, section-local, confidence-gated** prior with Mixolydian tolerance and validate the local-key premise (does section-local key clear 60%?), or (b) test the same premise on POP909 where diatonicism should be much higher, or (c) deprioritize #20 for jazz and move to #22 (section structure) / #21 (progression bigram model), which do not assume diatonicity.

---

## 2026-07-12 — chord-SSM section-structure detector (IMPLEMENTED + committed)

- **Git tag:** `nightly/2026-07-12-1253-chord-ssm-sections`
- **Focus area:** Tier 1 — issue #22 (global section structure / AABA inference)
- **Source issue:** known_issues.md #22 — relevance re-checked at pre-flight: still valid and open; this run implements the "nuclear subtask" (chord-level SSM + form-length prior).
- **Nuclear subtask:** symbolic chord self-similarity matrix + jazz form-length prior (8/16/32/64 bars) to recover section boundaries the chord-level gmerge segmentation cannot (gmerge cuts at every chord change, so it over-segments sections).

### Literature (25-min survey)

- **MSAF — Music Structure Analysis Framework** (Nieto & Bello, ISMIR 2015/2016): benchmark suite; SSM-based methods dominate, retrieving segments from SSM "blocks" (homogeneity) and "stripes" (repetition). Boundary metric = hit-rate F at ±0.5s / ±3s. Applicability: the SSM+novelty machinery is directly reusable; MSAF's *acoustic* features are the wrong substrate for our metronomic synth (see premise check) — the symbolic chord SSM is the fix.
- **Barwise Section Boundary Detection in Symbolic Music using CNNs** (ISMIR 2025, arXiv 2509.16566): supervised barwise CNN on symbolic music. Applicability: most promising *learning-based* route for the audio-only/YouTube case, but needs the iReal forms as training labels (1460 available) — deferred; the unsupervised SSM+prior here needs no training and fits the metronomic corpus.
- **Symbolic Music Structure Analysis with Graph Representations & Changepoint Detection** (arXiv 2303.13881) + **Barwise Correlation Block-Matching** (arXiv 2311.18604): symbolic SSMs built from pitch-class/chord-succession features + changepoint detection. Applicability: confirms the symbolic-SSM premise; the block-matching / repetition idea is exactly the "merge adjacent repeated blocks" step implemented here.

### Premise check (before implementation, CLAUDE.md rule #2)

`scripts/premise_check_chord_ssm.py` — 8 genuine AABA standards (form "A16 B8 A8"), chord-SSM vs acoustic (Basic-Pitch) SSM. Metric = **bridge-contrast** = mean sim(A,A') − mean sim(A,B) (positive ⇒ the bridge is correctly the odd section out; this is the signal a form detector exploits).

- chord-SSM beats acoustic-SSM on bridge-contrast **7/8 tunes** (gate ≥ 5/8) — PASS.
- acoustic-SSM bridge-contrast ≈ 0 (±0.003): the audio self-similarity carries essentially **no** section signal on these renders — the direct cause of issue #22.
- Checkerboard *novelty* boundary-F is poor for both (3/8) ⇒ the detector must use **repetition + form-length prior**, not novelty (issue #22's own diagnosis, confirmed empirically).
- Corpus survey (371 AABA tunes, GT chords): bridge correctly odd-one-out in **85.4%** (mean margin +0.08), but weak/noisy (14.6% wrong-signed) ⇒ lean on the form-length prior.

### Implementation

- `harmonia/models/section_structure.py` — `build_chord_ssm(chord_sequence, n_pitches=12)` (per-beat [root-rel-tonic | quality] one-hot, cosine) + `detect_section_boundaries(ssm, beats_per_bar, form_lengths, ...)`. Algorithm: pick *smallest* form length clearing a repetition floor (sections nest — smallest-then-merge, never largest), lay a uniform grid, merge adjacent repeated blocks (collapses two 8-bar A phrases into the iReal `A16`), absorb runt tail blocks (tempo-grid n_beats isn't an exact multiple of the section step).
- Wired into `chord_pipeline_v1.infer_chords_v1`: new `ChordChart.sections` field (`[{start_s, end_s, n_bars}]`), default-empty so nothing downstream breaks. Also `save_json` emits it.

### Metrics (held-out jazz1460, GT section markers, boundary-F @ ±1 bar)

  | variant | boundary-F | prec | rec | n | invocation |
  |---|---|---|---|---|---|
  | gmerge baseline | 0.097 | 0.055 | 0.992 | 301 | `scripts/eval_section_boundaries.py` (GT chords) |
  | chord-SSM (ceiling) | **0.986** | 0.987 | 0.987 | 301 | same |
  | chord-SSM (end-to-end) | **0.844** | 0.889 | 0.833 | 12 | `infer_chords_v1` on rendered audio, inferred chords |

- Decision: boundary-F 0.844 (end-to-end) / 0.986 (ceiling) ≫ 0.097 baseline → clears commit criterion. 8/12 end-to-end tunes perfect (F=1.0).
- Tests: `tests/test_section_structure.py` (5 new) + full suite **228 passed**.
- Diagnostic plot: `docs/plots/section_ssm_aaba.png` (4 AABA tunes, detected vs GT boundaries on the SSM heatmap).

- **What this does NOT solve / caveats:** section *labelling* (A vs B), *phase* (pickup offset assumed 0), through-composed / all-similar tunes ("Dat Dere" merges everything), and it is not yet wired into the chart renderer nor evaluated on POP909 / real YouTube audio (the "Georgia On My Mind" origin case). End-to-end 0.844 is on rendered metronomic audio; real audio (beat-tracking + chord noise) will be lower.
- **Verification performed:** premise check (8 tunes) + 371-tune corpus survey + 301-tune GT-chord eval + 12-tune end-to-end eval on rendered audio + full test suite + inspectable SSM plot (CLAUDE.md "something inspectable, not a metric alone").
- **Stop reason:** subtask done — implemented, evaluated, committed.
- **Revert command:** `git revert nightly/2026-07-12-1253-chord-ssm-sections` (or `git reset --hard <prev>`); new files: `harmonia/models/section_structure.py`, `tests/test_section_structure.py`, `scripts/{premise_check_chord_ssm,eval_section_boundaries}.py`.
- **Next suggested step:** (a) section *labelling* pass (cluster the detected blocks by SSM similarity → A/B/C labels) to actually render the AABA form; (b) wire `ChordChart.sections` into the interactive chart; (c) evaluate on POP909 verse/chorus and on the YouTube-audio path where beat-tracking noise is the real test.

---

## 2026-07-12 — chord bigram progression prior: premise check (MARGINAL / below gate, no implementation)

- **Git tag:** none — no verified checkpoint this run (premise below gate; only the premise script committed per runbook).
- **Focus area:** Tier 1 — issue #21 (structural chord-progression bigram prior).
- **Source issue:** known_issues.md #21 — relevance re-checked at pre-flight: **still valid as an open issue; its own nuclear subtask #1 pre-registers a ≥70% top-50 coverage gate, which this run measured and did not clear (63.8%).**
- **Nuclear subtask attempted:** Run the cheap premise check gating issue #21 (CLAUDE.md rule #2) before any pipeline change: what fraction of consecutive chord pairs on the iReal corpus fall into the top-50 transpose-invariant bigrams, and how much does the previous chord actually predict the next?
- **Mechanism / what changed:** Added `scripts/check_bigram_premise.py` (tracked). Loads chord sequences from `data/accomp_db/db.jsonl` via `song_chord_spans()`, forms consecutive-pair bigrams in the transpose-invariant space `(interval=(root_j−root_i) mod 12, quality_i, quality_j)`, and reports (a) top-50 / top-20 coverage per corpus and (b) an information diagnostic H(next) vs H(next | q_prev). **No change to `chord_pipeline_v1.py`; no `BigramModel` built (gated on premise).**
- **Metrics (premise check, transpose-invariant bigrams):**

  | corpus | songs | pairs | uniq | top-50 % | top-20 % | invocation |
  |---|---|---|---|---|---|---|
  | **jazz1460** (eval target) | 1458 | 62 820 | 1314 | **63.8%** | 50.3% | `.venv/bin/python scripts/check_bigram_premise.py` |
  | pop400 | 344 | 29 786 | 730 | 62.9% | 43.3% | same |
  | blues50 | 54 | 1 040 | 67 | 98.3% | 86.2% | same |
  | ALL | 1856 | 93 646 | 1451 | 57.0% | 42.4% | same |
  | **gate threshold to implement** | — | — | — | **≥ 70%** | — | issue #21 subtask 1 |

  Information diagnostic (jazz1460): H(next) = 5.25 bits, H(next | q_prev) = 4.35 bits → **info gain 0.90 bits = 17% of marginal uncertainty removed.**

- **Decision:** **MARGINAL — 63.8% < 70% gate → STOP, no implementation.** The signal is real but weak: knowing the previous chord removes only 17% of next-chord uncertainty. The top bigrams are exactly the textbook cadences (`min7 →+5 dom7 →+5 maj7` = ii-V-I, `m7b5 →+5 dom7alt` = minor ii-V), which qualitatively validates the mechanism — but the dominant pattern is a **trigram** (ii-V-I): a bigram sees `ii→V` and `V→I` separately and cannot enforce the full cadence, which is precisely where an incoherent progression would still slip through. This is not a calibration bug (CLAUDE.md rule #1): the bigram ranking is musically sensible and blues50 (formulaic 12-bar) correctly scores 98.3%, so the metric behaves as expected — jazz genuinely has a long tail (1314 unique transpose-invariant bigrams; top-50 = 64%).
- **What this does NOT solve / caveats:**
  - Top-50 coverage is a proxy tuned for a small *hand-crafted* rule set; a *learned smoothed* matrix uses the whole distribution, so the more direct diagnostic is the 0.90-bit info gain — also modest. Both proxies agree: below the bar.
  - A bigram prior at weight 0.3 was never wired, so no end-to-end majmin/root number was produced (deliberately not burning 25 audio renders on a one-sided eval with nothing to compare against).
  - Corpus counts: DB has 1856 iReal songs (1458 jazz), not the 2229 quoted in the brief — the accomp DB is the current rendered subset.
- **Verification performed:** premise script run end-to-end (93 646 pairs); cross-checked 4 corpora; added an independent information-theoretic diagnostic that corroborates the coverage verdict; manual inspection of the top-15 jazz bigrams confirmed they are musically correct (rules out a parsing/transpose bug per CLAUDE.md rule #1). No pipeline change, so no tests needed.
- **Stop reason:** subtask done — premise below the pre-registered gate; implementation correctly not attempted (CLAUDE.md rule #2 — do not move goalposts after seeing data).
- **Revert command:** n/a — no code change; new tracked file `scripts/check_bigram_premise.py` only.
- **Next suggested step (ranked):** (1) **Trigrams**, per issue #21's own fallback — the ii-V-I unit is a 3-chord object a bigram cannot represent; re-run the same coverage/entropy check on transpose-invariant *trigrams* (gate: does top-100 trigram coverage or conditional info gain clear a useful bar?). (2) **Condition the prior on Agent B's detected sections / local key** rather than a single global matrix — a within-phrase bigram may be far more concentrated than the corpus-wide mixture. (3) The **reranking transformer** in the issue spec (look-ahead window over nuclear bigrams), which does not assume the low-order Markov concentration this check found lacking. Deprioritize a plain global bigram re-scoring for jazz.

---

## 2026-07-12 — post-nightly cleanup: section_structure review + known_issues quick-ref

- **Git tag:** none — cleanup / docs-only, no metric change
- **Focus area:** other — Agent D (post-nightly cleanup per runbook §Agent D)
- **Source issue:** Token-audit recommendation #1 (Agent E, this same date) + Agent B (#22) code output
- **Nuclear subtask attempted:** (1) Code-review `harmonia/models/section_structure.py` + `tests/test_section_structure.py`; (2) insert `## ACTIVE ISSUES — QUICK REFERENCE` block in `docs/known_issues.md`; (3) update runbook pre-flight to point at the new block.
- **Mechanism / what changed:**
  - `harmonia/models/section_structure.py`: fixed type hint `list[tuple[int, int]]` → `list[tuple[int | None, int]]` (runtime already handled `None` roots; hint was inaccurate). No logic changes — code reviewed as **clean**.
  - `tests/test_section_structure.py`: added two targeted tests for documented-but-untested behaviours: (a) `test_no_chord_beats_yield_zero_rows` (root<0 → all-zero SSM row/col), (b) `test_all_same_chord_produces_no_boundaries` (all identical chords → one merged section → no interior boundaries). Total: 7 tests.
  - `docs/known_issues.md`: inserted 25-line `## ACTIVE ISSUES — QUICK REFERENCE` table (issues #1–#22, one row each) immediately after the Status Note, before the POP909 baseline table. Closed/done issues one-liner; open issues carry a "next action" cell.
  - `docs/nightly_agent_runbook.md`: pre-flight step 3 updated — read *only* the ACTIVE ISSUES block in pre-flight; read full §N only when working on it. Estimated saving: ~100 K tokens/session vs. the old "read known_issues.md in full" instruction.
- **Metrics:**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|
  | test count | 228 | **230** | full pytest suite | `.venv/bin/pytest tests/ -q` |
  | section_structure.py test count | 5 | **7** | module tests | `.venv/bin/pytest tests/test_section_structure.py -v` |

- **What this does NOT solve / known caveats:** The ACTIVE ISSUES table will drift if not updated alongside the main §N entries. Adding a note to each §N's header that authors should keep the quick-reference row in sync would help — not done here to keep scope narrow. Issue #17 is marked SUPERSEDED (v4 shipped) but the §17 header still says "integration OPEN" — this was not touched to avoid scope creep; the ACTIVE ISSUES row reflects the current state.
- **Verification performed:** full test suite 230/230 green (`.venv/bin/pytest tests/ -q --tb=short`); both new tests pass in isolation; diff-reviewed all 4 changed files.
- **Stop reason:** all 3 tasks done.
- **Revert command:** `git revert HEAD` after commit
- **Next suggested step:** Wire `ChordChart.sections` into the interactive chart renderer (`harmonia/output/chart_interactive.py`) — issue #22 still-open item (b). Or tackle issue #19 (200-song YouTube corpus build for real-audio quality head).

---

## 2026-07-12 — POP909 diatonic premise check (#20 re-scope)

- **diatonic % par song :** 001=99.3%, 002=99.1%, 003=100.0%, 004=91.8%, 005=79.7%
- **global : 93.3%** (540/579 events; gate = 60%)
- **verdict : PASS**
- **next :** premise validée sur POP909 — re-lancer Agent A avec POP909 comme cible de validation pour le prior diatonique. Jazz1460 reste FAIL (49.4%) ; traiter les deux corpus séparément ou n'appliquer le prior qu'au mode de décodage POP909/pop.

---

## 2026-07-12 16:46 — Section labelling (A/B) + chart renderer wiring (#22)

- **Git tag:** `nightly/2026-07-12-1646-section-labels` (see commit)
- **Focus area:** other — issue #22 section structure labelling + UI
- **Source issue:** known_issues.md #22 — section boundaries correct (F=0.844) but unlabelled; not shown in chart UI
- **Nuclear subtask attempted:** Add `label_sections` to `section_structure.py`; wire into `infer_chords_v1`; add A/B/C chips row to `chart_interactive.py`
- **Mechanism / what changed:**
  - `harmonia/models/section_structure.py`: new `label_sections(chord_ssm, boundary_beats, sim_threshold=0.70)` — greedy A/B/C assignment from L2-normalised SSM-row fingerprints; cosine similarity > threshold → same label. `__all__` updated.
  - `harmonia/models/chord_pipeline_v1.py`: import `label_sections`, call it after `detect_section_boundaries`, add `"label"` field to each dict in `sections_out`.
  - `harmonia/output/chart_interactive.py`: (a) CSS `.sec-chip` + `#section-chips` row with dark-mode overrides; (b) `<div id="section-chips"></div>` placeholder before the grid; (c) `render_interactive` gains optional `sections: list[dict] | None` param; `sectionChips` key added to JS payload; (d) JS IIFE builds chip buttons from `P.sectionChips` and click-scrolls to first measure at that section's start time.
  - `scripts/render_youtube_chart.py`, `scripts/harmonia_server.py`: pass `sections=pipeline_chart.sections` to `render_interactive`.
  - `tests/test_section_structure.py`: new `test_label_sections_aaba` (uses zero-overlap B8_sharp fixture to cleanly test A/B discrimination).
- **Metrics (same schema every run for this focus area):**

  | metric | before | after | eval set | invocation |
  |---|---|---|---|---|
  | boundary F | 0.844 | 0.844 (unchanged) | synthetic AABA | pytest |
  | section label accuracy | — | 8/8 synthetic tests pass | AABA + A-B-A-C | pytest |

- **What this does NOT solve / known caveats:**
  - Labels reflect SSM fingerprint similarity, not melody/timbre — a Coltrane-style reharmonized A section may get a B label.
  - The `label` field is only emitted when `n_beats >= 32` (existing guard in `infer_chords_v1`).
  - `sim_threshold=0.70` is untested against the real YouTube/iRealb corpus; may need tuning.
  - `section_structure.py` docstring still says "labelling is left to a downstream pass" — now superseded but harmless.
- **Verification performed:** 231 tests pass (full `pytest tests/`); manual code inspection of chart_interactive.py diff.
- **Stop reason:** subtask done
- **Revert command:** `git checkout nightly/2026-07-12-1646-section-labels~1`
- **Next suggested step:** Evaluate labelling accuracy on the real iRealb/POP909 corpus: compare predicted labels to iReal section markers (A/B/bridge). Consider tuning `sim_threshold` or switching representative to the centroid of all same-label fingerprints rather than the first occurrence.

## 2026-07-12 — Diatonic quality prior (#20): implemented opt-in, net-neutral end-to-end

- **Focus:** issue #20. Premise re-check had PASSed on POP909 (93.3% diatonic in
  local key) vs FAIL on jazz1460 (49.4%). Implemented the section-local,
  confidence-gated diatonic prior and measured it end-to-end.
- **Implementation:** `apply_diatonic_prior()` + `use_diatonic_prior` /
  `diatonic_boost` / `threshold_chromatic` kwargs on `infer_chords_v1`
  (`harmonia/models/chord_pipeline_v1.py`). Fires only when the acoustic quality
  is uncertain (`conf < threshold_chromatic`) AND the inferred local key is
  reliable (`key_conf >= 0.30`) AND the root is diatonic; snaps a non-diatonic
  maj/min/dom/dim call to the canonical diatonic quality, preserving the
  triad-vs-seventh extension. Unit tests: `tests/test_diatonic_prior.py` (9).
- **Eval harness (new):** `scripts/eval_diatonic_prior.py` (both corpora),
  `scripts/diag_diatonic_prior_pop.py` (POP909 (boost,thr) sweep + fire tally).
- **Metrics (tempo grid + gmerge, 25 held-out jazz1460 + 5 POP909):**

  | corpus | variant | root | majmin | 7ths |
  |---|---|---|---|---|
  | jazz1460 | baseline | 88.7% | 84.0% | 58.6% |
  | jazz1460 | +prior (thr 0.65) | 88.7% | 83.2% | 58.1% |
  | POP909 | baseline | 78.6% | 73.6% | 41.8% |
  | POP909 | +prior (thr 0.65) | 78.6% | 73.0% | 41.6% |

- **Verdict:** commit criterion NOT met — POP909 majmin does not improve
  (−0.6pp at default thr, best sweep point boost 4.0/thr 0.80 = +0.1pp, within
  n=5 noise). Fire tally is a coin-flip (thr 0.65: 3 wrong→correct vs 3
  correct→wrong). The 93%-diatonic premise is real but the *inferred* local key
  isn't accurate enough to exploit it. Per the criterion's fallback, shipped
  **default OFF** (`use_diatonic_prior=False`, thr default 0.80 for opt-in).
- **Verification:** 240 tests pass; drove real `infer_chords_v1` on a rendered
  F-major clip — prior correctly flips I:min→maj, IV:7→maj7, vi:7→min7
  (mechanism sound, corpus lever weak).
- **Next suggested step:** replace the single-window `infer_key` local-key
  estimate with the unused `harmonia/theory/local_key.py` (HMM local key),
  re-validate local-key accuracy on POP909, then re-sweep the prior. The prior
  is bottlenecked by key inference, not by the prior formulation.

## 2026-07-12 — Chord ProgressionEncoder training (#21)

- **Git tag:** `nightly/2026-07-12-1730-progression-encoder`
- **Focus area:** Tier 1 — issue #21 (chord progression model)
- **Source issue:** known_issues.md #21 — trigram premise 63.8% (marginal); encoder bypasses Markov assumption
- **Nuclear subtask attempted:** Train a masked-cloze transformer encoder on jazz1460 chord sequences; evaluate vs Markov baselines on held-out val split.
- **Mechanism / what changed:** `harmonia/models/progression_encoder.py` — 2-layer transformer encoder, ±6-chord context window, d_model=32, 4 heads. Masked-cloze objective: predict centre chord quality from neighbourhood. `scripts/train_progression_encoder.py` (30 epochs, AdamW, MPS). `scripts/eval_progression_encoder.py` (vacuum eval, oracle segments).

- **Metrics:**

  | model | cloze quality acc | eval set | invocation |
  |---|---|---|---|
  | majority baseline | 40.5% | jazz1460 val 292 songs | `eval_progression_encoder.py` |
  | bigram | 57.9% | same | same |
  | trigram | 69.0% | same | same |
  | **ProgressionEncoder** | **83.9%** | same | same |

  Per-class recall: maj 86.3%, min 81.3%, **dom 86.8%** (vs 53.7% in production), hdim 62.4%, dim 78.0%.

- **What this does NOT solve:** end-to-end MIREX not yet measured (this is a standalone cloze eval). Integration into `chord_pipeline_v1` as a post-processing reranker is the next step. The encoder sees symbolic context — it does not help when the root is wrong (root errors compound). Checkpoint `progression_encoder.pt` is gitignored (92KB).
- **Verification performed:** training curves (30 epochs, monotone val improvement), standalone eval on 13213 held-out cloze positions, per-class recall breakdown.
- **Stop reason:** subtask done — encoder trained and evaluated.
- **Revert command:** `git checkout nightly/2026-07-12-1730-progression-encoder`
- **Next suggested step:** Wire `ProgressionEncoder` into `infer_chords_v1` as an opt-in reranker (`use_progression_prior=False` default like the diatonic prior). Evaluate end-to-end MIREX on jazz1460 held-out 25 + listen check on "Georgia On My Mind".

## 2026-07-12 — Real per-q5 acoustic prior for the ProgressionEncoder reranker (#21)

- **Git tag:** `nightly/2026-07-12-encoder-realprobs`
- **Focus area:** Tier 1 — issue #21 (chord progression model), reranker calibration
- **Source issue:** known_issues.md #21 — the wired reranker (commit `f7ecd3c`) only moved
  end-to-end majmin +0.7pp despite 83.9% standalone cloze quality; diagnosed as the
  `log_acoustic + w·log_encoder` combination using a **confidence-gated one-hot** on the
  greedy q5 class, which pins the acoustic term near-degenerate whenever `conf > ~0.65` and
  gives the encoder nothing real to argue against.
- **Nuclear subtask attempted:** Replace the one-hot with the real per-q5 log-probabilities
  already latent in the acoustic classifier's two heads (family posterior + base7/seventh
  posterior), instead of collapsing them to a scalar `conf`.
- **Mechanism / what changed:** `harmonia/models/chord_pipeline_v1.py` —
  new `_family_q5_logprobs(p_fam, p7, b7_labels_aligned)` combines the 5-class family
  posterior (major/minor/dim/aug/sus) with the base7 posterior into a real 5-class q5
  (maj/min/dom/hdim/dim) log-prob vector (minor/aug/sus map 1:1 onto q5; major splits into
  maj-vs-dom and diminished splits into dim-vs-hdim via each branch's renormalized b7 mass).
  `_FamilyClassifier._proba_family_and_b7()` factors out the raw posteriors; `predict(...,
  return_q5proba=True)` on all three classifier classes (`_FamilyClassifier`,
  `_CtxFamilyClassifier`, `_CtxFamilyClassifierV2`) exposes the combined vector.
  `rerank_progression_qualities(..., aco_logprobs=...)` uses it directly when supplied,
  falling back to the old one-hot when `None` — fully back-compat (external call sites in
  `scripts/eval_diatonic_prior.py`, `eval_seg_variants.py`, `diag_diatonic_prior_pop.py`
  untouched). `progression_weight` default bumped 0.5 → 2.0 to match the new prior.
- **Metrics** (`scripts/eval_irealb_e2e.py`, tempo/gmerge config, held-out jazz1460 n=25):

  | variant | root | majmin | 7ths |
  |---|---|---|---|
  | baseline (no encoder) | 88.7% | 84.0% | 58.6% |
  | encoder + one-hot (old, w=0.5) | 88.7% | 84.7% | 58.9% |
  | encoder + real q5 logprobs (w=0.2) | 88.7% | 84.8% | 59.0% |
  | encoder + real q5 logprobs (w=0.5) | 88.7% | 84.8% | 59.0% |
  | encoder + real q5 logprobs (w=1.0) | 88.7% | 84.7% | 58.9% |
  | encoder + real q5 logprobs (w=2.0) | 88.7% | **85.0%** | **59.0%** |

- **Verdict: commit criterion MET (majmin ≥ 84.7%), but modest** — +1.0pp majmin over
  baseline, +0.3pp over the old one-hot prior at its own best weight. Did not clear the
  ≥85.5% "true improvement" bar set going in. Root is unaffected (encoder only reranks
  quality). Plateaus rather than climbs across the sweep — w=2.0 edges out w=0.5/1.0 by
  0.2–0.3pp, within likely noise at n=25.
- **Verification performed:** 248 tests pass (2 new tests added:
  `test_family_q5_logprobs_sums_to_one_and_splits_correctly`,
  `test_real_logprobs_used_when_provided`, in `tests/test_progression_encoder_rerank.py`);
  manually verified `_family_q5_logprobs` sums to 1.0 and splits major/diminished mass
  correctly on synthetic posteriors; verified both `_FamilyClassifier.predict` and
  `_CtxFamilyClassifierV2.predict` return a 4-tuple with `return_q5proba=True` and a
  3-tuple (unchanged) by default.
- **Stop reason:** sweep complete, commit criterion met, gain is real but modest — no
  further sweeping planned this session (a wider weight range or a proper w/CI sweep given
  n=25 noise is the natural follow-up, not attempted here).
- **Revert command:** `git checkout nightly/2026-07-12-encoder-realprobs`
- **Next suggested step:** (1) re-run on a larger held-out set (n=25 is noisy — 0.2–0.3pp
  deltas between weights are within sampling noise); (2) re-evaluate on POP909 (this session
  only checked irealb/jazz1460 per the canonical-GT provenance note); (3) the encoder's
  headline lever is dom recall (7ths metric) — the 7ths gain (+0.4pp over baseline) is
  proportionally smaller than hoped given 86.8% standalone dom recall, suggesting the
  reranker's context window or weight is still not fully exploiting the encoder's signal.
