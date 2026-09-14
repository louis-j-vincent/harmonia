# Harmonia refactor — plan (2026-09-14)

## Context

The live app is `harmonia_min/` (Flask on :7772, ~16k lines of Python + one 7,066-line
`app_shell.html`). It was written from scratch on 2026-07-30 because the previous package,
`harmonia/` (58k lines), was "presumed broken" nine days after a parity-gated rewrite of it
was declared complete. Since then `harmonia_min` accreted the same kind of debt in six weeks:
six section detectors (four wired in a fallback chain), 18 env flags read at point of use,
two silent fallbacks, forks of legacy modules with stale docstrings, a detector that imports
from `scripts/` at runtime, a frontend calling three dead endpoints, and a state directory
mixing live caches with backups and 98 MB of reports. The legacy package is untouched since
2026-07-30 but still supplies four modules, the musx weights and a checkpoint to the live app.

Louis wants one clean, modular, understandable codebase that does not carry these errors.
The history says how: the August 2026 screen-by-screen cleanup (small verified commits,
render-checked) landed; the July big-bang port of untrusted code did not. This plan is a
strangler-pattern port of the **trusted** app, gated by "today's charts come out identical".

## Decisions taken by Louis (grilling, 2026-09-14) — do not re-open

1. **Reference = today's `harmonia_min` output, but differences are arbitrated, not forbidden**
   (Louis, at plan review: "I don't have any rendered chart right now I'm adamant on not
   changing … a different chart doesn't mean you did bad work, you will have to tell me what
   the differences are and I'll arbitrate"). A pure port is still *expected* to reproduce the
   chart; every difference is surfaced as a before/after chart page he can open in the app and
   listen to, with one line saying what changed and why. He decides. Section *quality* remains
   a separate research track after the refactor. "Fold runs twice on annotated songs" is
   checked, fixed only if still real.
2. **Legacy deleted from the tree**: `harmonia/`, `scripts/harmonia_server.py`,
   `harmonia/serving`, `harmonia/output`, vendored ISMIR2019 Python. Restore tag first.
3. **Features kept**: analyze (YouTube / file / mic), chart, library + folders + titles,
   annotate (chord locks), Practise + piano, Jam, section tool + Soudure + `/ssm` + `/reports`,
   iReal search/export + UG tab search. **Dropped**: billboard screen, `/min/<f>`, chord-LM,
   stubs (`section-merge-game`, `section-merge-verdict`, `tab-import`, `irealb-import`), the
   retired reinfer path and the capability check that loads it.
4. **Only the production choice survives** per decision: songformer only (child death = loud
   failure), merge = mean, quarter-bar on, fold loop = occurrence, gate = letter, transpose
   off. Alternatives are deleted; git history keeps them.
5. **Frontend** split into per-screen ES modules, no build step, same pixels, dead calls removed.
6. **New package is `harmonia/`**; `harmonia_min` stays live as the reference until the swap,
   then is deleted. Server = `python -m harmonia.server`, Flask :7772, no reloader.
7. **Branch from `feat/section-criteres`** (prod), dedicated worktree, sprints of ~1–2 h, one
   verified commit each, app runnable after every commit. Merge back later.
8. **Archive, don't delete** non-code: referenced scripts → `tools/`, rest → `archive/scripts/`;
   `scratchpad/` untracked; docs untouched except new `docs/STATE.md` + fresh `known_issues.md`.
9. **Single-user tool** (Mac + iPhone via Tailscale). No auth, file state.
10. **Human state tracked in git**, separate from ignored caches.
11. **Philosophy** (confirmed verbatim): audio is the source; the bar is the unit of truth; one
    law per decision, read in one place; no silent fallbacks; under-fold never over-fold; Occam
    on the chart; one screen one thing; the ear judges, metrics illustrate; counting beats
    clever models where the question is repetition.
12. **Final ear check**: Stand By Me, This Love, Let It Be, Bora Bora, Sunny Afternoon.

## Target layout of `harmonia/` (~34 Python files, smaller than harmonia_min)

| Module | Responsibility |
|---|---|
| `settings.py` | ONE `Settings` object: paths (repo, state, data/cache, models), port, musx device. Algorithm choices are constants here, not env. Only `HARMONIA_PORT`, `HARMONIA_MUSX_DIR`, `HARMONIA_MUSX_DEVICE` are read from env, only in this file. |
| `cache.py` | ONE cache path convention for beats / musx_probs / nnls / songformer: `key = f"{stem}__{size_bytes}"` (what songformer already does). `get(kind, audio)` / `put(kind, audio, obj)`. No stage builds a cache path itself. |
| `chart_model.py` | Dataclasses + `to_dict/from_dict` documenting the served JSON exactly (schema in appendix). The only place the contract lives. |
| `labels.py`, `key_profiles.py` | Ported near-verbatim (leaves). |
| `beats.py` | Beat This! wrapper, hard-fail (leaf). |
| `musx.py` | ISMIR2019 ensemble: `frame_posteriors`, `redecode`, `label_confidence` (leaf). Weights resolved via settings. |
| `nnls_features.py` | NNLS bothchroma + `nnls24_heads.npz` from `harmonia/assets/` (leaf). |
| `bars.py` | **Extracted** from `pipeline.py:493-656`: beat-index bar layout + phase vote + Set-bar-1 re-phase. Typed in/out. |
| `sections/songformer.py` | Child-process runner + cache. Raises on child death (exit 137/-9 or timeout). |
| `sections/similarity.py` | The 4-line `_diag` from `voice_sections.py:168-171` that `soudure.py` and `section_tool.py` need. Nothing else of voice survives. |
| `folding.py` | `fold_letter_groups`, `minimal_fold`, `loi_de_merge` with the cqt / transpose / bar-gate / internal-loop branches deleted. |
| `harmonic_key.py`, `span_rescore.py` | Ported; `span_rescore` keeps `musx_suggestions` only (context-scorer / reinfer machinery dropped with `chord_context_prior.py`). |
| `pipeline.py` | `analyze_steps` as thin composition of the stages, yields head / raw / final as today. Duration computed once (ffprobe, raises) and passed in — today it is computed twice (`pipeline._duree` silently, `server._run_job` loudly). |
| `annotations.py`, `titles.py`, `jam.py`, `soudure.py`, `section_tool.py`, `phrases4.py`, `ssm_page.py`, `refold.py` | Ported; imports rewired. |
| `integrations/irealb_fetcher.py`, `irealb_export.py`, `tab_fetcher.py` | The only legacy Python ported (610 / 67 / 194 lines). |
| `server/app.py`, `server/jobs.py`, `server/youtube.py` | App factory; job registry (dict + lock) instead of module globals; yt-dlp + PO-token server. |
| `server/routes/{misc,library,analyze,annotate,sections,jam,irealb}.py` | One file per route group. Catch-all honest 404 stays. |
| `static/` | Split frontend (see below). |
| `assets/` | `nnls24_heads.npz`, `submission_chord_list.txt` (small, tracked). Musx `.sdict` weights → `data/models/musx/` (untracked, like today). |

State layout (decision 10), resolved from `settings.py` never from `__file__`:

```
state/human/    sections/ annotations/ marks/ chart_meta.json folders.json   ← git-tracked
state/cache/    charts/ beats/ musx_probs/ nnls/ songformer/ reports/        ← ignored
```
`marks/<stem>.json` holds the Set-bar-1 time. Today it lives only inside the regenerable
chart JSON (`bar1`), which is how it was lost on 2026-08-13; the chart still carries `bar1`
for the client, sourced from `marks/` at bake time. `charts.bak_*`, `intro_vocal/`,
`llm_forme/`, `charts_new/` are archived out of `state/`.

## Golden diff report (`tools/golden.py` + `tools/avant_apres.py`) — exists before any porting

Not a pass/fail gate: a report Louis arbitrates. Identical output is the default expectation
for a pure port; a difference is neither hidden nor a failure, it is explained and shown.

- `tools/golden.py` reproduces `scripts/rebake_library.py`'s composite exactly: `analyze(audio,
  title, file_key, audio_url, bar1_time)` then, if `state/sections/<stem>.json` is `validated`,
  `refold` + `sections_pour_chart` with `_completer` gap-filling. Engine is a parameter
  (`harmonia_min` or `harmonia`). Songs: the 44 library charts with audio + warm caches
  (`T64BgKEL-Sw_pile`, `h_D3VFfhvs4` excluded). A cold cache is an error, never a model run.
- Diff in song space (bar by bar via `barSpans`, ignoring `meta.musx_latency_ms`): count of
  changed bars per song, ranked. Zero changed bars = nothing to arbitrate.
- **Baseline = today's code, not the disk.** Disk charts were baked 2026-08-19/20 by older code.
  Sprint 0 runs harmonia_min twice (determinism proof), freezes run A as
  `state/golden/baseline/`. Every sprint diffs its output against that baseline, so a diff is
  attributable to the sprint's code, never to noise or stale disk files.
- **Before/after page**: `scripts/avant_apres.py` (moved to `tools/`) already does this —
  republishes the "before" charts under `old_<stem>` so they open in the real app with audio
  and transport, and writes `/reports/avant_apres.html` ranked by change magnitude, bar by
  bar, clickable to listen, with "open BEFORE / open NOW" buttons. Its inputs become
  (baseline dir, sprint output dir). Every sprint whose report shows ≥1 changed bar ships this
  page on the worktree server plus one line per song: what changed, and why the code did it.
  Louis arbitrates: accept (the new baseline is re-frozen from the accepted output), or
  reject (the sprint fixes it before commit).
- The disk-vs-baseline diff of sprint 0 is itself the first such page: it shows what a rebake
  with today's code would change in his library, before any refactor code exists.

## Sprint sequence (strangler pattern; each = one verified commit in the worktree)

"golden" in the table = the diff report against the frozen baseline; any changed bar ⇒ the
before/after page + one line per song, and Louis's arbitration before the commit.

| # | Sprint | Gate / deliverable |
|---|---|---|
| 0 | **Bootstrap + report tooling.** Tag `pre-refactor-2026-09-14`. Branch `refactor/clean-app` off `feat/section-criteres`, worktree under scratchpad. Symlink `.venv`, `data`, `docs/audio`; **copy** (snapshot) `harmonia_min/state/{charts,beats,sections,songformer,annotations,chart_meta.json,folders.json}`. Write `tools/golden.py`; move `scripts/avant_apres.py` → `tools/avant_apres.py` (parametrised on two dirs) and `handoff_cleanup/check.py` → `tools/render_check.py`. | old-vs-old: run A == run B on 44 charts (determinism); baseline frozen; disk-vs-today before/after page shown to Louis |
| 1 | **Free the name.** Port the three integrations into `harmonia/integrations/`; move musx weights to `data/models/musx/`, `nnls24_heads.npz` + chord list to `harmonia/assets/`; point `harmonia_min` at them. Delete legacy `harmonia/`, `scripts/harmonia_server.py`, the 83 legacy test files, `golden/`. Create `harmonia/{settings,cache,chart_model}.py` (cache.py serving old stem keys, logged). | golden; `pytest -k irealb`; `pytest --collect-only` clean |
| 2 | `labels.py`, `key_profiles.py` → harmonia_min imports them from `harmonia` | golden |
| 3 | `beats.py` via `cache.py` | golden; `test_beats_*` |
| 4 | `musx.py` | golden; `test_musx_*` |
| 5 | `nnls_features.py` | golden |
| 6 | `bars.py` extraction (pipeline.py shrinks) | golden (bars byte-identical) |
| 7 | `folding.py` locked to mean / occurrence / letter / no transpose; `refold.py` | golden; `test_fold_*`, `test_loi_de_merge` |
| 8 | `harmonic_key.py`, `span_rescore.py` (stripped) | golden; `test_musx_suggestions` |
| 9 | `sections/songformer.py` + `similarity.py`; delete voice/harmonic/chroma/retour/minimal_form; archive their tests | golden; `test_songformer_sections`; **new test**: child death raises |
| 10 | `pipeline.py` orchestration; `harmonia_min/pipeline.py` becomes a re-export; check fold-runs-twice (exactly one fold call per bake) | full golden; `pytest -k pipeline` |
| 11 | `server/app.py`, `jobs.py`, `routes/{misc,library}.py`; worktree server on `HARMONIA_PORT=7773` | golden; Playwright library + chart at 390 px |
| 12 | `routes/analyze.py`, `youtube.py` | pytest; Playwright analysing screen; yt-dlp smoke on 2 real URLs |
| 13 | `routes/annotate.py`, `routes/sections.py` (soudure, ssm, phrases4, section tool, inferer) | pytest -k "annot or soudure or ssm or phrases4"; Playwright |
| 14 | `routes/{jam,irealb}.py`; dead routes/stubs removed | pytest -k jam; manual jam smoke (mic not automatable) |
| 15 | **State split**: `state/human` + `state/cache`, `marks/` for bar 1, cache keys renamed **in place** to `stem__size` (no duplication: disk is 97 % full). Migration script idempotent, copies then verifies, never deletes source. | golden after migration; `df -h` before/after |
| 16 | Frontend: `static/{state,api,audio,router}.js` + `ui/kit.js`; `app_shell.html` becomes the module host | Playwright screenshot diff per screen |
| 17 | `screens/{library,chart}.js` | same |
| 18 | `screens/{analyse,annotate,sections_editor}.js` | same |
| 19 | `screens/{practise,jam}.js`; dead calls (`billboard-corpus`, `irealb-import`, `reinfer`) removed | same + `?open=` deep links |
| 20 | **Hygiene**: compute the import closure of the 10 referenced scripts (`bench.py` chain pulls `ssm_zoo`, `section_metric`, `order_lab`, `order_bundle` and `sys.path`-inserts `scratchpad/` — keep that closure together under `tools/sections_bench/`); everything outside → `archive/scripts/`; untrack `scratchpad/`; prune the 15 worktrees; remove zips / stray handoff dirs / `Harmonia.zip`; `docs/STATE.md`; archive `known_issues.md` → fresh one; update `CLAUDE.md`, `README.md`, `pyproject.toml`, `.claude/skills/ship/SKILL.md` (golden gate in, mirror step out, new paths). Unload the dead `com.harmonia.nightly-agent` plist (ask first: outside the repo). | every kept entry point runs; `pytest` full; golden |
| 21 | **Swap on prod** (ask Louis first, he may be annotating): merge into `feat/section-criteres`, run the state migration on the live tree, restart :7772 with `python -m harmonia.server`, ear check on the 5 songs, Playwright on the real phone URL. 24 h soak. | ear check + soak |
| 22 | Delete `harmonia_min/`, its `.gitignore`, remaining re-exports. | full golden + pytest + Playwright |

Estimated ~25–35 h of agent time across 23 sprints. Sprints 2–5 are short; 10, 15, 16 and 21
are the risky ones.

## Execution protocol

- Each sprint = one subagent brief (Sonnet by default; Haiku for pure moves), containing: files,
  gate commands, forbidden actions. The main session re-runs the gate before the commit.
- Git: `git add <paths> && git commit` in one command; never `-A`, `.`, `-a`, `--no-verify`.
  Load the `branch-safety` skill before any git action. Never touch :7772 before sprint 21.
- Do not touch `harmonia_min/chord_lm/acoustic.py` (untracked, another session's WIP).
- Check `df -h` at every sprint start; stop below 5 GB free.
- Check-in to Louis per sprint: one line + the artifact. If the report shows zero changed
  bars: the count and a screenshot link. If not: the before/after page at
  `http://100.89.209.63:7773/reports/avant_apres.html`, the `old_<stem>` charts openable in
  the app, and one line per changed song (what moved, why the code did it, my recommendation).
  Commit waits for his arbitration. No aggregate numbers without a per-song illustration.
- When unsure whether a difference is a regression or an improvement, do not guess: show it.

## "Errors not to carry" checklist (reviewed at every sprint)

1. No silent fallback: delete `pipeline.py:262-269` (`_duree` → 0.0) and
   `chord_context_prior.py:751-753` (`except Exception: pass`). Grep gate: bare `except`
   followed by `pass`/`continue`/`return <default>` without a log line.
2. No env read outside `settings.py`; no algorithm flag at all.
3. No docstring naming another file's path as "see X.py" (`chord_context_prior.py:1`,
   `nnls_features.py:59` today); say the thing or import it.
4. No two implementations of one concept (`minimal_form.py` vs `folding.minimal_fold`;
   six section detectors; two duration computations).
5. One cache convention, one place (`cache.py`).
6. No `sys.path` hacks (`voice_sections.py:94-108`).
7. No dead routes, stubs, or client calls to retired endpoints.
8. Every stage's docstring states what it does NOT solve (CLAUDE.md rule 4; songformer's
   docstring at `sections.py:252-256` is the house style).
9. Paths resolve from `settings`, never from `__file__` (`server.py:58-61`, `beats.py:34`,
   `songformer.py:75` today).

## Frontend split design

- `state.js`: `export const S = {}`; `API.build` fills it with `Object.assign` (today `S` is
  assigned inside `API.build`, `app_shell.html:1370-1375`). All `S.x` reads survive unchanged.
- `api.js` (fetch wrappers), `audio.js` (iOS traps: no rAF render loop, tick from `timeupdate`,
  fetch + blob URL), `router.js` (`go()`, `?open=`, `?seg=`, pushState), `ui/kit.js`
  (`kitButton`/`kitIcon`/`ctlSkin`), `screens/*.js`.
- `<script type="module">` served statically by Flask; no service worker exists, so nothing to
  precache. PWA manifest untouched.
- Verification: `tools/render_check.py` (from `handoff_cleanup/check.py`: Playwright, 375×667,
  DPR 2, mobile) against the worktree server, screenshot before/after per screen, zero
  horizontal overflow, zero console errors.

## Verification commands

```
.venv/bin/python tools/golden.py --engine harmonia_min --out state/golden/baseline          # sprint 0 (twice)
.venv/bin/python tools/golden.py --engine harmonia     --out state/golden/sprintNN \
                                 --baseline state/golden/baseline                          # every sprint ≥ 2
.venv/bin/python tools/avant_apres.py --before state/golden/baseline --after state/golden/sprintNN   # if changed bars > 0
.venv/bin/python -m pytest tests/ -q --no-cov -k "<touched>"
HARMONIA_PORT=7773 .venv/bin/python -m harmonia.server &   # worktree server, never :7772
.venv/bin/python tools/render_check.py out.png --url "http://127.0.0.1:7773/?open=<stem>"
df -h .
```

## Non-goals and risks

- Section quality, SaaS, multi-user, new features: out of scope.
- Songformer fail-loud is the one live behaviour change (a new song whose child dies now errors
  instead of degrading to voice). Louis chose it; say it once in the sprint-9 report.
- The diff report covers final charts only; head/raw charts never touch disk. Playwright covers
  them through the analysing screen.
- Arbitration is a human step on the critical path: a sprint with differences cannot commit
  until Louis has looked. Batch small diffs into one page per sprint, never one page per song.
- Concurrent sessions edit `docs/plots/` and `state/sections/`; unfamiliar diffs there are theirs.
- Disk at 97 %. Cache renames are in place, not copies. Musx caches (12 GB under `data/cache`)
  are not moved, only re-keyed.
- The 218 unmerged commits on `feat/section-criteres` are not this plan's problem; the merge to
  `main` is Louis's call after the swap.

---

## Appendix — exploration findings (reasoning trail for executing sessions)

### Facts verified 2026-09-14
- Live server: `.venv/bin/python -m harmonia_min.server`, PID 51968, `feat/section-criteres` @ ab8c68c.
- `harmonia/` last commit 2026-07-30. `harmonia_min` imports it at `server.py:593,631,1060`
  (irealb_fetcher, irealb_export, tab_fetcher — live routes), `chord_lm/corpus.py:23`
  (ireal_corpus, training only), `chord_context_prior.py:232` (only if vocab ≠ q8, never).
  Data: `nnls_features.py:31` loads `harmonia/models/nnls24_heads.npz`; `musx.py:_musx_dir()`
  falls back to `harmonia/third_party/ISMIR2019-…` (67 py files never imported; weights +
  `data/submission_chord_list.txt` read by path; MIT).
- Git: 1,090 commits, 28 branches, 15 registered worktrees, 517 dirty paths (docs/plots and
  scratchpad churn from other sessions). Tracked: 988 docs, 490 scripts, 488 scratchpad files.
- Nightly launchd agent `com.harmonia.nightly-agent` loaded but dead since 2026-07-29.
- Disk 97 % (8 GB free); `data/cache` 12 G, `scratchpad` 1.7 G, `state` 137 M.

### Live app map (harmonia_min)
- Pipeline `analyze_steps` (generator; one `bars` list mutated in place; caller must serialize
  raw before resuming): head 45 s → beats ∥ musx → Viterbi redecode on beats → bar layout +
  phase vote → draft key + suggestions → **raw** → sections cascade → `fold_letter_groups` →
  `minimal_fold` → `analyze_harmony` → suggestions → **final**.
- Reachability: scripts/tests only = `minimal_form.py`, `retour.py`,
  `chord_lm/{corpus,data,evaluate,repetition}.py`; orphan untracked = `chord_lm/acoustic.py`;
  `/min/<f>` route unlinked from the UI; `voice_sections._diag` used by soudure/section_tool.
- `server._capabilities()` (`server.py:479-497`) loads the context scorer on every
  `GET /api/library` to advertise `reinfer`, whose route returns 410.
- Frontend: `<style>` 14–36, one `<script>` 40–7064, screens library / charts / search / record
  / jam / analysing / chart / sections / prompter (= Practise) / billboard. Dead calls at
  `app_shell.html:1962` (`/api/billboard-corpus`), `:2234` (`/api/irealb-import`), `:6980` (`/api/reinfer/`).
- 18 env flags (list in the design agent's report; all collapse per decision 4).
- Chart JSON keys: audio_url, bar1, barGrid, beatTimes, bpb, file, fold, form (always null),
  key, keyName, keySegments, meta{bpm, musx_latency_ms, n_segments, engine, raw, pending},
  nBars, prompter, sections[{id,label,tag,reps,spans,barRanges,bars,barSpans}], title,
  video_id. Chord: {root,q,bass,nc,carry,c,t0,t1,colour,n,inflect?,flag?,sug?}.
- Library: 46 `min_*.json`; 44 with audio + warm caches; 20 validated hand sections.

### Legacy map (harmonia/)
- Old app = `scripts/harmonia_server.py` (3,950 lines, :7771) + `harmonia/serving/*` (8,296,
  "moved verbatim") + `output/chart_interactive.py` (4,077) + `chart_model.py` (1,742).
  Three docs say "never touch :7771". iOS playback bug never fixed there.
- Old pipeline `models/chord_pipeline_v1.py` (4,779 lines, 96 defs, 5 env flags).
- Dead (zero consumers): `core/audio.py`, `models/local_key_context.py`, `stages/beat_grid.py`,
  `stages/chords.py`. Kept alive only by own tests: 8 models files + 3 theory files +
  `chord_hmm.ChordInferrer`. Unwired research lanes: `align/`, `dataset/`.
- `eval/accuracy_score.py` scores against `golden/brick0/`, condemned by Louis 2026-08-07.
  Nothing in `eval/` is wired to the ship skill (`scripts/bench.py` is separate).
- Tests: 118 files, 1,655 collected; 28 import harmonia_min, ~83 harmonia, 7 scripts.

### History lessons
- 2026-07-15 refactor survey (never executed) diagnosed: same step re-implemented in two
  scripts, silent divergence on a constant, plausible metric.
- 2026-07-21 staged parity-gated port: shipped, "Phase 7 COMPLETE", discarded 2026-07-30.
- 2026-08-19 `handoff_cleanup/`: 5 screen-scoped tasks, one verified commit each — landed.
  Its README records a diff swallowed by a concurrent session's commit → stage+commit in one command.
- `docs/known_issues.md` is a 27,984-line narrative; `docs/STATE.md` promised twice, never built.
- `README.md`, `Makefile`, `setup.sh`, `pyproject.toml` describe the legacy package; the
  `harmonia.cli:main` entry point targets a file that does not exist.
