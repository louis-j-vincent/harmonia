# Handoff — continue the staged rewrite (serving tail + core Phases 3–5) — 2026-07-22

_You are the continuation of the **rewrite-orchestration thread**. The prior session
(Phase 0 benchmark + Phase 6 serving split + Phase 7 feature reroutes) is summarized
below with its full reasoning trail. Delegate everything to Opus subagents; you hold
only the gate decision, the commit, and the plan order._

## READ FIRST (in order)
1. `CLAUDE.md` — 6 error patterns + env gotchas (esp. the **stale-clone trap**: verify
   `harmonia.__file__` is under the canonical repo, not `~/harmonia`; and the
   `pytest -o addopts=""` note below).
2. `docs/rewrite_execution_plan_2026_07_21.md` §0 (two-net principle), §1 (GT provenance),
   §3 (the 8 phases). Phases 0,1,2 done earlier; this session did most of 6 + part of 7.
3. `docs/known_issues.md` → the sections `PHASE 6a`, `PHASE 7`, and
   **`REFACTOR-SURFACED POTENTIAL ISSUES (running log)`** — the live bug tracker. Keep
   appending to it; each porting subagent is asked to report smells.

## WHAT'S DONE + COMMITTED THIS SESSION (verify `git log --oneline -25`)
**Serving split — `scripts/harmonia_server.py` 10,334 → 5,668 lines (−45%).** Eleven
modules extracted under `harmonia/serving/`, 29 routes moved into `serving/api.py`, every
step gated and committed:
- `cache.py` (slug/stem, closes a stem-collision bug class) — 981f5a4, bc3b1fc
- `config.py` (path/cache constants) — 84b23f1
- `state.py` (yt/ireal registries, offset stores, +4 saver helpers) — 8bea5d7, 729d2e5
- `render.py` (`_chart_model_for`, overlay/PWA-head injectors) — d9c4e40, 96e315d
- `templates.py` (9 big HTML/JS template strings, ~40% of the old bulk) — 0e1ec5e
- `runtime.py` (`ARGS`/jobs/jam_sessions/analyze-config; the `_ARGS` reassignment trap
  handled via module-attribute access) — a18bc84
- `loaders.py` (annotation + iReal-alignment loaders) — 44295ad
- `billboard_gt.py` (billboard-GT stateful cluster; `_billboard_ds` reassignment trap
  handled) — 6151731
- `api.py` blueprint: 29 routes (SAFE-GET static/debug/page + 7 mutating) — 6a1574b,
  cd31e1c, 44295ad, 729d2e5, 6151731
**Frozen PARITY benchmark built** (was a half-built skeleton at 209de77) — 9563fe9:
`harmonia/eval/parity.py` (capture/diff/CLI), `benchmark_set.py`, `golden/frozen_parity/`
(8 songs), `tests/test_calibration_pins.py` (10/10, incl. a **fixed** song-002 pin that
was asserting the wrong tempo). Self-consistency gate = 8×11 stages zero divergence.
**Phase 7 feature reroute**: 6 sites moved to `core.features` bit-identically — 417257d,
1c4cadb (~75 sites remain, deferred — see `PHASE 7` in known_issues.md).

### Verification patterns that WORKED (reuse them)
- Read-only routes/helpers: **byte-identical HTTP** via Flask `app.test_client()` (status +
  sha256) baseline-before vs after; **move-identity** (function body byte-identical vs
  `git show HEAD:` except the `@app.route`→`@api.route` line); **`app.url_map` identity**
  (the full `sorted((rule, endpoint, sorted(methods)))` set stays IDENTICAL — 76 rules).
- The blueprint is registered with **`name=""`** so Flask doesn't namespace endpoints
  (keeps `serve_audio`, not `api.serve_audio`) — required for url_map byte-identity during
  the incremental move. Revert to a normal-name blueprint only once ALL routes are moved
  and url_for is audited (currently 0 `url_for(` in the repo).
- Mutating routes (approved by Louis): **pure-move + url_map identity + a few cheap
  smoke-runs against MONKEYPATCHED temp paths** (never mutate real data in gating).

## ⚠ CONCURRENT "GRID" LANE — COORDINATION IS LIVE
A second session (the grid/downbeat lane) is **actively editing** and **owns**:
`harmonia/output/chart_model.py`, `app_shell.html`, `chart_interactive.py`;
`harmonia/theory/local_key.py`, `harmonia/models/local_key_context.py`;
`chord_pipeline_v1.py` (decode region), `beat_grid.py`, `downbeat_anchor.py`,
`harmonic_downbeat.py`. **Import these; never edit them. Never `git add -A`** — those
files are dirty with the grid lane's WIP; stage only your specific files.
- **One lane owns `harmonia_server.py` at a time.** This lane has been refactoring it; if
  the grid lane needs it, checkpoint and hand off.
- **Two benchmarks, separate files.** Mine (`eval/parity.py`, `benchmark_set.py`,
  `golden/frozen_parity/`) is the **PARITY** net ("did a refactor change behavior?"). The
  grid lane's **"Brick 0"** is the **ACCURACY** net (hand-verified real GT — Louis's
  hand-verify). Do not clobber; keep Brick 0 under `eval/accuracy_*` + `golden/brick0/`.

## YOUR JOB — ranked try-order with per-item gate + stop criterion

### Priority 1 — Finish the serving split's SAFE remainder (low collision, no budget needed)
1a. **Cleanup round** (pure hygiene, byte-identical gate): delete dead `_SWIPE_NAV_JS`
    (templates.py — referenced nowhere); drop `render.py`'s remaining lazy
    `import scripts.harmonia_server` back-imports that are now redundant —
    `_load_annotation` is already in `loaders.py`, `_gt_chords_for_video` is in
    `billboard_gt.py` (import them directly); drop the `noqa: F401` re-exports in the
    server once nothing depends on the identity contract.
1b. **Billboard cohesion** (optional): pull `_load_billboard_corpus`, `_estimate_gt_offset`,
    `/api/billboard-corpus` into `billboard_gt.py` (billboard concept currently spread
    across 4 homes).
    *Gate:* byte-identical HTTP + url_map identity. *Stop* any route you can't prove.

### Priority 2 — Grid-coupled routes (needs grid-lane coordination FIRST)
`/gt-align` (needs `_waveform_peaks`, grid-shared), `/gt-playalong-corrected`,
`/gt-playalong-sectionwise` (`_perfect_grid_for`/`_sectionwise_for`), the beat-grid routes,
the annotators (`_build_annotator_data` pulls beats/grid). **Do not move these until the
grid lane confirms it's done with those helpers** — log a coordination note if blocked.

### Priority 3 — Heavy/network routes (needs an inference/disk budget from Louis)
`/api/analyze`, `/api/record-analyze`, `/api/reinfer*`, `/api/jam/*`, `/api/yt-search`,
`/api/tab-search`, `/api/tab-fetch`, `/api/render-tab`, `/api/irealb-align`,
`/api/irealb-render`, `/api/irealb-import`. These call the big `_run_analysis` helper +
network + inference. *Gate:* pure-move + url_map identity + **real one-shot end-to-end
smoke-runs** (Louis approved). *Disk is tight (~2.7 GiB, 99%)* — `df -h .` first, no wav
renders, one at a time. Get an explicit budget before starting.

### Priority 4 — Core Phases 3–5 (the more central work; heavy prerequisites)
- **⚠ BLOCKER (issue #2 in the log): the benchmark does NOT yet capture chord internals**
  (nnls24 chroma, pre-coalesce root/quality labels, section phase). **Extend
  `parity.py::capture()` to cover them BEFORE Phase 3**, or the PORT parity gate is blind.
- **Phase 3 (chords):** decompose `infer_chords_v1` into a `ChordHead` interface + config.
  ⚠ It lives in `chord_pipeline_v1.py` — **grid-lane-owned decode region**; needs heavy
  coordination. Gate: exact same labels on the (extended) benchmark. Do NOT re-add
  trigram/neighbor context (rejected twice, −7.9pp on real audio).
- **Phase 4 (key/structure):** port the working path (flat block8 V_F 0.68–0.70 +
  `chordtone` clustering). Do NOT resurrect symbolic learned-similarity (9/9 negative,
  dead end).
- **Phase 5 (pipeline.py):** one `PipelineConfig` dataclass replaces the ~30 kwargs; the
  capstone full-pipeline parity gate.

## Env / process gotchas (all hard-won)
- `pytest -o addopts=""` — pyproject hardcodes `--cov`; env-dependent whether pytest-cov is
  present, so override it (do NOT install deps unattended).
- Disk runs tight (~2.7 GiB / 99%). `df -h .` before big runs; render one wav at a time +
  delete; a real disk-full has happened.
- RWC full-song audio is GONE — use `docs/audio` + `aligned_corpus` + cached features.
- Delegation doctrine: Opus subagents research history first, return TIGHT reports, and do
  **NOT** commit — the orchestrator commits specific files at each green gate. Every number
  from a real run (a fabrication was caught by audit before). Commit at green gates so a
  compaction is a non-event.
- Never advance a red gate. Any behavior change → kill-switch, default OFF.

## Quantitative continue/stop criteria
- Serving: keep moving routes while each is provable (byte-identical HTTP, OR move-identity
  + url_map identity + a safe smoke-run) AND `app.url_map` stays at 76 rules with 0
  namespaced `api.*`. Target end-state: `harmonia_server.py` a thin shell (mostly `main()`
  + blueprint registration). Log-and-skip any route you can't prove.
- Core: do NOT start Phase 3 until `capture()` covers chord internals and grid-lane
  coordination on `chord_pipeline_v1.py` is settled.
