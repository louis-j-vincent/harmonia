# handoff 3 — analysis + implementation plan (2026-08-02)

**Question asked:** "there's a `handoff 3/` folder from a design agent — analyse it and
make an implementation plan."

**Answer:** the design itself is already shipped. What remains to implement is not in
the handoff folder — it is the **server side of harmonia_min**, where the copied UI
already calls endpoints that 404.

## 1. What `handoff 3/` is, and why we don't port it

- Design drop dated 2026-07-13, never committed. Five surfaces: iReal chart viewer
  (Read/Analyse/Annotate), chord editor (Compass/Guide/cylinder/chips), search-first
  import, re-infer loop (confirm hard-clamp → `POST {confirms, merges}` → diff by time
  overlap → propagation banner), launcher.
- A prior session already did the integration diff (2026-07-29,
  `docs/handoff3_integration_diff.md`): **every surface is live in the old app**
  (`:7771`), wired to real endpoints where the handoff's `js/` engines still use mocks.
  Porting the engines would *regress* live code to mocks. That verdict still holds.
- Of the three cosmetic chart deltas that report flagged, #1 (inline section letters +
  per-cell ×N badges) is now **obsolete**: Louis re-decided that design on 2026-08-02
  in harmonia_min (×N lives only in the Form strip, section badge above the row —
  commit `37824e8`). Deltas #2/#3 remain optional micro-decisions (see §4).

## 2. Where the handoff's intent is actually unfinished: harmonia_min

`harmonia_min/app_shell.html` is a near-verbatim copy of the old shell — **all five
handoff surfaces exist as client JS** — but `server.py` is deliberately milestone-1
only, with a catch-all 404 on `/api/*` (`server.py:324`). Inventory (2026-08-02):

| Handoff surface | Client JS | Server backing in harmonia_min |
|---|---|---|
| Chart viewer (3 modes, fold ×N, rotor, playhead, confidence) | ✅ | ✅ live (`minimal_fold()`, content-based playhead map) |
| Import / analysing narration | ✅ | ✅ live (local library + yt-dlp URLs); iReal import, Record, Jam → 404 |
| Chord editor (Compass/Guide/cylinder/chips, confirm c=1) | ✅ | ⚠ edits work **in memory only** — `POST /api/annotations/<file>` 404s **silently** (`app_shell.html:3321`, try/catch swallows it) |
| Re-infer loop + propagation banner | ✅ | ❌ dead — no `/api/reinfer`, no `/api/context_rescore`; `span_rescore.py` + `chord_context_prior.py` are in the package but unrouted |
| Launcher / library | ✅ | ✅ live; Billboard/Training, section-merge game tiles → 404 (game link = hard Flask 404 page) |

Constraint from `docs/handoff_2026-07-30_minimal_pipeline.md`: the old `/api/reinfer`
machinery is on the DO-NOT-IMPORT list. So milestone 2 = **rebuild a minimal re-infer
on the new bricks**, not port the old one.

## 3. The plan (ranked, with stop criteria)

### P1 — annotation persistence (~1–2 h) — *silent data loss, fix first*
Every confirm/edit in Annotate mode is lost on reload. This violates the
"silent fallbacks are URGENT" rule and blocks the whole correction loop.
- `POST /api/annotations/<file>` → write sidecar `state/annotations/<file>.json`
  (schema: `docs/annotation_sidecar_schema.md`).
- Re-apply the sidecar when serving `/api/chart-model/<file>` (or a GET route the
  shell fetches on open — check how the old app rehydrates before choosing).
- Remove the silent catch: surface a toast on save failure.
- **Done when:** confirm a chord → reload page → ✓ and c=1 survive.

### P2 — minimal re-infer endpoint (~½ day)
Wire the already-copied bricks to the route the shell already calls.
- Route `POST /api/context_rescore/<file>` (shell option at `app_shell.html:3222`;
  same request/response contract as `/api/reinfer`).
- Implementation: `chord_context_prior.py` / `span_rescore.py` over the saved
  ChartModel + confirms; confirms are hard evidence, confirmed chords never touched.
- Response = the handoff wire shape (Harte labels, `diff[]` with
  `start_s/end_s/old_label/new_label/old_confidence/new_confidence`) — the client
  `applyResp` applies by time overlap unchanged.
- **Done when:** fix one chord → Re-infer → propagation banner lists ≥1 sharpened
  neighbour on a song where the prior actually moves something; confirmed chords
  bit-identical before/after.

### P3 — graceful degradation of dead buttons (~1 h)
Milestone-1 said "degrade gracefully"; three spots don't:
- Section-merge game tile → full-page bare Flask 404 (`app_shell.html:550`) — hide the
  tile or route it.
- Hide/disable until backed: iReal search toggle, Record, Jam Mode, Training/Billboard
  tile, iReal-export button. A small server capability flag
  (`/api/library` already returns JSON — add `"capabilities": [...]`) beats hardcoding.
- **Done when:** no tap in the UI leads to a raw 404 or a silent no-op.

### P4 — section merge (later milestone, ~½ day, after P2)
`merges: [[A,B]]` in the same wire + `/api/section-merge-candidates`. Pooled
re-infer = run P2's rescore over the union of both sections' spans. Only worth doing
once P2 is validated by ear on 2–3 songs.

### P5 — cosmetic close-out + cleanup (~30 min)
- Decide the two surviving deltas from the July report: **Print · gig chart** button
  (demo has it, live doesn't) and the **"READING VIEW" intro card** vs the bottom
  caption. Both are take-it-or-leave-it; neither blocks anything.
- Archive `handoff 3/` (zip → `docs/archive/`), delete the working copy — its
  reference value is exhausted by the two reports.
- Commit the untracked `docs/handoff3_integration_diff.md` alongside this plan.

## 4. Explicitly out of scope / do-not-do

- **No port of `handoff 3/js/*` engines** — mock-driven, older than the live shell.
- **No un-folding of Louis's 2026-08-02 chart decisions** (×N in strip only, badge
  above row, quarter-position grid, règle d'or lattice) to match the July demo.
- **No `scripts/migrate_annotator_tool.py` run needed** — that migration applies to
  the old app's baked snapshots; `harmonia_min/app_shell.html` is a plain served file.
- **Never touch :7771** (old production server, other sessions own it).
