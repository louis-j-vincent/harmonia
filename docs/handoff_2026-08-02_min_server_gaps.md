# Handoff — harmonia_min server gaps (P1 → P2 → P3)

**For:** a fresh Claude session (agent 2), 2026-08-02.
**Context owner:** another session is concurrently implementing the New-Chord-UX
design handoff **inside `harmonia_min/app_shell.html` — that file is OFF-LIMITS
to you.** Your surface is `harmonia_min/server.py` + new server-side files.

## Read first (in this order)
1. `docs/plan_handoff3_implementation.md` §4 — your track, decided today.
2. `docs/handoff_2026-07-30_minimal_pipeline.md` — the milestone framing; the
   old `/api/reinfer` machinery is on its DO-NOT-IMPORT list.
3. `docs/minimal_pipeline_log.md` (tail) — current state + iOS audio fixes.
4. `docs/annotation_sidecar_schema.md` — sidecar shape used by the old app.
5. CLAUDE.md process rules (esp. #1 calibration checks, #2 premise screening).

## Hard constraints
- **Never edit `harmonia_min/app_shell.html`** — concurrent session owns it.
  If an endpoint contract seems to need a client change, write the need into
  this doc under "Client asks" and stop; the other session will pick it up.
- **Never touch port :7771** or `scripts/harmonia_server.py` (old prod, other
  sessions own it). harmonia_min runs on **:7772** (`python -m harmonia_min.server`,
  no reloader — **restart it after every server.py change** and say so).
- Git: work on branch `feat/chord-lm` (same as the UI session — file sets are
  disjoint, so no conflict). Stage explicit paths only; never `git add -A`.
  Commit your own files only: `harmonia_min/server.py`, new `harmonia_min/*.py`,
  `harmonia_min/state/annotations/` fixtures, tests, docs.
- Tests: red-first. Prefer RWC/real charts in `harmonia_min/state/charts/` as
  fixtures over synthetic JSON.

## P1 — annotation persistence (~1–2 h) — DO THIS FIRST (silent data loss)
The shell already POSTs `/api/annotations/<file>` on every chord lock
(`saveAnnotations`, app_shell.html:3312) and swallows the 404 silently. Every
confirmed/edited chord is lost on reload.

1. Characterize first: read `saveAnnotations()` + `confirmChord()`/`splitBar()`
   in app_shell.html (reading it is fine — editing is not) to see the exact
   payload; read the old app's annotations route (`harmonia/serving/api.py` or
   `scripts/harmonia_server.py`) for the sidecar precedent.
2. `POST /api/annotations/<file>` → write `harmonia_min/state/annotations/<file>.json`
   (atomic write; keep the schema compatible with `docs/annotation_sidecar_schema.md`
   unless the shell's payload says otherwise — payload wins).
3. Rehydrate: when `/api/chart-model/<file>` serves a chart, overlay the sidecar
   (confirmed roots/qualities, `c=1`, splits) onto the model **server-side** so the
   shell needs no change.
4. **Done when:** lock a chord in Annotate on :7772 → reload the page → the lock
   survives (✓ badge, c=1). Verify live in a browser, not just curl. Log the fix
   in `docs/minimal_pipeline_log.md` + commit in the same sitting.

## P2 — minimal re-infer endpoint (~½ day)
The shell POSTs `(endpoint||"/api/reinfer/")+file` from `runFlow`
(app_shell.html:3237-3288) and applies the response diff by TIME OVERLAP.
`harmonia_min/span_rescore.py` and `harmonia_min/chord_context_prior.py` are in
the package, import cleanly, and are reached by NO route ("unrouted milestone-2
bricks", minimal_pipeline_log.md:149).

1. Premise check first (rule #2): on one saved chart, run the context prior
   over the model + 1-2 fake confirms in a scratch script — does anything
   actually change? If nothing moves on any chart, STOP and report; don't build
   a dead endpoint.
2. Route `POST /api/context_rescore/<file>` (and alias `/api/reinfer/<file>` to
   it): body `{confirms:[{t0,t1,root,q}], merges:[]}`. Confirms are hard
   evidence — confirmed spans must come back bit-identical.
3. Response shape (the wire is Harte labels — see app_shell parseLabel:327):
   `{key, tempo_bpm, n_changed, diff:[{index,start_s,end_s,old_label,new_label,
   old_confidence,new_confidence}], chords:[]}`.
4. **Done when:** lock one wrong chord on a real chart → Re-infer → the
   propagation banner lists ≥1 changed neighbour, confirmed chords untouched,
   and the change is defensible by ear or against iReal/UG. Report the diff you
   got, not just "it worked".

## P3 — graceful degradation (~1 h, server-side only)
`/api/library` already returns JSON; add `"capabilities": [...]` listing the
live features (e.g. `["annotations","reinfer"]` once P1/P2 land). Do NOT hide
the buttons yourself — that's app_shell (client) work; just expose the field
and note it under "Client asks" below. Meanwhile: make `/debug/section-merge-game`
return a friendly 404 JSON/HTML instead of the bare Flask page (it is currently
a full-page dead end reachable from the library).

## Stopping criteria / budget
P1 is non-negotiable today. P2 stops either when the acceptance test passes or
when the premise check says the bricks don't move anything (report which). If
blocked >30 min on something owned by the other session, write it under
"Client asks" and move to the next item. Check disk space at every check-in.

## Client asks (write here; the UI session reads this)
- (none yet)
