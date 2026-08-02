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
- Git: work on branch `feat/chord-lm`, in this same working tree, alongside the
  UI session. **A separate worktree/branch was considered and rejected on
  facts**: `harmonia_min/state/` (all the charts), `docs/audio/` (all the
  audio), `.venv` and `data` are every one of them gitignored, so a fresh
  worktree starts with an app that has nothing to serve and no interpreter —
  and any symlink fix hands back the shared runtime state that the isolation
  was for. Two sessions on disjoint files (you: `server.py`; them:
  `app_shell.html`) is the cheaper, safer arrangement.
  Stage explicit paths only; never `git add -A`. Never `git checkout`,
  `stash`, or `reset` — the other session's uncommitted work lives here too.
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

Written by the server session, 2026-08-02, after P1–P3 landed (commits
`4c2152f`, `e6322c6`, `23bcd1c`). All three are client-side calls I did not
make, per the hard constraint.

1. **`/api/library` now returns `capabilities: ["annotations","reinfer"]`.**
   Nothing reads it yet. Hide/disable a feature's buttons when its capability
   is absent, rather than letting the call fail into a silent `catch`.
   `reinfer` is reported only when the trained prior table really loads —
   without it the re-score can never change anything, so the button would look
   alive and do nothing.

2. **`saveAnnotations()` hardcodes `bass:-1`** (app_shell.html ~:4011), so a
   confirm carries no bass opinion. The server therefore treats `-1` as "keep
   the model's bass" — otherwise confirming This Love's opening `G/B` would
   delete the `/B`. If you ever want a user to *remove* a slash bass, the
   payload needs a distinct value (e.g. `bass:null` = "no opinion",
   `bass:-1` = "explicitly none"); today those two cases are indistinguishable.

3. **`(bar, beat)` is not unique** in a folded ChartModel (Norah Jones `LA`,
   Stand By Me `LB` replay one bar identity). A lock is applied to **every**
   rendered copy — deliberate, since they are the same folded material. If the
   UI ever needs two passes of a repeat to differ, the payload needs a
   per-occurrence key; flagging it rather than inventing one.

4. **Re-infer's honest behaviour, for the banner's copy:** a *correct* lock in
   an already-confident neighbourhood propagates **zero** changes (measured on
   both verified corrections available — details in
   `docs/minimal_pipeline_log.md` P2). The existing "Locked in — nothing else
   moved" toast is exactly right for that case; no change needed, just don't
   read zero as a failure.

5. **A changed span loses its seventh** (the lattice decodes 5 families, so a
   rescored `C-7` returns as `Cm`). Unchanged spans keep their full quality.
   If that downgrade is visible enough to bother a user, the fix is server-side
   and I'd need to know it matters.
