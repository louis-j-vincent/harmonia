# 22 — The prompter: the /compare reel moves into the app

*2026-08-05*

Louis asked for the karaoke-style scrolling view — one big current chord,
and underneath a roll of what's coming — to graduate from the `/compare`
demo (commit `b7fb411`, old server's A/B/C ear test) into the harmonia_min
app, with the app's colours, fed by the new chord detection, and with
structure analysis deliberately left out ("on s'en fout de la détection de
sections, tu t'arrêtes avant").

## What shipped

**A new "prompter" screen in `harmonia_min/app_shell.html`**, reached from a
🎞 button in the transport (chart ↔ prompter switch keeps audio playing):

- the current chord huge in the middle, rendered with the app's own `glyph()`
  (serif italic, notation pref, key-aware enharmonics), coloured by the
  root's circle-of-fifths hue — theme-aware, and neutral ink when the "Key
  colours" pref is off, same contract as `lensBand`;
- at the bottom, a reel showing exactly the **next two bars** (window =
  2 × median bar from `barGrid`), chords as tinted blocks scrolling
  right→left past a fixed accent "now" marker; bar lines drawn from the
  chart's own `barGrid`; tap the reel to seek;
- scrolling is one `transform` write per rAF frame, only on this screen —
  not the 60/s full-DOM loop that starved iOS audio startup; paused scrubs
  reposition via the `timeupdate` hook like everything else, and the
  playhead lead (`PLAYHEAD_LEAD_S`) applies.

**The data is the pre-section decode.** `pipeline.analyze()` now captures
`model["prompter"] = {chords: [{t0,t1,root,q,bass,nc,c}, …]}` right after
the bar grid is computed and **before** `detect_sections` and before
folding's template re-decode rewrites bar chords. The prompter renders what
the chord detection heard, full stop.

**Backfill**: `scripts/backfill_prompter.py` stamped the key onto all 33
existing charts without touching anything else in them — it re-decodes from
each chart's *stored* `beatTimes`/`barGrid`/`bpb` (so the timeline matches
the grid the playhead already uses) with the cached musx posteriors.
33/33 ok, seconds per song.

## Verified

Playwright drive on This Love, light+dark themes, zero JS errors: reel
scrolls during playback, big chord follows N.C. → Cm → Fm7, and the reel
announces Ddim → Fm/Ab → G/B ahead of the pre-chorus. Server restarted so
new analyses carry the key; old charts without it fall back to the folded
chart chords (works, but re-bake to get the true pre-section stream).

## Not solved here

- The transport's time label still doesn't refresh on a paused seek
  (pre-existing, all screens).
- This Love's stored `keyName` says "F minor" — that's the chart's own
  harmonic-key verdict, displayed as-is; not touched by this feature.
