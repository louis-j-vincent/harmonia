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

## Round 2 (same evening)

Louis: 4 bars instead of 2, a piano so you can play along, and a
speed slider. Shipped:

- reel window = **four** bars (marker moved to 18% to keep the forward
  view long);
- a **play-along keyboard** between the big chord and the reel — the
  coach's voicing engine (`voicingByStyle` + `withBass` + `renderVoicing`)
  on the prompter's own pre-section chord; style follows the 🎹 coach pref
  ("smooth" falls back to close: it is chart-indexed voice leading);
- a **speed slider** (0.5–1×, pitch preserved) above the reel;
  `applyPlayRate()` re-applies on every fresh `Audio()` since the element
  resets to 1×. Verified: at 0.5×, 4 real seconds advance the audio 2 s.

## Round 3 (same evening): "plus user-friendly"

Louis: big play button, easy back/forward, and "think of other ideas".

- **Big transport in the middle**: 76px play/pause, flanked by round
  ‹‹4 / 4›› buttons (redo the phrase / skip ahead — also ←/→ keys, space
  = play/pause on desktop).
- **Direct manipulation**: drag the reel like a tape (right = earlier),
  tap = jump to that instant; plus a **full-song colour strip** (every
  chord as a thin root-hue block — the /compare demo's timeline, app
  colours) with tap/drag absolute seek and a playhead.
- Own ideas shipped: **LOOP 4** — cycles the four bars around the
  playhead (the practice feature: band drawn on the strip, auto-rewind
  at the end); **tap the piano card to hear the voicing** (playMidis,
  the coach's preview synth); **screen wake-lock while playing** so the
  phone doesn't sleep mid-practice; a one-line hint under the controls.
- Fixed along the way (all screens): a paused seek now refreshes the
  transport's time label + scrub position — seekFrac used to leave them
  frozen; only the timeupdate-while-playing path updated them.

## Round 4: standalone screen

Louis: kill the bottom bar (all duplicates), swipe to leave, and snap
the loop to the 4-bar grid. Done:

- the transport bar hides on the prompter (its big play/jumps/strip
  carry everything; the reel gains the freed bottom space);
- **swipe down** anywhere (outside the reel/strip/slider) goes back to
  the chart; the ‹ button and browser-back still work;
- **LOOP snaps to the 4-bar PHRASE grid** (bars 0-3, 4-7, …): pressed at
  bar 61 or 63, it cycles the same 61-64 phrase — a musical unit, not
  four bars from wherever the finger landed.

## Round 5: two hands

Louis: left hand / right hand, with the right hand moving as little as
possible between chords. The §9 smooth cascade already was that engine;
what changed:

- `vlCandidates` gains a **freeBass** mode: with the left hand owning
  the bass, the right hand drops the keep-the-bass-lowest rule (§11 is
  a one-hand doctrine) — inversions (index-0 lifts) and rootless shapes
  become legal candidates;
- the prompter precomputes the cascade over ITS chord stream (not the
  chart's): LH = sounding bass in the F2–E3 window (red on the keys),
  RH = minimal-movement voicing (blue), named "L B · R G–B–D (close)";
- tap-to-hear plays both hands, bass first.

Measured on This Love: G/B → Cm voices as G–B–D → E♭–G–C — G held,
B→C and D→E♭ a semitone each (2 semitones total instead of leaping to
root position). One real bug caught in the port: `renderKeys` appends
without clearing (renderVoicing's `clear` lived one level up), so keys
piled up on every chord change until a `clear(kb)` was added.

## Round 6: split-hands view + the loop wasn't smooth

- **Split option** on the piano card (persisted, default on): one small
  keyboard per hand — R on top (blue), L below (red) — each zoomed on
  its own range; JOINED brings back the single long keyboard.
- **Loop wrap fix** (Louis: "delta de temps avant de loop, pas smooth"):
  the wrap test ran on the lead-shifted display clock, so it fired
  0.18 s (PLAYHEAD_LEAD_S) before the bar line and cut the end of bar 4.
  Now it wraps on the RAW audio clock, carries the frame's overshoot
  into the next pass (currentTime = a + overshoot, capped 0.25 s) so the
  groove keeps phase, and a deliberate jump >1 s past the loop end
  disarms the loop instead of yanking the playhead back. Measured:
  overshoot 0.03 % of the song (~one frame), wrap lands on the bar line.

## Not solved here

- This Love's stored `keyName` says "F minor" — that's the chart's own
  harmonic-key verdict, displayed as-is; not touched by this feature.
- LOOP is fixed to the containing 4-bar phrase — no custom in/out
  points yet; the band is drawn on the strip but not draggable.
- The phrase grid assumes phrases start at bar 1; a chart with a pickup
  bar or odd intro length shifts the groups — section starts would be a
  better anchor once the prompter is allowed to know about sections.
