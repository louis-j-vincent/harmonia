# Waveform Annotator v2 — Rebuild Results

**Date:** 2026-07-14
**Route:** `GET /annotator-v2?song=<slug>` (in `scripts/harmonia_server.py`)
**Template:** `ANNOTATOR_SIMPLE_TEMPLATE` (single file, vanilla JS, ~430 lines JS)
**Old route left untouched:** `/annotator` (the cursed 900-line-JS template) still works.

## Step 1 — Diagnosis of the old `/annotator` failures

Root causes, from reading the route + template:

1. **Chords don't load / 404s.** `/annotator` hard-requires
   `docs/plots/irealb_<slug>.html` with a `window.P` payload. When missing it
   tries a *network* iReal community fetch inline in the request path
   (`search_community` → `render_irealb_chart`), which can hang or fail on a
   phone, then falls back to a 4-bar placeholder. Only **2 of 10** downloaded
   songs (`autumn_leaves`, `let_it_be`) actually have *both* a chart and audio.
2. **Audio didn't load** — the serving side is actually correct (`/audio/<f>`
   returns `206 Partial Content`, `audio/mp4`, `Accept-Ranges: bytes`, which is
   exactly what iOS Safari needs to seek). The real fragility was in the old
   template coupling *playback* to Web-Audio decoding: if `decodeAudioData`
   failed, the whole timeline broke.
3. **Waveform didn't render.** The old template decodes the m4a via
   `AudioContext.decodeAudioData` and draws to canvas. On iOS this is the
   flakiest link (older Safari only supports the *callback* form and returns
   `undefined` instead of a Promise; an `AudioContext` may also start
   suspended). A decode failure took the waveform *and* left no visible reason.
4. **Confusing UI.** Modals, band-labels, sync-drift dots, peak-snap cues,
   shift-click multi-select merge, keyboard nudges — none discoverable on a
   touch screen, no instructions.

## Step 2–4 — What the rebuild does

Single linear vertical flow, mobile-first, no tabs/modals:

1. **Loading overlay** with 3 explicit live steps (`✓ chords / ✓ audio /
   ✓ waveform`, each flips green or red).
2. **Instruction card** at the top in plain language.
3. **Player card:** 64px play/pause, `m:ss / m:ss` clock, seek slider, volume
   slider, and a big **current-chord** readout that updates during playback.
4. **Waveform card:** horizontally-scrollable canvas (60 px/s) with a red
   playhead that auto-scrolls to follow playback; chord regions overlaid as
   translucent blocks with **44px drag handles** on each edge.
5. **Chord list:** one row per chord — jump (`▸`), label, start time, and
   **−/+** buttons (±0.1s) for fine control. Rows highlight the playing chord.
6. **Sticky Save bar** with a live "N edited" counter; Save disabled at 0 edits.

Key robustness choices:
- **Playback and waveform are decoupled.** Sound uses a plain `<audio
  playsinline>` element (Range-served, reliable). The waveform is a *separate*
  fetch+decode that is purely cosmetic — if it fails the page says so
  ("Waveform preview unavailable (drag still works)") and everything else keeps
  working. No audio → "Audio not available — timings still editable."
- `decodeAudioData` handles **both** the Promise and callback signatures.
- A 6s watchdog guarantees the loader never hangs forever.
- Touch: `pointer*` events, `touch-action:none` on handles, all targets ≥44px,
  no hover-only affordances, viewport zoom locked.
- **Same save contract** as the old tool: `POST /api/annotations/<saveFile>`
  with `{annotator, chords:[{bar,beat,label,section,t0,t1,ts}], merges}`, keyed
  by `bar:beat`, so it resumes prior alignments and interoperates with the
  existing sidecars. (The reinfer / correction-log side-channels were dropped
  per the "just drag + play + save" brief.)

## Step 5 — Test results (server-side, this host)

| song | route | audio | chords | notes |
|------|-------|-------|--------|-------|
| `the_beatles_..._let_it_be...` | 200 | ✓ `/audio/…m4a` | ✓ | full chart+audio |
| `autumn_leaves` | 200 | ✓ | 66 | full chart+audio, dur 160s, 1291 beats |
| `my_baby_just_cares_for_me` | 200 | — (none) | ✓ | graceful no-audio mode |
| `ten_times_blue` | 200 | — (none) | ✓ | graceful no-audio mode |
| `doesnotexist123` | 404 | — | — | clean 404 fallback |

Verified directly:
- Injected `const D = {…}` payload is **valid JSON** (12 keys, 66 chords with
  well-formed `t0/t1`).
- Audio endpoint: `206 PARTIAL CONTENT`, `Content-Type: audio/mp4`,
  `Accept-Ranges: bytes` — iOS-seek-ready.
- **Save round-trip:** POST of the v2 payload shape is accepted and read back
  identically (`chords[0]` preserved bar/beat/label/section/t0/t1/ts).
- Old `/annotator` and `/api/beat-grid` still return 200 after the refactor.

## Not done / caveats

- **No on-device iPhone Safari screenshot.** This environment is headless (no
  browser), so touch-drag, Web-Audio decode, and `<audio>` playback are
  *designed and coded* for iOS Safari and verified server-side, but the actual
  on-glass behavior is the user's confirmation step. Open
  `http://<lan-ip>:<port>/annotator-v2?song=autumn_leaves` on the phone.
- Dragging moves a **shared boundary** (chord i's right edge == chord i+1's
  left edge stay contiguous); it does not insert/delete/merge chords — out of
  scope by design.
- Two songs (`autumn_leaves`, `let_it_be`) are the only ones with both chart
  and audio; the rest exercise the no-audio path.

## How to run

```
python scripts/harmonia_server.py --port 7771
# then on the iPhone (same Wi-Fi):
#   http://<lan-ip>:7771/annotator-v2?song=autumn_leaves
```

## Files changed

- `scripts/harmonia_server.py`:
  - extracted `_build_annotator_data(slug) -> (data, err)` (shared by both routes)
  - added route `annotator_v2()` → `/annotator-v2`
  - added `ANNOTATOR_SIMPLE_TEMPLATE`
- `docs/waveform_v2_redesign_results.md` (this file)

No commit made (left for review, per brief).
