# Harmonia UI/UX handoff — for Claude Code

## ⭐ START HERE — the unified app + how to feed it real data

`Harmonia App.dc.html` is the **whole app in one file**: launcher → search/library →
analysing → chart (iReal, Read/Analyse/Annotate) → tap-to-edit (Compass/Guide/By-hand)
→ re-infer + section merge → YouTube-audio playback synced to the grid. `window.APP` is
the engine; `harmonia_app_standalone.html` is the same thing pre-bundled (open it to see
the exact target). Everything below (the per-surface engines in `js/`) is the same code
split up — use the unified file as the source of truth; use the split engines only if you
prefer to wire one surface at a time.

**The real inference is messy and under-structured. Do NOT feed it raw.** The UI expects
one clean, normalised shape. Build a single adapter — `to_chart_model(inference) -> ChartModel`
— in Python (server side) and hand the UI *only* that. The rules below are the contract;
if the model can't fill a field, apply the stated default rather than omitting it.

### INPUT the UI needs — `ChartModel` (one JSON object)
```jsonc
{
  "title": "Autumn Leaves",
  "video_id": "sTfmMd0uOEc",     // YouTube id the audio was fetched from (for playback). REQUIRED for sound.
  "key":   { "tonic": 8, "mode": "major" },  // tonic = pitch-class 0..11 (C=0). mode "major"|"minor".
  "bpb": 4,                        // beats per bar
  "sections": [                    // the FORM, in play order, repeats folded (see rules)
    { "label": "A", "reps": 2, "bars": [ Bar, Bar, ... ] },
    { "label": "B", "reps": 1, "bars": [ ... ] },
    { "label": "C", "reps": 1, "bars": [ ... ] }
  ]
}
// Bar = array of 1–2 Chord (2 chords = split bar, 2 beats each; >2 not rendered — merge them first)
// Chord = {
//   "root": 0..11,               // pitch-class of the root (C=0, C#=1 … B=11)
//   "q": "-7",                   // iReal quality token (see table) — NOT a Harte label here
//   "c": 0.0..1.0,               // calibrated confidence (P the chord is correct)
//   "t0": 6.68, "t1": 8.64       // start/end in SECONDS of real audio (for playhead + re-infer spans)
// }
```
Quality tokens `q` the renderer understands: `"" ^7 6 7 - -7 -^7 o -7b5 o7 9 7b9`
(major triad is `""`, dim triad `o`, half-dim `-7b5`). Map anything exotic to its nearest
family; never pass a raw Harte string as `q` here (Harte is only used on the re-infer wire — see below).

### NORMALISATION RULES (turn messy inference into the shape above)
1. **Root as pitch-class 0..11**, not a name — resolve enharmonics yourself; the UI spells
   flats for display.
2. **One `key` for the tune** (or per-section if it truly modulates). If the inference emits
   a per-bar key that never changes, collapse it to one. Default mode `major` if unknown.
3. **Sections must be real spans, not per-bar labels.** If the inference tags every bar with
   the same letter, run your changepoint/segmentation to produce A/B/C blocks. If you can't,
   emit a single section `{ "label": "A", "reps": 1, "bars": [...] }` — the UI still works.
4. **Fold repeats.** If two consecutive sections are the same music, emit ONE section with
   `reps: N` (rendered once, badged `×N`) — never duplicate the bars. If two same-letter
   sections differ only because the model was unsure, keep them separate (`A`,`A`) so the
   user can **merge** them (that triggers a pooled re-infer — see outputs).
5. **≤2 chords per bar.** If inference gives 3+, keep the two strongest by confidence (or
   merge by beat) before sending.
6. **`c` is calibrated 0..1.** If you only have raw logits, calibrate (Platt/isotonic) so
   `0.4` genuinely means "40% likely" — the whole Annotate UX (colour ramp, family-only
   fallback under 0.42, the `?` flag) is driven by this number being honest.
7. **`t0`/`t1` in seconds, monotonic, covering the whole track.** These drive the synced
   playhead and are the spans the re-infer request sends back. Split-bar chords split the
   bar's time in two.
8. **Per-candidate suggestions (optional but recommended).** For each chord you may add
   `"sug": [ { "root", "q", "c" }, … ]` (2–3 alternates). The Compass/Guide use them; if
   omitted the UI derives fallback candidates, but real model alternates are far better.

### OUTPUT the UI produces — what your server must accept
- **Playback:** the UI plays `video_id`'s audio (hidden YouTube IFrame) and maps
  `currentTime` onto each chord's `[t0,t1]`. Your only job is a correct `video_id` + honest
  `t0/t1`. (If you serve your own audio instead of YouTube, swap the player in `ensureTransport`.)
- **Re-infer with corrections** — `POST /api/reinfer/<file>`:
  ```jsonc
  { "confirms": [ { "t0":6.68, "t1":8.64, "root":2, "q":"7" } ],   // chords the user hard-locked
    "merges":   [] }
  ```
- **Section merge** — same endpoint, its own call:
  ```jsonc
  { "confirms": [], "merges": [ ["A1","A2"] ] }   // pool these sections into one shared reading
  ```
- **Re-infer RESPONSE** the UI expects back (this is the ONE place labels are Harte
  `"root:quality"`, because that's your model's native tongue):
  ```jsonc
  { "key": "G# major", "tempo_bpm": 112.3, "n_changed": 3,
    "diff": [ { "index": 5, "start_s": 8.64, "end_s": 10.8,
                "old_label": "G:hdim7", "new_label": "G:7",
                "old_confidence": 0.33, "new_confidence": 0.71 } ],
    "chords": [] }
  ```
  The UI applies each `diff` entry by **time overlap** onto unconfirmed chords and shows a
  "what your fix sharpened" banner. `confirms` are hard-clamped (`c=1`) and never overwritten.
  Harte→token map used on this wire: `maj""  maj7^7  min-  min7-7  7→7  hdim7→-7b5  dim→o  dim7→o7  6→6`.

> Swap-to-live is one line in `window.APP`'s `reinfer()` (uncomment the `fetch`, delete the
> mock). The mock returns exactly the response shape above, so once the endpoint matches, it
> just works. `to_chart_model` + this endpoint are the entire integration surface.

---

Design pass on the Harmonia front-end. This folder contains 4 **design-source files**
(`.dc.html`). They open in a browser preview tool, but for integration you only care
about the **`<helmet><script>` blocks** in each — those are **plain vanilla JS/DOM**
(no framework, no build step) and are meant to be lifted into
`harmonia/output/chart_interactive.py` (the triple-quoted HTML/CSS/JS template).

> The device frames (phone/iPad/browser-window markup in the `<x-dc>` body), the
> `data-*="A:phone"` mount attributes, the `<meta design_doc_mode>`, and the trailing
> `class Component extends DCLogic` block are **preview scaffolding only** — ignore them.
> Everything real is inside the `window.XXX = (function(){…})()` engines and the
> `build*()` functions.

## Winners to ship
| Surface | Winner | Source file |
|---|---|---|
| Chord editor | **BOTH** — Compass + Guide (offer as two tabs/modes) | `Harmonia Chord Editor.dc.html` (`window.HZ`) |
| Chart viewer | **iReal-style** as the default view | `Harmonia Chart Viewer.dc.html` (`window.HZC`, `buildIReal`) |
| Import / search | **Search-first** | `Harmonia Import.dc.html` (`window.IMP`) |
| Re-infer + section merge | the collaborative loop | `Harmonia Re-infer.dc.html` (`window.RI`) |
| Launcher | the redesign (replaces `harmonia.html`) | `Harmonia Launcher.dc.html` (`window.HLAUNCH`) |

## ✅ Easiest path — the drop-in engines in `js/`
Each winner is also provided as a **single self-contained `.js`** plus a runnable
`*_demo.html` reference. No framework, no build, no dependencies — the file injects its
own CSS and auto-initialises on DOM ready.

| Drop-in | Reference to match |
|---|---|
| `js/harmonia_chart.js` | `js/harmonia_chart_demo.html` |
| `js/harmonia_chord_editor.js` | `js/harmonia_chord_editor_demo.html` |
| `js/harmonia_import.js` | `js/harmonia_import_demo.html` |
| `js/harmonia_reinfer.js` | `js/harmonia_reinfer_demo.html` |
| `js/harmonia_launcher.js` | `js/harmonia_launcher_demo.html` |

**To use one:** `<script src="harmonia_chart.js"></script>` then add the mount element
shown at the top of that `.js` (e.g. `<div data-hzc="C:desktop"></div>`). Open the
matching `*_demo.html` in a browser first — **that is the exact target to reproduce.**

**Adapting the data:** each engine keeps its demo data in a clearly-commented block
near the top (chart: `BARS`/`SECS`; editor: `SONG`; import: `RESULTS`/`LIBRARY`). The
shapes are already what Harmonia produces, so wiring real data is a rename/mapping, not
a rewrite — see each surface's section below for the exact `P.chords` mapping. If your
real data differs slightly, adapt the mapping; the rendering code doesn't need to change.

---

## Design tokens (already match the existing `:root`)
```
--paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a;
--faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --deep:#2a2622;
green:#1f8a5b  amber:#c58a2e
```
Type: chord glyphs = **Georgia italic** (root + `<sup>` quality); UI text = system-ui.
Logo wordmark: lowercase italic **harmon**(ink)**ia**(accent).

---

## 1. Chart viewer — iReal-style (default) + 3 modes
Source: `Harmonia Chart Viewer.dc.html`, engine `window.HZC`.

- **Three modes** (segmented control, replaces the old catch-all "Options" menu):
  - **Read** — clean chords, no colour (gig/reading default).
  - **Analyse** — colour ON, with a sub-toggle *Function* vs *Key*. Function colour
    (`fnOf`, `FN`) tints by harmonic job (Home/Setting-up/Pulling — the I/IV/V anchors
    from `docs/colour_system_2026-07-07.md`); Key colour uses the shipped
    circle-of-fifths hue (`keyHue`/`keyFill`, same `tonic*7 mod 12` mapping as `colOf`).
  - **Annotate** — chord text coloured by certainty (amber→red→black, `certaintyColor`),
    dotted underline when `c < 0.65`, tap-to-fix, and **merge sections** affordance.
- **iReal layout** (`buildIReal` + `irealGlyphs`): continuous vertical barlines, boxed
  maroon section letters on a double left barline, multi-chord bars **side by side**
  (a bar = `{ch, ch2?}`), maroon **playhead** bar on the right edge, `4/4 · key · style`
  header, "Auto: shows 7th/exact only where certainty ≥ 0.60…" caption.
- **Repeat folding** (`USECS`, `locate`): identical consecutive sections collapse to one
  block badged `×N` — never render the same bars twice; the playhead still reports the pass.
- **Transpose = a circle-of-fifths rotor** (`buildRotor`) behind the Key pill, NOT a
  stepper (key changes are rare; make it a deliberate "spin to a new key" moment).

**Wire to real data** — the demo uses a hand-typed `BARS`/`SECS`; replace with `P`:
```
P.chords[i] = { root, bass, bar, beat,
                lv:{ family:{q,c}, seventh:{q,c}, exact:{q,c} },  // c ∈ 0..1 = certainty
                t0, t1, sug:[{root,q,c}] }
P.sections (per-bar), P.sectionChips (form), P.bpb, P.home = {tonic,mode}
```
- `certaintyColor(c)` ← use the shown-depth `lv.<depth>.c` (respect the existing
  "Auto" depth slider — colour = certainty at the reported depth).
- `fnOf(root,q)` in the demo is keyed to a fixed relative-major ref; replace with the
  real per-chord function from `harmonia/theory/local_key.py` (`local_key_track` /
  `consolidate_dominant_chains`) so secondary-dominant/ii-V colour is exact.
- Section merge → persist to the annotation sidecar `store.merges` (see
  `docs/annotation_sidecar_schema.md`), same as the existing merge feature.

## 2. Chord editor — Compass + Guide
Source: `Harmonia Chord Editor.dc.html`, engine `window.HZ`. Drop into `#chordEditModal`.

- **Compass** (`buildCompass`) — NEW circular suggestion view (the deferred idea in
  `docs/architecture_extensions.md §1b`). Candidates from `P.chords[i].sug` orbit the
  current chord; **angle = circle-of-fifths position of the root** (`fifthsIndex`),
  **size = confidence** (`sqrt(c)`), **hue = root's key colour**. A relaxation pass
  (in `buildCompass`) guarantees no orbs overlap. Tap → arpeggio (`play`) + select.
- **Guide** (`buildCards`) — ranked cards with a plain-language "why" per candidate
  (`roleOf`): ii, V7, secondary dominant, tritone sub, borrowed, etc. The demo's
  `roleOf` is a good heuristic but **should call the real theory** (`jazz_priors.py`
  `PROGRESSIONS`, `local_key.py`) for exactness.
- Manual edit: **cylinder** (`buildCylinder`, iOS scroll-snap Root·Family·7th·Ext — the
  shipped v2) and/or **chip grid** (`buildChips`). Chord-tree data
  (`FAMILIES`/`SEVENTHS`/`EXT`/`Q2FS`) mirrors `harmonia/theory/chord_tree.py`.
- `play(root,q)` = Web-Audio arpeggio (replaces `playChordArpeggio`); `TOK`/`Q_FULL_LABEL`
  map quality tokens → labels.

## 3. Import / search — search-first + analysing
Source: `Harmonia Import.dc.html`, engine `window.IMP`.

- Entry: one field that **searches YouTube or takes a pasted link**, recent charts below.
- **Analysing state** (`analysing`) narrates the real pipeline with a result chip per
  stage — wire each stage to actual progress from `harmonia_server.py` (SSE or polling):
  `Fetching audio → Listening for notes (Basic Pitch) → Beat & tempo (madmom) →
   Sections (changepoint) → Key (Krumhansl) → Chords (HMM)` → "Chart ready → open".
- `RESULTS`/`LIBRARY` are mock — replace with the search endpoint + saved-charts list.

## 4. Re-infer with my corrections — collaborative loop + section merge
Source: `Harmonia Re-infer.dc.html`, engine `window.RI`. This is the *behavioural*
wrapper around editing — it hosts the real chord editors (§2 Compass/Guide), it is
not a third editor. In production, tapping a chord opens the Compass/Guide; the
confidence %, hard-clamp, Re-infer button, and propagation banner shown here wrap
around it.

- **Uncertainty as level-of-detail** (`depthOf`, `glyph`): confident chords render
  exact; shaky ones (`c < 0.42`) collapse to just the family (`G°`, not `Gm7♭5`) with
  a `?`. **Confidence % is always printed** under each unconfirmed chord
  (`confColor` ramp = calibrated certainty; the legend explains it).
- **Confirm = hard clamp** (`confirmChord`): sets `c=1`, marks `confirmed`, swaps the
  `?`/`%` for a maroon `✓`. Confirmed chords are never re-touched by a re-infer.
- **Re-infer** (`runReinfer` → `buildRequest` → `reinfer`): POSTs
  `{confirms:[{t0,t1,root,q}], merges:[]}`; applies the response `diff` by **time
  overlap** (`overlaps`, `parseLabel`) onto unconfirmed chords, then
  `showPropagation` reveals which nearby chords the fix sharpened (old→new label +
  confidence).
- **Section merge** (`SECTIONS`, form ribbon, `toggleSec` → `openMerge` → `runMerge`):
  two-tap two sections → confirm sheet → **its own re-infer** ("Pooling both
  sections…"), POSTing `{confirms:[], merges:[[idA,idB]]}`. Pooling doubles the
  evidence; the same `applyResp`/`showPropagation` payoff runs with a merge banner.
- **Live wiring**: `reinfer()` has the mock and the real `fetch("/api/reinfer/"+file)`
  side by side — uncomment one line. `mockReinfer` branches on `body.merges.length`
  to return `MERGE_DIFF` vs `REINFER_DIFF`; the live server makes that branch moot.
- Persist confirms + merges to the annotation sidecar (`docs/annotation_sidecar_schema.md`).

## 5. Launcher (`harmonia.html`)
Source: `Harmonia Launcher.dc.html`, `window.HLAUNCH`. Keep the **real** `check()`
(fetch `localhost:7771` no-cors, 3s poll, auto-redirect on success). **Delete the
bottom "preview" state switcher** — that is a review aid only; the live page drives
state from the actual server check. States: Checking / Not-running (start command +
click-to-copy + auto-watch) / Running (Open button + redirect).

---

## ⚠ Migration — do not skip (from `docs/handoff_2026-07-13_annotator_ui.md §5`)
`chart_interactive.py` is a **template**; editing it only affects charts rendered *after*.
After changing its HTML/CSS/JS you MUST re-render the baked snapshots:
```
.venv/bin/python scripts/migrate_annotator_tool.py
```
It idempotently resyncs the span between stable markers. **Any new CSS/HTML/JS you add
must sit INSIDE one of the `SPLICES` marker pairs** in
`scripts/migrate_annotator_tool.py`, or the migration silently drops it. Verify with a
headless-Chrome screenshot at 390×844 (trust rendered pixels, not JS-measured widths).

## Notes
- All engines are inline-styled vanilla DOM — no CSS classes, portable as-is.
- Colour meaning is intentional and legended on-screen; keep the legends when porting.
- The `srcmap "missing = in const"` console line in the preview tool is a harmless
  artifact of the preview's source-map layer — it is NOT in the shipped code.

---

## Acceptance checklist (self-verify against the `*_demo.html` references)
**Chart viewer (iReal default)**
- [ ] 3 modes as a segmented control: Read / Analyse / Annotate.
- [ ] Read = black chords, no colour. Analyse = colour + Function|Key sub-toggle.
      Annotate = chord text amber→red→black by certainty, dotted underline < 0.65.
- [ ] Continuous vertical barlines; boxed maroon section letters on a double left bar.
- [ ] Repeated sections collapse to one block badged `×N` (never printed twice).
- [ ] Multi-chord bars sit side by side; maroon playhead bar tracks on the right edge.
- [ ] Key pill opens a circle-of-fifths **rotor** (not a stepper).
- [ ] Colour legend + "Auto: … certainty ≥ 0.60" caption present.

**Chord editor (both)**
- [ ] Compass: candidates orbit by circle-of-fifths angle, size = confidence,
      hue = root key colour, none overlapping; tap plays an arpeggio + selects.
- [ ] Guide: ranked cards with a plain-language function line + confidence meter;
      top pick badged.
- [ ] Suggestions vs Edit-by-hand toggle; edit = cylinder (A) / chip grid (B).

**Import (search-first)**
- [ ] Entry = one search-or-paste field + recent charts.
- [ ] Analysing state ticks through 6 pipeline stages, each revealing a result chip,
      ending on a "Chart ready" card.

**Re-infer (collaborative loop + merge)**
- [ ] Confidence % printed on every unconfirmed chord; shaky (< 0.42) drop to family + `?`.
- [ ] Tap → confirm sheet → hard clamp (✓, c=1); confirmed chords survive re-infer.
- [ ] "Re-infer with N fixes" posts `{confirms, merges:[]}`, applies diff by time overlap,
      then shows the propagation banner (which nearby chords sharpened).
- [ ] Form ribbon: two-tap two sections → merge sheet → own "Pooling…" re-infer
      posting `{confirms:[], merges:[[A1,A2]]}` → "Merged — one shared reading" banner.
- [ ] Swap-to-live is one line in `reinfer()`; mock matches the `/api/reinfer` shape.

**Launcher**
- [ ] Checking / Not-running (command + copy + auto-watch) / Running (Open + redirect).
- [ ] Real `check()` polling kept; **preview state-switcher removed.**
