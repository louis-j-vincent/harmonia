# Handoff: annotator-tool UI — for a design-focused session (2026-07-13)

This hands off the mobile chord-annotator UI (inside the interactive chart
viewer) for design work. It's not a fresh feature request — it's a working
tool with one iteration of real user testing already behind it. Read this
before touching CSS/JS: there's a non-obvious build step (see §5) that will
silently no-op your changes if skipped, and one whole approach (§3) was
already tried and rejected on-device, so don't re-derive it from scratch.

## 1. What this is

A tap/long-press chord-correction tool layered onto Harmonia's existing
interactive chart viewer (an iReal-Pro-style chord sheet, rendered as
self-contained HTML files, opened as an installed iPhone PWA over Tailscale).
The chart viewer already had a settled visual language before this tool was
added — cream/paper background, maroon accent, Georgia-italic logo, an
existing bottom-sheet modal system used for the transpose wheel and options
panel. The annotator tool was built to extend that language, not introduce
a new one. See `harmonia/output/chart_interactive.py` — it's a single Python
file whose main content is a triple-quoted HTML/CSS/JS template (~3200
lines), not a separate frontend project.

**Design tokens already in use** (don't reinvent):
- `--paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b;`
  (defined at `:root` near the top of the `<style>` block)
- Modal system: `.modal` / `.modal-backdrop` / `.modal-panel` — a bottom
  sheet that slides up, `transform .32s cubic-bezier(.32,.9,.35,1)`.
- Buttons: `.modal-panel button` — cream fill, thin border, `border-radius:8px`.
- Selected/active state convention: `background:var(--accent); color:#f7f3e9;`

## 2. What's live right now

Chord editor opens two ways: tap a chord while "Annotate mode" is on
(toggled via Options → Annotate), or **long-press any chord** (~500ms,
haptic + toast feedback) — this also toggles Annotate mode on and opens
straight into that chord's editor, so it's the fast path.

The editor (`#chordEditModal`) has two tabs:
- **Wheel** — pick root + quality directly (§3/§4 below — this is the part
  that just got rebuilt).
- **Suggestions** — a ranked list of the model's actual alternative
  candidates (not fabricated), with probability bars, a temperature slider
  that reshapes the displayed spread (never reorders), tap-to-preview
  (arpeggio) and tap-to-select. Data: `P.chords[i].sug`, populated by
  `chord_pipeline_v1._top_chord_suggestions` — real posteriors the model
  already computed and used to discard after argmax.

A ▶ button plays the currently-dialed-in chord as a short arpeggio (Web
Audio, `playChordArpeggio` in the JS) — works from both tabs.

Section merge (tap two structural sections → confirm → they're marked as
the same underlying material) uses a **custom in-app confirm modal**, not
`window.confirm()` — the native dialog is unreliable inside an iOS
standalone PWA and was almost certainly why merge looked broken in testing.

## 3. Rejected: concentric nested rings (v1, same session)

First attempt at the quality picker: three concentric rings for
Family → Seventh → Extension (matching `harmonia/theory/chord_tree.py`'s
3-level tree exactly), tap-only (no drag), toggled in/out of view against
the existing chromatic root ring via a vertical swipe gesture, geometry via
`cos`/`sin` positioning like the existing transpose wheel.

**It worked** (verified via headless-Chrome + synthetic pointer events —
tap-to-select, cascade rebuilding, save/persist all functioned correctly)
but **failed real on-device use**: once you tap a family, the remaining
rings' targets are small radial arcs that are genuinely hard to hit
precisely on a touchscreen, especially the innermost extension ring at low
radius. User's words: *"once you select the family, everything is fixed,
it's hard to choose anything else."* Don't re-propose this shape without
solving the target-size problem first — it's not a matter of nudging sizes,
the fundamental issue is radial targets shrink toward the center by
construction.

## 4. Shipped: iOS-style rolling cylinder picker (v2, current)

User's own reference: the classic iOS date/alarm-time picker (multiple
side-by-side vertically-scrolling columns, viewed "from above," each with a
highlighted center band). Rebuilt as four columns — **Root · Family · 7th ·
Ext** — using native CSS `scroll-snap`, no custom drag math at all:

```
.ce-cyl-picker    — flex row container, one highlight band overlay
.ce-cyl-col       — one column: overflow-y:scroll; scroll-snap-type:y mandatory;
                    padding:72px 0 (= 2 × item-height, so first/last items
                    can still center); 180px tall = 5 rows @ 36px
.ce-cyl-item      — one row; .sel = bold + larger, set by JS after each
                    scroll settles (debounced, ~90ms)
.ce-cyl-highlight — absolutely-positioned overlay band at vertical center,
                    non-interactive, purely visual
```

Picking a family rebuilds the Seventh column for that branch and resets
Extension to match (same cascade the rings used — `SEVENTHS_BY_FAMILY`,
`EXTENSIONS_BY_SEVENTH`, `QUAL_TO_FAMILY_SEVENTH` in the JS, unchanged data
model from v1). All 4 columns are `flex:1 1 0` — genuinely fluid, no fixed
pixel widths — so they can't overflow the container by construction (v1's
sibling feature, the suggestions list, had exactly this overflow bug from a
fixed-width row; the cylinder columns were designed to not repeat it).

**One real bug already found and fixed**: columns other than Root had their
initial scroll position set *before* the modal was open (`display:none`
ancestor) — `scrollTop` is a no-op on a non-rendered element, so Family/
Seventh/Extension silently stayed at position 0 while Root (set after
`openModal()`) worked. Caught via screenshot: highlight band and the bold
`.sel` text were visually on different rows. Fixed by moving all 4 columns'
sync into the post-`openModal()` callback in `openChordEditor()`. **If you
touch the open/init sequence, re-verify this specific thing** — it's an easy
regression to reintroduce.

**Not yet verified**: real on-device feel of scroll-snap responsiveness,
haptic timing, whether 36px row height / 180px column height is the right
scale for a thumb (these were reasonable first guesses, not measured
against anything). This is the natural next design pass — the mechanism
works, the tuning hasn't been validated against a real hand.

## 5. The step that will silently eat your work: migration

`chart_interactive.py` is a **template** — editing it only affects charts
rendered *after* your edit. All the already-rendered chart files
(`docs/plots/inferred_*.html`, ~15 of them, what's actually served) are
static HTML snapshots baked at analysis time. After any HTML/CSS/JS change
to the template, you must run:

```
.venv/bin/python scripts/migrate_annotator_tool.py
```

This does an **idempotent resync**: it finds stable markers in the old
files and replaces the span between them with whatever's currently in the
template, every run, regardless of whether that file was migrated before.
(It used to be skip-if-already-migrated, which silently missed every
increment after the first migration — including once nearly losing this
session's own CSS. Don't revert to that pattern.) If you add a new
CSS/HTML/JS block, **check it's actually inside one of `SPLICES`' marker
pairs** in `scripts/migrate_annotator_tool.py` — CSS added just past the
old end-marker is exactly how the bug above happened, twice.

After migrating, verify with a headless-Chrome screenshot before trusting
it (`google-chrome --headless --screenshot=... --window-size=390,844
--virtual-time-budget=N <url>`, server must be running). One gotcha:
`window.innerWidth` reports incorrectly (≈500 regardless of `--window-size`)
in this specific headless setup while the actual screenshot pixels are
correct at 390×844 — don't trust JS-measured layout widths from that tool,
trust the rendered pixels (or better, real on-device testing when possible).

## 6. Data availability caveat

Only 3 of ~15 charts currently have real Suggestions data (`autumn_leaves`,
`nina_simone_feeling_good_lyric_video`, `the_beatles_let_it_be`) — those are
the only songs with cached local audio (`docs/audio/*.m4a`) to backfill
from without re-downloading. **Test on Autumn Leaves** for the fullest
experience (suggestions + section merge both have real data there — most
other charts predate the section-structure feature entirely and won't show
section chips at all). Any *newly* analyzed song (via the app's own
YouTube search) gets suggestions automatically now — the backend fix is at
the source, backfilling was only needed for already-baked files.

## 7. Explicitly deferred — noted, not designed

`docs/architecture_extensions.md` §1b: a **circular/radial suggestion view,
color-coded by probability**, was requested as a follow-up to the (shipped)
flat suggestions list — "for the suggested chords... a circle like view
would be awesome... color code the probabilities." Not scoped or
prototyped. The doc suggests circle-of-fifths-by-root-angle as a starting
geometry (reusing the existing root-ring math), with probability →
saturation/lightness on top of the chart's existing per-family color scheme
(`FAMILY_COLOR`/`motifColor()` in the JS) so it stays visually consistent.
Explicitly flagged there as needing "a research pass on existing
chord-embedding/chord-space visualization work before designing this, not
just inventing a layout" — that's still true, nobody's done that pass yet.

## 8. How to see it live

Server: `.venv/bin/python scripts/harmonia_server.py --no-open --port 7771`
(from repo root, `.venv` already has all deps). Reachable over Tailscale at
the host's Tailscale IP, port 7771 — ask the user for the current IP
(`ifconfig | grep utun` on their machine), it can change between sessions.
Add-to-Home-Screen on iPhone gives the full standalone-PWA experience
(matters for testing — some things, like native `confirm()`, behave
differently in standalone mode vs. Safari-in-browser, see §2).
