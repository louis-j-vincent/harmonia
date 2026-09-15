# Harmonia UI refresh — integration brief

Design source: **`Harmonia UI Refresh.dc.html`** (turn 1 = chart chrome + control
system, turn 2 = prompter + loading). Reference ids used below (`1a`, `2b`) are the
badges on that board.

Target file: **`harmonia_min/app_shell.html`**, read at commit `3b462b68af43`.
Everything here is written in that file's own idiom — `el()`, `T`, `UI`, `SERIF`,
`S`, `go()`, `glyph()`, `note()` — and belongs **inside `window.APP`'s IIFE**. None of
these files load as a `<script src>`; they are paste-in blocks. Paste order:

```
ui_kit.js  →  form_rail.js  →  chart_chrome.js  →  prompter_keys.js  →  loading_chart.js
```

`progressive_job.py` is the one backend change, and it gates §5 only.

---

## The rule the whole refresh encodes

Three sizes, three radii, three states. **56** primary · **48** tab · **44** everything
else; radius **12** / **16** / full; state **selected** (accent fill) / **available**
(card + hairline) / **off** (muted). A control that fits none of these is a design
question — ask, don't invent a fourth.

Concretely, this deletes from the codebase: outlined-accent pill rows, ghost text
buttons, symbol-prefixed labels (`◎ Set bar 1`, `⧉ Check merges`, `⇧`), the 34px icon
circles, the 26px form chips, the 8.5px uppercase toggles, and every emoji label
(💡 🧩 🎤 🔁 🎹). Emoji are not in the design system.

---

## §1 — `ui_kit.js`

Paste after `haptic()` (~line 300). Gives you `SZ`, `ctlSkin`, `kitButton`, `kitIcon`,
`kitSegmented`, `kitChip`, `kitLegend`. Nothing else in this handoff works without it.

## §2 — chart chrome (`chart_chrome.js`) → design `1a`

Today `renderChart()` stacks **four rows** before any music: app bar, mode bar, legend,
form strip. It collapses to one toolbar and one dock.

In `renderChart()`:

| remove | replace with |
| --- | --- |
| `appBar(...)` + `keyPill` + `formPill` + `shareBtn` + `prefsBtn` | `chartToolbar()` |
| `modeBar` + `seg` + `learnPill` + the `S.mode==="analyse"` pill row | `chartDock()` at the end, `analyseLens()` when `S.mode==="analyse"` |
| `cap` / `legendFor()` | the one-line legend inside `analyseLens()` |
| `buildLadder()` | a level row in the `Aa` prefs sheet (`openPrefsSheet`) |

Key and form become the title's subtitle line; tapping it opens the rotor. Share moves
into the `Aa` sheet next to the existing export row.

**Bug to fix in the same pass:** in Annotate the FORM strip renders **twice** —
`buildRibbon()` emits its own form chips and `buildFormStrip()` then appends another.
Verified live. Keep the rail (§3), drop the ribbon's copy.

`Set bar 1` and `Check merges` stay, as `kitButton(label, fn, {height: SZ.control})` in
the annotate ribbon — words, no symbol prefix.

## §3 — form rail (`form_rail.js`) → design `1a`

`buildFormRail()` replaces `buildFormStrip()`. The whole song fits on one line as
`A B A C B A D B` — equal flex columns, letter size steps down with run count
(16/15/14/13), row height stays 44. Past 14 runs it wraps to two lines instead of
shrinking further.

`formRuns()` and `paintFormChip()` are unchanged and still keyed on **time**, not bar
index — do not "simplify" that; it is the fix for the 2026-07-30 stuck-highlight bug.

## §4 — prompter (`prompter_keys.js`) → design `2a`

`buildHandsCard(getMidis, onPlay)` replaces the `kbCard` block in `renderPrompter()`
(~line 3054). Split is the default and now means **two stacked keyboards, one per
hand** — right above left, each zoomed on its own octave, note names on the lit keys,
left hand in accent, right in blue. `joined` still calls the existing `renderVoicing()`.

The 8.5px `joined`/`split` buttons become a 34px `kitSegmented`. Repaint on chord
change only, via the existing `_lastCoachIdx` guard: `card._repaint()`.

Rest of that screen, from the design: current chord 78px, role on one line under it,
next two chords greyed; play 80 / jumps 60; the speed **slider** becomes three presets
(`kitSegmented` of `0.5× 0.75× 1×`, 56×44 each); `LOOP 4` becomes a 52px button that
names the bars it will loop. Drop the `tap the piano to hear…` micro-caption — the card
is the affordance.

## §5 — loading (`loading_chart.js` + `progressive_job.py`) → design `2b`

**Read `progressive_job.py`'s header before writing any of this.** The obvious design —
bars trickling into the grid one at a time — is not implementable, and an earlier draft
of this handoff got it wrong. `_musx.redecode()` in `harmonia_min/pipeline.py` is a
**single blocking call that returns every segment at once**; the bar layout after it is
integer arithmetic, i.e. instant. The `musx_fold` / `n_folds` fields the analysing
screen still reads belong to the OLD pipeline and are never set here. Anything that
animates bars arriving would be a fake progress bar.

What is real is **two waits with one hand-off**:

1. **the wait** — `beats.track`, then `musx.frame_posteriors` (minutes on a fresh song,
   cache-hit on a library one), then `redecode`. Nothing to show; show a quiet waiting
   state, not a fake grid.
2. **the raw chart, whole** — one section, `reps=1`, every bar written out, no letters.
   Readable and playable immediately. This is the state `HARMONIA_RAW_CHART=1` already
   produces; the payload just makes it the normal intermediate step.
3. **the letters fold in** — `detect_sections` + `fold_letter_groups` + `minimal_fold`
   land A/B/C and the repeats on the *same* grid.

**Backend.** `progressive_job.py` gives `phase`, `raw_model`, `n_bars`, `n_chords`,
`sections_found`, the `raw_chart_model()` builder, the five `report(...)` call sites in
`analyze()` order, and a four-point acceptance list. The load-bearing one: **bar
indices, `beat`, and every `(t0, t1)` in `raw_model` must be identical to the final
model's**, or the grid jumps when the letters land — which is the one thing this screen
exists to avoid. (It is also exactly what today's `renderChordPreview()` gets wrong: it
infers bars from `tempo/4` and never lines up with the chart that opens after it.)

**Shell.** `renderLoading()` / `paintLoading()` replace `renderAnalysing()` /
`paintAnalysing()`; `renderChordPreview()`, `drawCircleOfFifths()` and `STAGES` are
retired (keep `STAGES` for naming a failed stage on the error path). Phase `raw` hands
the shell a real ChartModel, so it renders through the **same `loadModel()` +
`buildIReal()`** the final chart uses — that is what makes the hand-off invisible.
Progress is two segments, not one bar: a single bar that leaps to 100% when the chart
appears reads as a lie.

Under `HARMONIA_RAW_CHART=1` phase `sections` never fires and the job goes `raw` →
`done` with `sections_found=1`. The screen handles it; the footer button just appears
sooner.

---

## Acceptance

1. Chart screen: exactly **two** chrome rows above the grid in Read (toolbar, form
   rail), three in Analyse (+ lens). No duplicated FORM strip in Annotate.
2. Every interactive element ≥ 44px in both axes. Quick audit in the console:
   `[...document.querySelectorAll('button')].filter(b=>{const r=b.getBoundingClientRect();return r.width<44||r.height<44}).map(b=>b.textContent)`
   must come back empty.
3. No emoji in any label. `grep -P '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]' harmonia_min/app_shell.html` — empty.
4. Prompter split mode draws two keyboards; hand assignment matches `splitHands()`.
5. Loading: the whole raw chart is readable before `phase` becomes `"sections"`, and no
   cell moves when the letters land or when the chart opens. Diff `raw_model` against
   the final model on This Love — bar, beat and `(t0, t1)` must match exactly.
   Nothing appends to a per-bar list across polls.
6. Nothing in `harmonia_min/*.py` other than the job payload changed. Chord vocabulary
   stays `labels.py`'s; do not send musx label strings to the shell.
