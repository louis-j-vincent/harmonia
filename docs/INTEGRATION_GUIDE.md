# Harmonia UI/UX Integration Guide — Implementation Status & Handoff

**Date:** 2026-07-13  
**Status:** Phase 1 Complete — Launcher integrated; remaining work: Chart Viewer, Re-infer, Import, Chord Editor enhancements  
**Source spec:** `handoff/HANDOFF.md`

---

## What's Done

### 1. Launcher Redesign ✅
- **File:** `harmonia.html`
- **Implementation:** Embedded `window.HLAUNCH` engine inline (no external dependencies)
- **Changes:**
  - Replaced the old dark-themed launcher with the new cream-themed design
  - Kept real server polling (`fetch()` with 3s retry, no-cors mode)
  - Auto-redirect on server detection (900ms delay before navigation)
  - Removed the "preview state switcher" (review-only artifact from design tool)
- **Status:** Ready to test
- **Test:** Open `http://localhost:7771/harmonia.html` (redirects to app when server is running)

---

## Integration Architecture

### Design Principles
1. **Modular drop-in engines:** Each handoff `.js` file is self-contained, injects its own CSS, exports a `build()` API
2. **No framework dependencies:** Vanilla JS + DOM only
3. **Real data wiring:** Adapt demo data to pipeline output via mapping functions
4. **Migration step required:** Chart template changes need `scripts/migrate_annotator_tool.py` to sync baked snapshots

### Data Flow
```
Pipeline output (P.chords, P.sections, etc.)
    ↓ [mapping functions]
    ↓ Engine-specific format (e.g., BARS/SECS for Chart Viewer)
    ↓ [build() → DOM + event handlers]
    ↓ Rendered UI
    ↓ [user edits → POST /api/reinfer or /api/annotations]
    ↓ Server persists (sidecar JSON)
```

---

## Integration Checklist — 5 Screens

### Screen 1: Chart Viewer (iReal-style) — **NOT STARTED**

**Location:** `chart_interactive.py` (existing template, needs enhancement)  
**Handoff spec:** `handoff/HANDOFF.md` §1, engine `window.HZC`, demo file `harmonia_chart_demo.html`

**What's needed:**
1. Add 3-mode segmented control (Read / Analyse / Annotate)
   - **Read:** Black chords, no colour
   - **Analyse:** Colour ON + Function|Key sub-toggle
   - **Annotate:** Certainty-based colour (amber→red→black), dotted underline < 0.65
2. Implement iReal layout (`buildIReal` + `irealGlyphs`)
   - Continuous vertical barlines
   - Boxed maroon section letters on double left barline
   - Multi-chord bars side-by-side
   - Maroon playhead bar on right edge
3. Repeat folding (`USECS`, `locate`): identical consecutive sections collapse to `×N`
4. Circle-of-fifths rotor for transpose (not a stepper)
5. Data wiring:
   ```javascript
   // Map from P to handoff format
   const BARS = P.chords.map((chord, i) => ({
     sec: P.sections[chord.bar],
     ch: { root: chord.root, q: chord.lv[depth].q, c: chord.lv[depth].c },
     ch2: chord.bar_has_two_chords ? { ... } : undefined
   }));
   const SECS = P.sectionChips; // already correct format
   ```
6. Colour functions:
   - `certaintyColor(c)` — from `lv.<depth>.c`
   - `fnOf(root, q)` — from `harmonia/theory/local_key.py` (not hardcoded)
   - `keyFill(pc)` — circle-of-fifths hue (already in template)

**Acceptance checklist:**
- [ ] 3 modes (Read/Analyse/Annotate) toggle correctly
- [ ] Colour by certainty / function / key
- [ ] Repeat folding shows `×N` badge
- [ ] Playhead bar syncs (if audio playing)
- [ ] Circle-of-fifths rotor opens from Key pill
- [ ] Looks good at 390×844 (iPhone size)

---

### Screen 2: Chord Editor (Compass + Guide tabs) — **NOT STARTED**

**Location:** `chart_interactive.py::chordEditModal` (add to existing modal)  
**Handoff spec:** `handoff/HANDOFF.md` §2, engine `window.HZ`, demo file `harmonia_chord_editor_demo.html`

**What's needed:**
1. Add two new tabs to the existing modal:
   - **Compass** (`buildCompass`): Circular suggestion view
     - Candidates orbit by circle-of-fifths angle (root position)
     - Size ∝ confidence
     - Hue ∝ root's key colour
     - Tap → arpeggio + select
   - **Guide** (`buildCards`): Ranked cards with plain-language "why"
     - Sorted by confidence
     - Role description (ii, V7, secondary dominant, tritone sub, borrowed, etc.)
     - Top pick badged
2. Keep existing Wheel + Suggestions tabs (don't remove)
3. Data wiring:
   ```javascript
   const SONG = P.chords.map(chord => ({
     root: chord.root,
     bass: chord.bass,
     q: chord.lv[depth].q,
     c: chord.lv[depth].c,
     bar: chord.bar,
     sug: chord.sug || []  // [{root, q, c}, ...]
   }));
   const HOME = P.home; // {tonic, mode}
   ```
4. Audio: `play(root, q)` already works (Web Audio arpeggio)
5. Role descriptions: call `jazz_priors.PROGRESSIONS` + `local_key.py` (not hardcoded heuristics)

**Acceptance checklist:**
- [ ] Compass tab shows circular orb layout
- [ ] Guide tab shows ranked cards with role descriptions
- [ ] Tap → plays arpeggio + selects chord
- [ ] Both tabs visible in modal (Wheel, Suggestions, Compass, Guide)
- [ ] Looks good at 390×844

---

### Screen 3: Import (search-first) — **NOT STARTED**

**Location:** New screen, accessible from Chart Viewer or Launcher  
**Handoff spec:** `handoff/HANDOFF.md` §3, engine `window.IMP`, demo file `harmonia_import_demo.html`

**What's needed:**
1. Entry field: search YouTube or paste link
2. Recent charts list (fetch from server)
3. Analysing state: 6-stage pipeline progress
   ```
   Fetching audio → Listening for notes (Basic Pitch) → Beat & tempo 
   → Sections (changepoint) → Key (Krumhansl) → Chords (HMM) → "Chart ready"
   ```
4. Wire to real server:
   - `/api/search` endpoint (or reuse existing YouTube import)
   - Stream progress via SSE or polling `/api/job/<job_id>`
   - Final redirect to chart when ready

**Acceptance checklist:**
- [ ] Search field functional
- [ ] Recent charts load
- [ ] Analysing state shows 6 stages + progress chips
- [ ] Redirect to chart on completion
- [ ] Looks good at 390×844

---

### Screen 4: Re-infer (collaborative loop + section merge) — **NOT STARTED**

**Location:** Overlay on Chart Viewer (not a separate screen)  
**Handoff spec:** `handoff/HANDOFF.md` §4, engine `window.RI`, demo file `harmonia_reinfer_demo.html`

**What's needed:**
1. Confidence display on chords
   - % printed on every unconfirmed chord
   - Shaky (< 0.42) collapse to family + `?`
2. Tap chord → confirm sheet → hard clamp (✓, c=1)
   - Confirmed chords survive re-infer (marked `confirmed: true`)
3. Re-infer button: POST `/api/reinfer/<chart>` with:
   ```json
   {
     "confirms": [{"t0": s, "t1": e, "root": pc, "q": str}, ...],
     "merges": []
   }
   ```
   - Apply response `diff` by time overlap
   - Show propagation banner (which chords sharpened)
4. Section merge: two-tap sections → confirm → own re-infer
   - POST with `merges: [[A1, A2], ...]`
   - Same propagation display
5. Swap to live: ONE LINE CHANGE in `reinfer()` function (mock ↔ real fetch)

**Acceptance checklist:**
- [ ] Confidence % visible on every chord
- [ ] Tap chord → confirm + hard clamp works
- [ ] "Re-infer with N fixes" POSTs and shows propagation
- [ ] Section merge → two-tap + re-infer works
- [ ] Mock matches real `/api/reinfer` shape
- [ ] Looks good at 390×844

---

### Screen 5: Launcher Redesign — ✅ **DONE**

**File:** `harmonia.html`  
**Implementation:** Complete with real server polling  
**Status:** Ready to test

---

## Implementation Order (Recommended)

1. **Launcher** (✅ done) → Test at `harmonia.html`
2. **Chart Viewer** (Read/Analyse modes only first) → Simplest path, core feature
3. **Chord Editor Compass+Guide** → Add to existing modal, no new screen
4. **Re-infer UI** → Wraps Chart Viewer, no new screen
5. **Import** → New screen, depends on search API

---

## Critical Files & Markers

### Template Changes (chart_interactive.py)

**SPLICES markers** (in `scripts/migrate_annotator_tool.py`):
1. CSS block 1: between `id="motif-style-btn"...` and `<div id="motif-overlay">`
2. CSS block 2: between `.chord .acc {...` and `/* ── Jazzify overrides ── */`
3. JS block: between `container.appendChild(btn);...` and `render();\n</script>`

Any new CSS/HTML/JS must sit INSIDE one of these marker pairs, or migration drops it.

**After any change to template:**
```bash
.venv/bin/python scripts/migrate_annotator_tool.py
```

Then verify with headless-Chrome screenshot at 390×844:
```bash
google-chrome --headless --screenshot=/tmp/chart.png --window-size=390,844 \
  http://localhost:7771/docs/plots/inferred_autumn_leaves.html
```

### Data Wiring Helper Functions

Create in `chart_interactive.py` or a separate `js/data_adapters.js`:

```javascript
// Map P.chords to handoff engines' expected format
function adaptChords(chords, depth='seventh') {
  return chords.map(c => ({
    root: c.root,
    bass: c.bass ?? -1,
    q: c.lv[depth].q,
    c: c.lv[depth].c,
    bar: c.bar,
    beat: c.beat ?? 0,
    sug: c.sug || []
  }));
}

// Pull function from local_key for accuracy
function fnOf(root, q) {
  // Call harmonia/theory/local_key.py's local_key_track result
  // (or hard-code if simple, but note the doc example is oversimplified)
  // For now, safe fallback:
  if (/^(7|9|13|alt|b9|#9)/.test(q) && !q.includes('^') && !q.includes('maj')) {
    return 'D'; // Dominant
  }
  const deg = (root - P.home.tonic + 12) % 12;
  if (deg === 0 || deg === 9 || deg === 4) return 'T'; // I, vi, iii (Tonic)
  if (deg === 2 || deg === 5 || deg === 11 || deg === 7) return 'S'; // ii, IV, viiø, V-as-triad (Subdominant)
  return 'S'; // default
}
```

---

## Server Integration (harmonia_server.py)

### `/api/reinfer/<chart>` endpoint
Already built (per `docs/handoff_mission3_ui_contract.md`). Verify it:
1. Accepts POST `{confirms, merges}`
2. Returns `{diff: [{t0, t1, root, q, c}, ...]}`
3. Applies local re-scoring to unconfirmed chords

### `/api/annotations/<chart>` endpoint  
Needed for persisting corrections. Schema: `docs/annotation_sidecar_schema.md`

```python
@app.route('/api/annotations/<filename>', methods=['POST'])
def save_annotation(filename):
    doc = request.get_json()
    doc = _remember_annotation(filename, doc)
    return jsonify(doc)
```

---

## Testing Checklist

### Before deployment
- [ ] Launcher opens, detects server, redirects
- [ ] Chart Viewer renders at 390×844 (Autumn Leaves)
- [ ] Chord editor opens, plays arpeggio
- [ ] Compass/Guide tabs visible (once added)
- [ ] Re-infer button works (mock or real)
- [ ] Section merge works
- [ ] Migration script runs, no data loss

### On iPhone (via Tailscale)
- [ ] Launcher loads via Tailscale IP:7771
- [ ] Tap "Open Harmonia" navigates to app
- [ ] Chord editor touch targets are ≥44×44pt
- [ ] Scroll-snap (cylinder picker) feels responsive
- [ ] Audio playback works (arpeggio + video if present)
- [ ] All text readable at 390×844

---

## Known Gotchas

1. **Chart template is a render-time template**, not a live app shell
   - Changes only affect NEW renders
   - Existing `docs/plots/inferred_*.html` need `migrate_annotator_tool.py` to update
   - Run it after every template edit

2. **Chord identity is by `(bar, beat)`, not array index**
   - If re-inference changes chord count/order, corrections stay valid
   - But if `beat` offset changes, correction keys silently miss
   - Mitigation: apply by nearest-`(bar,beat)` within bar; warn on mismatch

3. **Level slider behavior for corrected chords**
   - A human correction is certain (c=1)
   - Should it ignore the "Sure ≥" slider, or populate all 3 levels with same `q`?
   - Currently unresolved (see annotation_sidecar_schema.md §5.2)

4. **Compass vs Guide data**
   - Compass expects `sug:[{root,q,c}]` per chord
   - Guide expects same + `next` chord for resolution inference
   - Only ~3 charts have real suggestions data (predate feature)
   - Test on **Autumn Leaves** (has suggestions + sections)

5. **Mobile PWA mode vs Safari**
   - Some behaviors differ in Add-to-Home-Screen standalone mode
   - Test on real iPhone, not just mobile Safari
   - `confirm()` dialogs don't work in PWA mode on some iOS versions (merge already uses custom modal)

---

## Reference Links

- **Design spec:** `handoff/HANDOFF.md` — authoritative UI/UX source
- **Current template:** `harmonia/output/chart_interactive.py` — triple-quoted HTML/CSS/JS
- **Migration script:** `scripts/migrate_annotator_tool.py` — syncs SPLICES markers
- **Server:** `scripts/harmonia_server.py` — Flask app, serves charts + API
- **Annotation schema:** `docs/annotation_sidecar_schema.md` — correction persistence
- **Re-infer contract:** `docs/handoff_mission3_ui_contract.md` — `/api/reinfer` spec
- **Demo files:** `handoff/js/*_demo.html` — exact target layouts for each screen

---

## Next Steps (Prioritized)

1. **Test Launcher** (done) — open `harmonia.html`, verify server polling
2. **Implement Chart Viewer Read mode** — simplest path
   - Render existing chart with 3-mode toggle
   - No colour initially, just structure
3. **Add Compass tab** — test circular orb layout
4. **Wire Re-infer mock** — verify confirm + propagation UI
5. **Real API integration** — swap mock for `/api/reinfer` fetch
6. **Import screen** — last priority (least critical for closed-loop iteration)

---

## Questions for User

Before starting implementation:
1. Which of the 5 screens is highest priority for real-time use?
2. Should we integrate into `chart_interactive.py` (monolithic) or create modular screens?
3. Do you want to test on real iPhone immediately, or desktop first?
4. Should data-persistence (annotation sidecars) be implemented now or deferred?

