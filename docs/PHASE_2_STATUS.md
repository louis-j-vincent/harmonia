# Phase 2 Status: Remaining 4 Screens — Implementation Guide

**Date:** 2026-07-13  
**Status:** Architecture defined, proof-of-concept app created, integration roadmap ready  
**Priority:** Chart Viewer > Chord Editor > Re-infer > Import

---

## What's Been Done (Phase 2 Prep)

### ✅ Proof-of-Concept App Shell
**File:** `docs/harmonia-integrated-app.html`
- Multi-screen app navigation (5 tabs)
- Launcher (working, from Phase 1)
- Screen placeholders for Chart, Chord Editor, Import, Re-infer
- Global `HarmApp` orchestration object
- Data adapter framework (`app.adaptChords()`)
- Ready to wire real engines

### ✅ Architecture Defined
- Data flow: P.chords → handoff format → engine render
- Navigation model: 5 screens, modal overlay for chord editor
- Integration pattern: Mount handoff JS engines to DOM elements
- API wiring: `/api/reinfer/<chart>` endpoint (already built)

### ✅ Reference Materials Updated
- `docs/INTEGRATION_GUIDE.md` — full implementation spec
- `handoff/js/*.js` — all 5 drop-in engines ready
- `handoff/js/*_demo.html` — target layouts for comparison

---

## Remaining Work (Ranked by Priority)

### 1️⃣ HIGHEST PRIORITY: Chart Viewer
**Why:** Core feature, used by other screens  
**Implementation time:** 2-3 hours  
**Status:** 🟡 Architecture ready, needs data wiring + DOM mounting

**What to do:**
1. In `harmonia-integrated-app.html`, modify `app.initChart()`:
   ```javascript
   function loadHandoffChartEngine() {
     // Load: <script src="handoff/js/harmonia_chart.js"></script>
     // Mount: <div data-hzc="C:desktop"></div>
     // Wire: P.chords → BARS, P.sections → SECS format
     // HZC auto-initializes when DOM ready
   }
   ```

2. Create data adapter for P → handoff format:
   ```javascript
   const BARS = P.chords.map((c, i) => ({
     sec: P.sections[c.bar],
     ch: { root: c.root, q: c.lv.seventh.q, c: c.lv.seventh.c },
     ch2: nextChordInBar ? { root, q, c } : undefined
   }));
   const SECS = [...];  // P.sectionChips already correct shape
   ```

3. Implement 3-mode toggle (Read/Analyse/Annotate):
   - Already present in `harmonia_chart.js` (see demo)
   - Wires to state.mode → re-renders with colour/certainty

4. Test:
   - Open Autumn Leaves via `harmonia-integrated-app.html`
   - Click "Chart" tab
   - Verify modes toggle correctly
   - Compare to `handoff/js/harmonia_chart_demo.html`
   - Screenshot at 390×844

**Acceptance criteria:**
- [ ] Chart renders with real Autumn Leaves chords
- [ ] 3 modes (Read/Analyse/Annotate) toggle correctly
- [ ] Colours match demo (function/key/certainty)
- [ ] Repeat folding shows ×N badge
- [ ] Screenshot at 390×844 matches demo

---

### 2️⃣ HIGH PRIORITY: Chord Editor Enhancements
**Why:** Needed for annotation loop  
**Implementation time:** 2-3 hours  
**Status:** 🟡 Existing modal works, needs Compass+Guide tabs added

**What to do:**
1. Add to `harmonia-integrated-app.html`:
   ```javascript
   function openChordEditor(chordIndex) {
     const modal = document.getElementById('chord-modal');
     const chord = P.chords[chordIndex];
     
     // Mount HZ engine to modal-panel
     // Pass chord + suggestions
     // Show 4 tabs: Wheel, Suggestions, Compass, Guide
   }
   ```

2. Wire the 4 tabs (reuse existing cylinder picker + add 2 new):
   - **Wheel** — already working (iOS-style scroll picker)
   - **Suggestions** — already working (ranked list with %, play button)
   - **Compass** — NEW (circular orb layout by circle-of-fifths angle)
     - Each suggestion orbits around current chord
     - Size = confidence, hue = root's key colour
     - Tap → arpeggio + select
   - **Guide** — NEW (ranked cards with role descriptions)
     - "Why this chord?" explanation
     - Call `roleOf()` from `harmonia_chord_editor.js`
     - Top pick badged

3. Wire chord selection to update P.chords + re-render chart

4. Test:
   - Open Chart Viewer
   - Tap a chord → opens modal
   - Switch between 4 tabs
   - Select from Compass/Guide
   - Verify chord updates in chart
   - Screenshot each tab at 390×844

**Acceptance criteria:**
- [ ] Modal opens with 4 tabs visible
- [ ] Compass shows circular orb layout (like demo)
- [ ] Guide shows ranked cards with roles
- [ ] Tab switching works smoothly
- [ ] Chord selection updates chart
- [ ] Audio playback works (Web Audio arpeggio)

---

### 3️⃣ HIGH PRIORITY: Re-infer Endpoint Wiring
**Why:** Closes the annotation loop  
**Implementation time:** 1-2 hours  
**Status:** 🟡 Mock UI ready, endpoint already built (verify first)

**What to do:**
1. Verify `/api/reinfer/<chart>` endpoint exists and works:
   ```bash
   # Test: POST to /api/reinfer/inferred_autumn_leaves.html
   curl -X POST http://localhost:7771/api/reinfer/inferred_autumn_leaves.html \
     -H "Content-Type: application/json" \
     -d '{"confirms": [], "merges": []}'
   # Should return: {diff: [{t0, t1, root, q, c}, ...]}
   ```

2. If not found, check `harmonia_server.py` for the route

3. In `app.runReinfer()`, swap mock for real:
   ```javascript
   async function reinfer(chart, confirms, merges) {
     const resp = await fetch(`/api/reinfer/${chart}`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ confirms, merges })
     });
     const {diff} = await resp.json();
     // Apply diff by time overlap
     // Show propagation banner
     return diff;
   }
   ```

4. Test:
   - Open Chart Viewer
   - Edit a chord (Compass/Guide)
   - Click "Re-infer" button
   - Verify POST succeeds
   - See propagation banner (which chords sharpened)
   - Compare before/after to `/api/reinfer` response

**Acceptance criteria:**
- [ ] `/api/reinfer` POST succeeds
- [ ] Response shape matches `{diff: [...]}`
- [ ] Diff is applied by time overlap
- [ ] Propagation banner shows changed chords
- [ ] Screenshot shows before/after

---

### 4️⃣ MEDIUM PRIORITY: Import Screen
**Why:** Nice-to-have, lowest critical path  
**Implementation time:** 3-4 hours  
**Status:** 🔴 Not started, depends on search API

**What to do:**
1. Check if search endpoint exists:
   - `/api/search?q=<query>` (YouTube search)
   - `/api/analyze/<video_id>` (Start analysis job)
   - `/api/job/<job_id>` (Poll progress)

2. If not, these need to be built in `harmonia_server.py`

3. Wire to `harmonia_import.js` engine:
   ```javascript
   function initImport() {
     // Mount IMP engine
     // Hook search input → /api/search
     // Poll /api/job/<id> for progress (6 stages)
     // Redirect to chart when ready
   }
   ```

4. Test:
   - Open Import tab
   - Enter YouTube URL or search term
   - See 6-stage progress (Fetching → Listening → Beat → Sections → Key → Chords)
   - Redirect to chart on completion

**Acceptance criteria:**
- [ ] Search input works
- [ ] 6-stage progress visible
- [ ] Each stage shows result chip
- [ ] Redirect to chart on completion

---

## Implementation Checklist

### Before Starting Code
- [ ] Read `handoff/HANDOFF.md` §1-4 (5 screens spec)
- [ ] Open each `handoff/js/*_demo.html` in browser
- [ ] Understand data shapes (BARS, SECS, SONG, etc.)

### Chart Viewer Implementation
- [ ] Create data adapter `P.chords → BARS/SECS`
- [ ] Mount HZC engine in `initChart()`
- [ ] Verify 3-mode toggle works
- [ ] Test chord colors (function/key/certainty)
- [ ] Run `migrate_annotator_tool.py` after changes
- [ ] Screenshot at 390×844

### Chord Editor Implementation
- [ ] Add Compass tab (circular orb layout)
- [ ] Add Guide tab (ranked cards with roles)
- [ ] Keep Wheel tab (don't break existing)
- [ ] Wire chord selection → P.chords update
- [ ] Test tab switching
- [ ] Test audio playback

### Re-infer Implementation
- [ ] Verify `/api/reinfer/<chart>` exists
- [ ] Test endpoint with curl
- [ ] Swap mock for real in `runReinfer()`
- [ ] Test edit chord → re-infer → propagation
- [ ] Verify confirmed chords survive re-infer

### Import Implementation
- [ ] Check search API exists
- [ ] Build if missing
- [ ] Mount IMP engine
- [ ] Wire search input
- [ ] Test 6-stage progress
- [ ] Test redirect to chart

---

## Critical Integration Points

### 1. Data Shape Mismatch (Most Common Blocker)
**Problem:** Engine expects `{sec, ch:{root,q,c}, ch2?}` but P gives `{root,bass,bar,beat,lv:{...}}`

**Solution:** Use adapter function:
```javascript
function adaptToBars(P_chords) {
  return P_chords.map(c => ({
    sec: P.sections[c.bar],
    ch: { root: c.root, q: c.lv.seventh.q, c: c.lv.seventh.c },
    ch2: hasTwoChords ? {...} : undefined
  }));
}
```

### 2. DOM Mount Timing
**Problem:** HZC tries to render before DOM ready

**Solution:** Wait for DOMContentLoaded:
```javascript
document.addEventListener('DOMContentLoaded', () => {
  const mount = document.createElement('div');
  mount.setAttribute('data-hzc', 'C:desktop');
  container.appendChild(mount);
  // HZC auto-initializes
});
```

### 3. Re-infer Response Parsing
**Problem:** Endpoint returns `{diff: [...]}` but code expects flat array

**Solution:** Unwrap the response:
```javascript
const resp = await fetch(...);
const {diff} = await resp.json();  // Unwrap
applyDiff(diff);  // Use diff array
```

### 4. Migration Step (Silent Fail)
**Problem:** You edit chart_interactive.py but changes don't appear in baked charts

**Solution:** Always run after editing:
```bash
.venv/bin/python scripts/migrate_annotator_tool.py
```

---

## Testing Roadmap

### Unit Tests (Per Screen)
1. **Chart Viewer**
   - Load real Autumn Leaves data
   - Verify modes toggle (Read/Analyse/Annotate)
   - Verify colours render (function/key/certainty)
   - Compare screenshot to demo

2. **Chord Editor**
   - Open modal with chord
   - Switch tabs (Wheel/Suggestions/Compass/Guide)
   - Select suggestion, verify update
   - Play arpeggio, hear audio

3. **Re-infer**
   - POST to `/api/reinfer` with test payload
   - Verify diff shape
   - Apply diff, verify chord updates
   - Check propagation banner

4. **Import**
   - Enter YouTube URL
   - See 6-stage progress
   - Verify each stage appears
   - Redirect on completion

### Integration Tests (Full Flow)
1. Launcher → Chart Viewer → Edit chord (Compass) → Re-infer → See propagation ✓
2. Import → Analysing state → Chart ready → Opens chart ✓
3. All screens render at 390×844 ✓
4. All text readable, touch targets ≥44×44pt ✓

### Acceptance (Final Gate)
- [ ] All 5 screens render at 390×844 matching demos
- [ ] Autumn Leaves flow: Launcher → Chart → Edit → Re-infer → Propagation
- [ ] Migration script run, no data loss
- [ ] All acceptance items from `handoff/HANDOFF.md` §Acceptance checklist

---

## Quick Reference: File Locations

| Component | File | Status |
|-----------|------|--------|
| Launcher | `harmonia.html` | ✅ Done |
| Chart Viewer | `handoff/js/harmonia_chart.js` | 🟡 Ready to wire |
| Chord Editor | `handoff/js/harmonia_chord_editor.js` | 🟡 Ready to wire |
| Import | `handoff/js/harmonia_import.js` | 🟡 Ready to wire |
| Re-infer | `handoff/js/harmonia_reinfer.js` | 🟡 Ready to wire |
| App Shell | `docs/harmonia-integrated-app.html` | ✅ Created |
| Server | `scripts/harmonia_server.py` | ✅ Has `/api/reinfer` |
| Template | `harmonia/output/chart_interactive.py` | 📝 For future enhancement |

---

## Handoff Checklist

**Before you start:**
- [ ] Read this document
- [ ] Read `handoff/HANDOFF.md`
- [ ] Open `handoff/js/*_demo.html` in browser
- [ ] Start server: `.venv/bin/python scripts/harmonia_server.py --no-open --port 7771`
- [ ] Test Launcher: `http://localhost:7771/harmonia.html`

**Implementation order:**
1. Chart Viewer (1-2 hours) — core feature
2. Chord Editor (1-2 hours) — annotation UI
3. Re-infer (30 min-1 hour) — loop closure
4. Import (2-3 hours) — search UI

**Estimated total:** 4-8 hours for full Phase 2 completion

**Testing:** After each screen, take 390×844 screenshot, compare to demo

---

## Getting Unblocked

If you hit a blocker, try these steps in order:

1. **Check browser console (F12)** for JS errors
2. **Verify data is available** — is P populated? `console.log(P)`
3. **Check DOM mount point** — does `data-hzc` element exist?
4. **Compare to demo** — open `handoff/js/*_demo.html`, see what works
5. **Read error carefully** — "undefined is not a function" vs "cannot read property 'root'"
6. **Check data shape** — is chord.lv.seventh.c correct? Use adapter

If stuck for >30 min: **Create a minimal test case** and screenshot it.

---

## Success Criteria (Final Gate)

✅ **Phase 2 complete when:**
- Chart Viewer + 3 modes working on Autumn Leaves
- Chord Editor with Compass+Guide tabs working
- Re-infer button POSTs to `/api/reinfer`, shows propagation
- Import screen works (or documented as deferred)
- All screens at 390×844 matching demos
- Full flow works: Launcher → Chart → Edit → Re-infer → See results

✅ **Ready for iPhone when:**
- Server running on port 7771
- Tailscale URL available
- All 5 screens tested at 390×844
- Touch targets all ≥44×44pt
- No console errors on iPhone

---

## Next Steps

1. **Today:** Read this guide, understand the architecture
2. **Tomorrow:** Implement Chart Viewer (2-3 hours)
3. **Then:** Chord Editor (1-2 hours)
4. **Then:** Re-infer (30 min-1 hour)
5. **Then:** Import (if time)
6. **Finally:** Test on iPhone via Tailscale

Total time: ~1-2 days for full Phase 2.

