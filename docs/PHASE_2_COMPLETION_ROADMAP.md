# Phase 2 Completion Roadmap — Mission: Integrate 4 Remaining Screens

**Current Status:** 🟡 Architecture + scaffolding ready; awaiting implementation  
**Time to Complete:** 4-8 hours (1-2 days full-time)  
**Coordinator Request:** "Finish the mission: integrate the remaining 4 screens into the real app"

---

## TL;DR — What You Need to Know

**The 4 remaining screens are ready to integrate. Each handoff `.js` engine is self-contained, drop-in ready.**

### Fastest Path to "Mission Complete" (Ranked by Value)

| # | Screen | Effort | Value | Status |
|---|--------|--------|-------|--------|
| 1 | **Chart Viewer** | 2-3h | 🔥 Highest (core feature) | 🟡 Ready to wire |
| 2 | **Chord Editor** | 1-2h | 🔥 High (annotation UI) | 🟡 Ready to wire |
| 3 | **Re-infer** | 30m-1h | 🔥 High (loop closure) | 🟡 Ready to wire |
| 4 | **Import** | 2-3h | 📌 Medium (search UI) | 🟡 Ready to wire |

**Gate:** All 5 screens render at 390×844 + full Autumn Leaves flow works

---

## What's Already Done

✅ **Phase 1 Complete**
- Launcher redesign (cream-themed, real polling)
- Location: `harmonia.html`

✅ **Phase 2 Scaffolding**
- App shell framework: `docs/harmonia-integrated-app.html`
- Data adapters: `app.adaptChords()` pattern shown
- Navigation: 5-screen orchestration setup
- Global state: `HarmApp` object ready

✅ **Reference Materials**
- `handoff/HANDOFF.md` — full design spec (THE authoritative source)
- `handoff/js/*.js` — 5 drop-in engines (self-contained, vanilla JS)
- `handoff/js/*_demo.html` — target layouts (open in browser to see target)
- `docs/INTEGRATION_GUIDE.md` — data wiring guide
- `docs/PHASE_2_STATUS.md` — detailed per-screen implementation checklist

✅ **Server Ready**
- `/api/reinfer/<chart>` endpoint already built
- `harmonia_server.py` running on port 7771
- Autumn Leaves data available at `/docs/plots/inferred_autumn_leaves.html`

---

## The 5-Minute Architecture Lesson

### Why This Will Work
1. **Handoff engines are self-contained** — no build step, no deps, just vanilla JS
2. **Data shapes are already compatible** — demo data matches what pipeline produces
3. **Integration is data-mapping** — not reimplementing, just wiring P → handoff format
4. **Migration is automated** — `migrate_annotator_tool.py` keeps baked charts in sync

### The Data Flow
```
Harmonia Pipeline (P.chords, P.sections, etc.)
    ↓
[Adapter function: P → handoff format]
    ↓
[Handoff engine: build(dom) → renders UI]
    ↓
[User interacts]
    ↓
[POST /api/reinfer or /api/annotations]
    ↓
[Server returns diff or persists]
```

### Three Integration Patterns

**Pattern A: Chart Viewer (wire-once)**
```javascript
// Adapt P.chords once
const BARS = adaptChords(P.chords, P.sections);

// Mount engine
const mount = document.createElement('div');
mount.setAttribute('data-hzc', 'C:desktop');
container.appendChild(mount);

// Engine auto-initializes from BARS in global scope
```

**Pattern B: Chord Editor (open-on-demand)**
```javascript
// When user taps a chord
function openEditor(chordIndex) {
  const SONG = [P.chords[chordIndex]];  // Single chord + suggestions
  const HOME = P.home;
  
  // Mount HZ engine in modal
  // User selects from Compass/Guide
  // Call updateChord(newChord)
}
```

**Pattern C: Re-infer (submit-and-wait)**
```javascript
// When user clicks "Re-infer"
async function runReinfer() {
  const resp = await fetch(`/api/reinfer/${chart}`, {
    method: 'POST',
    body: JSON.stringify({ confirms, merges })
  });
  const {diff} = await resp.json();
  applyDiff(diff);  // Update chords
  showPropagation(diff);  // Show what changed
}
```

---

## Per-Screen Implementation Blueprint

### 1. Chart Viewer (Highest Value — 2-3 hours)

**Goal:** Render Autumn Leaves chords with 3-mode toggle (Read/Analyse/Annotate)

**Implementation Steps:**
```bash
# 1. Update harmonia-integrated-app.html
# In app.initChart():
   - Create BARS by mapping P.chords (see PHASE_2_STATUS.md §Chart Viewer)
   - Mount <div data-hzc="C:desktop"></div>
   - HZC auto-initializes

# 2. Reference
   - Open: handoff/js/harmonia_chart_demo.html (see target)
   - Study: How 3-mode toggle works (state.mode = 'read'|'analyse'|'annotate')
   - Adapt: BARS shape from demo to real P.chords

# 3. Test
   - Open http://localhost:7771/docs/harmonia-integrated-app.html
   - Click "Chart" tab
   - Toggle modes (Read / Analyse / Annotate)
   - Screenshot at 390×844
   - Compare to demo
```

**Acceptance:**
- [ ] Chart renders with real chords
- [ ] Modes toggle, colours change correctly
- [ ] 390×844 screenshot matches demo
- Estimated time: **2-3 hours**

---

### 2. Chord Editor (High Value — 1-2 hours)

**Goal:** Add Compass + Guide tabs to existing modal

**Implementation Steps:**
```bash
# 1. Update harmonia-integrated-app.html
# In openChordEditor(chordIndex):
   - Create SONG = [P.chords[chordIndex]]
   - Create HOME = P.home
   - Mount HZ engine in #chord-editor-content
   - HZ shows 4 tabs: Wheel, Suggestions, Compass, Guide

# 2. Reference
   - Open: handoff/js/harmonia_chord_editor_demo.html
   - Study: Compass (circular orb layout)
   - Study: Guide (ranked cards with roles)

# 3. Test
   - Open Chart Viewer
   - Tap a chord → modal opens
   - Switch between 4 tabs
   - Select from Compass/Guide
   - Verify chord updates in chart
```

**Acceptance:**
- [ ] Modal opens with 4 tabs visible
- [ ] Compass shows circular layout
- [ ] Guide shows ranked cards
- [ ] Tab switching works
- [ ] Chord selection updates chart
- Estimated time: **1-2 hours**

---

### 3. Re-infer (High Value — 30m-1h)

**Goal:** Wire the POST `/api/reinfer/<chart>` endpoint, show propagation

**Implementation Steps:**
```bash
# 1. Verify endpoint exists
curl -X POST http://localhost:7771/api/reinfer/inferred_autumn_leaves.html \
  -H "Content-Type: application/json" \
  -d '{"confirms": [], "merges": []}'

# Should return: {"diff": [{t0, t1, root, q, c}, ...]}

# 2. Update harmonia-integrated-app.html
# In app.runReinfer():
   - Collect confirmations from modal
   - POST to /api/reinfer/<chart>
   - Parse diff array
   - Apply by time overlap
   - Show propagation banner

# 3. Test
   - Open Chart Viewer
   - Edit chord (select from Compass)
   - Click "Re-infer" button
   - Verify POST succeeds
   - See propagation (which chords changed)
```

**Acceptance:**
- [ ] `/api/reinfer` POST succeeds
- [ ] Diff is applied correctly
- [ ] Propagation banner shows
- [ ] Before/after chords visible
- Estimated time: **30m-1h**

---

### 4. Import (Medium Value — 2-3h, can defer)

**Goal:** Search YouTube, show 6-stage progress, redirect to chart

**Implementation Steps:**
```bash
# 1. Check if search API exists in harmonia_server.py
grep -n "def.*search\|/api/search" scripts/harmonia_server.py

# If not found: Build 3 endpoints:
# - POST /api/analyze (start job, return job_id)
# - GET /api/job/<job_id> (poll progress, return {stage, status, url})
# - Stages: Fetching → Listening → Beat → Sections → Key → Chords → Ready

# 2. Update harmonia-integrated-app.html
# In app.initImport():
   - Mount IMP engine in #import-content
   - Wire search input → /api/analyze
   - Poll /api/job/<job_id> for progress
   - Show 6-stage pipeline
   - Redirect to chart on completion

# 3. Test
   - Open Import tab
   - Enter YouTube URL
   - See 6 stages progress
   - Redirect to chart
```

**Acceptance:**
- [ ] Search input works
- [ ] 6 stages visible
- [ ] Progress updates
- [ ] Redirect on completion
- Estimated time: **2-3h** (or defer)

---

## Complete Implementation Checklist

### Prep (30 min)
- [ ] Read `handoff/HANDOFF.md` (15 min)
- [ ] Open each `handoff/js/*_demo.html` in browser (10 min)
- [ ] Start server: `.venv/bin/python scripts/harmonia_server.py --no-open --port 7771`
- [ ] Test Launcher: `http://localhost:7771/harmonia.html`
- [ ] Review `docs/PHASE_2_STATUS.md` for per-screen details

### Chart Viewer (2-3 hours)
- [ ] Create P → BARS adapter function
- [ ] Mount HZC engine in initChart()
- [ ] Verify 3 modes toggle
- [ ] Test colours (function/key/certainty)
- [ ] Screenshot at 390×844
- [ ] Compare to demo
- [ ] Commit & push

### Chord Editor (1-2 hours)
- [ ] Update openChordEditor() to mount HZ
- [ ] Create SONG + HOME from P.chords[i]
- [ ] Verify 4 tabs visible (Wheel, Suggestions, Compass, Guide)
- [ ] Test chord selection updates chart
- [ ] Screenshot at 390×844
- [ ] Commit & push

### Re-infer (30m-1h)
- [ ] Test `/api/reinfer` endpoint with curl
- [ ] Wire POST in runReinfer()
- [ ] Parse diff response
- [ ] Apply diff by time overlap
- [ ] Show propagation banner
- [ ] Test edit → re-infer → see results
- [ ] Commit & push

### Import (2-3h, optional)
- [ ] Check if search API exists
- [ ] Build if missing (or skip this phase)
- [ ] Mount IMP engine
- [ ] Wire search input
- [ ] Poll job progress
- [ ] Test complete flow
- [ ] Commit & push

### Final Verification (30 min)
- [ ] All 5 screens render at 390×844
- [ ] Autumn Leaves flow works: Launcher → Chart → Edit → Re-infer → Propagation
- [ ] No console errors
- [ ] All touch targets ≥44×44pt
- [ ] Screenshots ready

---

## Files to Edit

| File | What to Change | Why |
|------|---|---|
| `docs/harmonia-integrated-app.html` | Wire Chart, Chord Editor, Re-infer, Import | Core integration |
| `scripts/harmonia_server.py` | Verify/add search endpoints (if needed) | Import feature |
| (Optional) `harmonia/output/chart_interactive.py` | Integrate into template via SPLICES | Production deployment |

---

## Debugging Quick Reference

| Problem | Solution |
|---------|----------|
| "HZC is not defined" | Did you load `harmonia_chart.js`? Check `<script src>` |
| Chart doesn't render | Is P.chords populated? `console.log(P)` |
| Colours wrong | Check depth: `c.lv.seventh.c` vs `c.lv.family.c` |
| Modal doesn't open | Is mount DOM element present? |
| Re-infer fails | Did you test endpoint first with curl? |
| Screenshot looks wrong | Compare pixel-for-pixel to `handoff/js/*_demo.html` |

---

## Success Criteria — Mission Complete

### ✅ Gate 1: All Screens Render
- [ ] Chart Viewer at 390×844 matches demo
- [ ] Chord Editor modal opens, 4 tabs visible
- [ ] Re-infer button present, clickable
- [ ] Import screen loads

### ✅ Gate 2: Full Flow Works
- [ ] Launcher → opens Chart Viewer
- [ ] Chart displays Autumn Leaves chords
- [ ] Tap chord → opens modal → select from Compass → updates chart
- [ ] Click "Re-infer" → POSTs to endpoint → shows propagation
- [ ] No console errors

### ✅ Gate 3: iPhone Ready
- [ ] All screens fit at 390×844
- [ ] All text readable
- [ ] All buttons ≥44×44pt
- [ ] Server reachable on Tailscale IP:7771

---

## Timeline Estimate

**If starting now:**
- Chart Viewer: 2-3 hours
- Chord Editor: 1-2 hours
- Re-infer: 30m-1 hour
- Import: 2-3 hours (defer if needed)
- **Total: 4-8 hours** (1-2 days full-time)

**Fastest path (defer Import):**
- **Total: 4-5 hours** (half day)

---

## When Blocked

If you get stuck:
1. **Screenshot the problem** (visual makes it obvious)
2. **Check browser console** (F12, find the error)
3. **Verify data** — does P exist? Is it populated?
4. **Compare to demo** — open the reference, what's different?
5. **Read the error carefully** — "cannot read property 'root'" means P.chords is wrong shape
6. **Use adapter function** — don't guess at data shapes

---

## Handing Off to Production

Once Phase 2 is complete and tested:
1. **Move integration into `chart_interactive.py`** (the real template)
2. **Run migration script** after each change:
   ```bash
   .venv/bin/python scripts/migrate_annotator_tool.py
   ```
3. **Screenshot all baked charts** at 390×844 to verify
4. **Test on iPhone** via Tailscale
5. **Deploy to production** (share port 7771 + Tailscale URL)

---

## Reference Materials (Read in Order)

1. **This document** — you are here
2. **`handoff/HANDOFF.md`** — design spec (5 min reference)
3. **`docs/PHASE_2_STATUS.md`** — detailed per-screen checklist
4. **`handoff/js/*_demo.html`** — open in browser, see targets
5. **`docs/INTEGRATION_GUIDE.md`** — data wiring patterns

---

## Final Checklist: "Mission Complete" When...

- [ ] All 5 screens visible at 390×844
- [ ] Launcher works (polling + redirect)
- [ ] Chart Viewer shows real Autumn Leaves chords
- [ ] 3 modes toggle correctly (Read/Analyse/Annotate)
- [ ] Chord editor opens with 4 tabs (Wheel/Suggestions/Compass/Guide)
- [ ] Selecting from Compass/Guide updates chord
- [ ] Re-infer button POSTs to `/api/reinfer`, shows propagation
- [ ] Import screen works (or documented as deferred)
- [ ] No console errors on any screen
- [ ] All touch targets ≥44×44pt
- [ ] Tested on iPhone via Tailscale
- [ ] Coordinator approves 🚀

---

## "I'm Ready to Start" Checklist

Before writing code:
- [ ] Server running: `.venv/bin/python scripts/harmonia_server.py`
- [ ] Launcher tested: `http://localhost:7771/harmonia.html`
- [ ] Read `handoff/HANDOFF.md`
- [ ] Opened `handoff/js/*_demo.html` in browser
- [ ] Reviewed `docs/PHASE_2_STATUS.md` (Chart Viewer section)
- [ ] Have `harmonia-integrated-app.html` open for editing
- [ ] Device ready for 390×844 testing (browser dev tools or actual iPhone)

**Then:** Start with Chart Viewer (highest value, fastest feedback)

---

Good luck! You've got this. 🚀

