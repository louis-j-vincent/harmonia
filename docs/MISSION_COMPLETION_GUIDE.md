# MISSION COMPLETION GUIDE — Full Integration into Production App

**Status:** 🟡 Phase 2 ready to execute  
**Coordinator Request:** Full integration of 5 screens into real app with real data + screenshots  
**Timeline:** 2-4 hours for complete integration and testing

---

## TL;DR — What You Have vs What's Needed

### What Works Now ✅
- Launcher (harmonia.html) — tested, auto-redirects
- App shell scaffolded (harmonia-integrated-app.html)
- All 5 handoff engines ready to mount (handoff/js/*.js)
- Real Autumn Leaves data available (648 chords, 40 sections, all suggestions)
- Server running with /api/reinfer endpoint ready

### What's Needed 🟡
- Mount 5 handoff engines to real DOM elements
- Wire real P data (P.chords, P.sections, etc.)
- Add 3-mode control to Chart Viewer
- Add Compass+Guide tabs to Chord Editor
- Wire Re-infer POST to real endpoint
- Test at 390×844 with screenshots
- Run migration script after template changes

---

## Complete Integration Blueprint

### Step 1: Serve Handoff JS Files from Flask Server (15 min)

Edit `scripts/harmonia_server.py` to serve handoff files:

```python
# Add after the existing routes
@app.route('/handoff/<path:filename>')
def serve_handoff(filename):
    return send_from_directory(REPO / 'handoff' / 'js', filename)

# Add 'js' to the directory listing
@app.route('/handoff/js/')
def handoff_js_list():
    files = list((REPO / 'handoff' / 'js').glob('*.js'))
    return {'files': [f.name for f in files]}
```

**Test:** Open `http://localhost:7771/handoff/js/harmonia_chart.js` → should return JS file

---

### Step 2: Create Integrated Chart Template (45 min)

In `harmonia/output/chart_interactive.py`, modify the existing template to add:

**2a. Import handoff engines at top of HTML:**
```html
<script src="/handoff/js/harmonia_chart.js"></script>
<script src="/handoff/js/harmonia_chord_editor.js"></script>
<script src="/handoff/js/harmonia_reinfer.js"></script>
<script src="/handoff/js/harmonia_import.js"></script>
```

**2b. Add 3-mode control to Chart (after existing mode options):**
```html
<!-- In optionsModal section, add -->
<div class="opt-section">
  <div class="opt-title">View Mode</div>
  <div class="opt-row">
    <button type="button" id="mode-read">Read</button>
    <button type="button" id="mode-analyse">Analyse</button>
    <button type="button" id="mode-annotate">Annotate</button>
  </div>
</div>
```

**2c. Add JavaScript to wire modes:**
```javascript
// Add to existing JS block
const CHART_MODES = {
  read: function() { /* show chords, no colour */ },
  analyse: function() { /* show colours by function/key */ },
  annotate: function() { /* show certainty colours */ }
};

document.querySelectorAll('[id^="mode-"]').forEach(btn => {
  btn.onclick = function() {
    const mode = this.id.split('-')[1];
    CHART_MODES[mode]();
  };
});
```

**⚠️ CRITICAL:** All changes MUST sit INSIDE SPLICES markers in `migrate_annotator_tool.py`

---

### Step 3: Mount Chord Editor Tabs (30 min)

In the existing `#chordEditModal`, add tabs for Compass and Guide:

```html
<!-- In #chordEditModal, replace single editor with tabbed interface -->
<div class="opt-row" id="ce-mode-group">
  <button type="button" data-mode="wheel" class="sel">Wheel</button>
  <button type="button" data-mode="suggestions">Suggestions</button>
  <button type="button" data-mode="compass">Compass</button>
  <button type="button" data-mode="guide">Guide</button>
</div>

<div id="ce-wheel-mode"><!-- existing wheel picker --></div>
<div id="ce-suggestions-mode"><!-- existing suggestions --></div>
<div id="ce-compass-mode" style="display:none;">
  <div data-hz="A:desktop"></div>
</div>
<div id="ce-guide-mode" style="display:none;">
  <div data-hz="B:desktop"></div>
</div>
```

**Wire tab switching:**
```javascript
document.querySelectorAll('#ce-mode-group button').forEach(btn => {
  btn.onclick = function() {
    document.getElementById('ce-wheel-mode').style.display = this.dataset.mode === 'wheel' ? 'block' : 'none';
    document.getElementById('ce-suggestions-mode').style.display = this.dataset.mode === 'suggestions' ? 'block' : 'none';
    document.getElementById('ce-compass-mode').style.display = this.dataset.mode === 'compass' ? 'block' : 'none';
    document.getElementById('ce-guide-mode').style.display = this.dataset.mode === 'guide' ? 'block' : 'none';
  };
});
```

---

### Step 4: Wire Re-infer Endpoint (20 min)

In the existing chord editor JS, find the `reinfer()` function:

```javascript
// OLD (mock):
const resp = mockReinfer({confirms, merges});

// NEW (real endpoint):
async function reinfer(chart, confirms, merges) {
  try {
    const resp = await fetch(`/api/reinfer/${chart}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirms, merges })
    });
    if (!resp.ok) throw new Error(`${resp.status}`);
    const {diff} = await resp.json();
    
    // Apply diff by time overlap
    applyDiff(diff);
    
    // Show propagation
    showPropagation(diff);
    
    return diff;
  } catch (err) {
    console.error('Re-infer failed:', err);
    toast('Re-infer failed: ' + err.message);
    return null;
  }
}

// Call with real chart name
reinfer('inferred_autumn_leaves.html', confirms, merges);
```

---

### Step 5: Migration & Testing (45 min)

After each change to `chart_interactive.py`:

```bash
# 1. Run migration script
.venv/bin/python scripts/migrate_annotator_tool.py

# 2. Screenshot at 390×844 to verify
google-chrome --headless --screenshot=/tmp/chart.png --window-size=390,844 \
  http://localhost:7771/docs/plots/inferred_autumn_leaves.html

# 3. Compare to handoff demo
# Open: handoff/js/harmonia_chart_demo.html
# Check: visual match at 390×844
```

---

## Complete Test Flow: Autumn Leaves

### Setup
```bash
.venv/bin/python scripts/harmonia_server.py --no-open --port 7771
# Wait for "Running on http://localhost:7771"
```

### Test Sequence
1. **Launcher**
   - Open `http://localhost:7771/harmonia.html`
   - Should auto-detect server (green dot)
   - Click "Open Harmonia →" (or wait 900ms)
   - Screenshot at 390×844

2. **Chart Viewer**
   - Opens Autumn Leaves chart
   - Toggle 3 modes (Read/Analyse/Annotate)
   - Verify colours change correctly
   - Screenshot each mode at 390×844

3. **Chord Editor**
   - Tap a chord (e.g., bar 4, the low-confidence one)
   - Modal opens
   - Switch between 4 tabs (Wheel/Suggestions/Compass/Guide)
   - Select from Compass or Guide
   - Verify chord updates in chart
   - Screenshot each tab at 390×844

4. **Re-infer Loop**
   - Edit chord (select from Compass)
   - Confirm selection (hard-clamp c=1)
   - Click "Re-infer with Fixes"
   - POST to `/api/reinfer/inferred_autumn_leaves.html`
   - See propagation banner (which chords changed)
   - Screenshot before/after at 390×844

5. **Final Verification**
   - All 5 screens render correctly
   - No console errors (F12)
   - All text readable at 390×844
   - Touch targets ≥44×44pt

---

## Exact Integration Points in chart_interactive.py

### Location 1: HTML Head (Add Scripts)
**After:** `<meta name="viewport"...`  
**Before:** `<style>`

```html
<!-- Handoff engines -->
<script src="/handoff/js/harmonia_chart.js"></script>
<script src="/handoff/js/harmonia_chord_editor.js"></script>
<script src="/handoff/js/harmonia_reinfer.js"></script>
<script src="/handoff/js/harmonia_import.js"></script>
```

**Must be INSIDE SPLICES[0]:** between `id="motif-style-btn"...` and `<div id="motif-overlay">`

### Location 2: Options Modal (Add Mode Controls)
**In:** `#optionsModal`  
**Add:** After existing control sections, new section for 3-mode toggle

**Must be INSIDE SPLICES[0]** (same as above)

### Location 3: Chord Editor Modal (Add Compass/Guide Tabs)
**In:** `#chordEditModal`  
**Modify:** Tab buttons + div containers for each mode

**Must be INSIDE SPLICES[0]** (same as above)

### Location 4: JavaScript Re-infer Function
**In:** `<script>` block, find `reinfer()` function  
**Change:** Mock → real POST to `/api/reinfer/<chart>`

**Must be INSIDE SPLICES[2]:** between `container.appendChild(btn)...` and `render();`

---

## Critical Checklist

### Before You Start
- [ ] Read `handoff/HANDOFF.md` (understand the design)
- [ ] Open `handoff/js/*_demo.html` in browser (see targets)
- [ ] Have `chart_interactive.py` open for editing
- [ ] Have `migrate_annotator_tool.py` nearby (will need to run it)
- [ ] Server running: `.venv/bin/python scripts/harmonia_server.py`

### Implementation (In Order)
- [ ] Add handoff JS script tags to HTML head
- [ ] Add 3-mode control to options modal
- [ ] Add Compass/Guide tabs to chord editor
- [ ] Wire Re-infer function to real endpoint
- [ ] Run: `python scripts/migrate_annotator_tool.py`
- [ ] Screenshot Chart Viewer at 390×844
- [ ] Screenshot Chord Editor (each tab) at 390×844
- [ ] Test full Autumn Leaves flow: Edit → Re-infer → Propagation
- [ ] Screenshot Re-infer results at 390×844
- [ ] Verify no console errors (F12)

### Verification (Final Gate)
- [ ] All 5 screens render at 390×844 matching demos
- [ ] Chart Viewer: 3 modes toggle, colours correct
- [ ] Chord Editor: 4 tabs visible, Compass/Guide work
- [ ] Re-infer: POST succeeds, propagation shows
- [ ] Autumn Leaves flow: complete end-to-end success
- [ ] No console errors on any screen

---

## Troubleshooting Reference

| Problem | Solution |
|---------|----------|
| "Cannot read property 'chords' of undefined" | P is not loaded. Chart data not injected. Check `/docs/plots/inferred_autumn_leaves.html` has payload. |
| Compass tab shows nothing | Data not wired. Check SONG object created. Verify mount element exists. |
| Re-infer POST fails 404 | Endpoint doesn't exist. Check `/api/reinfer/<chart>` in `harmonia_server.py`. |
| Migration deletes my changes | Changes were OUTSIDE SPLICES markers. Edit again, ensure inside markers. |
| Screenshot looks wrong | Compare pixel-for-pixel to `handoff/js/*_demo.html`. Check data adapter mapping. |
| Touch target too small | Buttons must be ≥44×44pt. Check CSS padding/min-height. |

---

## Success Criteria: Mission Complete

When ALL of these are ✅:

✅ Chart Viewer renders with real Autumn Leaves chords  
✅ 3 modes (Read/Analyse/Annotate) toggle correctly  
✅ Chord Editor opens with 4 tabs (Wheel/Suggestions/Compass/Guide)  
✅ Selecting from Compass/Guide updates the displayed chord  
✅ Re-infer button POSTs to `/api/reinfer/<chart>`  
✅ Propagation banner shows (which chords changed, old→new labels, confidence deltas)  
✅ All 5 screens render at 390×844 matching `handoff/js/*_demo.html`  
✅ No console errors on any screen  
✅ All text readable, all touch targets ≥44×44pt  
✅ Autumn Leaves full flow works end-to-end  
✅ Screenshots ready for coordinator  

---

## Time Estimates

| Task | Time | Difficulty |
|------|------|-----------|
| Serve handoff JS files | 15 min | Easy |
| Add 3-mode control | 20 min | Medium |
| Add Compass/Guide tabs | 20 min | Medium |
| Wire Re-infer endpoint | 20 min | Medium |
| Testing + screenshots | 45 min | Medium |
| Debugging (if needed) | 30-60 min | Hard |
| **TOTAL** | **2-4 hours** | — |

---

## Next Action: Go Execute

1. **Read** this guide one more time
2. **Start server:** `.venv/bin/python scripts/harmonia_server.py`
3. **Edit** `harmonia/output/chart_interactive.py`
4. **Migrate** after changes: `python scripts/migrate_annotator_tool.py`
5. **Screenshot** at 390×844 to verify
6. **Test** full Autumn Leaves flow
7. **Iterate** if blocked

---

## Final Checklist Before Declaring Victory

```
LAUNCHER:
  ✓ Auto-detects server
  ✓ Auto-redirects on boot
  ✓ 390×844 screenshot matches new design

CHART VIEWER:
  ✓ Real Autumn Leaves data loaded
  ✓ 3 modes toggle (Read/Analyse/Annotate)
  ✓ Colours render correctly (function/key/certainty)
  ✓ Repeat folding works (×N badge)
  ✓ 390×844 screenshot matches demo

CHORD EDITOR:
  ✓ Modal opens with 4 tabs
  ✓ Compass shows circular orb layout
  ✓ Guide shows ranked cards with roles
  ✓ Tab switching works smoothly
  ✓ Selecting chord updates chart
  ✓ Audio playback works
  ✓ 390×844 screenshot matches demo

RE-INFER:
  ✓ Edit chord → confirm (hard-clamp)
  ✓ Re-infer button → POST to endpoint
  ✓ Propagation banner shows
  ✓ Before/after chords visible
  ✓ 390×844 screenshot shows results

FULL FLOW:
  ✓ Launcher → Chart Viewer
  ✓ Tap chord → editor opens
  ✓ Select from Compass
  ✓ Re-infer → see propagation
  ✓ End-to-end success

FINAL CHECKS:
  ✓ No console errors
  ✓ All text readable at 390×844
  ✓ All touch targets ≥44×44pt
  ✓ Migration script run
  ✓ Screenshots ready

STATUS: ✅ MISSION COMPLETE
```

---

## Deliver to Coordinator

When complete, provide:
1. Screenshots of all 5 screens at 390×844
2. Browser console showing no errors
3. Test report: "Full Autumn Leaves flow: [✅ SUCCESS / ❌ FAILURE — reason]"
4. Server URL (localhost:7771 or Tailscale IP:7771 for iPhone)
5. Any blockers encountered + solutions tried

---

**You've got this. Execute the plan, iterate on failures, and ship it. 🚀**

