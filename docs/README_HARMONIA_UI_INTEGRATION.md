# Harmonia UI/UX Integration — Master Index

**Project:** Integrate 5 redesigned UI screens for production app  
**Current Phase:** 1 of 3 (Launcher complete; roadmap ready for Phase 2)  
**Last Updated:** 2026-07-13

---

## Start Here

### 1. **Quick Test** (2 minutes)
```bash
cd ~/Documents/Projets\ Perso/Code/harmonia
.venv/bin/python scripts/harmonia_server.py --no-open --port 7771
# Then open: http://localhost:7771/harmonia.html
```
Expected: Server check → "Running on :7771" → Auto-redirect

---

### 2. **Understand the Scope** (15 minutes)
Read in order:
1. **`handoff/HANDOFF.md`** — 5 screens spec (what the designer built)
2. **`docs/SESSION_SUMMARY.md`** — What we did this session (what's done vs. next)
3. **`docs/INTEGRATION_GUIDE.md`** — How to implement (detailed step-by-step)

---

### 3. **See the Target Designs** (5 minutes)
Open these in your browser to see what you're building toward:
- `handoff/js/harmonia_launcher_demo.html` — Launcher target ✅ (DONE)
- `handoff/js/harmonia_chart_demo.html` — Chart Viewer target
- `handoff/js/harmonia_chord_editor_demo.html` — Chord Editor target
- `handoff/js/harmonia_import_demo.html` — Import target
- `handoff/js/harmonia_reinfer_demo.html` — Re-infer target

---

## The 5 Screens

| Screen | Status | Where | Next Step |
|---|---|---|---|
| **Launcher** | ✅ Done | `harmonia.html` | Test on iPhone via Tailscale |
| **Chart Viewer** | ⏳ Not started | `harmonia/output/chart_interactive.py` | Add 3-mode toggle (Read/Analyse/Annotate) |
| **Chord Editor** | ⏳ Not started | `chart_interactive.py::chordEditModal` | Add Compass + Guide tabs |
| **Re-infer UI** | ⏳ Mock ready | `chart_interactive.py` + `/api/reinfer` endpoint | Wire real endpoint |
| **Import** | ⏳ Not started | New screen in app | Build search UI + 6-stage progress |

---

## Documentation Map

### For Understanding the Architecture
- **`handoff/HANDOFF.md`** — Full design spec, 5 winners, acceptance checklist
- **`docs/annotation_sidecar_schema.md`** — How corrections are stored/persisted
- **`docs/handoff_mission3_ui_contract.md`** — `/api/reinfer` endpoint shape

### For Implementing
- **`docs/INTEGRATION_GUIDE.md`** — Step-by-step for each screen, data wiring, markers
- **`docs/handoff_2026-07-13_annotator_ui.md`** — Migration script + template editing rules
- **`scripts/migrate_annotator_tool.py`** — Syncs template changes to baked charts

### For Testing
- **`docs/TESTING_QUICK_START.md`** — Commands to start server, test on desktop/iPhone
- **`docs/SESSION_SUMMARY.md`** — What works now, what's next, risk mitigation

### Reference
- **`handoff/js/*.js`** — 5 drop-in engines (Launcher, Chart, Chord Editor, Import, Re-infer)
- **`handoff/js/*_demo.html`** — Target layouts (open in browser to see what you're aiming for)

---

## Key Concepts

### Drop-in Engines
Each `handoff/js/*.js` file is:
- **Self-contained** — no framework, no build step
- **Auto-initializing** — finds its DOM mount point on page load
- **Data-driven** — you swap demo data for real P.chords etc.

Example:
```html
<script src="harmonia_chart.js"></script>
<div data-hzc="C:desktop"></div>  <!-- Mounts here automatically -->
```

### Data Wiring Pattern
```
P = window.P (from server)
  ↓ [mapping function]
  ↓ Engine-specific format
  ↓ [build() → DOM]
  ↓ Rendered UI
```

See `docs/INTEGRATION_GUIDE.md` §Data Wiring Helper Functions for code.

### Template Workflow
```
1. Edit harmonia/output/chart_interactive.py
2. Check changes sit INSIDE SPLICES markers (see migrate_annotator_tool.py)
3. .venv/bin/python scripts/migrate_annotator_tool.py
4. Take screenshot at 390×844: google-chrome --headless --screenshot=... --window-size=390,844
5. Compare to handoff/js/*_demo.html
```

**Golden rule:** No template change without running migration + taking screenshot.

---

## Immediate Next Steps (Ranked by Value)

### 🔥 Highest Priority (3 hours)
1. Add 3-mode control to Chart Viewer (Read/Analyse/Annotate toggles)
2. Add Compass tab to Chord Editor
3. Test full flow at 390×844

**Payoff:** Professional chart + new editor modes visible

### 🔥 High Priority (2 hours)
4. Add Guide tab to Chord Editor
5. Wire "Re-infer" button to `/api/reinfer` endpoint
6. Test on iPhone via Tailscale

**Payoff:** Full annotator loop working (edit → re-infer → see propagation)

### 📌 Medium Priority (4+ hours)
7. Implement iReal layout (barlines, section boxes, playhead)
8. Build Import screen (YouTube search + 6-stage progress)
9. Persist annotations to sidecars

**Payoff:** Polished, production-ready UI

---

## Testing Checklist

### Phase 1 ✅ (Done)
- [x] Launcher displays + detects server
- [x] Server polling works (3s retry)
- [x] Auto-redirect on boot

### Phase 2 (Next)
- [ ] Chart Viewer renders with 3 modes
- [ ] Chord Editor has Compass + Guide tabs
- [ ] Re-infer button works (mock or real)
- [ ] Full flow on Autumn Leaves
- [ ] Screenshots at 390×844 match demos
- [ ] Text readable on iPhone

### Phase 3 (Polish)
- [ ] Import screen works
- [ ] Annotations persist to sidecars
- [ ] All acceptance items from `handoff/HANDOFF.md`
- [ ] Real iPhone testing via Tailscale

---

## File Structure

```
harmonia/
├── harmonia.html ✅ (UPDATED)
│   └── New Launcher design with real polling
│
├── docs/
│   ├── README_HARMONIA_UI_INTEGRATION.md (you are here)
│   ├── SESSION_SUMMARY.md (what was done this session)
│   ├── INTEGRATION_GUIDE.md (how to implement each screen)
│   ├── TESTING_QUICK_START.md (copy-paste test commands)
│   ├── app-shell.html (modular screen reference)
│   ├── HANDOFF.md (design spec — read first)
│   ├── annotation_sidecar_schema.md (corrections storage)
│   ├── handoff_mission3_ui_contract.md (API spec)
│   ├── handoff_2026-07-13_annotator_ui.md (migration details)
│   └── plots/inferred_*.html (baked charts, updated by migration)
│
├── handoff/
│   ├── HANDOFF.md (design spec — THE SOURCE OF TRUTH)
│   └── js/
│       ├── harmonia_launcher.js
│       ├── harmonia_launcher_demo.html
│       ├── harmonia_chart.js
│       ├── harmonia_chart_demo.html
│       ├── harmonia_chord_editor.js
│       ├── harmonia_chord_editor_demo.html
│       ├── harmonia_import.js
│       ├── harmonia_import_demo.html
│       ├── harmonia_reinfer.js
│       └── harmonia_reinfer_demo.html
│
├── harmonia/
│   └── output/
│       └── chart_interactive.py (TEMPLATE — edit here for Chart Viewer changes)
│
├── scripts/
│   ├── harmonia_server.py (Flask app, already has `/api/reinfer`)
│   └── migrate_annotator_tool.py (syncs template changes to baked charts)
│
└── ... (rest of repo)
```

---

## Quick Reference: Data Format

### Pipeline Output (P.chords)
```javascript
{
  root: 0-11,           // Pitch class
  bass: 0-11 or -1,     // Bass note or none
  bar: 0+,              // Bar index
  beat: 0-4,            // Beat offset in bar
  lv: {
    family:  { q: str, c: 0-1 },  // Triad only (C, Cm, C°, C+, Csus)
    seventh: { q: str, c: 0-1 },  // 7th-level (C, C7, Cm7, etc)
    exact:   { q: str, c: 0-1 }   // Full quality (C, C9, Cm7♭5, etc)
  },
  t0: seconds?,         // Start time (if available)
  t1: seconds?,         // End time (if available)
  sug: [                // Model's top suggestions
    { root: 0-11, q: str, c: 0-1 },
    ...
  ]
}
```

### Engine Format (Handoff)
Engines expect the same shape, just different naming:
- `c` → confidence at that level (not `conf`)
- `q` → iReal quality token (not `quality`)
- `sug` → suggestion list (keep as-is)

---

## Support & Troubleshooting

### Server won't start
```bash
lsof -i :7771  # Check if port is in use
kill -9 <PID>  # Kill if needed
```

### Launcher keeps checking
- Is the server actually running? (`http://localhost:7771` should return 200)
- Check console for errors (F12)
- Network may be blocked (firewall)

### Charts look wrong after migration
- Did you run `migrate_annotator_tool.py`?
- Compare to `handoff/js/*_demo.html`
- Check browser console for JS errors
- Take screenshot, compare pixels to demo

### Compass/Guide not showing
- Is data wired correctly? (P.chords format matches engine expectation)
- Mount element exists? (check DOM)
- Browser console errors?
- Try opening the demo file first to verify engine works

---

## Session Wrap-Up

**What was accomplished:**
- ✅ Launcher redesign (embedded, real polling, auto-redirect)
- ✅ Integration guide (full spec for 5 screens)
- ✅ Testing guide (copy-paste commands)
- ✅ Architecture documented (data wiring, migration, template rules)

**Ready to go:**
- The Launcher at `harmonia.html` (test immediately)
- Clear roadmap for Chart Viewer → Chord Editor → Re-infer
- All reference materials (design specs, target layouts, schemas)

**Next priority:**
- Implement Chart Viewer 3-mode toggle
- Add Compass + Guide tabs
- Wire Re-infer endpoint
- Test on iPhone via Tailscale

---

## Questions?

Refer to these docs in order:
1. **What should I build?** → `handoff/HANDOFF.md`
2. **How do I build it?** → `docs/INTEGRATION_GUIDE.md`
3. **How do I test it?** → `docs/TESTING_QUICK_START.md`
4. **What's the current state?** → `docs/SESSION_SUMMARY.md`
5. **Where does this go?** → File structure (above)

---

**Happy coding! 🎸**

