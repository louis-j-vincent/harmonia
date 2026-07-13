# Testing the Redesigned UI — Quick Start

**Last updated:** 2026-07-13

---

## 1. Start the Server

From the repo root:

```bash
.venv/bin/python scripts/harmonia_server.py --no-open --port 7771
```

Expected output:
```
 * Running on http://localhost:7771
 * Press CTRL+C to quit
```

---

## 2. Test the Launcher

### Desktop
Open in your browser: **http://localhost:7771/harmonia.html**

Expected sequence:
1. "Looking for Harmonia…" (animated bars, dot pulsing amber)
2. Server found → "Running on :7771" (green dot, glowing ring)
3. "Open Harmonia →" button appears
4. After ~900ms: Auto-redirect to main app (or click button)

### iPhone (via Tailscale)
1. Get the host's Tailscale IP (on host machine): `ifconfig | grep utun`
2. On iPhone: open `http://<tailscale-ip>:7771/harmonia.html` (add to Home Screen for PWA experience)
3. Verify 390×844 viewport
4. Tap "Open Harmonia →"

---

## 3. Current Status

### ✅ Complete
- **Launcher redesign** (`harmonia.html`) — new cream-themed UI with real server polling
- **Integration guide** (`docs/INTEGRATION_GUIDE.md`) — full spec for 5 screens
- **App shell reference** (`docs/app-shell.html`) — modular screen architecture (demo, not live yet)

### ⏳ Not Yet Integrated
- **Chart Viewer** (iReal with 3 modes) — currently uses old template
- **Chord Editor Compass+Guide** — existing Wheel/Suggestions tabs only
- **Import screen** — doesn't exist yet
- **Re-infer UI** — mock ready, needs real `/api/reinfer` wiring
- **Re-infer migration** — annotation sidecars not persisted yet

---

## 4. If You Want to Test Chart Viewer Now

The existing chart template is at: `/docs/plots/inferred_autumn_leaves.html`

1. Start server: `.venv/bin/python scripts/harmonia_server.py`
2. Open: `http://localhost:7771/docs/plots/inferred_autumn_leaves.html`
3. Click "Options" (⋮) button
4. Toggle "Annotate" → opens chord editor
5. Tap a chord → long-press (500ms) → opens editor with Wheel + Suggestions tabs

This is the OLD design. The NEW design (with Compass + Guide) isn't deployed yet.

---

## 5. If You Want Screenshots

### Desktop (full width)
```bash
# Basic chart screenshot
google-chrome --headless --screenshot=/tmp/chart.png \
  http://localhost:7771/docs/plots/inferred_autumn_leaves.html
```

### iPhone (390×844)
```bash
# iPhone viewport with chart
google-chrome --headless --screenshot=/tmp/chart-iphone.png \
  --window-size=390,844 \
  http://localhost:7771/docs/plots/inferred_autumn_leaves.html

# Launcher
google-chrome --headless --screenshot=/tmp/launcher-iphone.png \
  --window-size=390,844 \
  http://localhost:7771/harmonia.html
```

---

## 6. Migration Step (After Any Template Edit)

If you edit `harmonia/output/chart_interactive.py`, you MUST then run:

```bash
.venv/bin/python scripts/migrate_annotator_tool.py
```

This syncs your changes from the template to all already-rendered charts (docs/plots/inferred_*.html).

**Do NOT commit chart_interactive.py changes without running this.**

---

## 7. Known Limitations Right Now

1. **Chart Viewer** still shows old design (Wheel/Suggestions only)
2. **Compass/Guide tabs** not yet added to chord editor
3. **Import screen** doesn't exist
4. **Re-infer** has mock UI, but `/api/reinfer` endpoint may need verification
5. **Annotations** (corrections) not yet persisted to sidecars

---

## 8. Next Priority

Once you confirm Launcher works:
1. Add the new Chart Viewer modes to the existing template
2. Add Compass + Guide tabs to the chord editor modal
3. Wire the Re-infer button to real `/api/reinfer` endpoint
4. Test full flow on iPhone

See `docs/INTEGRATION_GUIDE.md` for detailed step-by-step instructions.

---

## 9. Troubleshooting

### Server won't start
```bash
# Check if port 7771 is in use
lsof -i :7771
# Kill the process if needed
kill -9 <PID>
```

### Launcher shows "Checking" forever
- Make sure server is actually running
- Check `http://localhost:7771/` returns a 200 (even in no-cors mode)
- Check firewall isn't blocking local connections

### Charts look weird after migration
- Did you run `scripts/migrate_annotator_tool.py`?
- Check browser console for JS errors
- Take a screenshot, compare to `handoff/js/*_demo.html`

### Compass/Guide not showing (after you add them)
- Check they're mounted to the right DOM element
- Verify data is wired correctly (P.chords format)
- Check browser console for errors

---

## 10. Support Links

- **Design spec:** `handoff/HANDOFF.md`
- **Implementation guide:** `docs/INTEGRATION_GUIDE.md`
- **Migration docs:** `docs/handoff_2026-07-13_annotator_ui.md`
- **Annotation schema:** `docs/annotation_sidecar_schema.md`

