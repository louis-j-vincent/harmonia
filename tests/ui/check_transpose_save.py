"""Red-first check (2026-10-01, UX audit): transposing the DISPLAY must never
change what Annotate saves. Before the fix, transposeTo() shifted S.chords in
place and saveAnnotations() posted those shifted roots — every confirmed chord
of the song was silently rewritten in the transposed key.

Run against the live app (the POST is intercepted, nothing is written):
    .venv/bin/python tests/ui/check_transpose_save.py [file]
Exit 0 = the saved roots/basses are those of the original key.
"""
import json, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    posted = []
    def grab(route):
        posted.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, body="{}", content_type="application/json")
    pg.route("**/api/annotations/**", lambda r: grab(r) if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_timeout(3500)
    out = pg.evaluate("""async () => {
      const {S} = await import('/static/state.js');
      const {transposeTo} = await import('/static/screens/prefs.js');
      const {saveAnnotations} = await import('/static/screens/annotate.js');
      const c = S.chords.find(x => !x.nc && x.bass >= 0 && x.bass !== x.root) || S.chords.find(x => !x.nc);
      const before = {root: c.root, bass: c.bass};
      transposeTo((S.key + 5) % 12);
      const shown = {root: c.root, bass: c.bass};
      c.confirmed = true; c.bassPicked = c.bass >= 0;
      await saveAnnotations();
      return {bar: c.bar, beat: c.beat, before, shown};
    }""")
    b.close()
sent = next(x for x in posted[-1]["chords"] if x["bar"] == out["bar"] and x["beat"] == out["beat"])
print("original", out["before"], "shown after transpose", out["shown"], "saved", {"root": sent["root"], "bass": sent["bass"]})
ok = sent["root"] == out["before"]["root"] and sent["bass"] in (-1, out["before"]["bass"]) \
     and out["shown"]["bass"] == (out["before"]["bass"] + 5) % 12 if out["before"]["bass"] >= 0 else True
print("OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
