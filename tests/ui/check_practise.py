"""Check (2026-10-01, UX #5): Practise loop length 1/2/4/8, visible speed.
    .venv/bin/python tests/ui/check_practise.py [file]
Exit 0 = the loop band spans N bars of the bar grid, the speed chip matches
S.playRate, the length persists, no console errors. POSTs are intercepted."""
import sys
from playwright.sync_api import sync_playwright
FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
bad = []
with sync_playwright() as p:
  for name in ("chromium", "webkit"):
    b = getattr(p, name).launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json") if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_timeout(3500)
    pg.evaluate("localStorage.removeItem('harmLoopBars')")
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}")
    pg.wait_for_timeout(1500)
    pg.screenshot(path=f"/tmp/practise_{name}.png")
    def band():
        return pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); const e=S._loopEngine;
          const g=S.model.barGrid; if(!e) return null;
          const i=g.findIndex(x=>Math.abs(x-e.a)<1e-6), j=g.findIndex(x=>Math.abs(x-e.b)<1e-6);
          return {i,j,n:j-i};}""")
    chip = pg.locator("button[title^=\"Longueur\"]")
    loopb = pg.locator("button", has_text="boucle")
    seen = []
    for want in (4, 8, 1, 2, 4):
        while chip.inner_text().split()[0] != str(want): chip.click()
        if not pg.evaluate("!!window.__x") : pass
        loopb.click(); pg.wait_for_timeout(300)
        bd = band(); seen.append((want, bd))
        if not bd or bd["n"] != want or bd["i"] % want: bad.append((name, "band", want, bd))
        loopb.click(); pg.wait_for_timeout(200)
    # chip change while looping re-bands
    loopb.click(); pg.wait_for_timeout(200)
    while chip.inner_text().split()[0] != "2": chip.click()
    pg.wait_for_timeout(300); bd = band()
    if not bd or bd["n"] != 2: bad.append((name, "rechange", bd))
    pg.screenshot(path=f"/tmp/practise_{name}_loop.png")
    # speed chip
    sp = pg.locator("button[title^=\"Vitesse\"]")
    for _ in range(6):
        sp.click()
        rate = pg.evaluate("async()=>(await import('/static/state.js')).S.playRate")
        lab = sp.inner_text(); sub = pg.locator("body").inner_text()
        if lab != {0.5: "0,5×", 0.6: "0,6×", 0.75: "0,75×", 0.9: "0,9×", 1: "1×"}[rate]: bad.append((name, "speed", rate, lab))
    ls = pg.evaluate("localStorage.getItem('harmLoopBars')")
    if ls != "2": bad.append((name, "persist", ls))
    if errs: bad.append((name, "console", errs[:3]))
    print(name, seen, "ls", ls, "errs", errs[:2]); b.close()
print("FAIL" if bad else "OK", bad); sys.exit(1 if bad else 0)
