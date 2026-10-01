"""UX lot 2, rows #20-#26 (chart.js / prefs.js / kit.js). Chromium AND webkit,
390x844, every POST intercepted.   .venv/bin/python tests/ui/check_chart_lot2.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lot2"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def run(p, name):
    if name == "webkit":
        b = p.webkit.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    else:
        b = p.chromium.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844})
    ctx.add_init_script("try{localStorage.setItem('harmPianoOffByDefault','1')}catch(e){}")
    pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json") if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pre = f"{PRE}_{name}"
    def tap(loc):
        loc.tap() if name == "webkit" else loc.click()
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)

    # ---- #20
    meta = pg.get_by_role("button", name="Transposer").first
    check("Transposer" in meta.inner_text(), f"{name} #20 toolbar says Transposer")
    pg.screenshot(path=pre + "_20_toolbar.png")
    tap(meta); pg.wait_for_timeout(500)
    txt = pg.inner_text("body")
    check("minor" not in txt and "major" not in txt, f"{name} #20 rotor sheet without minor/major")
    check(re.search(r"majeur|mineur", txt) is not None, f"{name} #20 rotor hub in French")
    pg.screenshot(path=pre + "_20_rotor.png")
    tap(pg.get_by_role("button", name="Annuler").first); pg.wait_for_timeout(300)
    # ---- #21
    rate = pg.locator('[data-role="rate"]')
    check(rate.count() == 1 and rate.inner_text().strip() == "1×", f"{name} #21 speed pill shows 1×")
    bb = rate.bounding_box(); check(bb["width"] >= 44 and bb["height"] >= 44, f"{name} #21 pill >=44px {bb}")
    pg.screenshot(path=pre + "_21_dock.png")
    tap(rate); pg.wait_for_timeout(400)
    pg.screenshot(path=pre + "_21_sheet.png")
    for lb in ("0,5×", "0,6×", "0,75×", "0,9×", "1×"):
        check(pg.get_by_role("button", name=lb, exact=True).count() >= 1, f"{name} #21 sheet offers {lb}")
    tap(pg.get_by_role("button", name="0,75×", exact=True)); pg.wait_for_timeout(500)
    check(ev("return S.playRate") == 0.75, f"{name} #21 S.playRate = 0.75")
    check(pg.locator('[data-role="rate"]').inner_text().strip() == "0,75×", f"{name} #21 pill now 0,75×")
    pg.screenshot(path=pre + "_21_after.png")
    ev("S.playRate=1")
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('chart',{push:false});}"); pg.wait_for_timeout(400)
    check(not errs, f"{name} no page errors: {errs}")
    return pg, tap, ev, pre

with sync_playwright() as p:
    for name in ("chromium", "webkit"):
        run(p, name)
print("FAILS:", fails); sys.exit(1 if fails else 0)
