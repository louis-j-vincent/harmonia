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
    # ---- #22
    zs = pg.evaluate("""()=>[...document.querySelectorAll('[data-form-band] button')].map(b=>{const r=b.getBoundingClientRect(); return [Math.round(r.width),Math.round(r.height)]})""")
    segs = pg.evaluate("""()=>[...document.querySelectorAll('[data-form-band] > div')].map(b=>Math.round(b.getBoundingClientRect().width))""")
    print("     form zones", zs, "visible", segs)
    check(len(zs) >= 2 and all(w >= 44 and h >= 44 for w, h in zs), f"{name} #22 every form-rail zone >= 44x44")
    hit = pg.evaluate("""()=>{const r=document.querySelectorAll('[data-form-band] button')[0].getBoundingClientRect(); const e=document.elementFromPoint(r.right-3, r.top+r.height/2); return e===document.querySelectorAll('[data-form-band] button')[0]}""")
    check(hit, f"{name} #22 the narrow segment keeps its widened zone (on top of its neighbour)")
    pg.screenshot(path=pre + "_22_rail.png", clip={"x": 0, "y": 40, "width": 390, "height": 60})
    # ---- #23
    sh = pg.locator('[data-role="share"]'); check(sh.count() == 1, f"{name} #23 share icon under the title")
    bb = sh.bounding_box(); check(bb["width"] >= 44 and bb["height"] >= 44, f"{name} #23 share >=44 {bb}")
    pg.screenshot(path=pre + "_23_toolbar.png", clip={"x": 0, "y": 0, "width": 390, "height": 80})
    tap(sh); pg.wait_for_timeout(400)
    check(pg.get_by_text("Lien de partage").count() == 1, f"{name} #23 share sheet opens")
    pg.screenshot(path=pre + "_23_share.png")
    tap(pg.get_by_role("button", name="Annuler").first); pg.wait_for_timeout(300)
    tap(pg.get_by_role("button", name="Réglages").first); pg.wait_for_timeout(400)
    check("Partager" not in pg.inner_text("body"), f"{name} #23 Réglages sheet no longer has Partager")
    pg.screenshot(path=pre + "_23_reglages.png")
    tap(pg.get_by_role("button", name="Fermer").first); pg.wait_for_timeout(300)
    # ---- #24
    tap(pg.get_by_role("button", name="Réglages").first); pg.wait_for_timeout(400)
    sm = pg.locator("summary").first
    check("Plus de réglages" in sm.inner_text(), f"{name} #24 summary says « Plus de réglages »")
    bb = sm.bounding_box(); check(bb["height"] >= 44, f"{name} #24 summary >= 44 high {bb['height']}")
    col = sm.evaluate("e=>getComputedStyle(e).color"); ink = pg.evaluate("async()=>{const {T}=await import('/static/ui/kit.js'); return T.ink}")
    print("     summary colour", col, "ink", ink)
    pg.screenshot(path=pre + "_24_closed.png", clip={"x": 0, "y": 560, "width": 390, "height": 284})
    tap(sm); pg.wait_for_timeout(400)
    check(sm.inner_text().strip().startswith("▾"), f"{name} #24 chevron turns")
    pg.screenshot(path=pre + "_24_open.png")
    ev("S._prefsPlusOpen=false")
    tap(pg.get_by_role("button", name="Fermer").first); pg.wait_for_timeout(300)
    # ---- #25
    tap(pg.get_by_role("button", name="Réglages").first); pg.wait_for_timeout(400)
    txt = pg.inner_text("body")
    check("Détail des accords" in txt and "Apprendre" not in txt and "Niveau" not in txt, f"{name} #25 one control, no Apprendre/Niveau")
    pg.screenshot(path=pre + "_25_full.png", clip={"x": 0, "y": 440, "width": 390, "height": 404})
    tap(pg.get_by_role("button", name="Triades", exact=True)); pg.wait_for_timeout(500)
    check(ev("return [S.learn,S.level]") == [True, 1], f"{name} #25 Triades = learn on, level 1")
    pg.screenshot(path=pre + "_25_triades.png")
    tap(pg.get_by_role("button", name="Réglages").first); pg.wait_for_timeout(400)
    pg.screenshot(path=pre + "_25_sheet_triades.png", clip={"x": 0, "y": 440, "width": 390, "height": 404})
    tap(pg.get_by_role("button", name="Complet", exact=True)); pg.wait_for_timeout(500)
    check(ev("return S.learn") is False, f"{name} #25 Complet = learn off")
    check(not errs, f"{name} no page errors: {errs}")
    # ---- #26 : a chart without audio
    pg.goto("http://localhost:7772/?open=ireal_autumn_leaves"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    gb = pg.locator('[data-role="goto-bar"]')
    check(gb.count() == 1 and gb.bounding_box()["height"] >= 44, f"{name} #26 « Aller à la mesure… » on an audio-less chart")
    pg.screenshot(path=pre + "_26_noaudio.png")
    tap(gb); pg.wait_for_timeout(400)
    inp = pg.get_by_label("Numéro de mesure"); inp.fill("20"); pg.keyboard.press("Enter"); pg.wait_for_timeout(1200)
    pg.screenshot(path=pre + "_26_after.png")
    check(ev("return S.playBar") == 19, f"{name} #26 goto 20 -> bar index 19, got " + str(ev("return S.playBar")))
    check(not errs, f"{name} no page errors (ireal): {errs}")
    return pg, tap, ev, pre

with sync_playwright() as p:
    for name in ("chromium", "webkit"):
        run(p, name)
print("FAILS:", fails); sys.exit(1 if fails else 0)
