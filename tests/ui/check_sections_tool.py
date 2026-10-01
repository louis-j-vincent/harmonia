"""Item 11 of the 2026-10-01 UX list: the sections tool (Corriger > ··· >
« Redessiner les sections »). Tempo shown, « Défaire » works, no dead « tri »
chips, every tap target >= 44 px. Every POST is intercepted (nothing reaches
state/).   .venv/bin/python tests/ui/check_sections_tool.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sectool"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def run(p, name):
    b = getattr(p, name).launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    pg = ctx.new_page(); errs = []; posts = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def route(r):
        if r.request.method == "POST":
            posts.append(r.request.url)
            if "/inferer/" in r.request.url or "/soudure/" in r.request.url:
                r.fulfill(status=200, body='{"error":"test"}', content_type="application/json")
            else: r.fulfill(status=200, body='{"ok":true,"written":0}', content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    pg.get_by_text("···").first.click(); pg.wait_for_timeout(400)
    pg.get_by_text("Redessiner les sections").first.click(); pg.wait_for_timeout(1500)
    tag = f"[{name}] "
    txt = pg.evaluate("document.body.innerText")
    check(re.search(r"♩ = \d+", txt) is not None, tag + "tempo « ♩ = N » shown")
    check(pg.get_by_text(re.compile("^tri$")).count() == 0, tag + "no dead « tri » chips")
    und = pg.locator("[data-sect-undo]")
    check(und.count() == 1 and und.is_disabled(), tag + "« Défaire » present, disabled at start")
    # two taps on bars = one mark
    pg.evaluate("window.scrollTo(0,0)")
    def tap_bar(i):
        pg.evaluate("(i)=>document.querySelector(`[data-bar=\"${i}\"]`).scrollIntoView({block:'center'})", i); pg.wait_for_timeout(250)
        r = pg.evaluate("(i)=>{const r=document.querySelector(`[data-bar=\"${i}\"]`).getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}}", i)
        pg.touchscreen.tap(r["x"] + r["w"]/2, r["y"] + r["h"]/2); pg.wait_for_timeout(350)
    ev = lambda js: pg.evaluate("async()=>{const {S}=await import('/static/state.js'); " + js + "}")
    n0 = ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)")
    pg.get_by_role("button", name="D", exact=True).first.click(); pg.wait_for_timeout(200)
    check("1/2 : touche le début de D" in pg.evaluate("document.body.innerText"), tag + "30: step « 1/2 : touche le début de D »")
    tap_bar(1)
    check(re.search(r"2/2 : début mesure \d+ — touche la fin de D", pg.evaluate("document.body.innerText")) is not None, tag + "30: step « 2/2 : début mesure N — touche la fin »")
    pg.screenshot(path=f"{PRE}_{name}_step2.png")
    tap_bar(2)
    n1 = ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)")
    check(n1 == n0 + 1, tag + f"two taps add one mark ({n0}->{n1})")
    check(pg.locator("[data-sect-undo]").is_enabled(), tag + "« Défaire » enabled after a mark")
    pg.evaluate("window.scrollTo(0,0)"); pg.screenshot(path=f"{PRE}_{name}.png")
    pg.wait_for_timeout(600)   # past the 450 ms ghost-click guard
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(400)
    n2 = ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)")
    check(n2 == n0, tag + f"« Défaire » removes it ({n2})")
    # UX #13c: Défaire undoes only the END tap -> the start stays pending
    check(ev("return S.sectTool.pendingStart") == 1, tag + "13c: after Défaire the start is pending again (bar 1)")
    check("touche la fin" in pg.evaluate("document.body.innerText"), tag + "13c: « touche la fin » shown")
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    check(ev("return S.sectTool.pendingStart") is None and ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)") == n0, tag + "13c: a second Défaire cancels the pending start")
    tap_bar(1); tap_bar(2)
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    # tapping the end again recreates the section from the kept start
    tap_bar(2)
    check(ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)") == n0 + 1, tag + "13c: touching the end again re-draws the section")
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    # half-done tap is undone first
    tap_bar(3)
    check(ev("return S.sectTool.pendingStart") == 3, tag + "pending start set")
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    check(ev("return S.sectTool.pendingStart") is None, tag + "« Défaire » cancels the half-done start")
    # tap targets
    small = pg.evaluate("""()=>{const c=document.querySelector('[data-sect-undo]').closest('div').parentElement;
      return [...c.querySelectorAll('button,summary')].map(b=>{const r=b.getBoundingClientRect();return [b.innerText.trim().slice(0,20),Math.round(r.width),Math.round(r.height)]}).filter(x=>x[2]<44||x[1]<44)}""")
    check(not small, tag + f"all tool buttons >= 44 px (small: {small})")
    tap_bar(1); tap_bar(2)
    pg.wait_for_timeout(600)
    check(ev("return S.sectTool.marks.length") == 1 and ev("return S.sectTool.undo.length") >= 1, tag + "no ghost click: the mark just drawn survives the card growing under the finger")
    pg.locator("summary").first.tap(); pg.wait_for_timeout(300)
    small = pg.evaluate("[...document.querySelectorAll('[data-sect-run]')].map(b=>{const r=b.getBoundingClientRect();return [b.innerText,r.width,r.height]}).filter(x=>x[1]<44||x[2]<44)")
    check(not small, tag + f"segment chips >= 44 px (small: {small})")
    pg.screenshot(path=f"{PRE}_{name}_open.png")
    check(not [u for u in posts if "/section-marks/" not in u], tag + f"only draft posts intercepted ({set(posts)})")
    check(not errs, tag + f"no console errors {errs[:3]}")
    b.close()

with sync_playwright() as p:
    for n in ("chromium", "webkit"): run(p, n)
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
