"""Sections tool, suggestions (Louis, 2026-10-01): after the 2nd tap the other
occurrences of the block appear PALE (dashed, « A ? »); tap one = confirm,
« Garder les N » = confirm all, Défaire steps back one confirm, picking another
letter clears them. /api/section-repeats is read-only and goes through to the
live server; every POST that writes is intercepted (nothing reaches state/).
  .venv/bin/python tests/ui/check_section_suggestions.py [file] [shots_prefix]
Song min_08DjMT-qR9g, A on bars 3-10 (0-based 2..9) -> 14-21, 24-31, 32-39,
44-51, 56-63 (0-based).
"""
import re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sectsugg"
EXPECT = [(14, 21), (24, 31), (32, 39), (44, 51), (56, 63)]
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
        if r.request.method == "POST" and "/section-repeats/" not in r.request.url:
            posts.append(r.request.url)
            r.fulfill(status=200, body='{"ok":true,"written":0}', content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    pg.get_by_text("···").first.click(); pg.wait_for_timeout(400)
    pg.get_by_text("Redessiner les sections").first.click(); pg.wait_for_timeout(1500)
    tag = f"[{name}] "
    ev = lambda js: pg.evaluate("async()=>{const {S}=await import('/static/state.js'); " + js + "}")
    def tap_bar(i):
        pg.evaluate("(i)=>document.querySelector(`[data-bar=\"${i}\"]`).scrollIntoView({block:'end'})", i); pg.wait_for_timeout(250)
        r = pg.evaluate("(i)=>{const r=document.querySelector(`[data-bar=\"${i}\"]`).getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}}", i)
        pg.touchscreen.tap(r["x"] + r["w"]/2, r["y"] + r["h"]/2); pg.wait_for_timeout(350)
    txt = lambda: pg.evaluate("document.body.innerText")
    marks = lambda: ev("return S.sectTool.marks.map(m=>[m.label,m.occurrences.map(o=>[o.b0,o.b1,o.source])])")
    nocc = lambda: ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)")
    pale = lambda: ev("return S.sectTool.sugg?S.sectTool.sugg.items.map(x=>[x.b0,x.b1]):null")
    pick = lambda L: pg.evaluate("(L)=>{const c=document.querySelector('[data-sect-undo]').closest('div').parentElement;[...c.querySelectorAll('button')].find(b=>b.innerText.trim()===L).click()}", L)
    n0 = nocc()
    check(n0 == 0, tag + f"starts with no marks ({n0}) — draft empty for this song")

    check("1/2 : touche le début de A" in txt(), tag + "step text 1/2 kept")
    tap_bar(2)
    check("2/2 : début mesure 3" in txt(), tag + "step text 2/2 kept")
    check(pale() is None, tag + "no proposals before the 2nd tap")
    tap_bar(9)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    p1 = pale()
    check(p1 == [list(x) for x in EXPECT], tag + f"pale zones = bars 14-21,24-31,32-39,44-51,56-63 (0-based): {p1}")
    check(nocc() == 1, tag + "nothing confirmed yet (1 drawn)")
    check(pg.locator("[data-sect-pale]").count() >= 1, tag + "« A ? » tags drawn")
    check("A : 1 tracé · 5 proposés" in txt(), tag + "banner « A : 1 tracé · 5 proposés »")
    check(pg.get_by_role("button", name="Garder les 5").count() == 1 and pg.get_by_role("button", name="Ignorer").count() == 1, tag + "buttons Garder les 5 / Ignorer")
    pg.evaluate("window.scrollTo(0,0)")
    tap_bar(14)
    # tap bar 14 already confirmed above -> state check
    check(nocc() == 2 and pale() == [list(x) for x in EXPECT[1:]], tag + "tap on a pale zone confirms just it")
    check(["A", [2, 9, "humain"]] == marks()[0][:1] + [marks()[0][1][0]] and [14, 21, "humain"] in marks()[0][1], tag + "confirmed = normal placed mark (source humain)")
    check("A : 2 tracés · 4 proposés" in txt(), tag + "banner updates: 2 tracés · 4 proposés")
    pg.screenshot(path=f"{PRE}_{name}_one_confirmed.png")
    # Défaire steps back one confirm
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(400)
    check(nocc() == 1 and pale() == [list(x) for x in EXPECT], tag + "Défaire undoes the confirm; the 5 pale zones are back")
    # Garder les 5
    pg.get_by_role("button", name="Garder les 5").tap(); pg.wait_for_timeout(400)
    check(nocc() == 6 and pale() is None, tag + f"Garder les 5 confirms all ({nocc()} marks), no pale left")
    pg.wait_for_timeout(600)
    pg.evaluate("window.scrollTo(0,0)"); pg.screenshot(path=f"{PRE}_{name}_all_kept.png")
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(400)
    check(nocc() == 1 and len(pale() or []) == 5, tag + "one Défaire undoes the whole « Garder les 5 »")
    # picking another letter clears pale
    pick("B"); pg.wait_for_timeout(300)
    check(pale() is None and nocc() == 1, tag + "picking letter B clears the pale zones, keeps the drawn one")
    # Ignorer, and a new stroke elsewhere clears
    pick("A"); pg.wait_for_timeout(200)
    tap_bar(2); tap_bar(9)   # redraw A over the same bars: replaces itself
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    pg.wait_for_timeout(900)   # the smooth scroll that brings the first pale zone into view
    pg.screenshot(path=f"{PRE}_{name}_pale.png")
    pg.wait_for_timeout(600)
    pg.get_by_role("button", name="Ignorer").tap(); pg.wait_for_timeout(300)
    check(pale() is None and nocc() == 1, tag + "Ignorer drops the proposals, nothing confirmed")
    tap_bar(2); tap_bar(9)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    tap_bar(12)   # a free bar (not pale): starts a new stroke, clears the pale zones
    check(pale() is None and ev("return S.sectTool.pendingStart") == 12, tag + "tap outside a pale zone = new stroke, pale cleared")
    check(not [u for u in posts if "/section-marks/" not in u], tag + f"only draft posts intercepted ({set(posts)})")
    check(not errs, tag + f"no console errors {errs[:3]}")
    b.close()

with sync_playwright() as p:
    for n in ("chromium", "webkit"): run(p, n)
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
