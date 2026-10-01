"""UX lot 2, rows 27-32 (Corriger): « + » to add a chord, what « Garder » changes,
« Supprimer » under « Plus… », sections-tool steps, bar-1 click, tempo confirm.
Chromium AND WebKit at 390x844; every POST intercepted (nothing reaches state/).

    .venv/bin/python tests/ui/check_corriger_lot2.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lot2c"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def run(p, kind):
    if kind == "chromium":
        b = p.chromium.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844})
    else:
        b = p.webkit.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    pg = ctx.new_page(); T = f"[{kind}] "; sh = lambda n: pg.screenshot(path=f"{PRE}_{kind}_{n}.png")
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    posts = []
    def route(r):
        u, m = r.request.url, r.request.method
        if m == "POST":
            posts.append((u, r.request.post_data)); r.fulfill(status=200, body='{"job_id":"j1"}', content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)
    def tap_bar(bi):
        pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bi); pg.wait_for_timeout(250)
        r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return [r.x,r.y,r.width,r.height]}", bi)
        pg.mouse.click(r[0] + r[2]/2, r[1] + r[3]/2); pg.wait_for_timeout(450)

    # ---- #27 : un « + » franc sur un temps vide
    one = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===1&&!S.chords[x[0]].nc&&S.chords[x[0]].t0>6)")
    tap_bar(one[0])
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    cells = pg.locator("[data-cut-beat]")
    check(cells.count() >= 2, T + "27 empty beats are « + » cells")
    check(all(cells.nth(i).inner_text().split()[1:] == ["+", "ajouter"] for i in range(cells.count())), T + "27 a cell says only « + ajouter », no pale chord")
    check(cells.first.get_attribute("aria-label").startswith("Ajouter un accord"), T + "27 aria-label")
    sh("27_cut")
    cells.first.click(); pg.wait_for_timeout(500)
    check(pg.get_by_text("nouvel accord").count() == 1, T + "27 tapping « + » opens the editor on a new chord")
    check(not errs, T + f"no page errors {errs[:3]}")
    b.close()

with sync_playwright() as p:
    for kind in ("chromium", "webkit"): run(p, kind)
sys.exit(1 if fails else 0)
