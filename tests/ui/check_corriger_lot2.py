"""UX lot 2, rows 27-32 (Corriger): « + » to add a chord, what « Garder » changes,
« Supprimer » under « Plus… », sections-tool steps, bar-1 click, tempo confirm.
Chromium AND WebKit at 390x844; every POST intercepted (nothing reaches state/).

    .venv/bin/python tests/ui/check_corriger_lot2.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
FOLDED = "min_0DdCoNbbRvQ"   # a folded chart: A sung 11 times
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
    def reopen(f=FILE):
        posts.clear(); errs.clear()
        pg.goto(f"http://localhost:7772/?open={f}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
        pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    reopen()
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
    reopen(FOLDED)
    # ---- #28 : « Garder corrige les N passages »
    multi = ev("return S.chords.findIndex(c=>!c.nc&&(c.spans||[]).length>1)")
    single = ev("return S.chords.findIndex(c=>!c.nc&&!c.confirmed&&(c.spans||[]).length===1)")
    check(multi >= 0, T + "28 the chart has a folded chord (spans>1)")
    if multi >= 0:
        n = ev("return S.chords[a].spans.length", multi)
        ev("const E=await import('/static/screens/annotate.js'); E.openEditor(a)", multi); pg.wait_for_timeout(500)
        t = pg.get_by_text(re.compile("^Garder corrige les ")).first.inner_text()
        check(t.startswith(f"Garder corrige les {n} passages (0:") or re.match(rf"Garder corrige les {n} passages \(\d+:\d\d", t), T + f"28 text names the {n} passages: {t}")
        sh("28_multi")
        ev("S._closeEditor()")
    ev("const E=await import('/static/screens/annotate.js'); E.openEditor(a)", single); pg.wait_for_timeout(500)
    check(pg.get_by_text(re.compile("^Garder corrige")).count() == 0, T + "28 a one-pass chord says nothing about passes")
    ev("S._closeEditor()")
    # ---- #29 : « Supprimer ce morceau » sous « Plus… »
    reopen()
    pg.get_by_role("button", name="Outils").click(); pg.wait_for_timeout(400)
    sup = pg.get_by_text("Supprimer ce morceau")
    check(not sup.first.is_visible(), T + "29 « Supprimer ce morceau » is hidden until « Plus… » is opened")
    sh("29_outils")
    pg.locator("summary", has_text="Plus d'outils").click(); pg.wait_for_timeout(300)
    check(sup.first.is_visible(), T + "29 it appears under « Plus… »")
    sh("29_outils_plus")
    pg.get_by_role("button", name="Fermer").last.click()
    # ---- #31 : Mesure 1 — « Écouter un clic sur ce temps »
    reopen()
    pg.add_init_script("window.__osc=0; for(const C of [window.AudioContext, window.webkitAudioContext]){ if(C&&!C.__w){ const o=C.prototype.createOscillator; C.prototype.createOscillator=function(){ window.__osc++; return o.apply(this,arguments); }; C.__w=1; } }")
    pg.reload(); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    pg.get_by_role("button", name="Outils").click(); pg.wait_for_timeout(400)
    pg.get_by_text("Caler la mesure 1").first.click(); pg.wait_for_timeout(900)
    ev("S.audio.currentTime=10"); pg.wait_for_timeout(300)
    check(ev("return !S._span && S.audio.paused"), T + "31 nothing autoplays when the sheet opens")
    clk = pg.get_by_role("button", name="Écouter un clic sur ce temps")
    check(clk.count() == 1, T + "31 the button exists")
    sh("31_bar1")
    ref = ev("return S.audio.currentTime")
    clk.click(); pg.wait_for_timeout(700)
    check(ev("return !!S._span"), T + "31 tapping plays a short span of the music")
    check(pg.get_by_role("button", name="■ Arrêter").count() == 1, T + "31 the button turns into Arrêter")
    pg.wait_for_timeout(2600)
    check(ev("return !S._span") and ev("return window.__osc") == 1, T + f"31 exactly one tick was fired ({ev('return window.__osc')}), span over")
    check(abs(ev("return S.audio.currentTime") - ref) < 0.3, T + "31 the marker is back where it was")
    check(len([x for x in posts if "/api/bar1/" in x[0]]) == 0, T + "31 listening posts nothing")
    pg.get_by_role("button", name="Annuler").click(); pg.wait_for_timeout(300)
    check(not errs, T + f"no page errors {errs[:3]}")
    b.close()

with sync_playwright() as p:
    for kind in ("chromium", "webkit"): run(p, kind)
sys.exit(1 if fails else 0)
