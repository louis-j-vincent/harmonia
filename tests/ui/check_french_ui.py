"""Item 4 of the 2026-10-01 UX top-5: the chart / annotate / practise / prefs
screens speak French with plain names (dock: Lire / Corriger / S'entraîner;
editor tabs: Roue / Extensions / Propositions / À la main; « Garder Gm » active
as soon as the editor opens on an existing chord). Fails on any listed English
UI string. Nothing is written (POSTs intercepted).

    .venv/bin/python tests/ui/check_french_ui.py [file] [shots_prefix]
"""
import re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fr"
BAD = re.compile(r"\b(Read|Annotate|Practise|Compass|By hand|Lock this|Cancel|Tap a chord|previewing|Split|Preferences|Change key|Close|played \d|candidates|the model|Level \d|loop|bars? \d)\b", re.I)
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    posts = []
    pg.route("**/api/**", lambda r: (posts.append(r.request.url), r.fulfill(status=200, body="{}", content_type="application/json"))
             if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    def english(where):
        txt = pg.evaluate("""()=>{const o=[...document.querySelectorAll('[data-overlay]')].map(e=>e.innerText).join(' | ');
          return document.getElementById('app').innerText+' | '+[...document.querySelectorAll('[aria-label],[title]')].map(e=>(e.getAttribute('aria-label')||'')+' '+(e.title||'')).join(' | ')}""")
        bad = sorted(set(m.group(0) for m in BAD.finditer(txt)))
        check(not bad, f"{where}: no English UI string {bad}")
        return txt
    txt = english("Lire")
    check(all(w in txt for w in ("Lire", "Corriger", "S'entraîner")), "dock tabs are Lire / Corriger / S'entraîner")
    pg.screenshot(path=PRE + "_read.png")
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    english("Corriger")
    pg.screenshot(path=PRE + "_annotate.png")
    # open the editor on a plain model chord
    idx = pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); return S.chords.findIndex(c=>!c.nc&&!c.confirmed&&c.sug&&c.sug.length)}""")
    pg.evaluate("""async(i)=>{const A=await import('/static/screens/annotate.js'); A.openEditor(i)}""", idx); pg.wait_for_timeout(500)
    txt = english("éditeur")
    check(all(w in txt for w in ("Propositions", "À la main", "Plus…")), "editor tabs are Propositions / À la main, Roue and Extensions under « Plus… »")
    keep = pg.get_by_role("button", name=re.compile(r"^Garder "))
    check(keep.count() == 1, "« Garder <accord> » is there")
    check(keep.count() == 1 and keep.evaluate("e=>getComputedStyle(e).pointerEvents") != "none", "« Garder » is active as soon as the editor opens")
    pg.screenshot(path=PRE + "_editor.png")
    for tab in ("Propositions", "À la main", "Extensions"):
        if tab == "Extensions":   # sous « Plus… » depuis b4bb343
            pg.locator("summary", has_text="Plus…").last.click(); pg.wait_for_timeout(250)
        pg.get_by_role("button", name=tab, exact=True).click(); pg.wait_for_timeout(350); english("onglet " + tab)
    pg.screenshot(path=PRE + "_editor_hand.png")
    pg.evaluate("async()=>{const {S}=await import('/static/state.js'); S._closeEditor()}"); pg.wait_for_timeout(400)
    # a NEW chord must not be keepable before a choice
    pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); const A=await import('/static/screens/annotate.js'); A.openNewChord(2,2)}"""); pg.wait_for_timeout(500)
    k = pg.get_by_role("button", name=re.compile(r"^Choisis un accord pour le garder"))
    check(k.count() == 1 and k.evaluate("e=>getComputedStyle(e).pointerEvents") == "none", "a brand-new chord cannot be kept before a choice is made")
    pg.screenshot(path=PRE + "_newchord.png")
    pg.evaluate("async()=>{const {S}=await import('/static/state.js'); S._closeEditor()}"); pg.wait_for_timeout(400)
    # practise + prefs
    pg.locator("button:text-is(\"S'entraîner\")").first.click(); pg.wait_for_timeout(900)
    english("S'entraîner"); pg.screenshot(path=PRE + "_practise.png")
    pg.go_back(); pg.wait_for_timeout(600)
    pg.get_by_role("button", name="Réglages").first.click(); pg.wait_for_timeout(500)
    english("Réglages"); pg.screenshot(path=PRE + "_prefs.png")
    check(not [u for u in posts if "annotations" in u], "nothing saved")
    check(not errs, f"no console errors {errs[:2]}")
    b.close()
sys.exit(1 if fails else 0)
