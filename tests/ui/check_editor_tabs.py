"""Item 2 of docs/ux_todo_2026-10-01.md: the chord editor shows 2 tabs,
« Propositions » (default) and « À la main »; « Roue » and « Extensions » live
under a « Plus… » <details>. Chromium AND WebKit, 390x844, touch; POSTs are
intercepted (nothing reaches state/).

    .venv/bin/python tests/ui/check_editor_tabs.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tabs"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
  for name in ("chromium", "webkit"):
    print("==", name)
    b = getattr(p, name).launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    posts = []
    def route(r):
        if r.request.method == "POST":
            posts.append(r.request.url); r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.tap(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)
    def open_ed(idx):
        ev("const A=await import('/static/screens/annotate.js'); if(S._closeEditor) S._closeEditor(); A.openEditor(a,{})", idx); pg.wait_for_timeout(450)
    # a chord with model proposals, one without
    withsug = ev("const A=await import('/static/screens/annotate.js'); return S.chords.findIndex((c,i)=>!c.nc&&!c.removed&&!c.confirmed&&A.candList(i).length>0)")
    vis_tabs = lambda: [t for t in ("Roue", "Extensions", "Propositions", "À la main") if pg.get_by_role("button", name=t, exact=True).is_visible()]
    open_ed(withsug)
    check(vis_tabs() == ["Propositions", "À la main"], f"2 visible tabs: {vis_tabs()}")
    plus = pg.locator("details", has=pg.locator("summary", has_text="Plus"))
    check(plus.count() == 1 and plus.evaluate("d=>!d.open"), "« Plus… » is a closed <details>")
    guide_on = pg.evaluate("""()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Propositions'); return getComputedStyle(b).boxShadow!=='none'}""")
    check(guide_on, "default tab is Propositions (selected)")
    pg.screenshot(path=f"{PRE}_{name}_default.png")
    plus.locator("summary").tap(); pg.wait_for_timeout(300)
    check(plus.evaluate("d=>d.open") and pg.get_by_role("button", name="Roue", exact=True).is_visible()
          and pg.get_by_role("button", name="Extensions", exact=True).is_visible(), "« Plus… » reveals Roue and Extensions")
    pg.screenshot(path=f"{PRE}_{name}_plus.png")
    pg.get_by_role("button", name="Roue", exact=True).tap(); pg.wait_for_timeout(400)
    check(pg.get_by_text(re.compile("Touche un accord pour l'entendre")).count() >= 1 and ev("return !!document.querySelector('svg')"), "Roue still works")
    pg.screenshot(path=f"{PRE}_{name}_roue.png")
    pg.get_by_role("button", name="Extensions", exact=True).tap(); pg.wait_for_timeout(400)
    pg.screenshot(path=f"{PRE}_{name}_ext.png")
    pg.get_by_role("button", name="À la main", exact=True).tap(); pg.wait_for_timeout(400)
    pg.screenshot(path=f"{PRE}_{name}_hand.png")
    pg.get_by_role("button", name="Propositions", exact=True).tap(); pg.wait_for_timeout(400)
    # reopening on another chord goes back to Propositions even after Roue was used
    pg.get_by_role("button", name="Roue", exact=True).tap() if pg.get_by_role("button", name="Roue", exact=True).is_visible() else None
    open_ed(withsug)
    guide_on = pg.evaluate("""()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Propositions'); return getComputedStyle(b).boxShadow!=='none'}""")
    check(guide_on, "a re-opened editor is back on Propositions")
    # no proposals -> opens straight on À la main
    nosug = ev("const A=await import('/static/screens/annotate.js'); return S.chords.findIndex((c,i)=>!c.nc&&!c.removed&&A.candList(i).length===0)")
    if nosug >= 0:
        open_ed(nosug)
        on_hand = pg.evaluate("""()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='À la main'); return getComputedStyle(b).boxShadow!=='none'}""")
        check(on_hand, "a chord with no model proposals opens on À la main")
    check(not posts, "nothing was written")
    check(not errs, f"no console errors {errs[:3]}")
    b.close()
sys.exit(1 if fails else 0)
