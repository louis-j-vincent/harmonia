"""Item 1 of docs/ux_todo_2026-10-01.md: in the chord editor, ‹ › go to the
previous / next chord without closing (a running ▶ Original stops), and
« Déverrouiller » returns a kept chord to the model's chord, with « Annuler ».
Chromium AND WebKit, 390x844, touch taps; every POST is intercepted
(nothing reaches state/).

    .venv/bin/python tests/ui/check_editor_nav.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nav"
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
    annots = []
    def route(r):
        u, m = r.request.url, r.request.method
        if m == "POST" and "/api/annotations/" in u:
            annots.append(json.loads(r.request.post_data or "{}")); r.fulfill(status=200, body="{}", content_type="application/json")
        elif m == "POST": r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.tap(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)
    def tap_chord(idx):
        bi = ev("return S.bars.findIndex(b=>b.idxs.includes(a))", idx)
        pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bi); pg.wait_for_timeout(250)
        r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}}", bi)
        k = ev("return S.bars[a[0]].idxs.indexOf(a[1])", [bi, idx]); n = ev("return S.bars[a].idxs.length", bi)
        pg.touchscreen.tap(r["x"] + r["w"]*(k+.5)/n, r["y"] + r["h"]/2); pg.wait_for_timeout(500)
    prev_b = lambda: pg.get_by_role("button", name=re.compile("^Accord précédent"))
    next_b = lambda: pg.get_by_role("button", name=re.compile("^Accord suivant"))
    orig_b = lambda: pg.locator("button", has_text=re.compile("Original|Stop"))
    cur = lambda: ev("return S._editorIdx")
    snack = lambda: pg.get_by_role("button", name=re.compile("^(Annuler|Réessayer)$"))
    unlock_b = lambda: pg.get_by_role("button", name=re.compile("^Déverrouiller"))

    # playable chords in chart order (what ‹ › must walk)
    order = ev("return S.bars.flatMap(b=>b.idxs.slice().sort((x,y)=>(S.chords[x].beat||0)-(S.chords[y].beat||0))).filter(i=>!S.chords[i].removed&&!S.chords[i].nc&&!S.chords[i].confirmed&&S.chords[i].t1>S.chords[i].t0)")
    mid = order[len(order)//2]; i0 = order.index(mid)

    # --- navigation
    tap_chord(mid)
    check(cur() == mid, f"editor opened on chord {mid}")
    check(prev_b().count() == 1 and next_b().count() == 1, "‹ and › are present")
    box = next_b().bounding_box()
    check(box and box["width"] >= 44 and box["height"] >= 44, f"› is a 44px target ({box})")
    pg.screenshot(path=f"{PRE}_{name}_open.png")
    orig_b().tap(); pg.wait_for_timeout(400)
    check(ev("return !!S._span") is True, "▶ Original is playing")
    next_b().tap(); pg.wait_for_timeout(500)
    check(cur() == order[i0+1], f"› goes to the next chord ({cur()} vs {order[i0+1]})")
    check(ev("return !S._span") is True, "› stops the running ▶ Original")
    check(orig_b().first.inner_text().strip().endswith("Original"), "the new chord's button says ▶ Original again")
    pg.screenshot(path=f"{PRE}_{name}_next.png")
    prev_b().tap(); pg.wait_for_timeout(400); prev_b().tap(); pg.wait_for_timeout(500)
    check(cur() == order[i0-1], f"‹ twice goes to the chord before ({cur()} vs {order[i0-1]})")
    check(pg.get_by_role("button", name=re.compile("^Garder ")).count() == 1, "exactly one editor is open (one « Garder » button)")
    check(not annots, "navigating wrote nothing")
    # first chord: ‹ disabled
    ev("const A=await import('/static/screens/annotate.js'); S._closeEditor(); A.openEditor(a,{})", order[0]); pg.wait_for_timeout(400)
    check(prev_b().is_disabled(), "‹ is disabled on the first chord")
    ev("const A=await import('/static/screens/annotate.js'); S._closeEditor(); A.openEditor(a,{})", order[-1]); pg.wait_for_timeout(400)
    check(next_b().is_disabled(), "› is disabled on the last chord")
    ev("S._closeEditor()"); pg.wait_for_timeout(300)

    # --- unlock
    c0 = ev("const c=S.chords[a]; return {root:c.root,q:c.q}", mid)
    tap_chord(mid)
    check(unlock_b().count() == 0, "no « Déverrouiller » on a chord that is not kept")
    pg.get_by_role("button", name=re.compile("^Garder ")).tap(); pg.wait_for_timeout(600)
    check(len(annots) == 1 and len(annots[0]["chords"]) == 1, "keeping saves one chord")
    snack().first.wait_for(timeout=3000); pg.wait_for_timeout(300)
    tap_chord(mid)
    check(unlock_b().count() == 1, "« Déverrouiller » appears on a kept chord")
    pg.screenshot(path=f"{PRE}_{name}_locked.png")
    unlock_b().tap(); pg.wait_for_timeout(1500)
    check(len(annots) == 2 and annots[-1]["chords"] == [], "unlocking rewrites the sidecar without it")
    st = ev("const c=S.chords[a]; return {root:c.root,q:c.q,confirmed:c.confirmed}", mid)
    check(not st["confirmed"] and st["root"] == c0["root"] and st["q"] == c0["q"], f"the chord is back to the model's ({st} vs {c0})")
    check(pg.get_by_text(re.compile("Déverrouillé")).count() == 1 and snack().count() == 1, "« Déverrouillé · Annuler » snackbar")
    pg.screenshot(path=f"{PRE}_{name}_unlocked.png")
    snack().tap(); pg.wait_for_timeout(1500)
    check(ev("return S.chords[a].confirmed", mid) is True, "Annuler: the chord is kept again")
    check(len(annots[-1]["chords"]) == 1, "Annuler rewrites the sidecar with it")
    check(not errs, f"no console errors {errs[:3]}")
    b.close()
sys.exit(1 if fails else 0)
