"""Sections tool, suggestions (Louis, 2026-10-01/02). After the 2nd tap the other
SAME-KEY occurrences of the block appear pale (each one framed as ONE dashed block,
« A ? » on its first bar). While some are pending the card is ONE line
(« A : N proposés [Garder les N] [Ignorer] ») and the letters row / Valider /
Enregistrer / Recommencer are hidden. Tap a pale block = confirm; « Garder les N » =
confirm all (one Défaire step); picking another letter clears. A search with no
same-key hit says « A : aucune reprise trouvée » for ~2 s. /api/section-repeats is
read-only and goes through; every writing POST is intercepted (nothing reaches state/).
  .venv/bin/python tests/ui/check_section_suggestions.py [shots_prefix]
Lazy Song: A on 0-based bars 1..8 -> same-key 9-16, 19-26, 27-34, 41-48, 61-68.
Chasing Pavements: A on 2..9 -> nothing same-key (verse never returns).
"""
import sys
from playwright.sync_api import sync_playwright

PRE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sectsugg"
LAZY = "min_bruno_mars_the_lazy_song_official_music_video"
CHASE = "min_08DjMT-qR9g"
EXPECT = [(9, 16), (19, 26), (27, 34), (41, 48), (61, 68)]
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def session(p, name, file):
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
    pg.goto(f"http://localhost:7772/?open={file}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    pg.get_by_text("···").first.click(); pg.wait_for_timeout(400)
    pg.get_by_text("Redessiner les sections").first.click(); pg.wait_for_timeout(1500)
    ev = lambda js: pg.evaluate("async()=>{const {S}=await import('/static/state.js'); " + js + "}")
    def tap_bar(i):
        pg.evaluate("(i)=>document.querySelector(`[data-bar=\"${i}\"]`).scrollIntoView({block:'end'})", i); pg.wait_for_timeout(250)
        r = pg.evaluate("(i)=>{const r=document.querySelector(`[data-bar=\"${i}\"]`).getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}}", i)
        pg.touchscreen.tap(r["x"] + r["w"]/2, r["y"] + r["h"]/2); pg.wait_for_timeout(350)
    txt = lambda: pg.evaluate("document.body.innerText")
    nocc = lambda: ev("return S.sectTool.marks.reduce((n,m)=>n+m.occurrences.length,0)")
    pale = lambda: ev("return S.sectTool.sugg&&!S.sectTool.sugg.none?S.sectTool.sugg.items.map(x=>[x.b0,x.b1]):null")
    pick = lambda L: pg.evaluate("(L)=>{const c=document.querySelector('[data-sect-undo]').closest('div').parentElement;[...c.querySelectorAll('button')].find(b=>b.innerText.trim()===L).click()}", L)
    return b, pg, errs, posts, ev, tap_bar, txt, nocc, pale, pick

def lazy(p, name):
    tag = f"[{name}/lazy] "
    b, pg, errs, posts, ev, tap_bar, txt, nocc, pale, pick = session(p, name, LAZY)
    check(nocc() > 1, tag + f"song opens with its validated sections ({nocc()} marks)")
    pg.get_by_role("button", name="Recommencer").tap(); pg.wait_for_timeout(400)   # client-side reset; the draft POST is intercepted
    check(nocc() == 0, tag + "after Recommencer: no marks, A = the first occurrence only gets redrawn")
    pick("A") if ev("return S.sectTool.label") != "A" else None
    tap_bar(1); tap_bar(8)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    check(pale() == [list(x) for x in EXPECT], tag + f"pale blocks = same-key only {EXPECT}: {pale()}")
    t = txt()
    check("A : 5 proposés" in t, tag + "one-line banner « A : 5 proposés »")
    check(pg.get_by_role("button", name="Garder les 5").count() == 1 and pg.get_by_role("button", name="Ignorer").count() == 1, tag + "Garder les 5 / Ignorer")
    check("Valider les sections" not in t and "Recommencer" not in t and "Enregistrer comme" not in t and "touche le début" not in t, tag + "letters/Valider/Enregistrer/Recommencer/help hidden while pending")
    check("↗" not in t and pg.locator("[data-sect-pale]").count() >= 5, tag + "« A ? » tags drawn, no ↗")
    h = pg.evaluate("()=>{const e=document.querySelector('[data-sect-sugg]');let w=e;while(w&&getComputedStyle(w).position!=='sticky')w=w.parentElement;return (w||e).getBoundingClientRect().height}")
    check(h <= 844 * 0.2, tag + f"card height {h:.0f}px <= 20% of the screen")
    # each block framed on its own: the first cell of 19-26 has a left+top edge, and 27-34 starts its own frame
    fr = ev("return [26,27].map(b=>{const c=S._cells.find(c=>c.bar===b);return [b,['borderTopStyle','borderBottomStyle','borderLeftStyle','borderRightStyle'].map(k=>getComputedStyle(c.sel)[k]).join(',')]})")
    check(all('dashed' in x[1] for x in fr), tag + f"blocks have their own dashed frame ({fr})")
    pg.wait_for_timeout(900); pg.screenshot(path=f"{PRE}_{name}_lazy_pale.png")
    tap_bar(9)
    check(nocc() == 2 and pale() == [list(x) for x in EXPECT[1:]], tag + "tap on a pale block confirms just it")
    check("A : 4 proposés" in txt(), tag + "banner « A : 4 proposés »")
    pg.screenshot(path=f"{PRE}_{name}_lazy_one_confirmed.png")
    pg.wait_for_timeout(600)
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(400)
    check(nocc() == 1 and pale() == [list(x) for x in EXPECT], tag + "Défaire undoes the confirm, pale blocks back")
    pg.get_by_role("button", name="Garder les 5").tap(); pg.wait_for_timeout(400)
    check(nocc() == 6 and pale() is None, tag + "Garder les 5 confirms all")
    check("Valider les sections" in txt(), tag + "normal card is back when nothing is pending")
    pg.wait_for_timeout(600); pg.screenshot(path=f"{PRE}_{name}_lazy_all_kept.png")
    pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(400)
    check(nocc() == 1 and len(pale() or []) == 5, tag + "one Défaire undoes the whole « Garder les 5 »")
    pg.evaluate("window.scrollTo(0,0)")
    pg.get_by_role("button", name="Ignorer").tap(); pg.wait_for_timeout(300)
    check(pale() is None and nocc() == 1 and "Valider les sections" in txt(), tag + "Ignorer drops proposals, normal card back")
    tap_bar(1); tap_bar(8)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    tap_bar(4)    # not pale: new stroke, pale cleared
    check(pale() is None and ev("return S.sectTool.pendingStart") == 4, tag + "tap outside a pale block = new stroke, pale cleared")
    pg.wait_for_timeout(600); pg.locator("[data-sect-undo]").tap(); pg.wait_for_timeout(300)
    tap_bar(1); tap_bar(8)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.items.length}", timeout=15000)
    check(not [u for u in posts if "/section-marks/" not in u], tag + f"only draft posts intercepted ({set(posts)})")
    check(not errs, tag + f"no console errors {errs[:3]}")
    b.close()

def chase(p, name):
    tag = f"[{name}/chasing] "
    b, pg, errs, posts, ev, tap_bar, txt, nocc, pale, pick = session(p, name, CHASE)
    tap_bar(2); tap_bar(9)
    pg.wait_for_function("async()=>{const {S}=await import('/static/state.js'); return S.sectTool.sugg&&S.sectTool.sugg.none}", timeout=15000)
    check("A : aucune reprise trouvée" in txt() and nocc() == 1, tag + "zero same-key hit -> « A : aucune reprise trouvée »")
    pg.screenshot(path=f"{PRE}_{name}_chasing_none.png")
    pg.wait_for_timeout(2600)
    check("aucune reprise" not in txt() and ev("return S.sectTool.sugg") is None, tag + "message gone after ~2 s, normal state")
    check(not errs, tag + f"no console errors {errs[:3]}")
    b.close()

with sync_playwright() as p:
    for n in ("chromium", "webkit"):
        lazy(p, n); chase(p, n)
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
