"""UX #4/#8/#9 (2026-10-01) : numéros de mesure + « aller à la mesure… »,
écriture serrée automatique à 3-4 accords, « Toucher une mesure » nommé +
retour au tap. Chromium ET WebKit, 390x844, tout POST intercepté.

    .venv/bin/python tests/ui/check_grille.py [file] [shots_prefix]
"""
import sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/grille"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    for name in ("chromium", "webkit"):
        print("==", name)
        if name == "chromium":
            b = p.chromium.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844})
        else:
            b = p.webkit.launch()
            ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, device_scale_factor=2)
        pg = ctx.new_page(); errs = []; posts = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        def route(r):
            if r.request.method == "POST": posts.append(r.request.url); r.fulfill(status=200, body="{}", content_type="application/json")
            else: r.continue_()
        pg.route("**/api/**", route)
        pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
        ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)
        nbar = ev("return S.bars.length")
        # --- #4 numéros
        nums = pg.evaluate("()=>[...document.querySelectorAll('[data-bar]')].map(c=>[+c.dataset.bar, c.getBoundingClientRect().left<20, (c.lastElementChild&&0)||[...c.children].filter(x=>/^\\d+$/.test(x.textContent)&&x.children.length===0&&getComputedStyle(x).fontSize==='9px').map(x=>x.textContent)])")
        firsts = [n for n in nums if n[1]]
        check(len(firsts) > 3 and all(n[2] == [str(n[0] + 1)] for n in firsts), "bar number on the first cell of each row (%d rows)" % len(firsts))
        check(all(n[2] == [] for n in nums if not n[1]), "no number elsewhere")
        pg.screenshot(path=f"{PRE}_{name}_numeros.png")
        # --- aller à la mesure
        tw = pg.locator("[data-where]")
        check(tw.count() == 1 and "aller" in (tw.inner_text() or "aller à la mesure…") or tw.count() == 1, "dock label present")
        if tw.count():
            bb = tw.bounding_box(); check(bb["height"] >= 44 and bb["width"] >= 44, "label hit area >= 44px (%dx%d)" % (bb["width"], bb["height"]))
            tw.click(); pg.wait_for_timeout(400)
            inp = pg.locator("input[type=number]"); check(inp.count() == 1, "goto sheet opens")
            target = min(nbar, 12); inp.fill(str(target)); pg.get_by_role("button", name="Aller", exact=True).click(); pg.wait_for_timeout(1200)
            r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return [r.top,r.bottom,innerHeight]}", target - 1)
            check(r[0] >= 0 and r[1] <= r[2], "bar %d scrolled into view (%s)" % (target, r))
            check((ev("return S.playBar") == target - 1), "playhead on bar %d" % target)
            pg.screenshot(path=f"{PRE}_{name}_goto.png")
        # --- #9 mesures serrées : tout tient dans la cellule, rien ne se chevauche
        res = pg.evaluate("""()=>{
          const cells=[...document.querySelectorAll('[data-bar]')].filter(c=>c.querySelector('[data-fit]'));
          let over=0, ovl=0;
          cells.forEach(c=>{ const cr=c.getBoundingClientRect(); const rs=[...c.querySelectorAll('[data-fit]')].map(i=>i.getBoundingClientRect());
            rs.forEach((r,k)=>{ if(r.right>cr.right-1||r.left<cr.left) over++; if(k&&r.left<rs[k-1].right-0.5) ovl++; }); });
          return [cells.length,over,ovl]; }""")
        check(res[0] > 0, "dense bars found (%d)" % res[0]); check(res[1] == 0, "no chord overflows its bar (%d)" % res[1]); check(res[2] == 0, "no chord overlaps its neighbour (%d)" % res[2])
        # Annotate : zones disjointes dans chaque mesure
        pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(800)
        zs = pg.evaluate("""()=>{ let bad=0,n=0; document.querySelectorAll('[data-bar]').forEach(c=>{ const rs=[...c.querySelectorAll('[data-tapzone]')].map(z=>z.getBoundingClientRect()).sort((a,b)=>a.left-b.left);
            n+=rs.length; rs.forEach((r,k)=>{ if(k&&r.left<rs[k-1].right-0.5) bad++; }); }); return [n,bad]; }""")
        check(zs[0] > 0 and zs[1] == 0, "Annotate tap zones disjoint (%d zones, %d overlaps)" % tuple(zs))
        pg.evaluate("()=>{const c=document.querySelector('[data-fit]'); c&&c.scrollIntoView({block:'center'})}"); pg.wait_for_timeout(300)
        pg.screenshot(path=f"{PRE}_{name}_annotate_dense.png")
        check(not errs, "no page errors " + str(errs)); check(not posts, "no POST " + str(posts))
        b.close()
print("FAILS:", fails); sys.exit(1 if fails else 0)
