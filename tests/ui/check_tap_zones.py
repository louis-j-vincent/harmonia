"""Item 1 of the 2026-10-01 UX top-5: in Annotate, each chord's own share of
the bar is its tap zone, and a tap opens THAT chord's editor directly (no
intermediate "bar expanded" sheet). Nothing is written (POSTs intercepted).

    .venv/bin/python tests/ui/check_tap_zones.py [file]
"""
import json, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
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
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached")
    pg.locator('button:text-is("Annotate"), button:text-is("Corriger")').first.click()
    pg.wait_for_timeout(800)

    def editor_idx():
        return pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); return S._editorIdx==null?null:S._editorIdx;}""")
    def close():
        pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); if(S._closeEditor) S._closeEditor();}""")
        pg.wait_for_timeout(700)

    info = pg.evaluate("""async()=>{const {S}=await import('/static/state.js');
      const rows=S.bars.map((b,i)=>({i,n:b.idxs.length,idxs:b.idxs}));
      return {one:rows.find(r=>r.n===1&&S.chords[r.idxs[0]]&&!S.chords[r.idxs[0]].nc), two:rows.find(r=>r.n===2), held:rows.find(r=>r.n===0), four:rows.find(r=>r.n>=3)};}""")
    print(info)
    def rect(bar):
        return pg.evaluate("(bi)=>{const c=document.querySelector(`[data-bar=\"${bi}\"]`); c.scrollIntoView({block:'center'}); const r=c.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height};}", bar)

    # 1. a one-chord bar: tap anywhere in it -> editor of that chord, directly
    bi = info["one"]["i"]; r = rect(bi); pg.wait_for_timeout(200); r = rect(bi)
    for frac in (.2, .8):
        pg.mouse.click(r["x"] + r["w"]*frac, r["y"] + r["h"]/2); pg.wait_for_timeout(300)
        check(editor_idx() == info["one"]["idxs"][0], f"1-chord bar {bi}: tap at {frac:.0%} opens its editor directly")
        close()
    # 2. a two-chord bar: left half -> chord 1, right half -> chord 2
    if info["two"]:
        bi = info["two"]["i"]; rect(bi); pg.wait_for_timeout(200); r = rect(bi)
        for k, frac in enumerate((.12, .88)):
            pg.mouse.click(r["x"] + r["w"]*frac, r["y"] + r["h"]/2); pg.wait_for_timeout(300)
            check(editor_idx() == info["two"]["idxs"][k], f"2-chord bar {bi}: tap at {frac:.0%} opens chord #{k+1}")
            close()
    # 2b. a 3-chord bar: each chord's tap zone lands on that chord (beat-proportional)
    if info["four"]:
        bi = info["four"]["i"]; idxs = info["four"]["idxs"]
        beats = pg.evaluate("async(ix)=>{const {S}=await import('/static/state.js'); return ix.map(i=>S.chords[i].beat||0)}", idxs)
        bpb = 4; edges = [0 if k == 0 else beats[k] for k in range(3)] + [bpb]
        rect(bi); pg.wait_for_timeout(200); r = rect(bi)
        for k in range(3):
            mid = (edges[k] + edges[k+1]) / 2 / bpb
            # 3+ accords : mesure serrée automatique (UX #9), les accords sont empilés à
            # gauche — on tape au milieu de chaque glyphe, pas à la position du temps.
            fx = pg.evaluate("""([bi,k])=>{const it=[...document.querySelector(`[data-bar="${bi}"]`).querySelectorAll('[data-fit]')]; if(!it[k]) return null; const q=it[k].getBoundingClientRect(); return q.left+q.width/2;}""", [bi, k])
            pg.mouse.click(fx if fx is not None else r["x"] + r["w"]*mid, r["y"] + r["h"]/2); pg.wait_for_timeout(300)
            check(editor_idx() == idxs[k], f"3-chord bar {bi} (beats {beats}): tap in zone #{k+1} opens chord #{k+1}")
            close()
    # 3. zone height >= 44px
    hs = pg.evaluate("""()=>[...document.querySelectorAll('[data-tapzone]')].map(z=>z.getBoundingClientRect().height)""")
    check(len(hs) > 0 and min(hs) >= 44, f"tap zones exist and are >=44px tall (n={len(hs)}, min={min(hs) if hs else None})")
    # 4. no dotted 'bar expanded' sheet anywhere
    check(pg.get_by_text("filled dot").count() == 0, "no dot-sheet legend")
    check(not [u for u in posts if "annotations" in u], "nothing saved by opening editors")
    check(not errs, f"no console errors {errs[:2]}")
    pg.screenshot(path=sys.argv[2] if len(sys.argv) > 2 else "/tmp/tapzones.png")
    b.close()
sys.exit(1 if fails else 0)
