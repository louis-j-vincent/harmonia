"""UX Lot 2 rows #33-#38 (S'entraîner). Chromium AND WebKit, 390x844, touch, POSTs intercepted.
    .venv/bin/python tests/ui/check_practise_lot2.py [file]
#33 drag on the bar band chooses loop A/B (engine a,b; audio stays inside; tap still seeks)."""
import sys
from playwright.sync_api import sync_playwright
FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
fails = []
def SE(pg, expr): return pg.evaluate("async()=>{const {S}=await import('/static/state.js'); return "+expr+"}")
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)
with sync_playwright() as p:
  for name in ("chromium", "webkit"):
    tag = name + ": "
    b = getattr(p, name).launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    ctx.add_init_script("localStorage.setItem('harmPianoOffByDefault','1')")
    pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json") if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.evaluate("localStorage.removeItem('harmLoopBars')")
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1500)
    eng = lambda: pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); const e=S._loopEngine, g=S.model.barGrid;
      if(!e) return null; return {a:e.a,b:e.b,i:g.findIndex(x=>Math.abs(x-e.a)<1e-6),j:g.findIndex(x=>Math.abs(x-e.b)<1e-6),t:e.active.currentTime}}""")
    chips = pg.evaluate("async()=>{const {S}=await import('/static/state.js'); return S._formChips.map(c=>{const r=c.el.getBoundingClientRect(); return [r.left,r.right,r.top+r.height/2,c.t0,c.t1]})}")
    y = chips[0][2]; xa = chips[0][0] + 3; xb = chips[-1][1] - 3
    # ---- #33 drag
    pg.mouse.move(xa, y); pg.mouse.down(); pg.mouse.move((xa+xb)/2, y, steps=6); pg.mouse.move(xb, y, steps=6); pg.mouse.up()
    pg.wait_for_timeout(500)
    e1 = eng()
    check(e1 is not None and e1["j"] > e1["i"] >= 0, tag + f"drag across the whole band arms a loop {e1}")
    hi = pg.evaluate("()=>{const d=document.querySelector('[data-role=loop-range]'); const r=d.getBoundingClientRect(); return {disp:getComputedStyle(d).display,w:r.width}}")
    check(hi["disp"] == "block" and hi["w"] > 100, tag + f"range drawn on the band {hi}")
    # drag a short span in the middle of the first segment (right to left = same loop either way)
    c0 = max(chips, key=lambda c: c[1]-c[0]); x1 = c0[0] + (c0[1]-c0[0]) * 0.55; x0 = c0[0] + (c0[1]-c0[0]) * 0.2
    pg.mouse.move(x1, y); pg.mouse.down(); pg.mouse.move((x0+x1)/2, y, steps=4); pg.mouse.move(x0, y, steps=4); pg.mouse.up()
    pg.wait_for_timeout(500)
    pg.screenshot(path=f"/tmp/lot2_drag_loop_{name}.png")
    e2 = eng(); g = SE(pg,'S.model.barGrid')
    check(e2 is not None and e2["j"] > e2["i"], tag + f"short right-to-left drag gives a loop {e2}")
    if e2:
        tA = c0[3] + (x0-c0[0])/(c0[1]-c0[0])*(c0[4]-c0[3]); tB = c0[3] + (x1-c0[0])/(c0[1]-c0[0])*(c0[4]-c0[3])
        check(e2["a"] <= tA + 0.02 and e2["b"] >= tB - 0.02 and e2["b"] - e2["a"] < (tB-tA) + 2*(g[2]-g[1]) + 0.5, tag + f"loop covers the dragged span, whole bars (a={e2['a']:.2f} b={e2['b']:.2f} span {tA:.2f}-{tB:.2f})")
        # audio stays inside the band over time (loop wraps)
        ts = []
        pg.wait_for_timeout(300)
        for _ in range(8):
            ts.append(eng()["t"]); pg.wait_for_timeout(450)
        pl = SE(pg,'S.playing')
        check(all(e2["a"] - 0.3 <= t <= e2["b"] + 0.3 for t in ts), tag + f"audio clock stays inside [a,b] ({min(ts):.2f}..{max(ts):.2f}, playing={pl})")
    # ---- tap still seeks (a loop-free state)
    pg.evaluate("async()=>{const {loopEngineDispose}=await import('/static/audio.js'); loopEngineDispose(false);}")
    pg.reload(); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1200)
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1500)
    chips = pg.evaluate("async()=>{const {S}=await import('/static/state.js'); return S._formChips.map(c=>{const r=c.el.getBoundingClientRect(); return [r.left,r.right,r.top+r.height/2,c.t0,c.t1]})}")
    last = chips[-1]; SE(pg,'(S.audio.currentTime=100)'); pg.wait_for_timeout(300); t0 = SE(pg,'S.audio.currentTime')
    pg.touchscreen.tap((last[0]+last[1])/2, last[2]); pg.wait_for_timeout(500)
    t1 = SE(pg,'S.audio.currentTime'); lp = eng()
    check(abs(t1 - last[3]) < 0.6 and t0 > 90 and lp is None, tag + f"plain tap on the last segment still seeks to its start ({t0:.1f}s -> {t1:.1f}s, segment starts {last[3]:.1f}s), no loop armed")
    pg.screenshot(path=f"/tmp/lot2_drag_{name}.png")
    check(not errs, tag + f"no page error {errs}")
    b.close()
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
