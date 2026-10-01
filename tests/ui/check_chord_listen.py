"""Item 2 of the 2026-10-01 UX top-5: the chord editor has ONE small play
button that plays the ORIGINAL recording over that chord's span (t0->t1), only
on demand: no autoplay, no loop, tap again = stop, closing the editor = stop.
Nothing is written (POSTs intercepted).

    .venv/bin/python tests/ui/check_chord_listen.py [file] [screenshot.png]
"""
import sys, re
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    posts = []
    pg.route("**/api/**", lambda r: (posts.append(r.request.url), r.fulfill(status=200, body="{}", content_type="application/json"))
             if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Annotate"), button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    st = lambda: pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); const a=S.audio; return {paused:a.paused, t:a.currentTime, span:!!S._span, playing:S.playing}}""")
    # open the editor of a chord with a span of 1.5-6 s
    idx = pg.evaluate("""async()=>{const {S}=await import('/static/state.js'); return S.chords.findIndex(c=>!c.nc&&c.t1-c.t0>1.5&&c.t1-c.t0<6&&c.t0>6)}""")
    ch = pg.evaluate("""async(i)=>{const {S}=await import('/static/state.js'); const c=S.chords[i]; return {t0:c.t0,t1:c.t1}}""", idx)
    bar = pg.evaluate("""async(i)=>{const {S}=await import('/static/state.js'); return S.bars.findIndex(b=>b.idxs.includes(i))}""", idx)
    pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bar); pg.wait_for_timeout(300)
    r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}}", bar)
    # tap the zone of that chord (find which zone: use idxs order)
    k = pg.evaluate("""async(a)=>{const {S}=await import('/static/state.js'); return S.bars[a[0]].idxs.indexOf(a[1])}""", [bar, idx])
    n = pg.evaluate("""async(bi)=>{const {S}=await import('/static/state.js'); return S.bars[bi].idxs.length}""", bar)
    pg.mouse.click(r["x"] + r["w"]*(k+.5)/n, r["y"] + r["h"]/2); pg.wait_for_timeout(500)
    btn = pg.get_by_role("button", name=re.compile("Écouter l'enregistrement"))
    check(btn.count() == 1, "editor shows ONE play-original button")
    s0 = st(); check(s0["paused"] and not s0["span"], "no autoplay on opening the editor")
    if btn.count() == 1:
        sz = btn.bounding_box(); check(sz["height"] >= 44, f"button >=44px tall ({sz['height']:.0f})")
        btn.click(); pg.wait_for_timeout(600)
        s1 = st(); check(not s1["paused"] and ch["t0"] - .1 <= s1["t"] <= ch["t1"] + .3, f"plays inside the chord span {ch['t0']:.2f}-{ch['t1']:.2f} (t={s1['t']:.2f})")
        check("Stop" in btn.inner_text(), "button turns into Stop while playing")
        pg.screenshot(path=sys.argv[2] if len(sys.argv) > 2 else "/tmp/listen.png")
        btn.click(); pg.wait_for_timeout(300)
        s2 = st(); check(s2["paused"] and not s2["span"], "second tap stops")
        btn.click(); pg.wait_for_timeout((ch["t1"] - ch["t0"]) * 1000 + 1200)
        s3 = st(); check(s3["paused"] and s3["t"] <= ch["t1"] + .4, f"stops by itself at t1 (t={s3['t']:.2f}, t1={ch['t1']:.2f}) - no loop")
        check("Original" in btn.inner_text(), "button back to play after the end")
        btn.click(); pg.wait_for_timeout(400)
        pg.get_by_text(re.compile("^(Cancel|Annuler|Fermer)$")).first.click(); pg.wait_for_timeout(300)
        s4 = st(); check(s4["paused"] and not s4["span"], "closing the editor stops the recording")
    check(not [u for u in posts if "annotations" in u], "nothing saved")
    check(not errs, f"no console errors {errs[:2]}")
    b.close()
sys.exit(1 if fails else 0)
