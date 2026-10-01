"""UX Lot 2 rows #33-#38 (S'entraîner). Chromium AND WebKit, 390x844, touch, POSTs intercepted.
    .venv/bin/python tests/ui/check_practise_lot2.py [file]
#34 speeds 0,5 0,6 0,75 0,9 1 (comma; audio.playbackRate follows).
#35 tapping « mesure N · v× » opens « Aller à la mesure… » and seeks the audio to that bar.
#36 help line under Piano + 🎹 button toggling the keyboard card.
#37 « Tic à chaque accord » rename + count-in (2 bars of ticks, then audio starts; default Non; tap cancels).
#38 the note under the big chord is short and French (all degrees x qualities, and on screen).
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
    ctx.add_init_script("""(()=>{ window.__sq=[]; const st=OscillatorNode.prototype.start;
      OscillatorNode.prototype.start=function(t){ if(this.type==='square') window.__sq.push([t, performance.now()]); return st.apply(this,arguments); }; })()""")
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
    # ---- #38 short French role note
    res = pg.evaluate("""async()=>{ const {roleCourt}=await import('/static/screens/practise.js'); const {roleOf}=await import('/static/ui/kit.js');
      const out=new Set(); for(const q of ['','-','7','-7','h7','o','sus','+','^7']) for(let r=0;r<12;r++) for(const nx of [null,{root:(r+5)%12,q:'7'}]) out.add(roleCourt(roleOf(0,r,q,nx).text)); return [...out]; }""")
    bad = [x for x in res if len(x) > 34 or "(" in x or "," in x or " — " in x]
    check(len(res) > 15 and not bad, tag + f"{len(res)} distinct role notes, all <= 34 chars, no clause/parenthesis; longest: {max(res, key=len)!r}")
    shown = pg.evaluate("()=>[...document.querySelectorAll('div')].map(d=>d.textContent).filter(t=>/^(I|II|III|IV|V|VI|VII|i|ii|iii|iv|v|vi|vii|♭\\w+|V⁷[^ ]*)( de \\w+)? : /.test(t))")
    check(shown and all(len(x) <= 34 for x in shown), tag + f"on screen: {shown[:2]}")
    # ---- #37 count-in
    pg.locator('button[aria-label="Réglages"]').first.tap(); pg.wait_for_timeout(500)
    sh = pg.locator("[data-overlay]").inner_text()
    check("Tic à chaque accord" in sh and "Clic sur les accords" not in sh, tag + "setting renamed « Tic à chaque accord »")
    check("Décompte" in sh and "2 mesures" in sh, tag + "« Décompte : Non / 2 mesures » offered")
    check(SE(pg,'S.countIn') is False or SE(pg,'S.countIn') == False, tag + "default is Non")
    pg.screenshot(path=f"/tmp/lot2_countin_sheet_{name}.png")
    pg.reload(); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1200)
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1200)
    g = SE(pg,'S.model.barGrid'); bpb = SE(pg,'S.model.bpb') or 4
    play = lambda: pg.locator('button[title^="Lecture / pause"]')
    def audio(): return SE(pg,'S.audio.currentTime')
    def run_play(tag2, secs, cancel_at=None):
        SE(pg,'(S.audio.pause(), S.playing=false, S.audio.currentTime=11.7)'); pg.evaluate("window.__sq.length=0"); pg.wait_for_timeout(300)
        play().tap(); trace = []
        for i in range(int(secs*4)):
            pg.wait_for_timeout(250)
            trace.append((i*0.25+0.25, audio(), SE(pg,'S.playing'), SE(pg,'S.audio.paused')))
            if cancel_at is not None and abs(trace[-1][0]-cancel_at) < 0.13: play().tap()
        return trace
    # default Non: plays immediately, no ticks
    tr = run_play("off", 2.5)
    check(tr[-1][1] > 11.7 + 1.0 and not tr[-1][3], tag + f"default Non: audio starts at once ({tr[-1][1]:.2f}s)")
    check(pg.evaluate("window.__sq.length") == 0, tag + "default Non: no count-in tick")
    SE(pg,'(S.audio.pause(), S.playing=false)')
    # turn it on through the sheet
    pg.locator('button[aria-label="Réglages"]').first.tap(); pg.wait_for_timeout(500)
    pg.locator("[data-overlay]").get_by_role("button", name="2 mesures", exact=True).tap(); pg.wait_for_timeout(300)
    check(SE(pg,'S.countIn') is True and pg.evaluate("localStorage.getItem('harmCountIn')") == "1", tag + "Décompte 2 mesures saved")
    pg.evaluate("document.querySelector('[data-overlay]')&&document.querySelector('[data-overlay]').remove()")
    bi = next(i for i in range(len(g)-1) if g[i+1] > 11.7 + 0.18)   # bar under the playhead (+lead)
    bar = g[bi+1] - g[bi]; D = 2 * bar
    tr = run_play("on", D + 3.5)
    start = 11.7
    moved = next((t for t, a, pl, pa in tr if a > start + 0.3), None)
    sq = pg.evaluate("window.__sq")
    ts = [x[0] for x in sq]; gaps = [b - a for a, b in zip(ts, ts[1:])]
    check(len(sq) == 2 * bpb, tag + f"{len(sq)} ticks = 2 bars x {bpb} beats")
    check(gaps and all(abs(x - bar/bpb) < 0.02 for x in gaps), tag + f"tick spacing {sum(gaps)/max(1,len(gaps)):.3f}s = beat {bar/bpb:.3f}s")
    check(all(abs(a - start) < 0.05 for t, a, pl, pa in tr if t < D - 0.6), tag + "audio does NOT move during the count-in")
    check(moved is not None and abs(moved - D) < 0.9, tag + f"audio starts after ~2 bars (moved at t={moved}, 2 bars = {D:.2f}s)")
    check(tr[-1][1] > start + 1.5 and tr[-1][2], tag + f"then plays normally ({tr[-1][1]:.2f}s, playing={tr[-1][2]})")
    pg.screenshot(path=f"/tmp/lot2_countin_{name}.png")
    SE(pg,'(S.audio.pause(), S.playing=false)')
    # tap during the count-in cancels it
    tr = run_play("cancel", D + 2.0, cancel_at=1.0)
    check(all(abs(a - start) < 0.05 for t, a, pl, pa in tr), tag + "second tap during the count-in cancels: audio never starts")
    pg.evaluate("localStorage.removeItem('harmCountIn')")
    # ---- #36 piano button + help
    card = lambda: pg.evaluate("()=>!![...document.querySelectorAll('div')].find(d=>d.textContent.startsWith('piano : gauche'))")
    pb = pg.locator('[data-role=piano-toggle]')
    check(pb.count() == 1 and pb.bounding_box()["width"] >= 44, tag + "🎹 button on the screen, >= 44 px")
    check(SE(pg,'S.piano') == "never" and not card(), tag + "starts at Jamais, no keyboard card")
    pb.tap(); pg.wait_for_timeout(1000)
    check(SE(pg,'S.piano') == "tap" and card(), tag + f"🎹 -> Au tap, keyboard card appears (S.piano={SE(pg,'S.piano')})")
    pg.screenshot(path=f"/tmp/lot2_piano_on_{name}.png")
    pg.locator('[data-role=piano-toggle]').tap(); pg.wait_for_timeout(1000)
    check(SE(pg,'S.piano') == "never" and not card(), tag + "🎹 again -> Jamais, card gone")
    pg.locator('button[aria-label="Réglages"]').first.tap(); pg.wait_for_timeout(500)
    ph = pg.locator('[data-role=piano-help]'); txt = ph.inner_text() if ph.count() else ""
    check("Jamais" in txt and "Au tap" in txt and "Suit" in txt and "clavier" in txt, tag + f"help line under Piano explains the three choices")
    pg.screenshot(path=f"/tmp/lot2_piano_help_{name}.png")
    pg.keyboard.press("Escape"); pg.evaluate("localStorage.removeItem('harmPiano');localStorage.removeItem('harmCoach');localStorage.removeItem('harmPianoLast')")
    pg.reload(); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1200)
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1200)
    # ---- #35 go to bar
    g = SE(pg,'S.model.barGrid'); SE(pg,'(S.audio.currentTime=3)')
    sub = pg.locator('[data-where]').first
    bx = sub.bounding_box(); check(bx["height"] >= 44, tag + f"subtitle tap area {bx['width']:.0f}x{bx['height']:.0f} >= 44 high")
    sub.tap(); pg.wait_for_timeout(500)
    check(pg.locator("[data-overlay] input").count() == 1 and "Aller à la mesure" in pg.locator("[data-overlay]").inner_text(), tag + "tap opens « Aller à la mesure… »")
    pg.screenshot(path=f"/tmp/lot2_goto_{name}.png")
    pg.locator("[data-overlay] input").fill("12"); pg.locator("[data-overlay] button", has_text="Aller").tap(); pg.wait_for_timeout(700)
    t = SE(pg,'S.audio.currentTime')
    check(abs(t - g[11]) < 0.5, tag + f"audio went to bar 12 ({t:.2f}s, grid says {g[11]:.2f}s)")
    check("mesure 12" in pg.locator("header, body").first.inner_text(), tag + "subtitle now says mesure 12")
    check(pg.locator("[data-overlay]").count() == 0, tag + "sheet closed")
    # ---- #34 speeds
    sp = pg.locator('button[title^="Vitesse"]'); seen = []
    for _ in range(6):
        sp.click(); pg.wait_for_timeout(120)
        seen.append((sp.inner_text().strip(), SE(pg,'S.playRate'), SE(pg,'S.audio.playbackRate')))
    labs = [x[0] for x in seen]
    check(labs[:5] == ["0,5×", "0,6×", "0,75×", "0,9×", "1×"] or labs[1:6] == ["0,5×", "0,6×", "0,75×", "0,9×", "1×"], tag + f"chip cycles 0,5 0,6 0,75 0,9 1 with commas {labs}")
    check(all(abs(r - a) < 1e-6 for _, r, a in seen), tag + "audio.playbackRate follows S.playRate at every step")
    check("0.5" not in pg.locator("body").inner_text() and "0.75" not in pg.locator("body").inner_text(), tag + "no decimal point shown on the screen")
    sp.click(); pg.wait_for_timeout(100)   # leave at a non-1 speed for the sheet check
    pg.locator('button[aria-label="Réglages"]').first.click(); pg.wait_for_timeout(500)
    sheet = pg.evaluate("()=>{const o=document.querySelector('[data-overlay]'); return o? o.innerText:''}")
    check("0,6×" in sheet and "0,9×" in sheet and "0.6" not in sheet, tag + "Réglages sheet offers the five speeds with commas")
    pg.screenshot(path=f"/tmp/lot2_speed_sheet_{name}.png")
    pg.keyboard.press("Escape"); pg.reload(); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1200)
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1200)
    check(not errs, tag + f"no page error {errs}")
    b.close()
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
