"""Item 3 of the 2026-10-01 UX top-5: ONE « Couper la mesure » tool (a beat
ruler: tap the beat where the new chord starts, 3+1 possible) + « Recoller ».
The new chord is born UNLOCKED (nothing is saved until it is kept; cancelling
undoes the cut). Every POST is intercepted: nothing is written to state/.

    .venv/bin/python tests/ui/check_cut_bar.py [file] [screenshot.png]
"""
import json, re, sys
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
    def route(r):
        if r.request.method == "POST":
            if "/api/annotations/" in r.request.url: posts.append(json.loads(r.request.post_data or "{}"))
            r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Annotate"), button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); const A=await import('/static/screens/annotate.js'); " + js + "}", arg)
    def tap_chord(idx):
        bi = ev("return S.bars.findIndex(b=>b.idxs.includes(a))", idx)
        pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bi); pg.wait_for_timeout(250)
        r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}}", bi)
        k = ev("return S.bars[a[0]].idxs.indexOf(a[1])", [bi, idx]); n = ev("return S.bars[a].idxs.length", bi)
        pg.mouse.click(r["x"] + r["w"]*(k+.5)/n, r["y"] + r["h"]/2); pg.wait_for_timeout(450)
        return bi
    def close_sheet():
        pg.get_by_text(re.compile("^(Cancel|Annuler|Fermer)$")).first.click(); pg.wait_for_timeout(450)

    one = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===1&&!S.chords[x[0]].nc&&S.chords[x[0]].t0>6)")
    bi, [idx] = one[0], one[1]
    before = ev("const c=S.chords[a]; return {t0:c.t0,t1:c.t1,n:S.chords.length}", idx)
    bar = ev("const b=S.bars[a]; return {t0:b.t0,t1:b.t1,bpb:S.model.bpb||4}", bi)

    # --- A. cancel path: cut at beat 3, cancel -> nothing changes, nothing posted
    tap_chord(idx)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    cells = pg.locator("[data-cut-beat]")
    check(cells.count() == bar["bpb"] - 1, f"ruler offers {bar['bpb']-1} free beats (all but the chord's own start) - got {cells.count()}")
    check(min(c["height"] for c in [cells.nth(i).bounding_box() for i in range(cells.count())]) >= 44, "ruler cells >=44px tall")
    pg.screenshot(path=(sys.argv[2] if len(sys.argv) > 2 else "/tmp/cut.png").replace(".png", "_ruler.png"))
    pg.locator('[data-cut-beat="3"]').click(); pg.wait_for_timeout(500)
    new = ev("const i=S._editorIdx; const c=S.chords[i]; return {i, beat:c.beat, confirmed:c.confirmed, t0:c.t0, t1:c.t1, host:S.chords[a].t1, n:S.chords.length}", idx)
    dur = (bar["t1"] - bar["t0"]) / bar["bpb"]
    check(new["beat"] == 3 and abs(new["t0"] - (bar["t0"] + 3*dur)) < 1e-6, "3+1 cut: the new chord starts on beat 4 (index 3) of the bar")
    check(not new["confirmed"], "the new chord is NOT locked")
    check(abs(new["host"] - new["t0"]) < 1e-6 and abs(new["t1"] - before["t1"]) < 1e-6, "host shrinks to the cut, new chord runs to the bar end")
    check(pg.get_by_role("button", name=re.compile("Écouter l'enregistrement")).count() == 1, "new chord has its own play-original button")
    check(not posts, "nothing saved yet")
    close_sheet()
    after = ev("const c=S.chords[a]; return {t0:c.t0,t1:c.t1,n:S.chords.length,ids:S.bars[S.bars.findIndex(b=>b.idxs.includes(a))].idxs.length}", idx)
    check(after["t1"] == before["t1"] and after["n"] == before["n"] and after["ids"] == 1, "cancel restores the bar exactly")
    check(not posts, "cancel saved nothing")

    # --- B. keep path: cut at beat 2 (2+2), keep -> saved with beat 2, t0/t1
    tap_chord(idx)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    pg.locator('[data-cut-beat="2"]').click(); pg.wait_for_timeout(500)
    ev("const i=S._editorIdx; A.confirmChord(i, 5, '-7', null)")
    pg.wait_for_timeout(500)
    check(len(posts) == 1, "keeping the new chord saves once")
    sent = [c for c in posts[-1]["chords"] if c["bar"] == ev("return S.chords[a].bar", idx) and c["beat"] == 2]
    check(len(sent) == 1 and abs(sent[0]["t0"] - (bar["t0"] + 2*dur)) < 1e-6 and abs(sent[0]["t1"] - bar["t1"]) < 1e-6,
          f"saved chord = beat 2, t0/t1 of its half ({sent[0] if sent else None})")
    check(posts[-1].get("removed") == [], "no removal recorded for a hand-added chord")

    # --- C. recoller an ADDED chord: it just leaves `chords`, no marker
    tap_chord(idx)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    pg.get_by_role("button", name=re.compile("^Recoller")).click(); pg.wait_for_timeout(400)
    pg.locator("[data-keep-idx]").first.click(); pg.wait_for_timeout(500)   # « garder lequel ? » : le premier
    st = ev("const bi=S.bars.findIndex(b=>b.idxs.includes(a)); return {n:S.bars[bi].idxs.length, t1:S.chords[a].t1}", idx)
    check(st["n"] == 1 and abs(st["t1"] - before["t1"]) < 1e-6, "recoller: one chord again, full length back")
    check(not any(c["beat"] == 2 and c["bar"] == sent[0]["bar"] for c in posts[-1]["chords"]) and posts[-1]["removed"] == [], "recoller of an added chord: gone from chords, no marker")

    # --- D. recoller a MODEL chord -> removal marker
    two = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===2&&!x.some(j=>S.chords[j].added))")
    first, second = two[1]
    c2 = ev("const c=S.chords[a]; return {bar:c.bar, beat:c.beat}", second)
    tap_chord(first)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    pg.screenshot(path=(sys.argv[2] if len(sys.argv) > 2 else "/tmp/cut.png").replace(".png", "_recoller.png"))
    pg.get_by_role("button", name=re.compile("^Recoller")).click(); pg.wait_for_timeout(400)
    pg.locator("[data-keep-idx]").first.click(); pg.wait_for_timeout(500)   # « garder lequel ? » : le premier
    check(posts[-1]["removed"] == [{"bar": c2["bar"], "beat": c2["beat"]}], f"recoller of a model chord records it in `removed` ({posts[-1]['removed']})")
    check(ev("return S.bars[a].idxs.length", two[0]) == 1, "bar shows one chord")
    check(not errs, f"no console errors {errs[:2]}")
    b.close()
sys.exit(1 if fails else 0)
