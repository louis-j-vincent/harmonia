"""UX #12 (b) : « Recoller » demande « garder lequel ? » quand la mesure a >=2
accords, et la mesure garde l'accord choisi (pas seulement le premier).
(a) : « Annuler » renvoie les `ledger_ids` de la sauvegarde d'origine dans
`annule`. Toutes les requêtes POST sont interceptées : rien n'écrit dans state/.
Chromium ET WebKit mobile, 390x844.

    .venv/bin/python tests/ui/check_recoller.py [file]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def run(p, name):
    print("==", name)
    if name == "webkit":
        b = p.webkit.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    else:
        b = p.chromium.launch(); ctx = b.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    posts = []
    def route(r):
        if r.request.method == "POST" and "/api/annotations/" in r.request.url:
            posts.append(json.loads(r.request.post_data or "{}"))
            r.fulfill(status=200, body=json.dumps({"ledger_ids": ["L%d" % len(posts)]}), content_type="application/json")
        elif r.request.method == "POST": r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); const A=await import('/static/screens/annotate.js'); " + js + "}", arg)
    def tap_chord(idx):
        bi = ev("return S.bars.findIndex(b=>b.idxs.includes(a))", idx)
        pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bi); pg.wait_for_timeout(250)
        r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}}", bi)
        k = ev("return S.bars[a[0]].idxs.indexOf(a[1])", [bi, idx]); n = ev("return S.bars[a].idxs.length", bi)
        x, y = r["x"] + r["w"]*(k+.5)/n, r["y"] + r["h"]/2
        (pg.touchscreen.tap(x, y) if name == "webkit" else pg.mouse.click(x, y)); pg.wait_for_timeout(450)
    two = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===2&&!x.some(j=>S.chords[j].added||S.chords[j].confirmed||S.chords[j].nc))")
    check(two is not None, "a 2-chord model bar exists")
    bi, (first, second) = two
    c1 = ev("const c=S.chords[a]; return {root:c.root,q:c.q,bar:c.bar,beat:c.beat,t0:c.t0,t1:c.t1}", first)
    c2 = ev("const c=S.chords[a]; return {root:c.root,q:c.q,bar:c.bar,beat:c.beat,t0:c.t0,t1:c.t1}", second)
    tap_chord(first)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    pg.get_by_role("button", name=re.compile("^Recoller")).click(); pg.wait_for_timeout(400)
    pg.screenshot(path=f"/tmp/recoller_{name}.png")
    check(pg.get_by_text("garder lequel").count() == 1, "chooser « garder lequel ? » shown")
    n = pg.locator("[data-keep-idx]").count()
    check(n == 2, f"one button per chord ({n})")
    h = pg.locator("[data-keep-idx]").nth(1).bounding_box()["height"]
    check(h >= 44, f"button tall enough ({h})")
    pg.locator("[data-keep-idx]").nth(1).click(); pg.wait_for_timeout(600)
    st = ev("const x=S.chords[a[0]]; return {n:S.bars[a[2]].idxs.length, root:x.root, q:x.q, conf:x.confirmed, t0:x.t0, t1:x.t1}", [first, second, bi])
    check(st["n"] == 1 and st["root"] == c2["root"] and st["q"] == c2["q"], f"bar keeps the SECOND chord ({st} vs {c2})")
    check(abs(st["t0"] - c1["t0"]) < 1e-6 and abs(st["t1"] - c2["t1"]) < 1e-6, "kept chord spans the whole bar")
    last = posts[-1]
    sent = [c for c in last["chords"] if c["bar"] == c1["bar"] and c["beat"] == c1["beat"]]
    check(len(sent) == 1 and sent[0]["root"] == c2["root"] and sent[0]["q"] == c2["q"], "sidecar locks the kept chord at the bar's first slot")
    check({"bar": c2["bar"], "beat": c2["beat"]} in last["removed"], "second slot recorded in `removed`")
    # undo: sends the ids of the ORIGINAL save back
    np_ = len(posts)
    pg.get_by_role("button", name="Annuler").click(); pg.wait_for_timeout(700)
    check(len(posts) == np_ + 1 and posts[-1].get("annule") == ["L%d" % np_], f"undo sends annule={posts[-1].get('annule')}")
    check(ev("return S.bars[a].idxs.length", bi) == 2 and posts[-1]["removed"] == [], "undo restores both chords, `removed` empty")
    # default path: keep the first
    tap_chord(first)
    pg.get_by_role("button", name=re.compile("^Couper la mesure")).click(); pg.wait_for_timeout(400)
    pg.get_by_role("button", name=re.compile("^Recoller")).click(); pg.wait_for_timeout(400)
    pg.locator("[data-keep-idx]").nth(0).click(); pg.wait_for_timeout(600)
    check(ev("return S.chords[a].root", first) == c1["root"] and ev("return S.bars[a].idxs.length", bi) == 1, "keeping the first still works")
    check(not any(c["bar"] == c1["bar"] for c in posts[-1]["chords"]), "keeping the first locks nothing")
    check(not errs, f"no page errors {errs[:2]}")
    b.close()

with sync_playwright() as p:
    run(p, "chromium"); run(p, "webkit")
sys.exit(1 if fails else 0)
