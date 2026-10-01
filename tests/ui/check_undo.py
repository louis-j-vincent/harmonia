"""Item 5 of the 2026-10-01 UX top-5: a safety net. « Annuler » after keeping a
chord, cutting a bar, gluing it back, tempo ÷2/×2 and bar 1; and a VISIBLE
error when saving fails. Every POST is intercepted (nothing reaches state/).

    .venv/bin/python tests/ui/check_undo.py [file] [shots_prefix]
"""
import json, re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/undo"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "500" not in m.text else None)  # the 500 is our simulated save failure
    annots, tempos, bar1s = [], [], []
    state = {"fail_save": False, "prev_bar1": 4.2}
    def route(r):
        u, m = r.request.url, r.request.method
        if m == "POST" and "/api/annotations/" in u:
            annots.append(json.loads(r.request.post_data or "{}"))
            if state["fail_save"]: r.fulfill(status=500, body='{"error":"x"}', content_type="application/json")
            else: r.fulfill(status=200, body="{}", content_type="application/json")
        elif m == "POST" and "/api/tempo/" in u:
            tempos.append(json.loads(r.request.post_data)); r.fulfill(status=200, body='{"job_id":"j1"}', content_type="application/json")
        elif m == "POST" and "/api/bar1/" in u:
            bar1s.append(json.loads(r.request.post_data)); r.fulfill(status=200, body=json.dumps({"job_id": "j1", "previous_bar1": state["prev_bar1"]}), content_type="application/json")
        elif m == "GET" and "/api/job/j1" in u:
            r.fulfill(status=200, body=json.dumps({"status": "done", "phase": "done", "chart_url": "/chart/" + FILE}), content_type="application/json")
        elif m == "POST": r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(500)
    ev = lambda js, arg=None: pg.evaluate("async(a)=>{const {S}=await import('/static/state.js'); " + js + "}", arg)
    def tap_chord(idx):
        bi = ev("return S.bars.findIndex(b=>b.idxs.includes(a))", idx)
        pg.evaluate("(bi)=>document.querySelector(`[data-bar=\"${bi}\"]`).scrollIntoView({block:'center'})", bi); pg.wait_for_timeout(250)
        r = pg.evaluate("(bi)=>{const r=document.querySelector(`[data-bar=\"${bi}\"]`).getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}}", bi)
        k = ev("return S.bars[a[0]].idxs.indexOf(a[1])", [bi, idx]); n = ev("return S.bars[a].idxs.length", bi)
        pg.mouse.click(r["x"] + r["w"]*(k+.5)/n, r["y"] + r["h"]/2); pg.wait_for_timeout(450)
        return bi
    snack = lambda: pg.get_by_role("button", name=re.compile("^(Annuler|Réessayer)$"))
    cut_btn = lambda: pg.get_by_role("button", name=re.compile("^Couper la mesure"))

    one = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===1&&!S.chords[x[0]].nc&&!S.chords[x[0]].confirmed&&S.chords[x[0]].t0>6)")
    bi, [idx] = one[0], one[1]
    c0 = ev("const c=S.chords[a]; return {root:c.root,q:c.q,t1:c.t1,n:S.chords.length}", idx)

    # --- 1. keep a chord, then undo it
    tap_chord(idx)
    pg.get_by_role("button", name=re.compile("^Garder ")).click(); pg.wait_for_timeout(500)
    check(len(annots) == 1 and len(annots[0]["chords"]) >= 1, "keeping saves")
    check(snack().count() == 1, "« Annuler » appears after keeping a chord")
    pg.screenshot(path=PRE + "_lock.png")
    snack().click(); pg.wait_for_timeout(500)
    check(not ev("return S.chords[a].confirmed", idx), "undo: the chord is unlocked again")
    bar_no = ev("return S.chords[a].bar", idx)
    check(not any(c["bar"] == bar_no and c["beat"] == 0 for c in annots[-1]["chords"]), "undo rewrites the sidecar without it")

    # --- 2. cut + keep + undo
    n_before = len(annots)
    tap_chord(idx); cut_btn().click(); pg.wait_for_timeout(400)
    pg.locator('[data-cut-beat="2"]').click(); pg.wait_for_timeout(500)
    ev("const A=await import('/static/screens/annotate.js'); A.confirmChord(S._editorIdx, 5, '-7', null)"); pg.wait_for_timeout(500)
    check(snack().count() == 1, "« Annuler » appears after cutting a bar")
    snack().click(); pg.wait_for_timeout(500)
    st = ev("return {n:S.chords.length, ids:S.bars[a[0]].idxs.length, t1:S.chords[a[1]].t1}", [bi, idx])
    check(st["n"] == c0["n"] and st["ids"] == 1 and abs(st["t1"] - c0["t1"]) < 1e-6, f"undo restores the uncut bar ({st})")
    check(not any(c["beat"] == 2 and c["bar"] == bar_no for c in annots[-1]["chords"]), "undo removes the cut chord from the sidecar")

    # --- 3. recoller + undo (a model 2-chord bar)
    two = ev("return S.bars.map((b,i)=>[i,b.idxs]).find(([i,x])=>x.length===2&&!x.some(j=>S.chords[j].added||S.chords[j].confirmed))")
    tap_chord(two[1][0]); cut_btn().click(); pg.wait_for_timeout(400)
    pg.get_by_role("button", name=re.compile("^Recoller")).click(); pg.wait_for_timeout(400)
    pg.locator("[data-keep-idx]").first.click(); pg.wait_for_timeout(500)   # « garder lequel ? » : le premier
    check(snack().count() == 1 and ev("return S.bars[a].idxs.length", two[0]) == 1, "« Annuler » appears after gluing a bar")
    snack().click(); pg.wait_for_timeout(500)
    check(ev("return S.bars[a].idxs.length", two[0]) == 2 and annots[-1]["removed"] == [], "undo brings the second chord back and clears `removed`")

    # --- 4. save failure is visible, retry works
    state["fail_save"] = True
    tap_chord(idx); pg.get_by_role("button", name=re.compile("^Garder ")).click(); pg.wait_for_timeout(600)
    check(pg.get_by_text(re.compile("Pas enregistré")).count() == 1, "a failed save is announced")
    pg.screenshot(path=PRE + "_fail.png")
    state["fail_save"] = False; n = len(annots)
    pg.get_by_role("button", name="Réessayer").click(); pg.wait_for_timeout(500)
    check(len(annots) == n + 1 and any(c["bar"] == bar_no for c in annots[-1]["chords"]), "« Réessayer » re-sends the whole state")

    # --- 5. tempo ÷2 then undo
    pg.get_by_role("button", name="Outils").click(); pg.wait_for_timeout(400)
    pg.get_by_role("button", name="÷2").click(); pg.wait_for_timeout(2500)
    check(tempos == [{"factor": 0.5}], f"÷2 posted {tempos}")
    pg.screenshot(path=PRE + "_analysing.png")
    ev("S.job=null; const C=await import('/static/screens/chart.js'); await C.openChart(a)", FILE); pg.wait_for_timeout(1800)   # = the « Open the chart » button of the loading screen
    check(pg.get_by_text("Tempo ÷2 appliqué").count() == 1, "after the job, the chart shows « Tempo ÷2 appliqué · Annuler »")
    pg.screenshot(path=PRE + "_tempo.png")
    snack().click(); pg.wait_for_timeout(1500)
    check(tempos[-1] == {"factor": 2.0}, f"undo of ÷2 posts the exact inverse ×2 ({tempos[-1]})")

    # --- 6. bar 1 then undo, with and without a previous mark
    for prev, expect in ((4.2, {"t": 4.2}), (None, {"clear": True})):
        state["prev_bar1"] = prev
        pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1200)
        pg.locator('button:text-is("Corriger")').first.click(); pg.wait_for_timeout(400)
        pg.get_by_role("button", name="Outils").click(); pg.wait_for_timeout(400)
        pg.get_by_text("Caler la mesure 1").first.click(); pg.wait_for_timeout(700)
        pg.get_by_role("button", name="La mesure 1 commence ici").click(); pg.wait_for_timeout(2500)
        ev("S.job=null; const C=await import('/static/screens/chart.js'); await C.openChart(a)", FILE); pg.wait_for_timeout(1800)   # = the « Open the chart » button of the loading screen
        check(pg.get_by_text("Mesure 1 calée").count() == 1, f"bar 1: undo offered (previous={prev})")
        snack().click(); pg.wait_for_timeout(1500)
        check(bar1s[-1] == expect, f"bar 1 undo posts {expect} (got {bar1s[-1]})")
    check(not errs, f"no console errors {errs[:3]}")
    b.close()
sys.exit(1 if fails else 0)
