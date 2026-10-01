"""Item 7 of the 2026-10-01 UX list: the home screen. « Reprendre » on top, a
REAL search field (YouTube link announced), and « Importer un fichier audio »
(multipart POST /api/import-audio -> loading screen). Every POST is
intercepted: nothing reaches state/.

    .venv/bin/python tests/ui/check_home.py [shots_prefix]
"""
import json, sys, tempfile, os
from playwright.sync_api import sync_playwright

PRE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/home"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

wav = os.path.join(tempfile.mkdtemp(), "Ma_chanson.wav")
open(wav, "wb").write(b"RIFF....WAVEfmt " + b"\0" * 64)

def run(name, launcher, **ctx):
    print("==", name)
    b = launcher.launch()
    pg = b.new_context(viewport={"width": 390, "height": 844}, **ctx).new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    posts = []
    def route(r):
        u, m = r.request.url, r.request.method
        if m == "POST":
            posts.append((u, r.request.headers.get("content-type", ""), r.request.post_data_buffer or b""))
            if "/api/import-audio" in u or "/api/analyze" in u:
                r.fulfill(status=200, body='{"job_id":"jx"}', content_type="application/json")
            else: r.fulfill(status=200, body="{}", content_type="application/json")
        elif "/api/job/jx" in u:
            r.fulfill(status=200, body=json.dumps({"status": "running", "stage": 1, "phase": "listening"}), content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(2000)
    pg.screenshot(path=f"{PRE}_{name}_home.png")

    txt = pg.inner_text("body")
    ys = pg.evaluate("""()=>{const f=t=>{const e=[...document.querySelectorAll('div,span')].find(x=>x.children.length===0&&x.textContent.trim().toUpperCase()===t);return e?e.getBoundingClientRect().y:null};return {rep:f('REPRENDRE'),charts:f('MES CHARTS'),seek:f('CHERCHER')}}""")
    check(ys["rep"] is not None and ys["rep"] < ys["seek"], f"Reprendre au-dessus de Chercher ({ys})")
    inp = pg.locator("input[type=text], input:not([type])").first
    check(inp.count() == 1 and inp.is_visible(), "the home search is a real <input>")
    check("lien" in (inp.get_attribute("placeholder") or "").lower() and "youtube" in (inp.get_attribute("placeholder") or "").lower(), "placeholder announces the YouTube link")
    fi = pg.locator("input[type=file]")
    check(fi.count() == 1 and "audio" in (fi.get_attribute("accept") or ""), "a file input accepts audio")
    check("Importer" in txt, "an « Importer » control is visible")

    # paste a YouTube link -> /api/analyze
    inp.fill("https://youtu.be/abcdefghijk"); inp.press("Enter"); pg.wait_for_timeout(800)
    check(any("/api/analyze" in u for u, _, _ in posts), "Enter on a YouTube link starts an analysis")
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1500)
    # typed title -> search screen with the query
    pg.locator("input[type=text], input:not([type])").first.fill("autumn leaves"); pg.keyboard.press("Enter"); pg.wait_for_timeout(800)
    check(pg.evaluate("async()=>(await import('/static/state.js')).S.screen") == "search", "Enter on a title opens the search screen")
    check(pg.locator("input").first.input_value() == "autumn leaves", "the query carries over")
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1500)

    # import a file
    posts.clear()
    pg.locator("input[type=file]").set_input_files(wav); pg.wait_for_timeout(1000)
    imp = [x for x in posts if "/api/import-audio" in x[0]]
    check(len(imp) == 1, "one POST /api/import-audio")
    if imp:
        u, ct, body = imp[0]
        check(ct.startswith("multipart/form-data"), "multipart/form-data")
        check(b'name="audio"' in body and b'filename="Ma_chanson.wav"' in body, "field `audio`, original filename kept")
    scr = pg.evaluate("async()=>(await import('/static/state.js')).S.screen")
    check(scr == "analysing", f"loading screen shown (screen={scr})")
    pg.screenshot(path=f"{PRE}_{name}_loading.png")
    check(not errs, f"no console errors {errs}")
    b.close()

with sync_playwright() as p:
    run("chromium", p.chromium)
    run("webkit", p.webkit, is_mobile=True, has_touch=True)
print("FAILS:", fails); sys.exit(1 if fails else 0)
