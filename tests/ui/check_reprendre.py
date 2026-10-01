"""« Reprendre · mesure N » de bout en bout, sur le VRAI lecteur de l'app
(2026-10-01). Le premier essai écoutait `document`, or l'élément audio n'est
pas dans le DOM : rien n'était jamais enregistré, et la carte n'y ramenait pas.
Seul le localStorage est écrit (par appareil), rien côté serveur.

    .venv/bin/python tests/ui/check_reprendre.py
"""
import sys
from playwright.sync_api import sync_playwright
F = "min_08DjMT-qR9g"; fails = []
def ck(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: fails.append(m)
with sync_playwright() as p:
    for eng in (p.chromium, p.webkit):
        b = eng.launch(); pg = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=eng is p.webkit, has_touch=True).new_page()
        pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json") if r.request.method == "POST" else r.continue_())
        ev = lambda js: pg.evaluate("async()=>{const {S}=await import('/static/state.js'); const M=await import('/static/audio.js'); " + js + "}")
        pg.goto(f"http://localhost:7772/?open={F}"); pg.wait_for_selector("[data-bar]", timeout=30000, state="attached"); pg.wait_for_timeout(1200)
        ev("const g=S.model.barGrid; M.seekFrac((g[39]+0.2)/M.totalDur())"); pg.wait_for_timeout(600)
        pos = pg.evaluate(f"JSON.parse(localStorage.getItem('harmPos:{F}')||'null')")
        ck(pos is not None and pos["bar"] == 40, f"[{eng.name}] a paused seek to bar 40 is remembered ({pos})")
        pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1500)
        ck(pg.get_by_text("mesure 40").count() == 1, f"[{eng.name}] home card says « mesure 40 »")
        pg.get_by_text("mesure 40").click(); pg.wait_for_selector("[data-bar]", timeout=30000, state="attached"); pg.wait_for_timeout(2500)
        t = ev("return M.nowTime()"); g40 = ev("return S.model.barGrid[39]")
        ck(abs(t - g40) < 0.5, f"[{eng.name}] tapping the card resumes at bar 40 ({t:.2f} vs {g40:.2f})")
        b.close()
sys.exit(1 if fails else 0)
