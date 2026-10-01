"""Check (2026-10-01, UX audit #10 Bibliothèque), viewport 390x844.
 - la visite n'écrit PAS /api/folders (fusion serveur -> local seulement)
 - champ de filtre >= 44 px ; libellés français ; pastille « corrigé » ;
   tri Récents / A–Z visible et effectif ; aucune erreur console.
/api/library est intercepté (corrected posé sur le 1er chart) : marche même
avant le redémarrage du serveur. Rien n'est écrit dans state/.
    .venv/bin/python tests/ui/check_bibliotheque.py
"""
import json, sys
from playwright.sync_api import sync_playwright

fails = []
def check(c, msg):
    print(("OK   " if c else "FAIL ") + msg)
    if not c: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs, posts = [], []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def lib(route):
        r = route.fetch(); d = r.json()
        for i, c in enumerate(d["charts"]): c["corrected"] = (i == 0)
        route.fulfill(json=d)
    pg.route("**/api/library", lib)
    # le serveur « sait » un dossier que cet appareil ignore -> la fusion bouge
    pg.route("**/api/folders", lambda r: posts.append(r.request.post_data) or r.fulfill(status=200, body="{}", content_type="application/json")
             if r.request.method == "POST" else
             r.fulfill(json={"order": ["Dossier test"], "of": {"zz_inconnu": "Dossier test"}}))
    pg.goto("http://localhost:7772/"); pg.evaluate("localStorage.removeItem('harmFolders')")
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(2500)
    pg.evaluate("""async()=>{const {go}=await import('/static/router.js');go('charts',{push:false});}""")
    pg.wait_for_timeout(1000)
    pg.screenshot(path="/tmp/claude-501/bib_charts.png")
    check(not posts, f"aucun POST /api/folders à la visite ({len(posts)})")
    h = pg.evaluate("document.querySelector('input').getBoundingClientRect().height")
    check(h >= 44, f"champ de filtre {h:.0f}px >= 44")
    txt = pg.inner_text("body")
    check("bars" not in txt and "no audio" not in txt, "plus de « bars »/« no audio »")
    check("mesures" in txt, "« mesures » affiché")
    check(pg.locator("[data-pastille=corrige]").count() == 1, "une seule pastille « corrigé »")
    check(pg.locator("[data-tri=az]").count() == 1, "tri visible")
    def titles(): return pg.evaluate("[...document.querySelectorAll('.ap-scroll > div:last-child > div')].map(e=>e.innerText.split('\\n')[0])")
    pg.click("[data-tri=az]"); pg.wait_for_timeout(500)
    t = titles(); k = [x.split(" / ")[-1].lower().split()[0] for x in t]
    check(len(t) > 3 and k[:6] == sorted(k[:6], key=lambda x: x.lstrip("'\"(")) , "A–Z appliqué")
    pg.screenshot(path="/tmp/claude-501/bib_az.png")
    pg.click("[data-tri=recent]")
    b.close()
check(not errs, f"aucune erreur console {errs[:2]}")
sys.exit(1 if fails else 0)
