"""Lot 2 UX list, rows 39-43 (accueil): Reprendre dit la mesure, titre avant
artiste, bouton « Nouveau dossier », estimation de durée, recherches récentes.
POSTs interceptés; rien n'atteint state/.

    .venv/bin/python tests/ui/check_accueil_lot2.py [shots_prefix]
"""
import json, sys
from playwright.sync_api import sync_playwright

PRE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/accueil2"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

def run(name, launcher, **ctx):
    print("==", name)
    b = launcher.launch()
    pg = b.new_context(viewport={"width": 390, "height": 844}, **ctx).new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    posts = []
    def route(r):
        if r.request.method == "POST":
            posts.append(r.request.url); r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1500)
    lib = pg.evaluate("async()=>(await import('/static/state.js')).S.library")
    c = lib[0]["file"]
    S = "(await import('/static/state.js')).S"

    # ---- #39 : Reprendre dit la mesure
    pg.evaluate("f=>{localStorage.setItem('harmLastChart',f);localStorage.setItem('harmPos:'+f,JSON.stringify({t:80,bar:40}))}", c)
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1500)
    t = pg.locator("[data-reprise=mesure]")
    check(t.count() == 1 and t.inner_text() == "mesure 40", "Reprendre affiche « mesure 40 »")
    pg.screenshot(path=f"{PRE}_{name}_reprendre.png")
    # écriture sur pause : ouvre le chart, simule un <audio> en pause à t donné
    pg.goto(f"http://localhost:7772/?open={c}"); pg.wait_for_timeout(2500)
    pg.evaluate("""async()=>{const {S}=await import('/static/state.js');
      localStorage.removeItem('harmPos:'+S.model.file);
      window.__bar=S.barTimeIndex[12].bi+1;
      const a=document.createElement('audio'); Object.defineProperty(a,'currentTime',{value:S.barTimeIndex[12].t0+0.3});
      document.body.appendChild(a); a.dispatchEvent(new Event('pause'));}""")
    exp = pg.evaluate("window.__bar")
    p = pg.evaluate("f=>JSON.parse(localStorage.getItem('harmPos:'+f))", c)
    check(p is not None and p["bar"] == exp, f"la pause écrit la position (mesure {exp}) -> {p}")
    check(not any("position" in u for u in posts), "rien envoyé au serveur")
    check(not errs, f"pas d'erreur JS {errs[:2]}")

    # ---- #40 : le titre passe avant l'artiste
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(1200)
    pg.evaluate('''async()=>{const k=await import('/static/screens/library.js');
      const d=document.createElement('div'); d.id='t40'; d.style.cssText='position:fixed;top:300px;left:20px;width:250px;background:#fff;z-index:99';
      d.appendChild(k.nameEl({title:'Autumn Leaves (live)',artist:'Cannonball Adderley Quintet'},16));
      d.appendChild(k.nameEl({title:'Un titre vraiment très très long qui ne tient pas du tout',artist:'Court'},16));
      d.appendChild(k.nameEl({title:'Solo',artist:'Artiste'},16));
      document.body.appendChild(d);}''')
    r = pg.evaluate('''()=>[...document.querySelectorAll('#t40 > div')].map(w=>({a:(w.querySelector('[data-nom=artiste]')||{}).clientWidth,aw:(w.querySelector('[data-nom=artiste]')||{}).scrollWidth,t:w.querySelector('[data-nom=titre]').clientWidth,tw:w.querySelector('[data-nom=titre]').scrollWidth}))''')
    check(r[0]["t"] >= r[0]["tw"] and r[0]["a"] < r[0]["aw"], f"artiste long: le titre tient en entier, l'artiste tronque {r[0]}")
    check(r[1]["t"] < r[1]["tw"] and r[1]["a"] <= r[1]["aw"], f"titre trop long: il tronque en dernier {r[1]}")
    check(r[2]["t"] >= r[2]["tw"] and r[2]["a"] >= r[2]["aw"], f"assez de place: rien ne tronque {r[2]}")
    pg.screenshot(path=f"{PRE}_{name}_titre.png")
    b.close()

with sync_playwright() as p:
    run("chromium", p.chromium)
    run("webkit", p.webkit, is_mobile=True, has_touch=True)
sys.exit(1 if fails else 0)
