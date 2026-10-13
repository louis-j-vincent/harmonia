"""Changements de gamme (2026-10-13) : pastille « → A♭ majeur » + orthographe
des accords par zone. WebKit 390x844, tout POST intercepté.

    .venv/bin/python tests/ui/check_key_zones.py PORT FILE SHOTS_PREFIX [--single]

`--single` : morceau à un seul ton — aucune pastille, orthographe d'avant.
"""
import sys
from playwright.sync_api import sync_playwright

PORT, FILE, PRE = sys.argv[1], sys.argv[2], sys.argv[3]
SINGLE = "--single" in sys.argv
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


SHARPS, FLATS = "♯", "♭"

with sync_playwright() as p:
    b = p.webkit.launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    ctx.add_init_script("try{localStorage.setItem('harmPianoOffByDefault','1')}catch(e){}")
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    # un fichier audio absent du disque (404) n'est pas une erreur de rendu :
    # on le signale à part, on ne le compte pas
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
    pg.on("response", lambda r: print("     (404 hors rendu)", r.url) if r.status == 404 else None)
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json")
             if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:{PORT}/?open={FILE}")
    pg.wait_for_selector("[data-bar]", timeout=20000, state="attached")
    pg.wait_for_timeout(1500)
    info = pg.evaluate("""async()=>{const {S}=await import('/static/state.js');
      const cells=[...document.querySelectorAll('[data-bar]')];
      return {zones:S.keyZones, bars:cells.map(c=>{const bi=+c.dataset.bar; const pill=c.querySelector('[data-key-pill]');
        return {bi, kz:S.bars[bi].kz, txt:c.innerText.replace(/\\s+/g,' ').trim(), pill:pill?pill.innerText:null,
                top:Math.round(c.getBoundingClientRect().top)};})};}""")
    zones, bars = info["zones"], info["bars"]
    pills = [x for x in bars if x["pill"]]
    print("zones", zones)
    for x in pills:
        print("pastille", x)
    over = pg.evaluate("()=>document.documentElement.scrollWidth-document.documentElement.clientWidth")
    check(over <= 0, f"pas de débordement horizontal ({over}px)")
    if SINGLE:
        check(len(zones) == 1, "une seule zone")
        check(not pills, "aucune pastille")
    else:
        check(len(zones) >= 2, "au moins deux zones servies")
        changes = [x for i, x in enumerate(bars) if i and x["kz"] != bars[i - 1]["kz"]]
        check(len(pills) == len(changes) and all(x["pill"].startswith("→") for x in pills),
              f"une pastille par changement de zone ({len(pills)} / {len(changes)})")
        avant = {bars[i - 1]["bi"] for i, x in enumerate(bars) if i and x["kz"] != bars[i - 1]["kz"]}
        for x in bars:
            if x["bi"] in avant:      # la mesure qui annonce : accords pivots
                print("     mesure d'avant la pastille :", x["txt"])
                continue
            z = zones[x["kz"]]
            maj = z["tonic"] if z["mode"] != "minor" else (z["tonic"] + 3) % 12
            flat = maj in (0, 1, 3, 5, 6, 8, 10)
            txt = x["txt"].replace(x["pill"] or "", "")
            bad = SHARPS if flat else FLATS
            if bad in txt:
                check(False, f"mesure {x['bi']} (zone {x['kz']}) écrit « {txt} »")
        # capture centrée sur la première pastille
        if pills:
            pg.evaluate("(bi)=>document.querySelector(`[data-bar='${bi}']`).scrollIntoView({block:'center'})", pills[0]["bi"])
            pg.wait_for_timeout(400)
    pg.screenshot(path=f"{PRE}.png")
    check(not errs, f"zéro erreur console {errs[:3]}")
    b.close()

print("FAILS:", len(fails))
sys.exit(1 if fails else 0)
