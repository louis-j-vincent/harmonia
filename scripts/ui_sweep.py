"""Balayage de l'app au navigateur, à la chasse aux bugs (390 px, tactile).

Chaque scénario est joué puis VÉRIFIÉ par une assertion ; tout ce qui casse
est collecté et rendu à la fin, avec les erreurs JS et les requêtes réseau
en échec. Rien n'est réparé ici : ce script constate.

    python uisweep.py [http://127.0.0.1:7778]
"""
import sys
import traceback

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7778"
BUGS, NOTES = [], []


def bug(area, what, detail=""):
    BUGS.append((area, what, detail))
    print(f"  ✗ [{area}] {what}" + (f" — {detail}" if detail else ""), flush=True)


def note(msg):
    NOTES.append(msg)
    print(f"  · {msg}", flush=True)


def open_song(pg, name, idx=0):
    pg.goto(BASE)
    pg.wait_for_timeout(900)
    pg.locator("text=Mes charts").click()
    pg.wait_for_timeout(1100)
    loc = pg.locator(f"text={name}")
    if loc.count() <= idx:
        return False
    loc.nth(idx).click()
    pg.wait_for_timeout(2600)
    return True


def cells_visible(pg, lo=None, hi=640):
    """[(numéro de mesure AFFICHÉE, boîte)] — `data-bar`, jamais l'ordre DOM :
    la rangée d'intro repliée porte plusieurs mesures dans un seul élément,
    donc le n-ième élément n'est pas la n-ième mesure (piège qui m'a fait
    croire à un décalage de la tête de lecture)."""
    out = []
    if lo is None:
        # sous TOUT ce qui flotte en haut (barre d'outils collante) : une
        # mesure à moitié dessous reçoit le clic de la barre, pas le sien —
        # ce qui faisait échouer le test alors que l'app allait bien.
        lo = pg.evaluate("()=>{let m=70;document.querySelectorAll('*')"
                         ".forEach(e=>{const s=getComputedStyle(e);"
                         "if(s.position==='sticky'||s.position==='fixed'){"
                         "const r=e.getBoundingClientRect();"
                         "if(r.top<200&&r.bottom>m&&r.height<200) m=r.bottom;}});"
                         "return m+8;}")
    cells = pg.locator("[data-bar]")
    n = cells.count()
    bars = pg.evaluate("[...document.querySelectorAll('[data-bar]')]"
                       ".map(c=>+c.dataset.bar)")
    for i in range(n):
        bb = cells.nth(i).bounding_box()
        if bb and lo < bb["y"] < hi:
            out.append((bars[i], bb))
    return out


def long_press_drag(pg, boxes, n=4, hold=420):
    a = boxes[0][1]
    pg.mouse.move(a["x"] + a["width"] / 2, a["y"] + a["height"] / 2)
    pg.mouse.down()
    pg.wait_for_timeout(hold)
    for _, bb in boxes[1:n]:
        pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_timeout(70)
    pg.mouse.up()
    pg.wait_for_timeout(700)


def sel_text(pg):
    return pg.evaluate(
        "()=>{const m=document.body.innerText.match(/mesures \\d+–\\d+/);"
        "return m?m[0]:''}")


def strip(pg):
    return pg.evaluate(
        "()=>[...document.querySelectorAll('div')].map(e=>e.textContent||'')"
        ".filter(t=>/^(intro|outro|bridge|solo|queue|[A-E]|\\?) \\d+–\\d+/.test(t.trim()))"
        ".slice(0,14)")


def lit_bars(pg):
    return pg.evaluate("[...document.querySelectorAll('[data-bar]')]"
                       ".filter(c=>c.style.background).map(c=>+c.dataset.bar)")


def open_tool(pg):
    b = pg.locator("button:has-text('sections')")
    if not b.count():
        return False
    b.first.click()
    pg.wait_for_timeout(2600)
    return True


def run(pg, errs, fails):
    # ── 1. bibliothèque ────────────────────────────────────────────────────
    print("\n1. bibliothèque", flush=True)
    pg.goto(BASE)
    pg.wait_for_timeout(1200)
    if "harmonia" not in pg.inner_text("body").lower():
        bug("bibliothèque", "l'accueil ne s'affiche pas")
    pg.locator("text=Mes charts").click()
    pg.wait_for_timeout(1200)
    rows = pg.evaluate("document.body.innerText.split('\\n').filter(t=>/bars$/.test(t)).length")
    note(f"{rows} charts listés")

    # ── 2. modes du chart ──────────────────────────────────────────────────
    print("\n2. modes du chart", flush=True)
    if not open_song(pg, "This Love"):
        bug("chart", "This Love introuvable dans la bibliothèque")
        return
    for mode in ("Analyse", "Annotate", "Read"):
        m = pg.locator(f"text={mode}")
        if not m.count():
            bug("modes", f"onglet {mode} absent")
            continue
        try:
            m.first.click()
            pg.wait_for_timeout(1200)
            if not pg.locator("[data-bar]").count():
                bug("modes", f"{mode} : plus aucune mesure affichée")
        except Exception as e:
            bug("modes", f"{mode} : clic impossible", str(e)[:80])
    # Practise est un écran PLEIN (le prompter) : on en sort par le chevron,
    # pas par l'onglet — vérifier qu'on peut bien y revenir.
    try:
        pg.locator("text=Practise").first.click()
        pg.wait_for_timeout(2000)
        if pg.locator("text=Read").count():
            note("Practise garde les onglets")
        back = pg.locator("text=‹")
        if not back.count():
            bug("modes", "Practise : aucune sortie (pas de chevron retour)")
        else:
            back.first.click()
            pg.wait_for_timeout(1600)
            if not pg.locator("[data-bar]").count():
                bug("modes", "sortie de Practise : on ne revient pas sur le chart",
                    (pg.inner_text("body")[:60] or "").replace("\n", " "))
    except Exception as e:
        bug("modes", "Practise : navigation impossible", str(e)[:80])
    if pg.locator("text=Read").count():
        pg.locator("text=Read").first.click()
        pg.wait_for_timeout(900)
    else:
        open_song(pg, "This Love")

    # ── 3. tête de lecture ─────────────────────────────────────────────────
    print("\n3. tête de lecture", flush=True)
    vis = cells_visible(pg)
    if len(vis) < 4:
        bug("lecture", f"seulement {len(vis)} mesure(s) visible(s) — "
                       "il faut pouvoir en toucher plusieurs sans défiler")
    else:
        for k in (1, 3):
            i, bb = vis[k]
            pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
            pg.wait_for_timeout(700)
            got = lit_bars(pg)
            if got != [i]:
                bug("lecture", f"tap sur la mesure affichée {i} allume {got}")
        # double tap au même endroit
        i, bb = vis[1]
        pg.mouse.dblclick(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_timeout(700)
        if lit_bars(pg) != [i]:
            bug("lecture", "double tap : la tête de lecture se perd",
                str(lit_bars(pg)))

    # ── 4. outil sections : ouverture ──────────────────────────────────────
    print("\n4. outil sections", flush=True)
    if not open_tool(pg):
        bug("outil", "bouton Sections absent sur This Love (mode Read)")
        return
    if "chargé depuis" not in pg.inner_text("body"):
        note("aucune mention de la source du chargement")
    before = strip(pg)
    note(f"bandeau à l'ouverture : {before[:6]}")

    # 4a. appui long + glissé — position de défilement DÉTERMINISTE : la
    # tête de lecture posée à l'étape 3 peut avoir fait défiler la grille
    # entre le calcul des coordonnées et l'appui (faux négatif du test).
    pg.evaluate("()=>{const s=document.querySelector('.ap-scroll');"
                "if(s) s.scrollTop=0;}")
    pg.wait_for_timeout(500)
    vis = cells_visible(pg)
    if len(vis) < 4:
        bug("outil", f"panneau trop haut : {len(vis)} mesure(s) visible(s) "
                     "sous lui, impossible de sélectionner sans défiler")
    if len(vis) >= 4:
        long_press_drag(pg, vis, 4)
        if not sel_text(pg):
            bug("outil", "appui long + glissé : aucune sélection")
        else:
            note(f"sélection : {sel_text(pg)}")

    # 4b. changer de lettre PENDANT une sélection
    b_chip = pg.locator("button:text-is('B')")
    if b_chip.count():
        b_chip.first.click()
        pg.wait_for_timeout(600)
        if not sel_text(pg):
            bug("outil", "changer de lettre efface la sélection en cours")
        vb = pg.locator("button:has-text('Valider')")
        if vb.count() and "Valider B" not in vb.first.inner_text():
            bug("outil", "le bouton Valider ne suit pas la lettre choisie",
                vb.first.inner_text())

    # 4c. valider
    vb = pg.locator("button:has-text('Valider')")
    if not vb.count():
        bug("outil", "pas de bouton Valider après une sélection")
    else:
        vb.first.click()
        pg.wait_for_timeout(9000)
        if sel_text(pg):
            bug("outil", "la sélection reste affichée après validation")
        after = strip(pg)
        note(f"bandeau après validation : {after[:8]}")
        if after == before:
            bug("outil", "valider ne change rien au bandeau")

    # 4d. valider SANS sélection (le bouton doit être absent)
    if pg.locator("button:has-text('Valider')").count():
        bug("outil", "le bouton Valider survit à la validation (double envoi possible)")

    # 4e. badge ✕
    badges = pg.locator("[data-sect-badge]")
    nb = badges.count()
    if not nb:
        bug("outil", "aucun badge de lettre affiché sur les mesures marquées")
    else:
        s0 = strip(pg)
        badges.first.click()
        pg.wait_for_timeout(1200)
        if strip(pg) == s0:
            bug("outil", "le badge ✕ ne retire pas l'occurrence")

    # 4f. tap normal pendant l'outil = lecture
    vis = cells_visible(pg, lo=260)
    if vis:
        i, bb = vis[len(vis) // 2]
        pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_timeout(800)
        if lit_bars(pg) != [i]:
            bug("outil", f"outil ouvert : tap sur la mesure {i} allume "
                         f"{lit_bars(pg)}")

    # 4g. défilement au doigt
    ta = pg.evaluate("getComputedStyle(document.querySelector('[data-bar]')"
                     ".parentElement).touchAction")
    if ta != "pan-y":
        bug("outil", f"touch-action = {ta} (le défilement au doigt est bloqué)")

    # 4h. Recommencer / Fermer / rouvrir
    rec = pg.locator("button:has-text('Recommencer')")
    if rec.count():
        rec.first.click()
        pg.wait_for_timeout(1000)
        if [x for x in strip(pg) if not x.startswith("?")]:
            bug("outil", "Recommencer laisse des marques", str(strip(pg))[:90])
    fer = pg.locator("button:has-text('Fermer')")
    if fer.count():
        fer.first.click()
        pg.wait_for_timeout(900)
        if pg.locator("button:has-text('Recommencer')").count():
            bug("outil", "Fermer ne ferme pas le panneau")
    if open_tool(pg):
        note(f"réouverture : {strip(pg)[:5]}")
    else:
        bug("outil", "impossible de rouvrir l'outil après fermeture")

    # ── 5. sélection d'UNE seule mesure ────────────────────────────────────
    print("\n5. cas limites de sélection", flush=True)
    vis = cells_visible(pg)
    if vis:
        a = vis[0][1]
        pg.mouse.move(a["x"] + a["width"] / 2, a["y"] + a["height"] / 2)
        pg.mouse.down()
        pg.wait_for_timeout(430)
        pg.mouse.up()
        pg.wait_for_timeout(700)
        if not sel_text(pg):
            note("appui long sans glisser : pas de sélection d'une mesure")
        else:
            vb = pg.locator("button:has-text('Valider')")
            if vb.count():
                vb.first.click()
                pg.wait_for_timeout(9000)
                note(f"validation d'UNE mesure → {strip(pg)[:6]}")

    # ── 6. autres surfaces ─────────────────────────────────────────────────
    print("\n6. autres surfaces", flush=True)
    for label, sel in (("Aa (préférences)", "text=Aa"),
                       ("A–B loop", "button:has-text('loop')")):
        el = pg.locator(sel)
        if not el.count():
            bug("UI", f"{label} absent")
            continue
        try:
            el.first.click()
            pg.wait_for_timeout(900)
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(500)
            body = pg.inner_text("body")
            if not body.strip():
                bug("UI", f"{label} laisse un écran vide")
        except Exception as e:
            bug("UI", f"{label} : clic impossible", str(e)[:80])

    print("\n7. erreurs JS et requêtes en échec", flush=True)
    for e in errs[:10]:
        bug("JS", e[:150])
    for f in fails[:10]:
        bug("réseau", f)


def main():
    errs, fails = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 390, "height": 844},
                        has_touch=True, is_mobile=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console: " + m.text)
              if m.type == "error" else None)
        pg.on("requestfailed", lambda r: fails.append(f"{r.method} {r.url[:80]}"))
        pg.on("response", lambda r: fails.append(
            f"HTTP {r.status} {r.request.method} {r.url[:80]}")
            if r.status >= 400 else None)
        try:
            run(pg, errs, fails)
        except Exception:
            print("\nLE BALAYAGE LUI-MÊME A PLANTÉ :")
            traceback.print_exc()
        b.close()
    print("\n" + "=" * 60)
    print(f"{len(BUGS)} anomalie(s) :")
    for area, what, detail in BUGS:
        print(f"  [{area}] {what}" + (f" — {detail}" if detail else ""))


if __name__ == "__main__":
    main()
