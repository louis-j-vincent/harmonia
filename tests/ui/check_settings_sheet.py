"""UX #6 (2026-10-01): ONE « Réglages » sheet for all modes. Chart (Aa) and
S'entraîner (···) open the same sheet; Practise adds its play settings first
(Vitesse, Piano, Clic) and the loop calage in the shared « Plus… », whose
summary shows a visible chevron (▸ closed / ▾ open). Chromium AND WebKit,
390x844, touch; every POST is intercepted.

    .venv/bin/python tests/ui/check_settings_sheet.py [file]
"""
import sys
from playwright.sync_api import sync_playwright
FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
  for name in ("chromium", "webkit"):
    tag = name + ": "
    b = getattr(p, name).launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    ctx.add_init_script("localStorage.setItem('harmPianoOffByDefault','1')")
    pg = ctx.new_page()
    errs = []; posts = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    def route(r):
        if r.request.method == "POST":
            posts.append(r.request.url); r.fulfill(status=200, body="{}", content_type="application/json")
        else: r.continue_()
    pg.route("**/api/**", route)
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    sheet_text = lambda: pg.evaluate("()=>document.querySelector('[data-overlay]')?.innerText||''")
    summ = lambda: pg.locator("[data-overlay] summary")

    # --- chart: Aa -> Réglages, Plus… has a chevron, no play settings
    pg.get_by_role("button", name="Réglages").first.tap(); pg.wait_for_timeout(500)
    t = sheet_text()
    check("Thème" in t and "Vitesse" not in t, tag + "chart sheet: reading settings, no play settings")
    check(summ().count() == 1 and summ().inner_text().strip().startswith("▸"), tag + "chart: « Plus… » shows ▸ when closed")
    summ().tap(); pg.wait_for_timeout(300)
    check(summ().inner_text().strip().startswith("▾"), tag + "chart: « Plus… » shows ▾ when open")
    pg.screenshot(path=f"/tmp/settings_chart_{name}.png")
    pg.get_by_role("button", name="Fermer").tap(); pg.wait_for_timeout(500)
    check(pg.evaluate("async()=>(await import('/static/state.js')).S._prefsPlusOpen") is True, tag + "« Plus… » open state is remembered")
    pg.evaluate("async()=>{(await import('/static/state.js')).S._prefsPlusOpen=false}")

    # --- practise: same sheet, play settings first, calage in Plus…
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1500)
    check(pg.get_by_role("button", name="Réglages").count() == 1, tag + "practise: one « Réglages » button")
    pg.get_by_role("button", name="Réglages").tap(); pg.wait_for_timeout(500)
    t = sheet_text()
    check(t.lstrip().lower().startswith("réglages"), tag + "practise sheet is titled « Réglages »")
    pos = {k: t.find(k) for k in ("Vitesse", "Piano", "Clic sur les accords", "Thème")}
    check(all(v >= 0 for v in pos.values()) and pos["Vitesse"] < pos["Piano"] < pos["Clic sur les accords"] < pos["Thème"],
          tag + f"practise: play settings first, then reading settings {pos}")
    check(summ().count() == 1 and summ().inner_text().strip().startswith("▸"), tag + "practise: « Plus… » ▸ closed")
    check(pg.get_by_text("Calage de la boucle").count() == 0, tag + "no calage without a loop")
    pg.locator("[data-overlay]").get_by_role("button", name="0.5×", exact=True).tap(); pg.wait_for_timeout(300)
    check(pg.evaluate("async()=>(await import('/static/state.js')).S.playRate") == 0.5, tag + "Vitesse 0.5× applies")
    pg.get_by_role("button", name="Fermer").tap(); pg.wait_for_timeout(500)
    # arm a loop -> calage appears inside Plus…
    pg.locator("button", has_text="boucle").first.tap(); pg.wait_for_timeout(500)
    pg.get_by_role("button", name="Réglages").tap(); pg.wait_for_timeout(500)
    check(not pg.get_by_text("Calage de la boucle").is_visible(), tag + "calage hidden while « Plus… » is closed")
    summ().tap(); pg.wait_for_timeout(300)
    check(pg.get_by_text("Calage de la boucle").is_visible(), tag + "calage visible under « Plus… » during a loop")
    pg.screenshot(path=f"/tmp/settings_practise_{name}.png")
    pg.get_by_role("button", name="plus tôt").tap(); pg.wait_for_timeout(200)
    check("ms" in sheet_text(), tag + "calage buttons work")
    check(not posts, tag + f"no POST ({posts})")
    check(not errs, tag + f"no page error ({errs})")
    b.close()
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
