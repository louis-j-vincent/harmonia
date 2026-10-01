"""UX item #3 (2026-10-01): the rest of the UI in French — Practise (+ sheets),
« Caler la mesure 1 », the sections editor and the loading/analysis screen.
Fails on any visible English word from BAD. Nothing is written (POSTs
intercepted); no JS console error allowed.

    .venv/bin/python tests/ui/check_french_rest.py [file] [shots_prefix]
"""
import re, sys
from playwright.sync_api import sync_playwright

FILE = sys.argv[1] if len(sys.argv) > 1 else "min_08DjMT-qR9g"
PRE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fr3"
BAD = re.compile(r"\b(setup|loop|bars?|decoded|ready|listening|reading|finding|finding the|the|your song|song|something|went wrong|didn'?t work|back to|library|play it now|open the chart|raw chart|no letters|already|every|written out|cancel|close|copy|copied|could not|microphone|permission|denied|not found|hands?|left|right|undo|save|loading|chart|sections?\.\.\.|beat|first bars|grouping|repeats|this is the long part|bar by bar)\b", re.I)
ALLOWED = re.compile(r"\bchart\b", re.I)   # « chart » is the project's own word, kept in French UI
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json")
             if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    def english(where):
        txt = pg.evaluate("""()=>document.body.innerText+' | '+[...document.querySelectorAll('[aria-label],[title],[placeholder]')].map(e=>(e.getAttribute('aria-label')||'')+' '+(e.title||'')+' '+(e.getAttribute('placeholder')||'')).join(' | ')""")
        txt = ALLOWED.sub("", txt)
        bad = sorted(set(m.group(0).lower() for m in BAD.finditer(txt)))
        check(not bad, f"{where}: no English word {bad}")
    def shot(n): pg.screenshot(path=f"{PRE}_{n}.png")
    def closeall():
        pg.keyboard.press("Escape"); pg.mouse.click(195, 30); pg.wait_for_timeout(300)

    # Practise
    pg.locator("button:text-is(\"S'entraîner\")").first.click(); pg.wait_for_timeout(1200)
    english("Practise"); shot("practise")
    pg.get_by_role("button", name="Outils").first.click(); pg.wait_for_timeout(500)
    english("Practise · Outils"); shot("practise_outils")
    pg.get_by_role("button", name="Fermer").first.click(); pg.wait_for_timeout(400)
    pg.evaluate("async()=>{const {S}=await import('/static/state.js'); S.piano='tap'; const {go}=await import('/static/router.js'); go('prompter',{push:false})}")
    pg.wait_for_timeout(900)
    pg.evaluate("""async()=>{const P=await import('/static/screens/practise.js'); const {root}=await import('/static/ui/kit.js');
      const c=P.buildHandsCard(()=>[48,55,60,64],()=>[48,55,60,64],()=>{},null); c.style.cssText+=';position:absolute;top:120px;left:20px;z-index:60'; (root||document.body).appendChild(c)}""")
    pg.wait_for_timeout(500)
    r = pg.get_by_role("button", name="réglages")
    check(r.count() >= 1, "carte mains : bouton « réglages »")
    english("carte mains"); shot("mains")
    if r.count():
        r.first.click(); pg.wait_for_timeout(400); english("feuille piano"); shot("mains_setup")
        pg.mouse.click(195, 40); pg.wait_for_timeout(300)
    # Mesure 1 sheet
    pg.evaluate("async()=>{const {openBar1Sheet}=await import('/static/screens/sections_editor.js'); openBar1Sheet()}"); pg.wait_for_timeout(700)
    english("Caler la mesure 1"); shot("mesure1")
    pg.get_by_role("button", name="Annuler").first.click(); pg.wait_for_timeout(300)
    # sections editor
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('sections')}"); pg.wait_for_timeout(1000)
    english("Sections"); shot("sections")
    # loading screen, every phase + error
    for ph in ("listening", "decoding", "head", "raw", "sections", "done", "error"):
        pg.evaluate("""async(ph)=>{const {S}=await import('/static/state.js'); const {go}=await import('/static/router.js');
          const A=await import('/static/screens/analyse.js');
          S.job={title:'', stage:2, phase:ph==='error'?'listening':ph, error:ph==='error'?'boom':null, file:null, n_bars:32, sections_found:3, raw_model:(ph==='raw'||ph==='sections'||ph==='done')?S.model:null, file:S.model&&S.model.file};
          if(S.screen!=='analysing') go('analysing'); else A.paintLoading();
          A.paintLoading()}""", ph)
        pg.wait_for_timeout(500); english("Chargement · " + ph); shot("load_" + ph)
    b.close()
check(not errs, f"no JS console error {errs[:3]}")
print("OK" if not fails else "FAIL"); sys.exit(1 if fails else 0)
