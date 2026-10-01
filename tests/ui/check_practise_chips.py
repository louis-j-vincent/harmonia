"""UX #13b (2026-10-01): S'entraîner's loop-length and speed chips are 24 px
tall; their TAP AREA must be >= 44 x 44 (an invisible extension, the transport
row must not grow), the two zones must not overlap, and a tap on the extension
(outside the visible pill) must act. Chromium AND WebKit, 390x844, touch.

    .venv/bin/python tests/ui/check_practise_chips.py [file]
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
    pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/**", lambda r: r.fulfill(status=200, body="{}", content_type="application/json") if r.request.method == "POST" else r.continue_())
    pg.goto(f"http://localhost:7772/?open={FILE}"); pg.wait_for_selector("[data-bar]", timeout=20000, state="attached"); pg.wait_for_timeout(1500)
    pg.evaluate("localStorage.removeItem('harmLoopBars')")
    pg.evaluate("async()=>{const {go}=await import('/static/router.js'); go('prompter');}"); pg.wait_for_timeout(1500)
    # the hit zone of a chip = the union of its own box and its descendants' boxes
    zone = """(t)=>{const b=document.querySelector(`button[title^="${t}"]`); let r=b.getBoundingClientRect();
      let top=r.top,bot=r.bottom,l=r.left,rt=r.right;
      b.querySelectorAll('*').forEach(c=>{const q=c.getBoundingClientRect(); top=Math.min(top,q.top);bot=Math.max(bot,q.bottom);l=Math.min(l,q.left);rt=Math.max(rt,q.right);});
      return {top,bot,l,rt,h:bot-top,w:rt-l,pillH:r.height,rowH:b.closest('div').parentElement.getBoundingClientRect().height,cx:(l+rt)/2}}"""
    A = pg.evaluate(zone, "Longueur"); B = pg.evaluate(zone, "Vitesse")
    check(A["h"] >= 44 and A["w"] >= 44, tag + f"length chip tap area {A['w']:.0f}x{A['h']:.0f} >= 44")
    check(B["h"] >= 44 and B["w"] >= 44, tag + f"speed chip tap area {B['w']:.0f}x{B['h']:.0f} >= 44")
    check(A["bot"] <= B["top"] + 0.5, tag + "the two zones do not overlap")
    check(A["pillH"] < 30, tag + "the visible pill keeps its compact size")
    transport = pg.evaluate("()=>{const b=document.querySelector('button[title^=\"Longueur\"]'); return b.parentElement.parentElement.getBoundingClientRect().height}")
    check(transport <= 64.5, tag + f"transport row did not grow ({transport:.0f} px)")
    # tap OUTSIDE the visible pill but inside the zone: top edge of length zone, bottom edge of speed zone
    rl = pg.evaluate("()=>document.querySelector('button[title^=\"Longueur\"]').getBoundingClientRect().top")
    rs = pg.evaluate("()=>document.querySelector('button[title^=\"Vitesse\"]').getBoundingClientRect().bottom")
    t0 = pg.locator("button[title^=\"Longueur\"]").inner_text(); s0 = pg.locator("button[title^=\"Vitesse\"]").inner_text()
    pg.touchscreen.tap(A["cx"], rl - 8); pg.wait_for_timeout(300)
    check(pg.locator("button[title^=\"Longueur\"]").inner_text() != t0, tag + "tap 8 px above the length pill changes the length")
    pg.touchscreen.tap(B["cx"], rs + 8); pg.wait_for_timeout(300)
    check(pg.locator("button[title^=\"Vitesse\"]").inner_text() != s0, tag + "tap 8 px below the speed pill changes the speed")
    pg.screenshot(path=f"/tmp/chips_{name}.png")
    check(not errs, tag + f"no page error {errs}")
    b.close()
print("FAILED" if fails else "ALL OK"); sys.exit(1 if fails else 0)
