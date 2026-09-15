"""One-off migration #6: enlarge the rotor on already-rendered chart HTML —
it now lives in its own bottom sheet instead of a cramped toolbar, so there's
room for it to be bigger. Also drops the now-unnecessary mobile down-sizing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_CSS = """  .wheel { position:relative; width:112px; height:112px; border-radius:50%;
           flex:0 0 auto; touch-action:none; -webkit-user-select:none; user-select:none;
           background:radial-gradient(circle at 50% 42%,#f2ead4 0%,#ddd0ac 72%,#c3b48c 100%);
           box-shadow:inset 0 1px 4px #0003, 0 1px 2px #0002; }
  .wheel-ring { position:absolute; inset:0; transform-origin:50% 50%; will-change:transform; }
  .wheel-ring.snap { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel-ring.snap .lbl { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel button { position:absolute; left:50%; top:50%; width:31px; height:31px;
                  margin:-15.5px 0 0 -15.5px; border-radius:50%; padding:0;
                  border:1px solid #0003; color:#242018; font:700 11px system-ui,sans-serif;
                  cursor:grab; box-shadow:0 1px 2px #0002; -webkit-tap-highlight-color:transparent; }
  .wheel button .lbl { display:block; pointer-events:none; }
  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;
                                     color:#111; }
  .wheel-pointer { position:absolute; top:-6px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:6px solid transparent;
                   border-right:6px solid transparent; border-bottom:9px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }
  @keyframes wheelTick { 0%{transform:translateX(-50%) scale(1);}
                         45%{transform:translateX(-50%) scale(1.5);}
                         100%{transform:translateX(-50%) scale(1);} }
  .wheel-pointer.tick { animation:wheelTick .16s ease-out; }
  .wheel .hub { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                width:39px; height:39px; border-radius:50%; background:#f7f3e9e8;
                border:1px solid #cfc7ae; display:flex; align-items:center;
                justify-content:center; font:700 11px system-ui,sans-serif; color:#4a4636;
                pointer-events:none; box-shadow:inset 0 1px 2px #0002; }
  .transposeLabel { min-width:74px; font:700 12px system-ui,sans-serif; color:#4a4636; }"""

NEW_CSS = """  /* Sized generously — the rotor now has its own dedicated sheet instead of
     competing for space in a toolbar, so there's no reason to keep it small. */
  .wheel { position:relative; width:176px; height:176px; border-radius:50%;
           flex:0 0 auto; touch-action:none; -webkit-user-select:none; user-select:none;
           background:radial-gradient(circle at 50% 42%,#f2ead4 0%,#ddd0ac 72%,#c3b48c 100%);
           box-shadow:inset 0 1px 4px #0003, 0 1px 2px #0002; }
  .wheel-ring { position:absolute; inset:0; transform-origin:50% 50%; will-change:transform; }
  .wheel-ring.snap { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel-ring.snap .lbl { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel button { position:absolute; left:50%; top:50%; width:49px; height:49px;
                  margin:-24.5px 0 0 -24.5px; border-radius:50%; padding:0;
                  border:1px solid #0003; color:#242018; font:700 16px system-ui,sans-serif;
                  cursor:grab; box-shadow:0 1px 2px #0002; -webkit-tap-highlight-color:transparent; }
  .wheel button .lbl { display:block; pointer-events:none; }
  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;
                                     color:#111; }
  .wheel-pointer { position:absolute; top:-9px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:9px solid transparent;
                   border-right:9px solid transparent; border-bottom:13px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }
  @keyframes wheelTick { 0%{transform:translateX(-50%) scale(1);}
                         45%{transform:translateX(-50%) scale(1.5);}
                         100%{transform:translateX(-50%) scale(1);} }
  .wheel-pointer.tick { animation:wheelTick .16s ease-out; }
  .wheel .hub { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                width:61px; height:61px; border-radius:50%; background:#f7f3e9e8;
                border:1px solid #cfc7ae; display:flex; align-items:center;
                justify-content:center; font:700 16px system-ui,sans-serif; color:#4a4636;
                pointer-events:none; box-shadow:inset 0 1px 2px #0002; }
  .transposeLabel { min-width:90px; font:700 15px system-ui,sans-serif; color:#4a4636; }"""

OLD_MOBILE = """    /* the song info is duplicated compactly at the top of the Options sheet */
    .subhead { display:none; }
    .wheel { width:84px; height:84px; }
    .wheel button { width:24px; height:24px; margin:-12px 0 0 -12px; font-size:9px; }
    .wheel .hub { width:30px; height:30px; font-size:9px; }
    .transposeLabel { min-width:0; font-size:11px; }
    .legend .bar { width:70px; }"""

NEW_MOBILE = """    /* the song info is duplicated compactly at the top of the Options sheet */
    .subhead { display:none; }
    .legend .bar { width:70px; }"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD_CSS not in text:
            skipped += 1
            continue
        text = text.replace(OLD_CSS, NEW_CSS, 1)
        if OLD_MOBILE in text:
            text = text.replace(OLD_MOBILE, NEW_MOBILE, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
