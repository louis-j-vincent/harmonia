"""One-off migration: fix the rotor's overlapping-circle bug on already-
rendered chart HTML. 12 buttons around a 176px ring at 49px each overlapped
(chord length between neighbours < button diameter), reading as blobs
instead of circles. Re-derived sizing so button+gap fits the 30° spacing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_CSS = """  /* Sized generously — the rotor now has its own dedicated sheet instead of
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

NEW_CSS = """  /* Sized so 12 buttons around the ring have real gaps between them — a
     button diameter has to stay well under the 30°-apart chord length or
     neighbours overlap and merge into blobs instead of reading as circles.
     radius (0.40×wheel, set in JS) × 2×sin(15°) ≥ button + gap. */
  .wheel { position:relative; width:250px; height:250px; border-radius:50%;
           flex:0 0 auto; touch-action:none; -webkit-user-select:none; user-select:none;
           background:radial-gradient(circle at 50% 42%,#f2ead4 0%,#ddd0ac 72%,#c3b48c 100%);
           box-shadow:inset 0 1px 4px #0003, 0 1px 2px #0002; }
  .wheel-ring { position:absolute; inset:0; transform-origin:50% 50%; will-change:transform; }
  .wheel-ring.snap { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel-ring.snap .lbl { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel button { position:absolute; left:50%; top:50%; width:44px; height:44px;
                  margin:-22px 0 0 -22px; border-radius:50%; padding:0;
                  border:1px solid #0003; color:#242018; font:700 15px system-ui,sans-serif;
                  cursor:grab; box-shadow:0 1px 2px #0002; -webkit-tap-highlight-color:transparent; }
  .wheel button .lbl { display:block; pointer-events:none; }
  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;
                                     color:#111; }
  .wheel-pointer { position:absolute; top:-10px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:10px solid transparent;
                   border-right:10px solid transparent; border-bottom:14px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }
  @keyframes wheelTick { 0%{transform:translateX(-50%) scale(1);}
                         45%{transform:translateX(-50%) scale(1.5);}
                         100%{transform:translateX(-50%) scale(1);} }
  .wheel-pointer.tick { animation:wheelTick .16s ease-out; }
  .wheel .hub { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                width:80px; height:80px; border-radius:50%; background:#f7f3e9e8;
                border:1px solid #cfc7ae; display:flex; align-items:center;
                justify-content:center; font:700 18px system-ui,sans-serif; color:#4a4636;
                pointer-events:none; box-shadow:inset 0 1px 2px #0002; }
  .transposeLabel { min-width:90px; font:700 15px system-ui,sans-serif; color:#4a4636; }"""

OLD_JS = """    const size=wheel.getBoundingClientRect().width||112;
    const radius=size*0.375;   // matches the original 42/112 ratio"""
NEW_JS = """    const size=wheel.getBoundingClientRect().width||250;
    const radius=size*0.40;   // wide enough that 12×44px buttons don't overlap"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD_CSS not in text:
            skipped += 1
            continue
        text = text.replace(OLD_CSS, NEW_CSS, 1)
        if OLD_JS in text:
            text = text.replace(OLD_JS, NEW_JS, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
