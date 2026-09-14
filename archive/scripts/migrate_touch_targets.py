"""One-off migration #5: bump .pill/.icon-btn to the 44x44pt Apple HIG
minimum touch target size on already-rendered chart HTML.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD = """  .pill { display:inline-flex; align-items:center; gap:5px; background:#efe9d9;
          border:1px solid #e2dac4; border-radius:20px; padding:8px 15px;
          font:700 13px system-ui,sans-serif; color:#4a4636; cursor:pointer;
          flex:0 0 auto; transition:transform .1s ease, background .12s; }
  .pill::after { content:"⌄"; opacity:.5; font-size:10px; }
  .pill:active { transform:scale(.94); background:#e2d9c2; }
  .icon-btn { display:inline-flex; align-items:center; justify-content:center;
              width:36px; height:36px; border-radius:50%; background:#efe9d9;
              border:1px solid #e2dac4; color:#4a4636; cursor:pointer; flex:0 0 auto;
              transition:transform .1s ease, background .12s; }"""

NEW = """  .pill { display:inline-flex; align-items:center; gap:5px; background:#efe9d9;
          border:1px solid #e2dac4; border-radius:22px; padding:8px 16px;
          min-height:44px; box-sizing:border-box;
          font:700 13px system-ui,sans-serif; color:#4a4636; cursor:pointer;
          flex:0 0 auto; transition:transform .1s ease, background .12s; }
  .pill::after { content:"⌄"; opacity:.5; font-size:10px; }
  .pill:active { transform:scale(.94); background:#e2d9c2; }
  /* 44×44pt is Apple's minimum recommended touch target */
  .icon-btn { display:inline-flex; align-items:center; justify-content:center;
              width:44px; height:44px; border-radius:50%; background:#efe9d9;
              border:1px solid #e2dac4; color:#4a4636; cursor:pointer; flex:0 0 auto;
              transition:transform .1s ease, background .12s; }"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD not in text:
            skipped += 1
            continue
        f.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
