"""One-off migration #7: shrink multi-chord bars (e.g. "Aø7 D7♭9") so they
don't wrap onto a second line and blow out that grid row's height, on
already-rendered chart HTML.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_BASE = """  .chord .root { font-size:27px; font-style:italic; }
  .chord .qual { font-size:17px; font-style:italic; }
  .chord sup { font-size:.62em; }
  .chord .acc { font-size:.6em; margin-left:-.1em; vertical-align:.12em; }"""

NEW_BASE = """  .chord .root { font-size:27px; font-style:italic; }
  .chord .qual { font-size:17px; font-style:italic; }
  .chord sup { font-size:.62em; }
  .chord .acc { font-size:.6em; margin-left:-.1em; vertical-align:.12em; }
  /* a bar with 2+ chords shrinks so it doesn't wrap onto a second line and
     blow out that entire grid row's height (every measure in the row grows
     to match the tallest one) */
  .measure:has(.chords > .chord:nth-child(2)) .chords { gap:8px; }
  .measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:19px; }
  .measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:12px; }"""

OLD_M640 = """    .chord .root { font-size:24px; }
    .chord .qual { font-size:15px; }
    .seclabel { width:15px; height:15px; font-size:9px; }"""
NEW_M640 = """    .chord .root { font-size:24px; }
    .chord .qual { font-size:15px; }
    .measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:15px; }
    .measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:10px; }
    .seclabel { width:15px; height:15px; font-size:9px; }"""

OLD_M360 = """    .chord .root { font-size:30px; }
    .chord .qual { font-size:19px; }
  }"""
NEW_M360 = """    .chord .root { font-size:30px; }
    .chord .qual { font-size:19px; }
    .measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:19px; }
    .measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:12px; }
  }"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD_BASE not in text:
            skipped += 1
            continue
        text = text.replace(OLD_BASE, NEW_BASE, 1)
        if OLD_M640 in text:
            text = text.replace(OLD_M640, NEW_M640, 1)
        if OLD_M360 in text:
            text = text.replace(OLD_M360, NEW_M360, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
