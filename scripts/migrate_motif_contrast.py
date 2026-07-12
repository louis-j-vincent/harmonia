"""One-off migration: fix low-contrast text in the dark motif UI on already-
rendered chart HTML (UX axis 4 of 4). #4a6a88 on #1a2233/#0d1117 measured
2.81:1 / 3.34:1 — both fail WCAG AA (4.5:1) for normal-size text. Replaced
with #7590b0 (4.83:1 / 5.74:1 on the same backgrounds), same muted
blue-grey character, just light enough to read.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if "#4a6a88" not in text:
            skipped += 1
            continue
        n = text.count("#4a6a88")
        f.write_text(text.replace("#4a6a88", "#7590b0"), encoding="utf-8")
        patched += 1
        print(f"patched {f.name} ({n} occurrences)")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
