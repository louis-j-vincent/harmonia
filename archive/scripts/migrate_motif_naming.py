"""One-off migration: translate raw quality-bucket tokens ("dom7", "min7") to
real chord symbols ("7", "m7") inside unnamed-shape bracket notation
("[dom7 +11dom7]" -> "[7 +11 7]") baked into already-rendered chart JSON
payloads — the naming-inconsistency half of the "not clean" report (named
shapes like "ii-V" already used real symbols; unnamed ones didn't).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

QUAL_SYMBOL = {
    "maj": "", "min": "m", "dom7": "7", "maj7": "△7", "min7": "m7",
    "m7b5": "ø7", "dim7": "°7", "dim": "°", "6": "6", "sus4": "sus4", "aug": "+",
}
# longest-first so "dom7" doesn't get shadowed by a shorter alt, etc.
_ALTS = "|".join(re.escape(k) for k in sorted(QUAL_SYMBOL, key=len, reverse=True))
BRACKET_RE = re.compile(r"\[(" + _ALTS + r")((?:\s\+\d+(?:" + _ALTS + r"))*)\]")
TAIL_RE = re.compile(r"\+(\d+)(" + _ALTS + r")")


def _translate(m: re.Match) -> str:
    head = QUAL_SYMBOL[m.group(1)]
    # space before the quality symbol — "+11 7" not "+117", which misreads
    # as the single number "117" once the symbol itself is numeric ("7")
    tail = TAIL_RE.sub(lambda mm: f"+{mm.group(1)} {QUAL_SYMBOL[mm.group(2)]}", m.group(2))
    return f"[{head}{tail}]"


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        new_text, n = BRACKET_RE.subn(_translate, text)
        if n == 0:
            skipped += 1
            continue
        f.write_text(new_text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name} ({n} bracket names)")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
