"""One-off migration: default motif style to "overlay" (Lab — keeps the warm
paper page, only the grid darkens) instead of "full" (Neon Lights — whole
page goes black), on already-rendered chart HTML. UX axis 3 of 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD = """// Motif style: "full" (Neon Lights) or "overlay" (Lab)
document.body.dataset.motifStyle = 'full';
const MOTIF_STYLE_LABELS = {full: '🌃 Neon Lights', overlay: '🔬 Lab'};
const motifStyleBtn = document.getElementById('motif-style-btn');
function updateStyleBtn(style) {
  const next = style === 'full' ? 'overlay' : 'full';
  motifStyleBtn.textContent = MOTIF_STYLE_LABELS[style];
  motifStyleBtn.title = 'Switch to ' + (next === 'full' ? 'Neon Lights' : 'Lab') + ' style';
}
updateStyleBtn('full');"""

NEW = """// Motif style: "full" (Neon Lights, whole page goes dark) or "overlay"
// ("Lab" — just the grid gets a dark treatment, the warm paper page stays).
// Default is "overlay": switching the *entire* page to a black theme the
// instant you tap Motifs was a jarring way to answer "show me the patterns"
// — Neon Lights is still one tap away for whoever wants the dramatic look.
document.body.dataset.motifStyle = 'overlay';
const MOTIF_STYLE_LABELS = {full: '🌃 Neon Lights', overlay: '🔬 Lab'};
const motifStyleBtn = document.getElementById('motif-style-btn');
function updateStyleBtn(style) {
  const next = style === 'full' ? 'overlay' : 'full';
  motifStyleBtn.textContent = MOTIF_STYLE_LABELS[style];
  motifStyleBtn.title = 'Switch to ' + (next === 'full' ? 'Neon Lights' : 'Lab') + ' style';
}
updateStyleBtn('overlay');"""


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
