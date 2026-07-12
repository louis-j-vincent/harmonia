"""Migrate the annotator tool (chord editor rotor + quality picker + section
merge) onto already-rendered chart HTML.

Rather than hand-copying the new HTML/CSS/JS blocks into OLD/NEW string
literals (error-prone for a change this size), this extracts them directly
from the current harmonia/output/chart_interactive.py at migration time,
using stable markers on either side of each new block. Can't drift from the
source of truth because it *is* the source of truth, read fresh each run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "harmonia" / "output" / "chart_interactive.py"
PLOTS_DIR = REPO / "docs" / "plots"

# Each triple: (marker before the new block, marker after it, sentinel that
# means "this file already has the new block, skip it").
SPLICES = [
    (
        'id="motif-style-btn" title="Switch motif style">🌃 Neon Lights</button>',
        '<div id="motif-overlay">',
        'chordEditModal',
    ),
    (
        '.chord .acc { font-size:.6em; margin-left:-.1em; vertical-align:.12em; }',
        '/* a bar with 2+ chords shrinks',
        'annotate-active',
    ),
    (
        'container.appendChild(btn);\n  });\n})();\n\n',
        'render();\n</script>\n</body></html>',
        'openChordEditor',
    ),
]

# Fallback start marker for the JS splice (index 2): older chart files
# predate the section-chips IIFE that SPLICES[2]'s normal start marker sits
# inside of, so anchor on the always-present "initMotifMode();" call instead.
JS_FALLBACK_START = 'initMotifMode();\n\n'

# Whole-block find/replace fixes, applied to every chart regardless of
# whether it has the annotator tool. (old_text, sentinel) — old_text is
# replaced by the equivalent live block in chart_interactive.py, found via
# the same before/after markers convention as SPLICES; sentinel means
# "already fixed, skip".
REPLACE_MARKERS = [
    (
        '  .wheel .wheel-ring button { position:absolute;',
        '  .wheel .wheel-ring button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;\n'
        '                                     color:#111; }',
        '.wheel .wheel-ring button',
    ),
]


def extract_new_blocks(src_text: str) -> list[str]:
    blocks = []
    for start_marker, end_marker, _sentinel in SPLICES:
        i = src_text.index(start_marker) + len(start_marker)
        j = src_text.index(end_marker, i)
        blocks.append(src_text[i:j])
    return blocks


def extract_replace_blocks(src_text: str) -> list[str]:
    blocks = []
    for start_marker, end_marker, _sentinel in REPLACE_MARKERS:
        i = src_text.index(start_marker)
        j = src_text.index(end_marker, i) + len(end_marker)
        blocks.append(src_text[i:j])
    return blocks


OLD_WHEEL_BUTTON_BLOCK = (
    '  .wheel button { position:absolute; left:50%; top:50%; width:44px; height:44px;\n'
    '                  margin:-22px 0 0 -22px; border-radius:50%; padding:0;\n'
    '                  border:1px solid #0003; color:#242018; font:700 15px system-ui,sans-serif;\n'
    '                  cursor:grab; box-shadow:0 1px 2px #0002; -webkit-tap-highlight-color:transparent; }\n'
    '  .wheel button .lbl { display:block; pointer-events:none; }\n'
    '  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;\n'
    '                                     color:#111; }'
)


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    new_blocks = extract_new_blocks(src_text)
    replace_blocks = extract_replace_blocks(src_text)

    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = 0
        for idx, ((start_marker, end_marker, sentinel), new_block) in enumerate(zip(SPLICES, new_blocks)):
            if sentinel in text:
                continue  # already migrated
            marker = start_marker
            if marker not in text and idx == 2 and JS_FALLBACK_START in text:
                marker = JS_FALLBACK_START
            if marker not in text or end_marker not in text:
                continue  # legacy file predating this whole CSS/JS lineage
            i = text.index(marker) + len(marker)
            j = text.index(end_marker, i)
            text = text[:i] + new_block + text[j:]
            hits += 1
        for (_start, _end, sentinel), new_block in zip(REPLACE_MARKERS, replace_blocks):
            if sentinel in text:
                continue  # already fixed
            if OLD_WHEEL_BUTTON_BLOCK not in text:
                continue  # doesn't have this exact stale block (already patched some other way, or predates it)
            text = text.replace(OLD_WHEEL_BUTTON_BLOCK, new_block, 1)
            hits += 1
        if hits == 0:
            skipped += 1
            continue
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name} ({hits} fixes)")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
