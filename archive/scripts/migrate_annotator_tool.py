#!/usr/bin/env python3
"""Migrate existing baked charts AWAY FROM the iframe-hosted Annotate tab.

2026-07-16: reverted. The Annotate tab briefly loaded a standalone
dark-themed waveform tool (`/annotator`, from the older "Waveform V4" /
gt-align work) in a full-screen iframe, hiding this page's own on-brand
tap/long-press chord editor. That was jarring and inconsistent with the
Read/Analyse tabs' cream/maroon styling right above it (see
docs/handoff_2026-07-13_annotator_ui.md for the actual design intent — the
in-place Wheel/Suggestions editor IS the annotate tool). This migration now
does the reverse of what it used to:

1. **Remove** the `/annotator` topbar icon-link, if a chart still has it.
2. **Restore** `setViewMode` to the plain 3-line body that just toggles the
   existing tap-to-fix editor (no iframe, no slug extraction).
3. **Remove** the `#annotation-tool-container` overlay div, if present.

All three edits match the exact strings emitted by
`harmonia/output/chart_interactive.py`, so re-running is a no-op once a
chart is up to date, and running this on a chart that predates the iframe
tab entirely (i.e. already has the plain setViewMode and no icon-link) is
also a no-op.

Usage:
    python scripts/migrate_annotator_tool.py
"""
from pathlib import Path
import re


# ── setViewMode payloads — kept byte-identical to chart_interactive.py ───────

_PLAIN_SETVIEWMODE = """  window.setViewMode = function(id){
    window.harmViewMode = id; paint();
    setAnnotate(id==='annotate');
    if(typeof render==='function') render();  // re-render, then applyViewMode runs via the patch
    else window.applyViewMode();
  };"""

_IFRAME_SETVIEWMODE_RE = re.compile(
    r"[ \t]*/\*__HARM_ANNOTATE_TAB__\*/.*?/\*__END_HARM_ANNOTATE_TAB__\*/", re.S)

_ICON_LINK_RE = re.compile(
    r'\s*<a href="/annotator\?song=[^"]*" class="icon-btn"[^>]*>.*?</a>\n?', re.S)

_CONTAINER_RE = re.compile(
    r'(?:<!-- Annotate tab:.*?-->\n)?'
    r'<div id="annotation-tool-container"[^>]*>\s*'
    r'<iframe id="annotation-tool-iframe"[^>]*></iframe>\s*'
    r'</div>\n?', re.S)


def extract_slug_from_filename(filename: str) -> str:
    """Extract slug from filename pattern like 'inferred_autumn_leaves.html'."""
    stem = Path(filename).stem
    slug = "_".join(stem.split("_")[1:]) if "_" in stem else stem
    return slug


def migrate_chart(html_path: Path) -> bool:
    """Strip the iframe-hosted Annotate tab out of a single baked chart.
    Returns True if anything changed."""
    html = html_path.read_text(encoding="utf-8")
    original = html

    # ── 1. Remove the /annotator topbar icon-link, if present ───────────────
    html = _ICON_LINK_RE.sub("", html)

    # Splices 2/3 only apply to real interactive chord charts (the ones that
    # build the 3-mode segmented control). Diagnostic plots and the iReal
    # reference pages have no #harm-modebar, so leave them untouched.
    is_chart = "'harm-modebar'" in html or 'id="harm-modebar"' in html

    if is_chart:
        # ── 2. Restore the plain (non-iframe) setViewMode ────────────────────
        if "/*__HARM_ANNOTATE_TAB__*/" in html:
            # Consume the leading indent too, so re-runs don't drift indentation.
            html = _IFRAME_SETVIEWMODE_RE.sub(
                lambda _m: _PLAIN_SETVIEWMODE, html, count=1)

        # ── 3. Remove the annotation-tool overlay container ──────────────────
        if 'id="annotation-tool-container"' in html:
            html = _CONTAINER_RE.sub("", html)

    if html != original:
        html_path.write_text(html, encoding="utf-8")
        return True
    return False


def main():
    """Migrate all interactive HTML charts."""
    docs_plots = Path("docs/plots")

    if not docs_plots.exists():
        print("Error: docs/plots directory not found")
        return

    html_files = sorted(docs_plots.glob("*.html"))

    if not html_files:
        print("No HTML files found in docs/plots")
        return

    updated_count = 0
    skipped_count = 0

    for html_path in html_files:
        if migrate_chart(html_path):
            print(f"✓ Updated {html_path.name}")
            updated_count += 1
        else:
            print(f"  Skipped {html_path.name}")
            skipped_count += 1

    print(f"\nMigration complete: {updated_count} updated, {skipped_count} skipped")


if __name__ == "__main__":
    main()
