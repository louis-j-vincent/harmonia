#!/usr/bin/env python3
"""Migrate existing baked charts to the iframe-hosted Annotate tab.

Two independent, idempotent splices are applied to every chart in docs/plots:

1. **Annotate-alignment topbar button** — the original migration: inserts the
   `/annotator` icon-link into charts that predate it.

2. **Annotate *tab* → iframe overlay** — the current work: the 3-mode
   segmented control's `setViewMode('annotate')` used to only toggle the old
   tap-to-fix chord editor. It now loads the full `/annotator?song=<slug>`
   tool in a full-screen iframe overlay. This splice:
     - replaces the old `window.setViewMode` body with the new one (which adds
       `extractSlug()` + iframe show/hide + mode-bar float), and
     - injects the `#annotation-tool-container` overlay div before `</body>`.

   Both edits match the exact strings emitted by
   `harmonia/output/chart_interactive.py`, so re-running is a no-op once a
   chart is up to date.

Usage:
    python scripts/migrate_annotator_tool.py
"""
from pathlib import Path
import re


# ── Splice 2 payloads — kept byte-identical to chart_interactive.py ──────────

_OLD_SETVIEWMODE = """  window.setViewMode = function(id){
    window.harmViewMode = id; paint();
    setAnnotate(id==='annotate');
    if(typeof render==='function') render();  // re-render, then applyViewMode runs via the patch
    else window.applyViewMode();
  };"""

_NEW_SETVIEWMODE = """  /*__HARM_ANNOTATE_TAB__*/
  // Slug for the /annotator iframe: baked into every chart's payload as
  // P.slug; fall back to the filename convention (inferred_<slug>.html) for
  // any legacy chart whose payload predates the slug field.
  function extractSlug(){
    if(window.P && window.P.slug) return window.P.slug;
    const stem = (document.location.pathname.split('/').pop()||'').replace(/\\.html?$/,'');
    return stem.indexOf('_')>=0 ? stem.split('_').slice(1).join('_') : stem;
  }
  window.setViewMode = function(id){
    window.harmViewMode = id; paint();
    const container = document.getElementById('annotation-tool-container');
    const iframe    = document.getElementById('annotation-tool-iframe');
    const modebar   = document.getElementById('harm-modebar');
    if(id==='annotate'){
      // Load the isolated annotation tool over the chart. Only (re)assign src
      // when the slug changes so re-entering the tab keeps its state.
      const slug = extractSlug();
      if(iframe && iframe.getAttribute('data-slug') !== slug){
        iframe.src = '/annotator?song=' + encodeURIComponent(slug);
        iframe.setAttribute('data-slug', slug);
      }
      if(container) container.style.display = 'block';
      // Hide the server-injected chrome (back button + docked audio player) so
      // they don't sit above the overlay. Harmless no-ops on a bare file://.
      const back = document.getElementById('harm-back'); if(back) back.style.display='none';
      const dock = document.getElementById('yt-player-dock'); if(dock) dock.style.display='none';
      // Float the mode bar above the overlay so Read/Analyse remain the exit.
      if(modebar){ modebar.style.position='fixed'; modebar.style.top='0';
        modebar.style.left='0'; modebar.style.right='0'; modebar.style.zIndex='9991';
        modebar.style.margin='0'; modebar.style.padding='8px 12px';
        modebar.style.background='var(--paper,#f7f3e9)';
        modebar.style.boxShadow='0 2px 8px -4px rgba(0,0,0,.35)'; }
      // The old tap-to-fix editor is replaced by the iframe tool; keep it off.
      setAnnotate(false);
    } else {
      if(container) container.style.display = 'none';
      const back = document.getElementById('harm-back'); if(back) back.style.display='';
      const dock = document.getElementById('yt-player-dock'); if(dock) dock.style.display='';
      if(modebar){ modebar.style.position=''; modebar.style.top='';
        modebar.style.left=''; modebar.style.right=''; modebar.style.zIndex='';
        modebar.style.margin='6px auto 10px'; modebar.style.padding='0 12px';
        modebar.style.background=''; modebar.style.boxShadow=''; }
      setAnnotate(false);
    }
    if(typeof render==='function') render();  // re-render, then applyViewMode runs via the patch
    else window.applyViewMode();
  };
  /*__END_HARM_ANNOTATE_TAB__*/"""

_CONTAINER_HTML = """<!-- Annotate tab: full-screen overlay hosting the /annotator tool in an
     isolated iframe (no namespace collision with the chart). Shown/hidden by
     setViewMode('annotate'). The #harm-modebar is floated above this overlay
     so Read/Analyse stay tappable and act as the exit. -->
<div id="annotation-tool-container" style="display:none; width:100%; height:100%; position:fixed; inset:0; z-index:9990; background:var(--paper,#f7f3e9);">
  <iframe id="annotation-tool-iframe" title="Annotation tool" style="width:100%; height:100%; border:none;"></iframe>
</div>
</body></html>"""


def extract_slug_from_filename(filename: str) -> str:
    """Extract slug from filename pattern like 'inferred_autumn_leaves.html'."""
    stem = Path(filename).stem
    slug = "_".join(stem.split("_")[1:]) if "_" in stem else stem
    return slug


def migrate_chart(html_path: Path) -> bool:
    """Apply both splices to a single chart. Returns True if anything changed."""
    html = html_path.read_text(encoding="utf-8")
    original = html

    # ── Splice 1: Annotate-alignment topbar button ──────────────────────────
    if "/annotator" not in html:
        slug = extract_slug_from_filename(html_path.name)
        pattern = r'''(<div class="topbar">\s*<button[^>]*>\s*<span[^>]*>Key</span></button>\s*<h1>[^<]*</h1>)(\s*<button type="button" class="icon-btn" id="optionsBtn")'''
        annotate_btn = f'''    <a href="/annotator?song={slug}" class="icon-btn" aria-label="Annotate alignment" title="Annotate alignment">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z"/><path d="M20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
      </svg>
    </a>
'''
        html = re.sub(pattern, r'\1\n' + annotate_btn + r'\2', html)

    # Splices 2a/2b only apply to real interactive chord charts (the ones that
    # build the 3-mode segmented control). Diagnostic plots and the iReal
    # reference pages have no #harm-modebar, so leave them untouched.
    is_chart = "'harm-modebar'" in html or 'id="harm-modebar"' in html

    if is_chart:
        # ── Splice 2a: install / refresh the iframe-overlay setViewMode ──────
        # Sentinel-wrapped so re-runs replace an older injected version in
        # place (version-proof), while first-time charts match the original
        # 3-line setViewMode body.
        if "/*__HARM_ANNOTATE_TAB__*/" in html:
            # Consume the leading indent too — _NEW_SETVIEWMODE carries its own
            # 2-space indent, so not matching it would drift +2 spaces per run.
            html = re.sub(
                r"[ \t]*/\*__HARM_ANNOTATE_TAB__\*/.*?/\*__END_HARM_ANNOTATE_TAB__\*/",
                lambda _m: _NEW_SETVIEWMODE, html, count=1, flags=re.S)
        elif _OLD_SETVIEWMODE in html:
            html = html.replace(_OLD_SETVIEWMODE, _NEW_SETVIEWMODE)

        # ── Splice 2b: inject the annotation-tool overlay container ──────────
        if 'id="annotation-tool-container"' not in html and "</body></html>" in html:
            html = html.replace("</body></html>", _CONTAINER_HTML, 1)

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
