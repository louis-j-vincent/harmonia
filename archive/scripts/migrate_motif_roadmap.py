"""One-off migration: add the song-structure roadmap strip to already-
rendered chart HTML (UX axis 1 — "simplify the song at a glance").
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_HTML = """  <div id="motif-overlay">
    <div id="motifpanel">
      <label>Mode"""
NEW_HTML = """  <div id="motif-overlay">
    <div id="motifpanel">
      <div id="motifroadmap" aria-label="Song structure at a glance"></div>
      <label>Mode"""

OLD_CSS = """  #motifpanel label { display:flex; align-items:center; gap:6px; color:#a0b8d0; }"""
NEW_CSS = """  /* Song-structure-at-a-glance: the whole song compressed to its recurring
     motifs in sequence (e.g. "A A A' B") — the practical payoff of the
     motif-detection work, meant to be legible in one look, no scrolling
     the chart to piece it together. Non-motif stretches collapse to a
     single neutral dash instead of one chip per passing chord. */
  #motifroadmap { width:100%; display:flex; align-items:center; gap:5px;
                  overflow-x:auto; padding-bottom:2px; -webkit-overflow-scrolling:touch; }
  #motifroadmap:empty { display:none; }
  .roadmap-chip { flex:0 0 auto; padding:4px 10px; border-radius:8px;
                  font:700 11px system-ui,sans-serif; white-space:nowrap;
                  cursor:pointer; transition:filter .15s,transform .15s; }
  .roadmap-chip:hover { filter:brightness(1.3); transform:scale(1.06); }
  .roadmap-gap { flex:0 0 auto; width:14px; height:1px; background:#2a3a50; }
  #motifpanel label { display:flex; align-items:center; gap:6px; color:#a0b8d0; }"""

OLD_CLEAR = """  document.getElementById('motif-svg-overlay')?.remove();
  document.getElementById('motiflegend').innerHTML = '';
  document.getElementById('motifstats').innerHTML = '';"""
NEW_CLEAR = """  document.getElementById('motif-svg-overlay')?.remove();
  document.getElementById('motiflegend').innerHTML = '';
  document.getElementById('motifstats').innerHTML = '';
  document.getElementById('motifroadmap').innerHTML = '';"""

OLD_ANCHOR = """    parent.insertBefore(seg, els[0]);
    els.forEach(el => seg.appendChild(el));
  });"""
NEW_ANCHOR = """    parent.insertBefore(seg, els[0]);
    els.forEach(el => seg.appendChild(el));
  });

  // Roadmap — the whole song compressed to its recurring motifs in sequence,
  // legible in one look instead of piecing it together by scrolling the
  // chart. Non-motif stretches collapse to a single dash, not one chip per
  // passing chord — the point is to show what repeats, not to re-draw the
  // whole chart in miniature.
  (function renderRoadmap(){
    const road = document.getElementById('motifroadmap');
    let i = 0, lastWasGap = false;
    while (i < ann.length) {
      const a = ann[i];
      if (a && a.pos === 0) {
        const chip = document.createElement('span');
        chip.className = 'roadmap-chip';
        const col = motifColor(a.name);
        chip.style.background = col + '2a';
        chip.style.color = col;
        chip.textContent = a.name;
        chip.title = `${a.name} — ×${a.count}`;
        chip.addEventListener('mouseenter', () => {
          document.querySelectorAll('.motif-segment').forEach(seg => {
            seg.classList.toggle('motif-hi', seg.dataset.motifName === a.name);
          });
        });
        chip.addEventListener('mouseleave', () => {
          document.querySelectorAll('.motif-segment.motif-hi').forEach(seg => seg.classList.remove('motif-hi'));
        });
        road.appendChild(chip);
        i += a.len;
        lastWasGap = false;
      } else {
        if (!lastWasGap) {
          const gap = document.createElement('span');
          gap.className = 'roadmap-gap';
          road.appendChild(gap);
          lastWasGap = true;
        }
        i += 1;
      }
    }
  })();"""

REPLACEMENTS = [(OLD_HTML, NEW_HTML), (OLD_CSS, NEW_CSS), (OLD_CLEAR, NEW_CLEAR), (OLD_ANCHOR, NEW_ANCHOR)]


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = sum(1 for old, _ in REPLACEMENTS if old in text)
        if hits == 0:
            skipped += 1
            continue
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name} ({hits}/{len(REPLACEMENTS)} blocks)")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
