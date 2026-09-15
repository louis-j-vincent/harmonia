"""One-off migration: fix motif colours on already-rendered chart HTML.

Old bug: colour = order of first appearance in the greedy tiling (i % 8),
so "ii-V" was a different colour in every song, and two unrelated motifs
9+ apart in the same song silently shared a colour. Colour/family are now
computed client-side from the motif's *name* (a fixed lookup, grouped by
harmonic family), so this migration only needs to patch the JS — the
underlying motif names in the payload were already correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_CSS = """  #motiflegend { width:100%; display:flex; flex-wrap:wrap; gap:8px;
                 padding-top:8px; border-top:1px solid #1a2a3a; margin-top:4px; }
  #motiflegend .item { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
    border-radius:20px; cursor:pointer; font:600 12px system-ui,sans-serif;
    border:1px solid transparent; transition:filter .2s,transform .2s; }
  #motiflegend .item:hover { filter:brightness(1.3); transform:scale(1.05); }
  #motiflegend .item .sw { width:9px; height:9px; border-radius:50%; flex:0 0 auto; }
  #motiflegend .item .cnt { font-size:10px; opacity:.6; font-weight:400; }"""
NEW_CSS = """  #motiflegend { width:100%; display:flex; flex-direction:column; gap:7px;
                 padding-top:8px; border-top:1px solid #1a2a3a; margin-top:4px; }
  .motif-fam-group { display:flex; align-items:center; flex-wrap:wrap; gap:6px; }
  .motif-fam-label { font:700 10px system-ui,sans-serif; text-transform:uppercase;
    letter-spacing:.05em; color:#4a6a88; flex:0 0 auto; margin-right:2px; min-width:112px; }
  #motiflegend .item { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
    border-radius:20px; cursor:pointer; font:600 12px system-ui,sans-serif;
    border:1px solid transparent; transition:filter .2s,transform .2s; }
  #motiflegend .item:hover { filter:brightness(1.3); transform:scale(1.05); }
  #motiflegend .item .sw { width:9px; height:9px; border-radius:50%; flex:0 0 auto; }
  #motiflegend .item .cnt { font-size:10px; opacity:.6; font-weight:400; }"""

OLD_TABLE = """let motifModeActive = false;

// Derive per-colour CSS vars for a hex colour (bg, fg, glow, border)
function motifColorVars(hex) {
  return `--mc-bg:${hex}22;--mc-fg:${hex};--mc-glow:${hex}88;--mc-border:${hex}44`;
}"""
NEW_TABLE = """let motifModeActive = false;

// Derive per-colour CSS vars for a hex colour (bg, fg, glow, border)
function motifColorVars(hex) {
  return `--mc-bg:${hex}22;--mc-fg:${hex};--mc-glow:${hex}88;--mc-border:${hex}44`;
}

// Fixed colour per *named* shape, grouped by harmonic family — "ii-V" is
// always cyan, in every song, instead of whichever hue happened to be next
// in rotation (the old bug: colour = order of first appearance, so it
// silently collided past 8 motifs and never meant the same thing twice).
// Computed client-side from the motif's name so it applies uniformly
// whether this chart was rendered before or after this fix.
const MOTIF_PALETTE = ["#00d4ff","#39ff14","#ff6ec7","#ffe400","#bf5fff","#ff4444","#00ffb3","#ff8c00"];
const SHAPE_FAMILY = {
  "ii-V":"Cadential","ii-V-I":"Cadential","ii-V ii-V":"Cadential",
  "V-I":"Resolution",
  "I-IV":"Subdominant",
  "I-VI":"Turnaround",
  "vi-ii":"Diatonic chain",
  "V/V-V":"Secondary dominant","dom-cycle":"Secondary dominant",
  "+4th":"Root: 4ths/5ths","+5th":"Root: 4ths/5ths",
  "+4th chain":"Root: 4ths/5ths","4th cycle":"Root: 4ths/5ths",
  "+2nd":"Root: steps","-2nd":"Root: steps","+½step":"Root: steps","-½step":"Root: steps",
  "+m3rd":"Root: 3rds","+M3rd":"Root: 3rds",
  "tritone":"Root: tritone",
};
const FAMILY_COLOR = {
  "Cadential":"#00d4ff","Resolution":"#39ff14","Subdominant":"#ff8c00",
  "Turnaround":"#bf5fff","Diatonic chain":"#00ffb3","Secondary dominant":"#ff4444",
  "Root: 4ths/5ths":"#00d4ff","Root: steps":"#ffe400","Root: 3rds":"#bf5fff",
  "Root: tritone":"#ff4444",
};
function motifFamily(name) { return SHAPE_FAMILY[name] || "Other patterns"; }
function motifColor(name) {
  if (SHAPE_FAMILY[name]) return FAMILY_COLOR[SHAPE_FAMILY[name]];
  // unnamed shape or a literal ("exact") repeat — no fixed identity to
  // anchor a colour to, so hash the name into the palette deterministically
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return MOTIF_PALETTE[Math.abs(h) % MOTIF_PALETTE.length];
}"""

OLD_SEG = """    seg.dataset.motifName  = a.name;
    seg.dataset.motifRunId = rid;
    seg.style.setProperty('--chip-color', a.color);
    seg.style.setProperty('--mc-bg',      a.color + '18');
    seg.style.setProperty('--mc-fg',      a.color);
    seg.style.setProperty('--mc-glow',    a.color + '99');
    seg.style.setProperty('--mc-border',  a.color + '55');"""
NEW_SEG = """    seg.dataset.motifName  = a.name;
    seg.dataset.motifRunId = rid;
    const segColor = motifColor(a.name);
    seg.style.setProperty('--chip-color', segColor);
    seg.style.setProperty('--mc-bg',      segColor + '18');
    seg.style.setProperty('--mc-fg',      segColor);
    seg.style.setProperty('--mc-glow',    segColor + '99');
    seg.style.setProperty('--mc-border',  segColor + '55');"""

OLD_LEGEND = """  const leg = document.getElementById('motiflegend');
  layer.legend.forEach(m => {
    const item = document.createElement('span');
    item.className = 'item';
    item.style.background = m.color + '18';
    item.style.borderColor = m.color + '44';
    item.style.color = m.color;
    item.innerHTML = `<span class="sw" style="background:${m.color}"></span>`
      + `${m.name} <span class="cnt">×${m.count} &minus;${m.saving}</span>`;
    item.addEventListener('mouseenter', () => {
      document.querySelectorAll('.motif-segment').forEach(seg => {
        seg.classList.toggle('motif-hi', seg.dataset.motifName === m.name);
      });
      document.querySelectorAll('.motif-svg-rect').forEach(r => {
        r.classList.toggle('motif-svg-hi', r.dataset.motifName === m.name);
      });
    });
    item.addEventListener('mouseleave', () => {
      document.querySelectorAll('.motif-segment.motif-hi').forEach(seg => seg.classList.remove('motif-hi'));
      document.querySelectorAll('.motif-svg-rect.motif-svg-hi').forEach(r => r.classList.remove('motif-svg-hi'));
    });
    leg.appendChild(item);
  });"""
NEW_LEGEND = """  const leg = document.getElementById('motiflegend');
  const families = [];
  const byFamily = {};
  layer.legend.forEach(m => {
    const fam = motifFamily(m.name);
    if (!byFamily[fam]) { byFamily[fam] = []; families.push(fam); }
    byFamily[fam].push(m);
  });
  families.forEach(fam => {
    const group = document.createElement('div');
    group.className = 'motif-fam-group';
    const label = document.createElement('span');
    label.className = 'motif-fam-label';
    label.textContent = fam;
    group.appendChild(label);
    byFamily[fam].forEach(m => {
      const col = motifColor(m.name);
      const item = document.createElement('span');
      item.className = 'item';
      item.style.background = col + '18';
      item.style.borderColor = col + '44';
      item.style.color = col;
      item.innerHTML = `<span class="sw" style="background:${col}"></span>`
        + `${m.name} <span class="cnt">×${m.count} &minus;${m.saving}</span>`;
      item.addEventListener('mouseenter', () => {
        document.querySelectorAll('.motif-segment').forEach(seg => {
          seg.classList.toggle('motif-hi', seg.dataset.motifName === m.name);
        });
        document.querySelectorAll('.motif-svg-rect').forEach(r => {
          r.classList.toggle('motif-svg-hi', r.dataset.motifName === m.name);
        });
      });
      item.addEventListener('mouseleave', () => {
        document.querySelectorAll('.motif-segment.motif-hi').forEach(seg => seg.classList.remove('motif-hi'));
        document.querySelectorAll('.motif-svg-rect.motif-svg-hi').forEach(r => r.classList.remove('motif-svg-hi'));
      });
      group.appendChild(item);
    });
    leg.appendChild(group);
  });"""

REPLACEMENTS = [(OLD_CSS, NEW_CSS), (OLD_TABLE, NEW_TABLE), (OLD_SEG, NEW_SEG), (OLD_LEGEND, NEW_LEGEND)]


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
