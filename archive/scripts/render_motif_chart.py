"""Render interactive motif-stacking HTML for one or more tunes.

Shows a lead-sheet-style grid where recurring motifs are colour-coded and
bracketed — you can switch between exact and transpose-invariant (shape) views,
hover over a motif to highlight all its copies, and see the song's compression
stats. Outputs self-contained HTML (no dependencies).

Usage:
    .venv/bin/python scripts/render_motif_chart.py "Anthropology"
    .venv/bin/python scripts/render_motif_chart.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models.motif import Chord, PC_NAMES, find_motifs, reduce_song  # noqa
from analyze_accomp_emission import parse_chord  # noqa

DB = REPO / "data" / "accomp_db" / "db.jsonl"

# Categorical palette (CVD-safe, from dataviz skill reference)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
           "#e34948", "#e87ba4", "#eb6834"]


def load_chords(rec: dict) -> list[Chord]:
    chords = []
    for e in rec["chord_timeline"]:
        p = parse_chord(e["mma"])
        if p is None:
            continue
        chords.append(Chord(root=p[0], qual=p[1], label=e["ireal"], bar=e["bar"]))
    return chords


def _ireal_html(label: str) -> str:
    """iReal token → clean HTML (flats as ♭, triangle as △, h as ø, - as m)."""
    # root + accidental
    m = re.match(r"([A-G])([b#]?)(.*)", label)
    if not m:
        return label
    root = m.group(1) + ({"b": "♭", "#": "♯"}.get(m.group(2), ""))
    qual = m.group(3)
    qual = qual.replace("^", "△").replace("-", "m")
    qual = re.sub(r"h", "ø", qual)
    # handle bass notes: /Xb → /X♭
    qual = re.sub(r"/([A-G])b", r"/\1♭", qual)
    qual = re.sub(r"/([A-G])#", r"/\1♯", qual)
    # flatten remaining b in quality as ♭ (e.g. b9 → ♭9)
    qual = re.sub(r"b(\d)", r"♭\1", qual)
    return f'<span class="root">{root}</span><span class="qual">{qual}</span>'


def build_song_data(rec: dict) -> dict:
    """Build JSON-serializable data for one song, both views."""
    chords = load_chords(rec)
    if not chords:
        return None
    bpb = rec.get("beats_per_bar", 4)
    n_bars = rec["n_bars"]
    spb = rec.get("section_per_bar", [])

    # Build both views
    views = {}
    for mode in ("exact", "shape"):
        is_shape = mode == "shape"
        motifs = find_motifs(chords, shape=is_shape, min_len=2, max_len=8, min_count=2)
        timeline, used = reduce_song(chords, shape=is_shape, min_len=2, max_len=8)

        # Assign colour per unique motif (by order of appearance)
        motif_colours = {}
        for i, m in enumerate(used):
            motif_colours[m.key] = PALETTE[i % len(PALETTE)]

        # Build per-chord annotation (which motif, colour, etc.)
        chord_ann = [None] * len(chords)
        for kind, obj, start in timeline:
            if kind == "motif":
                col = motif_colours.get(obj.key, "#999")
                for k in range(obj.length):
                    chord_ann[start + k] = {
                        "motif_key": str(obj.key),
                        "display": obj.display,
                        "colour": col,
                        "pos_in_motif": k,
                        "motif_len": obj.length,
                        "count": obj.count,
                    }

        legend = []
        for m in used:
            legend.append({
                "display": m.display,
                "colour": motif_colours[m.key],
                "count": m.count,
                "length": m.length,
                "saving": m.saving,
                "keys": m.keys[:8] if is_shape else [],
            })

        views[mode] = {
            "chord_ann": chord_ann,
            "legend": legend,
            "n_units": len(timeline),
            "n_unique": len(used),
        }

    # Chord grid by bar
    bars_data = []
    ci = 0
    for b in range(1, n_bars + 1):
        bar_chords = []
        while ci < len(chords) and chords[ci].bar == b:
            bar_chords.append({
                "label": chords[ci].label,
                "html": _ireal_html(chords[ci].label),
                "idx": ci,
            })
            ci += 1
        sec = spb[b - 1] if b - 1 < len(spb) else ""
        bars_data.append({"bar": b, "chords": bar_chords, "section": sec})

    return {
        "title": rec["title"],
        "form": rec.get("form", ""),
        "key": rec.get("key", ""),
        "n_chords": len(chords),
        "n_bars": n_bars,
        "bars": bars_data,
        "views": views,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motif Stacking — %(title)s</title>
<style>
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e;
  --rule: #b9b09a; --accent: #8a2b2b; --paper: #f7f3e9;
}
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --ink:#fff; --muted:#c3c2b7; --rule:#555; --accent:#e87ba4; --paper:#222; }
}
* { box-sizing:border-box; margin:0; }
body { background:var(--paper); color:var(--ink); font-family:Georgia,serif; }
.sheet { max-width:1020px; margin:0 auto; padding:24px 28px 48px; }
h1 { text-align:center; font-size:26px; margin:0 0 4px; }
.sub { text-align:center; color:var(--muted); font-style:italic; font-size:14px; margin-bottom:14px; }
.controls { display:flex; gap:16px; flex-wrap:wrap; align-items:center;
            background:#efe9d9; border:1px solid #e2dac4; border-radius:10px;
            padding:10px 14px; margin-bottom:18px; font-family:system-ui,sans-serif; font-size:13px; }
@media (prefers-color-scheme: dark) { .controls { background:#2a2a26; border-color:#444; } }
.controls label { display:flex; align-items:center; gap:6px; }
select { font:inherit; padding:2px 6px; border-radius:6px; border:1px solid #cfc7ae; }
.stats { margin-left:auto; color:var(--muted); font-size:12px; }

.grid { display:grid; grid-template-columns:repeat(4,1fr); border-right:1px solid var(--rule); }
.bar { position:relative; min-height:68px; border-left:1px solid var(--rule);
       display:flex; align-items:center; justify-content:center; padding:6px 4px; gap:2px; }
.bar.sec-start { border-left:4px double var(--rule); }
.seclabel { position:absolute; top:3px; left:4px; width:18px; height:18px;
            border:1.4px solid var(--accent); border-radius:3px; color:var(--accent);
            font:700 11px/18px system-ui; text-align:center; }

.chord { display:inline-flex; align-items:baseline; padding:2px 5px; border-radius:4px;
         font-size:15px; cursor:default; transition:background .15s,outline .15s;
         position:relative; }
.chord .root { font-size:17px; font-weight:600; }
.chord .qual { font-size:12px; }
.chord .acc { font-size:11px; vertical-align:super; }

/* motif bracket */
.chord[data-motif] { outline:2px solid transparent; outline-offset:-1px; }
.chord[data-motif][data-pos="0"] { border-top-left-radius:8px; border-bottom-left-radius:8px; }
.chord[data-motif][data-last="1"] { border-top-right-radius:8px; border-bottom-right-radius:8px; }
.chord.hi { outline-color:currentColor !important; filter:brightness(1.1); z-index:2; }

/* legend */
.legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; padding:10px 12px;
          background:var(--surface); border:1px solid var(--rule); border-radius:8px;
          font:13px/1.4 system-ui; }
.legend-item { display:flex; align-items:center; gap:5px; cursor:pointer; padding:3px 8px;
               border-radius:6px; transition:background .15s; }
.legend-item:hover { background:#0001; }
.legend-item .swatch { width:14px; height:14px; border-radius:3px; }
.legend-item .name { font-weight:600; }
.legend-item .meta { color:var(--muted); font-size:11px; }
</style>
</head><body>
<div class="sheet">
<h1>%(title)s</h1>
<div class="sub">%(form)s · key %(key)s · %(n_chords)d chords · motif stacking</div>
<div class="controls">
  <label>View: <select id="mode">
    <option value="shape">Shape (transpose-invariant)</option>
    <option value="exact">Exact (literal repeats)</option>
  </select></label>
  <div class="stats" id="stats"></div>
</div>
<div class="grid" id="grid"></div>
<div class="legend" id="legend"></div>
</div>
<script>
const DATA = %(json_data)s;

const grid = document.getElementById('grid');
const legend = document.getElementById('legend');
const stats = document.getElementById('stats');
const modeSelect = document.getElementById('mode');

function render(mode) {
  const view = DATA.views[mode];
  grid.innerHTML = '';
  legend.innerHTML = '';
  stats.textContent = DATA.n_chords + ' chords → ' + view.n_units + ' units (' + view.n_unique + ' unique motifs)';

  DATA.bars.forEach((bar, bi) => {
    const div = document.createElement('div');
    div.className = 'bar' + (bi === 0 || bar.section !== DATA.bars[Math.max(0,bi-1)].section ? ' sec-start' : '');
    if (bar.section && (bi === 0 || bar.section !== DATA.bars[Math.max(0,bi-1)].section)) {
      div.innerHTML += '<span class="seclabel">' + bar.section + '</span>';
    }
    bar.chords.forEach(c => {
      const ann = view.chord_ann[c.idx];
      const span = document.createElement('span');
      span.className = 'chord';
      span.innerHTML = c.html;
      if (ann) {
        span.style.background = ann.colour + '22';
        span.style.outlineColor = ann.colour;
        span.style.outline = '2px solid ' + ann.colour;
        span.dataset.motif = ann.motif_key;
        span.dataset.pos = ann.pos_in_motif;
        span.dataset.last = ann.pos_in_motif === ann.motif_len - 1 ? '1' : '0';
        span.title = ann.display + ' ×' + ann.count;
      }
      div.appendChild(span);
    });
    grid.appendChild(div);
  });

  // legend
  view.legend.forEach(m => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.dataset.motif = m.display;
    const keys = m.keys.length ? ' (in ' + m.keys.join(', ') + ')' : '';
    item.innerHTML = '<span class="swatch" style="background:' + m.colour + '"></span>' +
      '<span class="name">' + m.display + '</span>' +
      '<span class="meta">×' + m.count + ' saves ' + m.saving + keys + '</span>';
    item.addEventListener('mouseenter', () => highlightMotif(m.display, true));
    item.addEventListener('mouseleave', () => highlightMotif(m.display, false));
    legend.appendChild(item);
  });
}

function highlightMotif(display, on) {
  grid.querySelectorAll('.chord[data-motif]').forEach(el => {
    if (el.title.startsWith(display + ' ')) el.classList.toggle('hi', on);
  });
}

// hover on chords highlights siblings
grid.addEventListener('mouseover', e => {
  const ch = e.target.closest('.chord[data-motif]');
  if (!ch) return;
  const key = ch.dataset.motif;
  grid.querySelectorAll('.chord[data-motif="' + key + '"]').forEach(el => el.classList.add('hi'));
});
grid.addEventListener('mouseout', e => {
  const ch = e.target.closest('.chord[data-motif]');
  if (!ch) return;
  grid.querySelectorAll('.chord.hi').forEach(el => el.classList.remove('hi'));
});

modeSelect.addEventListener('change', () => render(modeSelect.value));
render('shape');
</script>
</body></html>"""


def render_html(rec: dict, out_path: Path):
    data = build_song_data(rec)
    if not data:
        return
    html = HTML_TEMPLATE % {
        "title": data["title"],
        "form": data["form"],
        "key": data["key"],
        "n_chords": data["n_chords"],
        "json_data": json.dumps(data),
    }
    out_path.write_text(html, encoding="utf-8")
    print(f"→ {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*", default=["Anthropology"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    if args.all:
        targets = recs
    else:
        targets = [r for r in recs if any(t.lower() in r["title"].lower() for t in args.titles)]

    out_dir = REPO / "docs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in targets:
        slug = re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
        render_html(rec, out_dir / f"motif_{slug}.html")


if __name__ == "__main__":
    main()
