"""Self-contained interactive iReal-style chord chart (single HTML file).

Renders the same lead-sheet look as ``chart_render`` but in the browser, with two
live controls:
    • rendering level — Auto (confidence-gated), Family, 7th, or Exact
    • certainty colour scale — Red→Green, Warm, or Grayscale

All model output is embedded as JSON; a small script re-labels and re-colours every
chord on change (no server, no build step). "Auto" reproduces the confidence-gated
tree descent — it only shows a 7th / exact quality where the model is sure — and a
threshold slider exposes that gate. Feed it the per-chord ``levels`` dicts produced
by ``demo_infer_song.infer_song``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .chart_render import Chart, chord_html

_LEVELS = ("family", "seventh", "exact")


def render_interactive(chart: Chart, chords: list[dict], out_path: str | Path,
                       bars_per_row: int = 4, meta_line: str = "") -> Path:
    """Write an interactive HTML chart. ``chords`` items need ``bar``, ``beat`` and
    a ``levels`` dict {family/seventh/exact: {ireal, conf}}."""
    out_path = Path(out_path)
    n_bars = max(chart.n_bars, (max((c["bar"] for c in chords), default=-1) + 1))
    spb = chart.section_per_bar

    def section_of(b):
        return spb[b] if 0 <= b < len(spb) else ""

    # per-chord, per-level: precompute the typeset HTML + certainty
    by_bar: dict[int, list[dict]] = {}
    data = []
    for c in chords:
        idx = len(data)
        by_bar.setdefault(c["bar"], []).append({"idx": idx, "beat": c.get("beat", 0)})
        lv = c["levels"]
        data.append({k: {"h": chord_html(lv[k]["ireal"]), "c": round(lv[k]["conf"], 4)}
                     for k in _LEVELS})

    # build the static measure grid; JS fills each .chord span by index
    cells = []
    for bar in range(n_bars):
        start = bar == 0 or section_of(bar) != section_of(bar - 1)
        final = bar == n_bars - 1
        classes = "measure" + (" section-start" if start else "") + (" final" if final else "")
        inner = ""
        if start and section_of(bar):
            inner += f'<span class="seclabel">{html.escape(section_of(bar))}</span>'
        cs = sorted(by_bar.get(bar, []), key=lambda d: d["beat"])
        inner += '<span class="chords">' + "".join(
            f'<span class="chord" id="chord-{d["idx"]}"></span>' for d in cs) + "</span>"
        cells.append(f'<div class="{classes}">{inner}</div>')

    sub = "  ·  ".join(x for x in [f"Key {chart.key}" if chart.key else "",
                                   chart.style, meta_line] if x)
    doc = _TEMPLATE.format(
        title=html.escape(chart.title),
        composer=html.escape(chart.composer),
        sub=html.escape(sub),
        cols=bars_per_row,
        grid="\n".join(cells),
        data=json.dumps(data),
    )
    out_path.write_text(doc, encoding="utf-8")
    return out_path


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — chord chart</title>
<style>
  :root {{ --paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--paper); color:var(--ink); margin:0;
         font-family:Georgia,'Times New Roman',serif; }}
  .sheet {{ max-width:960px; margin:0 auto; padding:28px 32px 48px; }}
  h1 {{ text-align:center; font-size:30px; margin:0 0 4px; letter-spacing:.3px; }}
  .subhead {{ display:flex; justify-content:space-between; color:var(--faint);
             font-style:italic; font-size:14px; margin-bottom:18px; }}
  .controls {{ display:flex; gap:20px; flex-wrap:wrap; align-items:center;
              background:#efe9d9; border:1px solid #e2dac4; border-radius:10px;
              padding:12px 16px; margin-bottom:22px; font-family:system-ui,sans-serif;
              font-size:13px; color:#4a4636; }}
  .controls label {{ display:flex; align-items:center; gap:7px; }}
  select, input[type=range] {{ font:inherit; }}
  select {{ padding:3px 6px; border-radius:6px; border:1px solid #cfc7ae; background:#fff; }}
  .legend {{ display:flex; align-items:center; gap:8px; margin-left:auto; }}
  .legend .bar {{ width:120px; height:12px; border-radius:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat({cols},1fr);
          border-right:1px solid var(--rule); }}
  .measure {{ position:relative; min-height:74px; border-left:1px solid var(--rule);
             display:flex; align-items:center; justify-content:center; padding:6px 4px; }}
  .measure.section-start {{ border-left:4px double var(--rule); }}
  .measure.final {{ border-right:3px solid var(--accent); }}
  .seclabel {{ position:absolute; top:4px; left:5px; width:20px; height:20px;
              border:1.4px solid var(--accent); border-radius:4px; color:var(--accent);
              font-family:system-ui,sans-serif; font-weight:700; font-size:12px;
              display:flex; align-items:center; justify-content:center; }}
  .chords {{ display:flex; gap:18px; align-items:baseline; justify-content:center;
            width:100%; flex-wrap:wrap; }}
  .chord {{ white-space:nowrap; transition:color .15s; }}
  .chord .root {{ font-size:27px; font-style:italic; }}
  .chord .qual {{ font-size:17px; font-style:italic; }}
  .chord sup {{ font-size:.62em; }}
  .chord .acc {{ font-size:.6em; margin-left:-.1em; vertical-align:.12em; }}
  .caption {{ color:var(--faint); font-size:12px; font-style:italic; margin-top:22px; }}
</style></head>
<body><div class="sheet">
  <h1>{title}</h1>
  <div class="subhead"><span>{sub}</span><span>{composer}</span></div>
  <div class="controls">
    <label>Level
      <select id="level">
        <option value="auto">Auto (certainty-gated)</option>
        <option value="family">Family (triad)</option>
        <option value="seventh">7th</option>
        <option value="exact">Exact</option>
      </select>
    </label>
    <label>Scale
      <select id="scale">
        <option value="rg">Red → Green</option>
        <option value="warm">Warm</option>
        <option value="gray">Grayscale</option>
      </select>
    </label>
    <label id="gate">Sure ≥ <span id="thv">0.60</span>
      <input type="range" id="thresh" min="0.4" max="0.95" step="0.05" value="0.6">
    </label>
    <span class="legend">unsure<span class="bar" id="legbar"></span>sure</span>
  </div>
  <div class="grid">
{grid}
  </div>
  <div class="caption" id="caption"></div>
</div>
<script>
const DATA = {data};
const LEVELS = ["family","seventh","exact"];
const SCALES = {{
  rg:   [[0.0,[192,57,43]],[0.5,[224,195,26]],[1.0,[58,138,58]]],
  warm: [[0.0,[176,57,43]],[0.5,[201,123,30]],[1.0,[40,40,40]]],
  gray: [[0.0,[190,60,50]],[0.15,[120,120,120]],[1.0,[28,28,28]]],
}};
function lerp(a,b,t){{return Math.round(a+(b-a)*t);}}
function colour(scaleKey, conf){{
  const t = Math.min(1,Math.max(0,(conf-0.35)/0.6));
  const stops = SCALES[scaleKey];
  for(let i=0;i<stops.length-1;i++){{
    const [p0,c0]=stops[i],[p1,c1]=stops[i+1];
    if(t<=p1){{const k=(t-p0)/(p1-p0);
      return `rgb(${{lerp(c0[0],c1[0],k)}},${{lerp(c0[1],c1[1],k)}},${{lerp(c0[2],c1[2],k)}})`;}}
  }}
  const last=stops[stops.length-1][1]; return `rgb(${{last[0]}},${{last[1]}},${{last[2]}})`;
}}
function pickLevel(d, mode, th){{
  if(mode!=="auto") return mode;
  if(d.exact.c>=th && d.seventh.c>=th) return "exact";
  if(d.seventh.c>=th) return "seventh";
  return "family";
}}
function render(){{
  const mode=document.getElementById("level").value;
  const scale=document.getElementById("scale").value;
  const th=parseFloat(document.getElementById("thresh").value);
  document.getElementById("thv").textContent=th.toFixed(2);
  document.getElementById("gate").style.opacity = mode==="auto"?1:0.35;
  DATA.forEach((d,i)=>{{
    const lv=pickLevel(d,mode,th);
    const el=document.getElementById("chord-"+i);
    if(!el) return;
    el.innerHTML=d[lv].h;
    el.style.color=colour(scale,d[lv].c);
  }});
  // legend gradient
  const g=[]; for(let k=0;k<=10;k++){{g.push(colour(scale,0.35+0.6*k/10));}}
  document.getElementById("legbar").style.background=
    `linear-gradient(90deg, ${{g.join(",")}})`;
  document.getElementById("caption").textContent = mode==="auto"
    ? `Auto: shows 7th / exact only where certainty ≥ ${{th.toFixed(2)}}; colour = certainty at the shown depth.`
    : `Fixed level: ${{mode}}. Colour = the model's certainty about that ${{mode}} label.`;
}}
["level","scale","thresh"].forEach(id=>
  document.getElementById(id).addEventListener("input",render));
render();
</script>
</body></html>
"""
