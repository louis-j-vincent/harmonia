"""Self-contained interactive iReal-style chord chart (single HTML file).

Renders the same lead-sheet look as ``chart_render`` in the browser, with live
controls (no server, no build step):

    • Level          — Auto (certainty-gated), Family, 7th, or Exact
    • Colour scale   — certainty as Red→Green, Warm, or Grayscale
    • Sure ≥ (gate)  — the confidence threshold Auto uses to descend the tree
    • Transpose      — rewrite every chord into any of the 12 keys
    • Highlight keys — tint each section by its estimated local key/scale

Chords are stored structurally (root pitch-class + quality tail per depth), and
a small script typesets them in the DOM — so transposition is just a root shift
and re-spell, and switching depth just swaps the quality tail. Feed it the per-
chord ``levels`` dicts from ``demo_infer_song.infer_song``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ..theory.local_key import parse_token, section_keys
from .chart_render import Chart

_LEVELS = ("family", "seventh", "exact")
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _parse_home_key(key: str) -> tuple[int, str]:
    """DB key string ("Ab", "G-", "E-") → (tonic_pc, mode)."""
    key = key.strip()
    if not key or key[0] not in _LETTER:
        return 0, "major"
    pc = _LETTER[key[0]]
    i = 1
    if len(key) > 1 and key[1] in "b#":
        pc += -1 if key[1] == "b" else 1
        i = 2
    mode = "minor" if "-" in key[i:] or "m" in key[i:] else "major"
    return pc % 12, mode


def render_interactive(chart: Chart, chords: list[dict], out_path: str | Path,
                       bars_per_row: int = 4) -> Path:
    """Write an interactive HTML chart. ``chords`` items need ``bar``, ``beat``
    and a ``levels`` dict {family/seventh/exact: {ireal, conf}}."""
    out_path = Path(out_path)
    n_bars = max(chart.n_bars, (max((c["bar"] for c in chords), default=-1) + 1))
    spb = chart.section_per_bar

    def section_of(b):
        return spb[b] if 0 <= b < len(spb) else ""

    # structured per-chord data (root pc + quality tail per depth) + grid cells
    by_bar: dict[int, list[dict]] = {}
    data = []
    for c in chords:
        idx = len(data)
        by_bar.setdefault(c["bar"], []).append({"idx": idx, "beat": c.get("beat", 0)})
        lv = c["levels"]
        root, _, bass = parse_token(lv["exact"]["ireal"])
        data.append({
            "root": root, "bass": bass if bass is not None else -1,
            "lv": {k: {"q": parse_token(lv[k]["ireal"])[1], "c": round(lv[k]["conf"], 4)}
                   for k in _LEVELS},
        })

    cells = []
    for bar in range(n_bars):
        start = bar == 0 or section_of(bar) != section_of(bar - 1)
        final = bar == n_bars - 1
        klass = "measure" + (" section-start" if start else "") + (" final" if final else "")
        sec = section_of(bar)
        inner = ""
        if start and sec:
            inner += f'<span class="seclabel">{html.escape(sec)}</span>'
        cs = sorted(by_bar.get(bar, []), key=lambda d: d["beat"])
        inner += '<span class="chords">' + "".join(
            f'<span class="chord" id="chord-{d["idx"]}"></span>' for d in cs) + "</span>"
        cells.append(f'<div class="{klass}" data-sec="{html.escape(sec)}">{inner}</div>')

    # per-section estimated local key (symbolic, from the exact-depth tokens)
    sk_chords = [{"bar": c["bar"], "symbol": c["levels"]["exact"]["ireal"]} for c in chords]
    seckeys = section_keys(sk_chords, spb)

    home_tonic, home_mode = _parse_home_key(chart.key)
    payload = {
        "cols": bars_per_row,
        "home": {"tonic": home_tonic, "mode": home_mode},
        "chords": data,
        "sectionKeys": {k: {"tonic": v["tonic"], "mode": v["mode"]}
                        for k, v in seckeys.items()},
    }
    sub = "  ·  ".join(x for x in [f"Key {chart.key}" if chart.key else "", chart.style] if x)

    doc = (_TEMPLATE
           .replace("%%TITLE%%", html.escape(chart.title))
           .replace("%%COMPOSER%%", html.escape(chart.composer))
           .replace("%%SUB%%", html.escape(sub))
           .replace("%%COLS%%", str(bars_per_row))
           .replace("%%GRID%%", "\n".join(cells))
           .replace("%%PAYLOAD%%", json.dumps(payload)))
    out_path.write_text(doc, encoding="utf-8")
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%% — chord chart</title>
<style>
  :root { --paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; }
  * { box-sizing:border-box; }
  body { background:var(--paper); color:var(--ink); margin:0;
         font-family:Georgia,'Times New Roman',serif; }
  .sheet { max-width:980px; margin:0 auto; padding:28px 32px 48px; }
  h1 { text-align:center; font-size:30px; margin:0 0 4px; }
  .subhead { display:flex; justify-content:space-between; color:var(--faint);
             font-style:italic; font-size:14px; margin-bottom:16px; }
  .controls { display:flex; gap:18px; flex-wrap:wrap; align-items:center;
              background:#efe9d9; border:1px solid #e2dac4; border-radius:10px;
              padding:11px 16px; margin-bottom:14px; font-family:system-ui,sans-serif;
              font-size:13px; color:#4a4636; }
  .controls label { display:flex; align-items:center; gap:7px; }
  select, input[type=range] { font:inherit; }
  select { padding:3px 6px; border-radius:6px; border:1px solid #cfc7ae; background:#fff; }
  .legend { display:flex; align-items:center; gap:8px; margin-left:auto; }
  .legend .bar { width:110px; height:12px; border-radius:6px; }
  #keylegend { display:none; flex-wrap:wrap; gap:14px; font-family:system-ui,sans-serif;
               font-size:12.5px; color:#4a4636; margin:0 0 16px 2px; }
  #keylegend .item { display:flex; align-items:center; gap:6px; }
  #keylegend .sw { width:15px; height:15px; border-radius:4px; border:1px solid #0002; }
  .grid { display:grid; grid-template-columns:repeat(%%COLS%%,1fr);
          border-right:1px solid var(--rule); }
  .measure { position:relative; min-height:74px; border-left:1px solid var(--rule);
             display:flex; align-items:center; justify-content:center; padding:6px 4px;
             transition:background .15s; }
  .measure.section-start { border-left:4px double var(--rule); }
  .measure.final { border-right:3px solid var(--accent); }
  .seclabel { position:absolute; top:4px; left:5px; width:20px; height:20px;
              border:1.4px solid var(--accent); border-radius:4px; color:var(--accent);
              font-family:system-ui,sans-serif; font-weight:700; font-size:12px;
              display:flex; align-items:center; justify-content:center; background:#f7f3e9aa; }
  .chords { display:flex; gap:18px; align-items:baseline; justify-content:center;
            width:100%; flex-wrap:wrap; }
  .chord { white-space:nowrap; transition:color .15s; }
  .chord .root { font-size:27px; font-style:italic; }
  .chord .qual { font-size:17px; font-style:italic; }
  .chord sup { font-size:.62em; }
  .chord .acc { font-size:.6em; margin-left:-.1em; vertical-align:.12em; }
  .caption { color:var(--faint); font-size:12px; font-style:italic; margin-top:20px; }
</style></head>
<body><div class="sheet">
  <h1>%%TITLE%%</h1>
  <div class="subhead"><span>%%SUB%%</span><span>%%COMPOSER%%</span></div>
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
      <input type="range" id="thresh" min="0.4" max="0.95" step="0.05" value="0.6"></label>
    <label>Transpose <select id="transpose"></select></label>
    <label><input type="checkbox" id="hl"> Highlight keys</label>
    <span class="legend">unsure<span class="bar" id="legbar"></span>sure</span>
  </div>
  <div id="keylegend"></div>
  <div class="grid">
%%GRID%%
  </div>
  <div class="caption" id="caption"></div>
</div>
<script>
const P = %%PAYLOAD%%;
const LEVELS=["family","seventh","exact"];
const SHARP=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
const FLAT =["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"];
const MAJN=["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"];
const MINN=["C","C♯","D","E♭","E","F","F♯","G","G♯","A","B♭","B"];
const FLAT_MAJ=new Set([0,1,3,5,6,8,10]);
const SCALES={
  rg:[[0,[192,57,43]],[0.5,[224,195,26]],[1,[58,138,58]]],
  warm:[[0,[176,57,43]],[0.5,[201,123,30]],[1,[40,40,40]]],
  gray:[[0,[190,60,50]],[0.15,[120,120,120]],[1,[28,28,28]]]};
const HL_PALETTE=["#fbe3e3","#e2eefb","#e6f7e0","#fbf4dd","#efe2fb","#dff6f4","#fbe6f3","#eef0da"];
const mod=(n,m)=>((n%m)+m)%m;

function noteName(pc,flats){return (flats?FLAT:SHARP)[mod(pc,12)];}
function wrapAcc(s){return s.replace(/♭/g,"<span class='acc'>♭</span>").replace(/♯/g,"<span class='acc'>♯</span>");}
function alterations(s){
  if(!s) return "";
  s=s.replace("69","6/9");
  s=s.replace(/([b#])(\d+)/g,(m,a,d)=>(a==="b"?"♭":"♯")+d);
  s=s.replace(/\^/g,"△");
  return s;
}
function typesetQuality(q){
  if(q==="") return ["",""];
  let base="",rest=q;
  if(rest.startsWith("-^")){base="m";rest="△"+rest.slice(2);}
  else if(rest.startsWith("-7b5")){base="ø";rest=rest.slice(4);}
  else if(rest.startsWith("-")){base="m";rest=rest.slice(1);}
  else if(rest.startsWith("h")){base="ø";rest=rest.slice(1);}
  else if(rest.startsWith("o")){base="°";rest=rest.slice(1);}
  else if(rest.startsWith("^")){base="";rest="△"+rest.slice(1);}
  else if(rest.startsWith("+")){base="+";rest=rest.slice(1);}
  else if(rest.startsWith("sus")){base="sus"+(rest.slice(3)||"4");rest="";}
  else if(rest.startsWith("5")){base="5";rest=rest.slice(1);}
  if(rest.includes("sus")){const i=rest.indexOf("sus");
    return [base+"sus"+(rest.slice(i+3)||"4"),alterations(rest.slice(0,i))];}
  return [base,alterations(rest)];
}
function chordHTML(d,q,offset,flats){
  const [base,sup]=typesetQuality(q);
  let h='<span class="root">'+wrapAcc(noteName(d.root+offset,flats))+'</span>';
  if(base||sup){h+='<span class="qual">'+wrapAcc(base)+(sup?'<sup>'+sup+'</sup>':'')+'</span>';}
  if(d.bass>=0){h+='<span class="qual">/'+wrapAcc(noteName(d.bass+offset,flats))+'</span>';}
  return h;
}
function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function colour(key,conf){
  const t=Math.min(1,Math.max(0,(conf-0.35)/0.6)),st=SCALES[key];
  for(let i=0;i<st.length-1;i++){const[p0,c0]=st[i],[p1,c1]=st[i+1];
    if(t<=p1){const k=(t-p0)/(p1-p0);
      return `rgb(${lerp(c0[0],c1[0],k)},${lerp(c0[1],c1[1],k)},${lerp(c0[2],c1[2],k)})`;}}
  const l=st[st.length-1][1];return `rgb(${l[0]},${l[1]},${l[2]})`;
}
function pickLevel(d,mode,th){
  if(mode!=="auto") return mode;
  if(d.lv.exact.c>=th && d.lv.seventh.c>=th) return "exact";
  if(d.lv.seventh.c>=th) return "seventh";
  return "family";
}
function preferFlats(offset){
  const h=P.home; const maj=h.mode==="major"?h.tonic:mod(h.tonic+3,12);
  return FLAT_MAJ.has(mod(maj+offset,12));
}
function keyLabel(tonic,mode,offset){
  return (mode==="major"?MAJN:MINN)[mod(tonic+offset,12)]+" "+mode;
}

// ── build transpose dropdown (offset 0..11 → resulting home key) ──
(function(){
  const sel=document.getElementById("transpose");
  for(let off=0;off<12;off++){
    const o=document.createElement("option");o.value=off;
    o.textContent=keyLabel(P.home.tonic,P.home.mode,off)+(off===0?"  (original)":"");
    sel.appendChild(o);
  }
})();
// stable colour per distinct original section key
const keyColour={}; let ci=0;
for(const lab in P.sectionKeys){
  const kn=keyLabel(P.sectionKeys[lab].tonic,P.sectionKeys[lab].mode,0);
  if(!(kn in keyColour)) keyColour[kn]=HL_PALETTE[ci++ % HL_PALETTE.length];
}

function render(){
  const mode=document.getElementById("level").value;
  const scale=document.getElementById("scale").value;
  const th=parseFloat(document.getElementById("thresh").value);
  const offset=parseInt(document.getElementById("transpose").value);
  const hl=document.getElementById("hl").checked;
  const flats=preferFlats(offset);
  document.getElementById("thv").textContent=th.toFixed(2);
  document.getElementById("gate").style.opacity=mode==="auto"?1:0.35;

  P.chords.forEach((d,i)=>{
    const lv=pickLevel(d,mode,th), el=document.getElementById("chord-"+i);
    if(!el) return;
    el.innerHTML=chordHTML(d,d.lv[lv].q,offset,flats);
    el.style.color=colour(scale,d.lv[lv].c);
  });

  // section key highlight
  const leg=document.getElementById("keylegend");
  document.querySelectorAll(".measure").forEach(m=>{
    const sec=m.dataset.sec, sk=P.sectionKeys[sec];
    if(hl && sk){const kn=keyLabel(sk.tonic,sk.mode,0); m.style.background=keyColour[kn];}
    else m.style.background="";
  });
  if(hl){
    const byKey={};
    for(const lab in P.sectionKeys){const kn=keyLabel(P.sectionKeys[lab].tonic,P.sectionKeys[lab].mode,0);
      (byKey[kn]=byKey[kn]||[]).push(lab);}
    leg.innerHTML=Object.keys(byKey).map(kn=>{
      const sk=P.sectionKeys[byKey[kn][0]];
      const shown=keyLabel(sk.tonic,sk.mode,offset);
      return `<span class="item"><span class="sw" style="background:${keyColour[kn]}"></span>`+
             `${byKey[kn].sort().join(", ")} — ${shown}</span>`;}).join("");
    leg.style.display="flex";
  } else leg.style.display="none";

  const g=[];for(let k=0;k<=10;k++)g.push(colour(scale,0.35+0.6*k/10));
  document.getElementById("legbar").style.background=`linear-gradient(90deg, ${g.join(",")})`;
  const tnote=offset?` · transposed to ${keyLabel(P.home.tonic,P.home.mode,offset)}`:"";
  document.getElementById("caption").textContent=(mode==="auto"
    ? `Auto: shows 7th / exact only where certainty ≥ ${th.toFixed(2)}; colour = certainty at the shown depth.`
    : `Fixed level: ${mode}. Colour = certainty about that ${mode} label.`)
    + (hl?"  Section tint = estimated local key/scale (symbolic).":"") + tnote;
}
["level","scale","thresh","transpose","hl"].forEach(id=>{
  const el=document.getElementById(id);
  el.addEventListener("input",render); el.addEventListener("change",render);
});
render();
</script>
</body></html>
"""
