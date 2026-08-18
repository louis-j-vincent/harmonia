#!/usr/bin/env python3
"""scripts/page_another_day.py — Another Day : la tête de lecture et les temps.

Louis, 2026-08-18 : « montres moi le playhead another day + beats détectés ».

Deux rangées superposées, sur la même échelle de temps :
  * ce que **Beat This! a posé** (420 battues, dont 15 doublons en rouge) ;
  * ce que **l'app sert maintenant** — la grille rigide à 135,47 BPM, avec les
    barres de mesure.
La tête de lecture traverse les deux ; un clic n'importe où l'y place.

    .venv/bin/python scripts/page_another_day.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "another_day_playhead.html"
PXS = 34          # pixels par seconde


def main() -> None:
    d = json.loads(Path("/tmp/ad.json").read_text())
    dur = d["dur"]
    W = int(dur * PXS) + 40
    x = lambda t: 20 + t * PXS
    jum = set(d["jum"])
    db = set(d["db"])
    bar = set(d["barres"])

    H1, H2, H3 = 46, 52, 44
    t1 = "".join(
        f'<line x1="{x(t):.1f}" y1="10" x2="{x(t):.1f}" y2="{H1-6}" '
        f'stroke="{"#a4462b" if t in jum else "#3f3a31"}" '
        f'stroke-width="{2.2 if t in jum else 1}"/>' for t in d["brut"])
    t2 = "".join(
        f'<line x1="{x(t):.1f}" y1="{6 if t in db else 18}" x2="{x(t):.1f}" '
        f'y2="{H2-14}" stroke="{"#3d6b47" if t in db else "#8aa88f"}" '
        f'stroke-width="{1.8 if t in db else 1}"/>' for t in d["prod"])
    t2 += "".join(
        f'<line x1="{x(t):.1f}" y1="0" x2="{x(t):.1f}" y2="{H2}" '
        f'stroke="#3d6b47" stroke-width="0.8" stroke-dasharray="2 3" '
        f'opacity="0.45"/>' for t in d["barres"])
    t2 += "".join(f'<text x="{x(k):.1f}" y="{H2-2}" font-size="9" fill="#8a8371" '
                  f'text-anchor="middle">{k}s</text>'
                  for k in range(0, int(dur)+1, 5))
    mx = max((abs(v) for _, v in d["derive"]), default=1e-6)
    ech = max(mx, d["med"] * 0.5)
    pts = " ".join(f'{x(t):.1f},{H3/2 - v/ech*(H3/2-6):.1f}' for t, v in d["derive"])

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Another Day — la tête de lecture et les temps</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --vert:#3d6b47;--bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 50px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 4px}}
.chapo{{color:var(--doux);margin:0 0 10px;font-size:13.5px}}
.chiffres{{display:flex;gap:8px;margin:0 0 12px;flex-wrap:wrap}}
.chiffres div{{flex:1;min-width:120px;background:#fff;border:1px solid var(--trait);
 border-radius:8px;padding:7px 10px}}
.chiffres b{{display:block;font-size:17px;font-variant-numeric:tabular-nums}}
.chiffres i{{font-style:normal;font-size:11px;color:var(--doux)}}
.chiffres .r b{{color:var(--rouge)}} .chiffres .v b{{color:var(--vert)}}
audio{{width:100%;height:34px;margin:0 0 8px;display:block}}
.ens{{display:block;border:1px solid var(--trait);border-radius:7px;
 background:#fdfbf5;margin:0 0 3px}}
.leg{{font-size:11.5px;color:var(--doux);margin:0 0 12px}}
.leg b.r{{color:var(--rouge)}} .leg b.v{{color:var(--vert)}}
.scroll{{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;
 border:1px solid var(--trait);border-radius:7px;background:#fdfbf5}}
.lab{{font:600 10px -apple-system,sans-serif;fill:#8a8371}}
svg{{display:block}}
</style>
<h1>Another Day — la tête de lecture et les temps</h1>
<p class="chapo">En haut ce que <b>Beat This!</b> a posé, en dessous ce que
<b>l'app sert</b>. Fais glisser, ou clique pour placer la lecture.</p>
<div class="chiffres">
  <div class="r"><b>420</b><i>battues du traqueur<br>dont 15 doublons</i></div>
  <div class="v"><b>405</b><i>cases de la grille rigide<br>135,47 BPM</i></div>
  <div><b>101</b><i>mesures<br>toutes égales</i></div>
</div>
<audio controls preload="none"></audio>

<svg viewBox="0 0 {W} {H1+H2+H3}" width="100%" height="150" class="ens"
     preserveAspectRatio="none">
  <g>{t1}</g>
  <g transform="translate(0,{H1})">{t2}</g>
  <g transform="translate(0,{H1+H2})">
    <line x1="20" y1="{H3/2}" x2="{W-20}" y2="{H3/2}" stroke="#ddd6c7"/>
    <polyline points="{pts}" fill="none" stroke="#3f6f8f" stroke-width="1.4"
              vector-effect="non-scaling-stroke"/>
  </g>
  <line class="tete2" x1="20" y1="0" x2="20" y2="{H1+H2+H3}" stroke="#a4462b"
        stroke-width="1" opacity="0" vector-effect="non-scaling-stroke"/>
</svg>
<p class="leg">toute la chanson · en haut les battues du traqueur
(<b class="r">rouge</b> = les doublons) · au milieu la grille servie
(<b class="v">vert foncé</b> = début de mesure, pointillés = barres) · en bas
l'écart des battues du traqueur à la grille</p>

<div class="scroll">
  <svg width="{W}" height="{H1+H2}">
    <text x="4" y="12" class="lab">BTS</text>
    <g>{t1}</g>
    <text x="4" y="{H1+12}" class="lab">app</text>
    <g transform="translate(0,{H1})">{t2}</g>
    <line class="tete" x1="20" y1="0" x2="20" y2="{H1+H2}" stroke="#a4462b"
          stroke-width="1.6" opacity="0"/>
  </svg>
</div>
<script>
(function(){{
  var el = document.querySelector("audio"), pxs = {PXS};
  var box = document.querySelector(".scroll");
  var tete = document.querySelector(".tete"), tete2 = document.querySelector(".tete2");
  var ens = document.querySelector(".ens"), dur = 0, W = {W};
  el.src = "{d['audio']}";
  if (window.fetch) fetch("{d['audio']}")
    .then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  ens.addEventListener("click", function(ev){{
    var r = ens.getBoundingClientRect();
    if (!dur) return;
    try {{ el.currentTime = (ev.clientX-r.left)/r.width*dur; el.play(); }} catch(e){{}}
  }});
  box.addEventListener("click", function(ev){{
    var r = box.getBoundingClientRect();
    var t = (ev.clientX - r.left + box.scrollLeft - 20)/pxs;
    try {{ el.currentTime = Math.max(0, t); el.play(); }} catch(e){{}}
  }});
  el.addEventListener("loadedmetadata", function(){{ dur = el.duration || 0; }});
  /* timeupdate, jamais requestAnimationFrame (Safari iOS). */
  el.addEventListener("timeupdate", function(){{
    var x = 20 + el.currentTime*pxs;
    tete.setAttribute("x1", x); tete.setAttribute("x2", x);
    tete.setAttribute("opacity", 1);
    var xx = 20 + el.currentTime*pxs;
    tete2.setAttribute("x1", xx); tete2.setAttribute("x2", xx);
    tete2.setAttribute("opacity", 1);
    var w = box.clientWidth;
    if (x < box.scrollLeft + w*0.2 || x > box.scrollLeft + w*0.8)
      box.scrollLeft = Math.max(0, x - w*0.4);
  }});
}})();
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
