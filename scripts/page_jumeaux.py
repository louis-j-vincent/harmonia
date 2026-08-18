#!/usr/bin/env python3
"""scripts/page_jumeaux.py — la règle de temps, un trait par battue détectée.

Louis, 2026-08-18 : « ce que j'aimerais bien que tu montres plutôt, c'est juste
le normal avec les barres là où il y a les temps qu'il détecte. Pour moi ce sera
beaucoup plus clair. »

Donc pas d'histogramme, pas de statistique : une RÈGLE DE TEMPS. Six secondes
autour de chaque anomalie, un trait par battue que Beat This! a posée, les
mesures marquées plus haut, et le doublon en rouge. On voit le trait de trop, on
l'écoute, on tranche.

    .venv/bin/python scripts/page_jumeaux.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "beats_jumeaux.html"
W = 340          # largeur de la règle, en px — tient sur un iPhone
H = 74


def regle(z: dict, med: float) -> str:
    """Une fenêtre de temps : un trait par battue, le jumeau en rouge."""
    t0, t1 = z["t0"], z["t1"]
    x = lambda t: (t - t0) / max(t1 - t0, 1e-6) * W
    jum = set(z["jum"])          # TOUS les jumeaux de la fenêtre, pas que le centre
    dbs = set(z["db"])
    traits = []
    for t in z["beats"]:
        rouge = t in jum
        haut = 8 if t in dbs else 26
        traits.append(
            f'<line x1="{x(t):.1f}" y1="{haut}" x2="{x(t):.1f}" y2="56" '
            f'stroke="{"#a4462b" if rouge else "#3f3a31"}" '
            f'stroke-width="{2.4 if rouge else 1.4}"/>')
        if t in dbs:
            traits.append(f'<circle cx="{x(t):.1f}" cy="5" r="2.4" '
                          f'fill="{"#a4462b" if rouge else "#8a8371"}"/>')
    # l'écart de la paire CENTRALE, chiffré dessous
    a, b = sorted(z["paire"])
    mid = (x(a) + x(b)) / 2
    traits.append(f'<text x="{mid:.1f}" y="70" text-anchor="middle" '
                  f'font-size="10.5" fill="#a4462b" font-weight="600">'
                  f'{(b-a)*1000:.0f} ms</text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" '
            f'preserveAspectRatio="none">{"".join(traits)}</svg>')


def main() -> None:
    d = json.loads(Path("/tmp/jum2.json").read_text())
    blocs = []
    for s in d["songs"]:
        zones = "".join(f'''
      <div class="z">
        <div class="zt">à {z["t0"]+3:.1f} s
          <button data-t="{z["t0"]:.2f}" data-fin="{z["t1"]:.2f}">▶ écouter</button>
        </div>
        {regle(z, s["med"])}
      </div>''' for z in s["zones"])
        blocs.append(f'''
  <section data-audio="{s["audio"]}">
    <h2>{html.escape(s["titre"][:46])}
      <span>{s["bpm"]} BPM · un temps toutes les {s["med"]*1000:.0f} ms ·
      <b>{s["n_jum"]} doublon{"s" if s["n_jum"]>1 else ""}</b></span></h2>
    <audio controls preload="none"></audio>
    {zones}
  </section>''')

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Les temps que Beat This! détecte</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:20px 16px 70px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:620px}}
h1{{font-size:21px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 7px}}
h2 span{{display:block;font-weight:400;font-size:11.5px;color:var(--doux);
 margin-top:2px}}
h2 span b{{color:var(--rouge)}}
.chapo{{color:var(--doux);margin:0 0 14px}}
.lire{{background:#fff;border:1px solid var(--trait);border-radius:8px;
 padding:10px 13px;margin:0 0 20px;font-size:13px}}
.lire u{{text-decoration:none;color:var(--rouge);font-weight:600}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 12px}}
audio{{width:100%;height:32px;margin:0 0 8px;display:block}}
.z{{margin:0 0 6px;padding:4px 0 0;border-top:1px solid #f2ecdf}}
.zt{{display:flex;align-items:center;justify-content:space-between;
 font-size:11.5px;color:var(--doux)}}
svg{{display:block}}
button{{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--trait);
 background:#faf7ef;border-radius:7px;cursor:pointer;min-height:32px}}
button:active{{background:#efe9db}}
</style>
<h1>Les temps que Beat This! détecte</h1>
<p class="chapo">Six secondes autour de chaque anomalie. Un trait = une battue
posée par le traqueur.</p>
<div class="lire">Les traits <b>hauts, avec un point</b> sont les débuts de
mesure. En <u>rouge</u>, les deux traits que le traqueur pose pour un seul
temps — l'écart est chiffré dessous. Écoute : s'il n'y a qu'un temps là, le
trait rouge en trop est bien parasite.</div>
{''.join(blocs)}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), url = sec.dataset.audio, fin = null;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }}).catch(function(){{}});
  sec.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    fin = c.dataset.fin ? parseFloat(c.dataset.fin) : null;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
  el.addEventListener("timeupdate", function(){{
    if (fin !== null && el.currentTime >= fin) {{ el.pause(); fin = null; }}
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
