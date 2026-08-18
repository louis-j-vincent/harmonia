#!/usr/bin/env python3
"""scripts/page_jumeaux.py — le décalage de Beat This!, à l'œil et à l'oreille.

Louis, 2026-08-18 : « illustre moi le décalage de beat this avec une page html
accessible iphone comme ça je peux arbitrer d'une règle ».

La page montre les trois choses dont dépend la règle :

  1. **où tombent les intervalles courts**, sur les 145 morceaux du disque. Ils
     font trois paquets, et c'est entre le premier et le deuxième que le seuil
     se pose ;
  2. **ce que la réparation change**, morceau par morceau — combien de temps
     retirés, et l'irrégularité de la grille avant/après ;
  3. **ce que ça donne à l'oreille** : chaque paire de jumeaux est jouable, avec
     deux secondes avant et après, pour entendre s'il y a vraiment deux temps.

    .venv/bin/python scripts/page_jumeaux.py
"""
from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "beats_jumeaux.html"


def main() -> None:
    d = json.loads(Path("/tmp/jumeaux_data.json").read_text())
    tol = d["tol"]
    h = Counter(round(x / 0.05) * 0.05 for x in d["hist"])
    hmax = max(h.values())
    barres = "".join(
        f'<div class="b {"dedans" if k < tol else "dehors"}" '
        f'style="height:{max(3, round(100 * h[k] / hmax))}%" '
        f'title="{k:.2f}× : {h[k]}"><i>{h[k]}</i><u>{k:.2f}</u></div>'
        for k in sorted(h) if k <= 0.65)

    songs = sorted(d["songs"], key=lambda s: -s["retires"])
    blocs = []
    for s in songs:
        gain = s["irr0"] - s["irr1"]
        paires = ""
        if s["audio"] and s["paires"]:
            paires = "".join(
                f'<button data-t="{max(0, t0 - 2):.2f}" data-fin="{t1 + 2:.2f}">'
                f'▶ {t0:.1f} s <i>{r:.2f}×</i></button>'
                for t0, t1, r in s["paires"][:12])
            paires = f'<div class="ecoute">{paires}</div>'
        blocs.append(f'''
  <div class="song"{f' data-audio="{s["audio"]}"' if s["audio"] else ""}>
    <h3>{html.escape(s["stem"][:44])}<span>{s["bpm"]} BPM · {s["n"]} temps</span></h3>
    <div class="chiffres">
      <div><b>−{s["retires"]}</b><i>temps jumeaux</i></div>
      <div><b>{s["irr0"]:.1%}</b><i>irrégularité avant</i></div>
      <div class="{"mieux" if gain > 0.005 else ""}"><b>{s["irr1"]:.1%}</b><i>après</i></div>
    </div>
    {f'<audio controls preload="none"></audio>{paires}' if s["audio"] else
     '<p class="pas">audio pas sur ce disque — chiffres seulement</p>'}
  </div>''')

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beat This! — les temps jumeaux</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --vert:#3d6b47}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:20px 16px 70px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:760px}}
h1{{font-size:22px;margin:0 0 6px}}
h2{{font-size:16px;margin:30px 0 8px;border-bottom:1px solid var(--trait);
 padding-bottom:5px}}
h3{{font-size:14px;margin:0 0 8px;display:flex;justify-content:space-between;
 align-items:baseline;gap:8px;word-break:break-all}}
h3 span{{font-weight:400;font-size:11px;color:var(--doux);white-space:nowrap}}
.chapo{{color:var(--doux);margin:0 0 10px}}
.regle{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--vert);
 border-radius:8px;padding:11px 13px;margin:12px 0 18px}}
.hist{{display:flex;align-items:flex-end;gap:3px;height:150px;margin:14px 0 6px;
 padding-bottom:26px;position:relative}}
.b{{flex:1;border-radius:3px 3px 0 0;position:relative;min-width:0}}
.b.dedans{{background:var(--rouge)}} .b.dehors{{background:#d8d2c6}}
.b i{{position:absolute;top:-15px;left:50%;transform:translateX(-50%);
 font-style:normal;font-size:9.5px;color:var(--doux)}}
.b u{{position:absolute;bottom:-20px;left:50%;transform:translateX(-50%);
 text-decoration:none;font-size:9.5px;color:var(--doux)}}
.leg{{font-size:12px;color:var(--doux);margin:4px 0 0}}
.leg b.r{{color:var(--rouge)}}
.song{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 10px}}
.chiffres{{display:flex;gap:8px;margin:0 0 8px}}
.chiffres div{{flex:1;background:#faf7ef;border-radius:7px;padding:6px 8px;
 text-align:center}}
.chiffres b{{display:block;font-size:17px;font-variant-numeric:tabular-nums}}
.chiffres i{{font-style:normal;font-size:10.5px;color:var(--doux)}}
.chiffres .mieux b{{color:var(--vert)}}
audio{{width:100%;height:32px;margin:2px 0 6px;display:block}}
.ecoute{{display:flex;flex-wrap:wrap;gap:5px}}
button{{font:inherit;font-size:12px;padding:5px 9px;border:1px solid var(--trait);
 background:#faf7ef;border-radius:7px;cursor:pointer;min-height:34px}}
button i{{font-style:normal;color:var(--doux);font-size:10.5px}}
button:active{{background:#efe9db}}
.pas{{font-size:12px;color:var(--doux);margin:0}}
</style>
<h1>Beat This! — les temps jumeaux</h1>
<p class="chapo">Par endroits le traqueur pose <b>deux marques pour un seul
temps</b>, à moins d'un quart de temps l'une de l'autre. Les mesures deviennent
alors trop courtes, et la grille glisse sous la musique.</p>

<div class="regle"><b>La règle actuelle : on retire le jumeau sous {tol:.2f}×
le temps médian.</b> Des deux, on garde celui qui tombe le plus près de
l'attendu. On ne supprime jamais qu'un temps — on n'en invente aucun.</div>

<h2>Où tombent les intervalles courts</h2>
<p class="chapo">Les {len(d['hist'])} intervalles sous 0,70× le temps médian, sur
les 145 morceaux du disque. En rouge, ce que la règle retire.</p>
<div class="hist">{barres}</div>
<p class="leg">Trois paquets : <b class="r">0,10–0,20</b> les jumeaux ·
<b>0,35</b> un tiers de temps (autre phénomène, laissé intact) ·
<b>0,50</b> l'erreur d'octave, déjà traitée ailleurs. Le seuil se pose entre le
premier et le deuxième.</p>

<h2>Les {len(songs)} morceaux touchés</h2>
<p class="chapo">L'irrégularité est l'écart moyen d'une mesure au temps médian.
Écoute une paire : deux secondes avant, deux après — s'il n'y a qu'un temps, le
jumeau est bien parasite.</p>
{''.join(blocs)}
<script>
document.querySelectorAll(".song[data-audio]").forEach(function(sec){{
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
    print(f"  {SORTIE.relative_to(REPO)}")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
