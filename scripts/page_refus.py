#!/usr/bin/env python3
"""scripts/page_derive.py — toute la chanson, tous les temps de Beat This!.

Louis, 2026-08-18 : « montres moi les cas où il y a des dérives... je veux tout
le playhead avec les temps marqués par beatthis sur toute la chanson ».

Une bande par morceau, du début à la fin, qui défile horizontalement : un trait
par battue, les débuts de mesure plus hauts, les doublons en rouge, et la tête
de lecture qui suit le son. Dessous, la courbe d'écart à une grille rigide posée
sur les 24 premières battues — c'est la forme de la dérive : plate si le morceau
tient le tempo, en pente s'il ralentit, en marche d'escalier si le traqueur a
sauté des temps.

    .venv/bin/python scripts/page_derive.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "beats_refus.html"
PXS = 26        # pixels par seconde — un temps à 120 BPM fait 13 px
HB = 62         # hauteur de la bande des temps
HD = 54         # hauteur de la courbe d'écart


def ensemble(s: dict) -> str:
    """TOUTE la chanson d'un seul regard, ajustée à la largeur de l'écran.

    La bande qui défile montre le détail ; celle-ci montre la FORME — où le
    traqueur décroche, et sur combien de temps.
    """
    W, H = 1000, 66
    dur = max(s["dur"], 1e-6)
    x = lambda t: t / dur * W
    db = set(s["db"])
    jum = set()
    # densité des battues, par seconde : c'est elle qui montre les décrochages
    import collections
    dens = collections.Counter(int(t) for t in s["beats"])
    att = 1.0 / s["med"]
    barres = []
    for k in range(int(dur) + 1):
        v = dens.get(k, 0)
        h = min(1.6, v / att) * 22
        c = "#a4462b" if abs(v - att) > 0.45 * att else "#cfc7b4"
        barres.append(f'<rect x="{x(k):.2f}" y="{26-h:.1f}" '
                      f'width="{max(0.8, W/dur):.2f}" height="{h:.1f}" fill="{c}"/>')
    mx = max((abs(v) for _, v in s["derive"]), default=1e-6)
    ech = max(mx, s["med"] * 0.5)
    pts = " ".join(f'{x(t):.2f},{46 - v/ech*16:.2f}' for t, v in s["derive"])
    return f'''
    <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" class="ens"
         preserveAspectRatio="none">
      {"".join(barres)}
      <line x1="0" y1="46" x2="{W}" y2="46" stroke="#ddd6c7" stroke-width="0.7"/>
      <polyline points="{pts}" fill="none" stroke="#3f6f8f" stroke-width="1.2"
                vector-effect="non-scaling-stroke"/>
      <line class="tete2" x1="0" y1="0" x2="0" y2="{H}" stroke="#a4462b"
            stroke-width="1" opacity="0" vector-effect="non-scaling-stroke"/>
    </svg>
    <p class="leg2">toute la chanson · en haut la densité de battues par seconde
    (<b>rouge</b> = le traqueur en pose trop ou pas assez) · en bas l'écart à la
    grille du début</p>'''


def bande(s: dict) -> str:
    W = int(s["dur"] * PXS) + 40
    x = lambda t: 20 + t * PXS
    db = set(s["db"])
    jum = set()
    traits = []
    for t in s["beats"]:
        r = t in jum
        traits.append(
            f'<line x1="{x(t):.1f}" y1="{6 if t in db else 24}" '
            f'x2="{x(t):.1f}" y2="{HB - 8}" '
            f'stroke="{"#a4462b" if r else "#3f3a31"}" '
            f'stroke-width="{2.2 if r else 1}"/>')
    for k in range(0, int(s["dur"]) + 1, 10):
        traits.append(f'<text x="{x(k):.1f}" y="{HB - 1}" font-size="9" '
                      f'fill="#8a8371" text-anchor="middle">{k}s</text>')

    # la courbe d'écart, en fraction de temps
    pts, mx = [], max((abs(v) for _, v in s["derive"]), default=1e-6)
    ech = max(mx, s["med"] * 0.5)
    for t, v in s["derive"]:
        pts.append(f'{x(t):.1f},{HD/2 - v/ech*(HD/2 - 6):.1f}')
    courbe = (f'<polyline points="{" ".join(pts)}" fill="none" '
              f'stroke="#3f6f8f" stroke-width="1.6"/>')
    zero = f'<line x1="20" y1="{HD/2}" x2="{W-20}" y2="{HD/2}" stroke="#ddd6c7"/>'
    return f'''
    <div class="scroll">
      <svg width="{W}" height="{HB}" class="b">{"".join(traits)}
        <line class="tete" x1="20" y1="0" x2="20" y2="{HB}" stroke="#a4462b"
              stroke-width="1.6" opacity="0"/></svg>
      <svg width="{W}" height="{HD}" class="d">{zero}{courbe}</svg>
    </div>'''


def main() -> None:
    songs = json.loads(Path("/tmp/refus.json").read_text())
    blocs = []
    for s in songs:
        if not s["audio"]:
            continue
        blocs.append(f'''
  <section data-audio="{s["audio"]}" data-pxs="{PXS}">
    <h2>{html.escape(s["titre"][:44])}
      <span>{s["dur"]:.0f} s · BPM du traqueur <b>{s["bpm_med"]}</b>
      {f'· tempo rigide <b>{s["bpm_rig"]}</b>, mais seulement '
       f'<b class=j>{s["inliers"]:.0%}</b> des battues y tombent'
       if s["bpm_rig"] else '· <b class=j>aucun tempo stable trouvé</b>'}
      </span></h2>
    <audio controls preload="none"></audio>
    {ensemble(s)}
    <p class="det">le détail, à faire glisser :</p>
    {bande(s)}
  </section>''')

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Les temps de Beat This! sur toute la chanson</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 6px}}
h2 span{{display:block;font-weight:400;font-size:11.5px;color:var(--doux);
 margin-top:2px}}
h2 b{{color:var(--fg)}} h2 b.j{{color:var(--rouge)}}
.chapo{{color:var(--doux);margin:0 0 10px}}
.lire{{background:#fff;border:1px solid var(--trait);border-radius:8px;
 padding:10px 12px;margin:0 0 16px;font-size:12.5px}}
.lire u{{text-decoration:none;color:var(--rouge);font-weight:600}}
.lire i{{font-style:normal;color:var(--bleu);font-weight:600}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:10px 12px;margin:0 0 12px}}
audio{{width:100%;height:32px;margin:0 0 6px;display:block}}
.ens{{display:block;border:1px solid #f0ebde;border-radius:6px;background:#fdfbf5}}
.leg2{{font-size:11px;color:var(--doux);margin:4px 0 10px}}
.det{{font-size:11px;color:var(--doux);margin:0 0 3px}}
.scroll{{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;
 border:1px solid #f0ebde;border-radius:6px;background:#fdfbf5}}
svg.b, svg.d{{display:block}}
svg.d{{border-top:1px dashed #ece5d5}}
</style>
<h1>Ceux qui gardent la grille du traqueur</h1>
<p class="chapo">La grille rigide est passée en prod sur 70 morceaux. Voici
ceux qu'elle refuse, et pourquoi : soit aucun tempo stable ne se dégage, soit
il s'en dégage un mais trop peu de battues y tombent (moins de 85 %) — la
rigidifier déplacerait la musique au lieu de la décrire.</p>
<div class="lire">En haut, un trait par battue ; les traits <b>hauts</b> sont
les débuts de mesure, les <u>rouges</u> les doublons.<br>
En bas, la <i>courbe d'écart</i> à une grille rigide calée sur les 24 premières
battues : <b>plate</b> = le morceau tient le tempo · <b>en pente</b> = il
ralentit ou accélère · <b>en marche d'escalier</b> = le traqueur a sauté ou
ajouté des temps.</div>
{''.join(blocs)}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), url = sec.dataset.audio;
  var pxs = parseFloat(sec.dataset.pxs), box = sec.querySelector(".scroll");
  var tete = sec.querySelector(".tete");
  var tete2 = sec.querySelector(".tete2"), dur = 0;
  var ens = sec.querySelector(".ens");
  ens.addEventListener("click", function(ev){{
    var r = ens.getBoundingClientRect();
    if (!dur) return;
    try {{ el.currentTime = (ev.clientX - r.left)/r.width*dur; el.play(); }} catch(e){{}}
  }});
  el.addEventListener("loadedmetadata", function(){{ dur = el.duration || 0; }});
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }}).catch(function(){{}});
  box.addEventListener("click", function(ev){{
    var r = box.getBoundingClientRect();
    var t = (ev.clientX - r.left + box.scrollLeft - 20) / pxs;
    try {{ el.currentTime = Math.max(0, t); el.play(); }} catch(e){{}}
  }});
  /* timeupdate, jamais requestAnimationFrame : sur Safari iOS un rAF empeche
     le moteur audio de demarrer. */
  el.addEventListener("timeupdate", function(){{
    var x = 20 + el.currentTime * pxs;
    tete.setAttribute("x1", x); tete.setAttribute("x2", x);
    tete.setAttribute("opacity", 1);
    if (dur) {{
      var xx = el.currentTime/dur*1000;
      tete2.setAttribute("x1", xx); tete2.setAttribute("x2", xx);
      tete2.setAttribute("opacity", 1);
    }}
    var w = box.clientWidth;
    if (x < box.scrollLeft + w * 0.2 || x > box.scrollLeft + w * 0.8)
      box.scrollLeft = Math.max(0, x - w * 0.4);
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
