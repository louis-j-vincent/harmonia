#!/usr/bin/env python3
"""scripts/page_another_day.py — Another Day : où tombe la barre de mesure ?

Louis, 2026-08-18 : « je ne comprends pas la distinction entre début de mesure
et barre. Et je ne comprends pas ce que je suis censé voir. En fait j'ai
l'impression qu'il y a un drift, mais qu'on ne le voit pas. C'est quoi le
souci ? »

Deux reproches justes, et la deuxième question a une vraie réponse.

  * « début de mesure » et « barre », c'est LA MÊME CHOSE — je les avais
    dessinés séparément sans le dire. Sauf qu'ici ils ne tombent pas au même
    endroit, et c'est tout le sujet.
  * ce n'est pas un drift : c'est un décalage CONSTANT d'un temps. Les
    downbeats de Beat This! votent à 101 voix sur 103 pour une phase ; les
    débuts d'accord votent à 80 sur 89 pour la phase d'à côté, un temps plus
    tôt ; et c'est le vote des accords qui l'emporte (`_phase_correction`).
    La barre du chart tombe donc un temps AVANT le downbeat du traqueur, sur
    tout le morceau.

Qui a raison est une question d'oreille — la sienne. La page pose les deux
lectures l'une sous l'autre et les fait écouter.

    .venv/bin/python scripts/page_another_day.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "another_day_playhead.html"
PXS = 46          # pixels par seconde


def main() -> None:
    d = json.loads(Path("/tmp/ad.json").read_text())
    dur, P = d["dur"], d["P"]
    W = int(dur * PXS) + 40
    x = lambda t: 20 + t * PXS
    db = set(d["db"])
    ch = d["chords"]

    HB, HD, HC = 40, 34, 30
    beats = "".join(
        f'<line x1="{x(t):.1f}" y1="{HB-24 if t in db else HB-16}" '
        f'x2="{x(t):.1f}" y2="{HB-4}" '
        f'stroke="{"#3f3a31" if t in db else "#b9b2a1"}" '
        f'stroke-width="{2 if t in db else 1}"/>' for t in d["beats"])
    barres = "".join(
        f'<line x1="{x(t):.1f}" y1="4" x2="{x(t):.1f}" y2="{HD-4}" '
        f'stroke="#a4462b" stroke-width="1.8"/>' for t in d["bar"])
    accords = "".join(
        f'<circle cx="{x(t):.1f}" cy="{HC/2}" r="3" fill="#3f6f8f"/>'
        for t in ch)
    reg = "".join(f'<text x="{x(k):.1f}" y="{HC-2}" font-size="9" fill="#8a8371" '
                  f'text-anchor="middle">{k}s</text>'
                  for k in range(0, int(dur)+1, 5))
    T = HB + HD + HC

    vdb = d["vote_db"]; vch = d["vote_ch"]
    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Another Day — où tombe la barre de mesure ?</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 50px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 6px}}
.chapo{{color:var(--doux);margin:0 0 12px;font-size:13.5px}}
.rep{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--rouge);
 border-radius:8px;padding:12px 14px;margin:0 0 14px;font-size:13.5px}}
.rep b{{color:var(--rouge)}}
.votes{{display:flex;gap:8px;margin:0 0 14px;flex-wrap:wrap}}
.votes div{{flex:1;min-width:150px;background:#fff;border:1px solid var(--trait);
 border-radius:8px;padding:8px 11px;font-size:12px}}
.votes b{{display:block;font-size:15px;margin-bottom:2px}}
.votes i{{font-style:normal;color:var(--doux)}}
audio{{width:100%;height:34px;margin:0 0 8px;display:block}}
.scroll{{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;
 border:1px solid var(--trait);border-radius:7px;background:#fdfbf5}}
.lab{{font:600 9.5px -apple-system,sans-serif;fill:#8a8371}}
.leg{{font-size:12px;color:var(--doux);margin:6px 0 0}}
.leg b.n{{color:#3f3a31}} .leg b.r{{color:var(--rouge)}} .leg b.b{{color:var(--bleu)}}
svg{{display:block}}
</style>
<h1>Another Day — où tombe la barre de mesure&nbsp;?</h1>
<p class="chapo">Tu avais raison de ne pas comprendre : « début de mesure » et
« barre » sont la même chose, et je les avais dessinés comme deux choses. Le
vrai sujet, c'est qu'ici ils ne tombent pas au même endroit.</p>

<div class="rep">Ce n'est <b>pas un drift</b> : c'est un décalage
<b>constant d'un temps</b> ({d["decalage_ms"]} ms, et un temps en fait
{round(P*1000)}). Les downbeats de Beat This! votent pour une phase, les débuts
d'accord pour celle d'à côté — un temps plus tôt — et ce sont les accords qui
l'emportent. La barre du chart tombe donc systématiquement un temps avant le
downbeat du traqueur.</div>

<div class="votes">
  <div><b>{max(vdb.values())} sur {sum(vdb.values())}</b>
    <i>downbeats de Beat This! d'accord entre eux</i></div>
  <div><b>{max(vch.values())} sur {sum(vch.values())}</b>
    <i>débuts d'accord d'accord entre eux — un temps plus tôt</i></div>
  <div><b>les accords gagnent</b>
    <i>la barre recule d'un temps sur tout le morceau</i></div>
</div>

<audio controls preload="none"></audio>
<div class="scroll">
  <svg width="{W}" height="{T}">
    <text x="3" y="11" class="lab">BEAT THIS!</text>
    <g>{beats}</g>
    <text x="3" y="{HB+11}" class="lab">BARRES DU CHART</text>
    <g transform="translate(0,{HB})">{barres}</g>
    <text x="3" y="{HB+HD+11}" class="lab">ACCORDS</text>
    <g transform="translate(0,{HB+HD})">{accords}{reg}</g>
    <line class="tete" x1="20" y1="0" x2="20" y2="{T}" stroke="#a4462b"
          stroke-width="1.4" opacity="0"/>
  </svg>
</div>
<p class="leg"><b class="n">Traits noirs épais</b> = les downbeats de Beat
This!, traits fins = les autres temps · <b class="r">traits rouges</b> = les
barres de mesure du chart · <b class="b">points bleus</b> = les débuts
d'accord. Clique pour placer la lecture : écoute si le « un » tombe sur le
rouge ou sur le noir.</p>
<script>
(function(){{
  var el = document.querySelector("audio"), pxs = {PXS};
  var box = document.querySelector(".scroll"), tete = document.querySelector(".tete");
  el.src = "{d['audio']}";
  if (window.fetch) fetch("{d['audio']}")
    .then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  box.addEventListener("click", function(ev){{
    var r = box.getBoundingClientRect();
    var t = (ev.clientX - r.left + box.scrollLeft - 20)/pxs;
    try {{ el.currentTime = Math.max(0, t); el.play(); }} catch(e){{}}
  }});
  /* timeupdate, jamais requestAnimationFrame (Safari iOS). */
  el.addEventListener("timeupdate", function(){{
    var x = 20 + el.currentTime*pxs;
    tete.setAttribute("x1", x); tete.setAttribute("x2", x);
    tete.setAttribute("opacity", 1);
    var w = box.clientWidth;
    if (x < box.scrollLeft + w*0.25 || x > box.scrollLeft + w*0.75)
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
