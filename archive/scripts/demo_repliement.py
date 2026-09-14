#!/usr/bin/env python3
"""scripts/demo_repliement.py — replier les tours qui se ressemblent, à travers
les lettres, et ne parler de variation que quand il y a de quoi arbitrer.

Louis, 2026-08-18 :

  « c'est surtout que dans un cas il n'y a que 2 répétitions, ce n'est pas assez
    pour dire qu'il y a une variation, on considère que ce sont des mesures
    différentes ; dans les autres cas on en a assez pour arbitrer et dire qu'ils
    sont similaires (même dans This Love l'intro elle est similaire à d'autres
    parties, tu peux la replier pareil s'il n'y a pas de diffs énormes) —
    montres moi une demo sur un morceau ou deux »

LA RÈGLE DE LA DÉMO, dans l'ordre :

  1. **découper en tours** de 4 mesures, ancrés sur ses sections (pas pavés
     depuis la mesure 0 : une section de longueur impaire décale tout ce qui
     suit et le pavage coupe à cheval sur les phrases) ;
  2. **grouper par ressemblance harmonique**, SANS regarder les lettres — c'est
     ce qui permet à l'intro de This Love de rejoindre les couplets ;
  3. **compter** : un groupe de 3 tours ou plus, on replie et on note la
     variante ; un groupe de 1 ou 2, on ne replie pas — deux lectures ne
     suffisent pas à dire laquelle est la variante de l'autre, ce sont
     simplement deux passages différents ;
  4. **filtrer le bruit** : une lecture vue une seule fois sur huit tours n'est
     pas un choix du musicien, c'est du décodage. Une variante doit revenir au
     moins deux fois pour être écrite.

LE SEUIL vient de ses deux arbitrages, pas d'une optimisation : Easy On Me
A- contre C à 0,83 se replie, le pont de This Love C-7 contre G7 à 0,74 ne se
replie pas. Repères mesurés sur 5 morceaux : deux tours au texte identique sont
à 0,94, deux lettres différentes à 0,69.

CE QUE LA DÉMO NE FAIT PAS : elle n'écrit dans aucun chart.

    .venv/bin/python scripts/demo_repliement.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import math

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.folding import _bar_vecs                  # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma   # noqa: E402
from harmonia_min.sections import halfbar_features          # noqa: E402
from harmonia_min.soudure import accords_par_mesure         # noqa: E402

NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()
CHARTS = REPO / "harmonia_min" / "state" / "charts"
MARQUES = REPO / "harmonia_min" / "state" / "sections"
SORTIE = REPO / "docs" / "plots" / "demo_repliement.html"

P = 4          # le tour, en mesures
SEUIL = 0.80   # ses deux arbitrages : 0,83 se replie, 0,74 non
MIN_TOURS = 3  # « 2 répétitions, ce n'est pas assez pour dire qu'il y a une variation »


def min_var(n: int) -> int:
    """Combien de fois une lecture minoritaire doit revenir pour être écrite —
    EN RACINE DE N (Louis, 2026-08-18 : « tu peux peut-être faire une règle en
    racine de n où n est le nombre d'observations qu'on a ; plus on a
    d'observations, plus on peut être sûr qu'on a en effet différentes
    variations, modulo le fait qu'il y ait un pattern »). 2 sur 4 tours, 4 sur
    10 : plus on observe, plus une lecture doit s'imposer pour ne pas être du
    bruit de décodage."""
    return max(2, math.ceil(math.sqrt(n)))

MORCEAUX = [("min_maroon_5_this_love", "Maroon 5 — This Love"),
            ("min_X-yIEMduRXk", "Adele — Easy On Me")]


def nom(c: dict) -> str:
    return "N.C." if c.get("nc") else NOTE[c["root"] % 12] + (c.get("q") or "")


def mots(bar) -> str:
    return " ".join(nom(c) for c in bar) or "%"


def morceau(stem: str, titre: str) -> str:
    chart = json.loads((CHARTS / f"{stem}.json").read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    audio = REPO / "docs" / "audio" / Path(chart["audio_url"]).name
    if not audio.exists():
        return ""
    arr, times = extract_bothchroma(audio)
    Vb = _bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)
    bars = accords_par_mesure(chart)

    lettre = {}
    for s in chart.get("sections") or []:
        for b0, b1 in s.get("barRanges") or []:
            for b in range(b0, b1 + 1):
                lettre[b] = s["label"]

    fm = MARQUES / f"{Path(chart['audio_url']).stem}.json"
    if fm.exists():
        ancres = [(x["b0"], x["b1"])
                  for x in json.loads(fm.read_text(encoding="utf-8"))["sections"]]
    else:
        ancres = [tuple(r) for s in chart.get("sections") or []
                  for r in (s.get("barRanges") or [])]
    departs, rang = [], {}
    for b0, b1 in ancres:
        for t in range((b1 - b0 + 1) // P):
            departs.append(b0 + t * P)
            rang[b0 + t * P] = t      # le rang du tour DANS sa section
    departs = sorted(set(departs))
    if len(departs) < 2:
        return ""

    def sim(a: int, b: int) -> float:
        return float(np.mean([Vb[a + k] @ Vb[b + k] for k in range(P)]))

    # groupement glouton : le tour qui ressemble au plus de monde attire les
    # siens, on recommence sur ce qui reste.
    restants, groupes = list(departs), []
    while restants:
        centre = max(restants,
                     key=lambda a: sum(sim(a, b) >= SEUIL for b in restants))
        g = [b for b in restants if sim(centre, b) >= SEUIL]
        groupes.append((centre, sorted(g)))
        restants = [b for b in restants if b not in g]
    groupes.sort(key=lambda x: x[1][0])

    blocs = []
    for centre, g in groupes:
        txts = {a: [mots(bars[a + k]) for k in range(P)] for a in g}
        lts = sorted({lettre.get(a, "?") for a in g})
        replie = len(g) >= MIN_TOURS
        divergentes = [k for k in range(P)
                       if len({txts[a][k] for a in g}) > 1]
        # L'ALTERNANCE (Louis, 2026-08-18, sur le refrain de This Love : « il y
        # a un pattern sur l'alternance de la dernière mesure, un coup c'est Ab
        # un coup c'est Bb »). Si la lecture d'une case est une FONCTION du rang
        # du tour dans sa section, ce n'est pas une variante : la boucle est
        # deux fois plus longue, et ces deux lectures en sont la 1re et la 2e
        # fin. Testé AVANT de parler de variation — sinon une alternance
        # parfaitement régulière sort en « variante », ce qui est faux.
        #
        # Le test porte sur l'accord de TÊTE de la mesure (« un coup c'est Ab un
        # coup c'est Bb » parle du premier accord). Sur le texte entier il
        # échouait : la classe impaire du refrain de This Love dit `Ab G`,
        # `Ab` et `Ab N.C.` — trois chaînes différentes pour un seul et même
        # Ab, la queue de mesure étant du bruit de décodage.
        def tete(txt):
            return txt.split(" ")[0]

        alterne = None
        for k in divergentes:
            par = {}
            for a in g:
                par.setdefault(rang.get(a, 0) % 2, set()).add(tete(txts[a][k]))
            if (len(par) == 2 and all(len(v) == 1 for v in par.values())
                    and par[0] != par[1] and alterne is None):
                alterne = k

        lignes = "".join(
            f'<tr><td class="ou"><button data-t="{grid[a]:.2f}" '
            f'data-fin="{grid[min(a + P, len(grid) - 1)]:.2f}">▶</button>'
            f'mes. {a + 1}<i>{html.escape(lettre.get(a, "?"))}</i></td>'
            + "".join(f'<td class="c{" div" if k in divergentes else ""}">'
                      f'{html.escape(txts[a][k])}</td>' for k in range(P))
            + "</tr>" for a in g)

        seuil_var = min_var(len(g))

        def _ecrire(sous):
            out = []
            for k in range(P):
                vals = [txts[a][k] for a in sous]
                uniq = sorted(set(vals), key=lambda x: (-vals.count(x), x))
                maj = uniq[0]
                var = [u for u in uniq[1:]
                       if vals.count(u) >= min_var(len(sous))]
                sup = f'<sup>{html.escape(var[0])}</sup>' if var else ""
                out.append(f'<span class="ac">{html.escape(maj)}{sup}</span>')
            return "".join(out)

        if replie:
            dd = [sim(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
            if alterne is not None:
                pairs = [a for a in g if rang.get(a, 0) % 2 == 0]
                impairs = [a for a in g if rang.get(a, 0) % 2 == 1]
                note = (f'{len(g)} tours, distance médiane '
                        f'<b>{np.median(dd):.2f}</b> — mais la mesure '
                        f'{alterne + 1} <b>alterne avec le rang du tour</b> : '
                        f'{len(pairs)} fois une lecture, {len(impairs)} fois '
                        f"l'autre, jamais mélangées. Ce n'est pas une variante — "
                        f'la boucle fait {2 * P} mesures, et ces deux lectures '
                        f'en sont la 1re et la 2e fin.')
                corps = (f'<div class="propose fin"><i>1.</i>{_ecrire(pairs)}</div>'
                         f'<div class="propose fin"><i>2.</i>{_ecrire(impairs)}</div>')
            else:
                note = (f'{len(g)} tours, distance médiane '
                        f'<b>{np.median(dd):.2f}</b> — pas d\'alternance, et '
                        f'assez d\'observations pour arbitrer : on replie, et '
                        f'une variante doit revenir <b>{seuil_var} fois</b> '
                        f'(√{len(g)}) pour être écrite.')
                corps = f'<div class="propose">{_ecrire(g)}</div>'
        else:
            note = (f'{len(g)} tour{"s" if len(g) > 1 else ""} — pas assez pour '
                    f'dire qu\'un accord est la variante de l\'autre. '
                    f'Ce sont des mesures différentes, chacune garde son texte.')
            corps = ""
        blocs.append(f"""
  <div class="grp {'ok' if replie else 'non'}">
    <h4>{len(g)} tour{'s' if len(g) > 1 else ''}
      <span>lettres du chart : {html.escape(', '.join(lts))}</span></h4>
    <table>{lignes}</table>
    <p class="note">{note}</p>
    {corps}
  </div>""")

    return f"""
<section data-audio="{chart['audio_url']}">
  <h3>{html.escape(titre)}</h3>
  <audio controls preload="none"></audio>
  {''.join(blocs)}
</section>"""


def main() -> None:
    corps = "".join(morceau(s, t) for s, t in MORCEAUX)
    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Démo — replier les tours qui se ressemblent</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --vert:#3d6b47}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:24px;margin:0 0 4px}}
h3{{font-size:18px;margin:0 0 6px}}
h4{{font-size:13px;margin:0 0 7px;display:flex;justify-content:space-between;
 align-items:baseline;gap:10px}}
h4 span{{font-weight:400;font-size:11px;color:var(--doux)}}
section{{margin:0 0 40px;padding:0 0 16px;border-bottom:2px solid var(--trait)}}
.chapo{{color:var(--doux);max-width:78ch;margin:0 0 10px}}
.regle{{background:#fff;border:1px solid var(--trait);border-radius:8px;
 padding:12px 16px;margin:14px 0 24px;max-width:78ch}}
.regle ol{{margin:0;padding-left:20px}} .regle li{{margin:3px 0}}
audio{{width:100%;max-width:420px;height:34px;margin:4px 0 12px;display:block}}
.grp{{border:1px solid var(--trait);border-radius:9px;padding:11px 13px;
 margin:0 0 12px;background:#fff}}
.grp.ok{{border-left:4px solid var(--vert)}}
.grp.non{{border-left:4px solid var(--trait);background:#faf7ef}}
table{{border-collapse:collapse;font-size:13.5px;width:100%}}
td{{padding:3px 8px 3px 0;border-top:1px solid #f2ecdf;white-space:nowrap}}
td.ou{{color:var(--doux);font-size:12.5px}}
td.ou i{{font-style:normal;font-size:10.5px;margin-left:5px;
 background:#efe9db;border-radius:4px;padding:1px 5px}}
td.c{{font-variant-numeric:tabular-nums}}
td.c.div{{background:#f7e8e1;color:var(--rouge);font-weight:600;border-radius:4px}}
.note{{font-size:12.5px;color:var(--doux);margin:8px 0 0}}
.note b{{color:var(--fg)}}
.propose{{margin:9px 0 0;padding:9px 12px;background:#eef3ee;border-radius:8px;
 display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
.ac{{font:600 19px Georgia,serif}}
.ac sup{{font:600 11px Georgia,serif;color:var(--rouge);vertical-align:super;
 margin-left:1px}}
.propose.fin{{margin-top:6px;align-items:baseline}}
.propose.fin i{{font:700 13px Georgia,serif;font-style:normal;color:var(--doux);
 border:1px solid var(--trait);border-bottom:none;border-radius:4px 4px 0 0;
 padding:0 7px}}
button{{font:inherit;font-size:12px;padding:1px 7px;margin-right:6px;
 border:1px solid var(--trait);background:#fff;border-radius:6px;cursor:pointer}}
button:hover{{background:#f2ecdd}}
</style>
<h1>Replier les tours qui se ressemblent</h1>
<p class="chapo">On découpe le morceau en tours de {P} mesures, on les groupe par
ressemblance harmonique <b>sans regarder les lettres</b>, et on ne parle de
variation que dans les groupes assez fournis pour arbitrer.</p>

<div class="regle"><ol>
<li>les tours sont ancrés sur les sections, pas pavés depuis la mesure 1 ;</li>
<li>on groupe au-dessus de <b>{SEUIL:.2f}</b> de distance harmonique — seuil pris
sur tes deux arbitrages : A- contre C à 0,83 se replie, C-7 contre G7 à 0,74
non ;</li>
<li><b>{MIN_TOURS} tours minimum</b> pour replier. En dessous, deux lectures ne
disent pas laquelle est la variante de l'autre : ce sont des mesures
différentes ;</li>
<li>si une case <b>alterne avec le rang du tour</b>, ce n'est pas une variante :
la boucle est deux fois plus longue, on écrit une 1re et une 2e fin ;</li>
<li>sinon une variante doit revenir <b>√n fois</b> — 2 sur 4 tours, 4 sur 10 :
plus on observe, plus une lecture doit s'imposer pour ne pas être du bruit.</li>
</ol></div>
{corps}
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
    print(f"  {SORTIE.relative_to(REPO)}")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
