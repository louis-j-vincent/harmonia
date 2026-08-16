#!/usr/bin/env python3
"""scripts/demo_charts.py — les charts que produit `harmonia_min.retour`.

Louis, 2026-08-16 : « fais-moi les charts avec cet algo pour voir ».

La démo pas-à-pas (`retour_10morceaux.html`) montre COMMENT l'algo décide. Cette
page-ci montre CE QU'IL PRODUIT : le morceau replié, c'est-à-dire

  * la **forme** — la suite des sections dans l'ordre, avec le nombre de tours
    de chacune, les mesures libres comprises ;
  * chaque **section écrite une seule fois** — sa boucle, en accords ;
  * les **variantes** — les mesures où une reprise ne joue pas ce que le modèle
    écrit, et la mesure de cadence exemptée de chaque passage. C'est
    exactement ce que Louis avait demandé de ne pas perdre : « au repliement du
    chart il faudra les noter sur le chart ».

Tout est cliquable : un clic sur une mesure place la tête de lecture.

    .venv/bin/python scripts/demo_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonia_min import retour as R          # noqa: E402
from demo_retour import (MORCEAUX, TEINTES, accords_par_mesure,   # noqa: E402
                         annotation, e)

CHARTS = REPO / "harmonia_min" / "state" / "charts"
SORTIE = REPO / "docs" / "plots" / "retour_charts.html"


def forme(res: dict) -> list[dict]:
    """La suite des passages dans l'ordre du morceau, trous compris."""
    bouts = [{"label": s["label"], "L": s["L"], **su}
             for s in res["sections"] for su in s["suites"]]
    bouts.sort(key=lambda x: x["b0"])
    out, curseur = [], 0
    for b in bouts:
        if b["b0"] > curseur:
            out.append({"libre": True, "b0": curseur, "b1": b["b0"] - 1})
        out.append({"libre": False, **b})
        curseur = b["b1"] + 1
    if curseur < res["n_mesures"]:
        out.append({"libre": True, "b0": curseur, "b1": res["n_mesures"] - 1})
    return out


def bloc(fichier: str, titre: str) -> str:
    chart = json.loads((CHARTS / f"{fichier}.json").read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    stem = Path(chart["audio_url"]).stem
    S = R.ssm_mesures(chart)
    if S is None:
        return f'<section><h2>{e(titre)}</h2><p>pas de chart.</p></section>'
    res = R.sections(S)
    acc = accords_par_mesure(chart)
    n = res["n_mesures"]
    coul = {s["label"]: TEINTES[i % len(TEINTES)]
            for i, s in enumerate(res["sections"])}

    # ── la forme ────────────────────────────────────────────────────────────
    cases = []
    for b in forme(res):
        if b["libre"]:
            cases.append(
                f'<b class="bout libre" data-t="{grid[b["b0"]]:.3f}">'
                f'<u>·</u><s>{b["b0"]}–{b["b1"]}</s></b>')
        else:
            cases.append(
                f'<b class="bout" style="--c:{coul[b["label"]]}" '
                f'data-t="{grid[b["b0"]]:.3f}">'
                f'<u>{e(b["label"])} <em>×{b["tours"]}</em></u>'
                f'<s>{b["b0"]}–{b["b1"]}</s></b>')
    la_forme = f'<div class="forme">{"".join(cases)}</div>'

    # ── chaque section, écrite une fois ─────────────────────────────────────
    blocs = []
    for s in res["sections"]:
        m0, L = s["modele"], s["L"]
        grille = "".join(
            f'<b class="mes" data-t="{grid[m0 + t]:.3f}">'
            f'<i>{e(acc[m0 + t])}</i><s>{m0 + t}</s></b>' for t in range(L))

        notes = []
        for su in s["suites"]:
            bouts = []
            for v in su["variante"]:
                bouts.append(f'<b class="v">m.{v["mesure"]} '
                             f'<i>{e(acc[v["mesure"]])}</i> au lieu de '
                             f'<i>{e(acc[v["mesure_modele"]])}</i></b>')
            if su["exemptee"]:
                x = su["exemptee"]
                if e(acc[x["mesure"]]) != e(acc[x["mesure_modele"]]):
                    bouts.append(f'<b class="v cad">cadence m.{x["mesure"]} '
                                 f'<i>{e(acc[x["mesure"]])}</i> au lieu de '
                                 f'<i>{e(acc[x["mesure_modele"]])}</i></b>')
            if bouts:
                notes.append(f'<li><span class="ou" data-t="{grid[su["b0"]]:.3f}">'
                             f'mesures {su["b0"]}–{su["b1"]}</span> '
                             f'{"".join(bouts)}</li>')
        variantes = (f'<ul class="notes">{"".join(notes)}</ul>' if notes else
                     '<p class="rien">aucune variante : toutes les reprises '
                     'jouent le modèle.</p>')

        ou = " · ".join(f'<span data-t="{grid[su["b0"]]:.3f}">{su["b0"]}–'
                        f'{su["b1"]}</span>' for su in s["suites"])
        blocs.append(
            f'<div class="sec"><h3><b class="lettre" '
            f'style="--c:{coul[s["label"]]}">{e(s["label"])}</b> '
            f'boucle de {L} mesures'
            f'<span>{sum(su["tours"] for su in s["suites"])} tours en tout — '
            f'{ou}</span></h3>'
            f'<div class="grille">{grille}</div>{variantes}</div>')

    louis = annotation(stem, n)
    vus = []
    for b in range(n):
        if louis[b] and (b == 0 or louis[b - 1] != louis[b]):
            vus.append(f'{e(louis[b])} {b}')
    sien = (f'<p class="sien"><b>Ta forme à toi&nbsp;:</b> '
            f'{" · ".join(vus)}</p>') if vus else ""

    return f"""<section data-audio="/audio/{e(stem)}.m4a">
<h2>{e(titre)}</h2>
<audio controls preload="none"></audio>
<h4>La forme</h4>
{la_forme}
{sien}
<h4>Les sections, écrites une fois</h4>
{"".join(blocs)}
</section>"""


CSS = """
:root{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558}
*{box-sizing:border-box}
body{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:1080px}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:20px;margin:0 0 10px}
h3{font-size:14px;margin:0 0 8px;display:flex;gap:10px;align-items:baseline;
 flex-wrap:wrap}
h3 span{font-size:12px;color:var(--doux);font-weight:400}
h4{font-size:11px;margin:18px 0 6px;color:var(--doux);font-weight:600;
 text-transform:uppercase;letter-spacing:.05em}
section{margin:0 0 46px;padding:0 0 10px;border-bottom:2px solid var(--trait)}
.chapo{color:var(--doux);max-width:78ch;margin:0 0 26px}
audio{width:100%;max-width:420px;height:32px;margin:0 0 6px;display:block}
.forme{display:flex;flex-wrap:wrap;gap:4px}
.bout{--c:#3d7fa6;padding:4px 8px;border-radius:5px;cursor:pointer;
 display:flex;flex-direction:column;line-height:1.3;
 background:color-mix(in srgb,var(--c) 20%,white);
 border:1px solid color-mix(in srgb,var(--c) 55%,white)}
.bout u{text-decoration:none;font-size:13px;font-weight:700;
 color:color-mix(in srgb,var(--c) 82%,black)}
.bout u em{font-style:normal;font-weight:400;font-size:11px}
.bout s{text-decoration:none;font-size:10px;color:#a49c8c}
.bout.libre{--c:#c9c2b2;opacity:.75}
.sien{font-size:12.5px;color:var(--doux);margin:10px 0 0}
.sec{margin:0 0 18px;padding:12px 14px;background:#fff;border:1px solid var(--trait);
 border-radius:8px}
.lettre{--c:#3d7fa6;display:inline-block;min-width:20px;text-align:center;
 padding:0 6px;border-radius:4px;color:#fff;background:var(--c);font-size:14px}
.grille{display:flex;flex-wrap:wrap;gap:3px;margin:0 0 10px}
.mes{width:74px;padding:6px 3px;border-radius:5px;text-align:center;
 cursor:pointer;background:#f6f2e6;border:1px solid var(--trait);
 display:flex;flex-direction:column}
.mes i{font-style:normal;font-size:14px;font-weight:600}
.mes s{text-decoration:none;font-size:9px;color:#a49c8c}
.notes{margin:0;padding-left:18px;font-size:12.5px}
.notes li{margin:0 0 4px}
.ou{color:var(--doux);cursor:pointer;text-decoration:underline dotted}
.v{display:inline-block;margin:0 4px 3px 0;padding:1px 6px;border-radius:5px;
 background:#fae7d2;font-weight:400;font-size:12px}
.v.cad{background:#e8e3f2}
.v i{font-style:normal;font-weight:700}
.rien{font-size:12.5px;color:var(--doux);margin:0}
"""

JS = """
document.querySelectorAll("section[data-audio]").forEach(function(sec){
  var el = sec.querySelector("audio"), url = sec.dataset.audio;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){ return r.ok ? r.blob() : null; })
    .then(function(b){ if (!b) return;
      var t = el.currentTime, j = !el.paused;
      el.src = URL.createObjectURL(b); el.currentTime = t; if (j) el.play();
    }).catch(function(){});
  sec.addEventListener("click", function(ev){
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try { el.currentTime = parseFloat(c.dataset.t); el.play(); } catch(e){}
  });
});
"""


def main() -> None:
    blocs = [bloc(f, t) for f, t in MORCEAUX]
    page = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Les charts de l'algo du retour</title>
<style>{CSS}</style>
<h1>Les charts que produit l'algo</h1>
<p class="chapo">Le morceau replié&nbsp;: sa <b>forme</b> — la suite des
sections avec leur nombre de tours, les passages sans section compris — puis
chaque <b>section écrite une seule fois</b>, sa boucle en accords. Sous chaque
section, les <b>variantes</b>&nbsp;: les mesures où une reprise ne joue pas ce
que le modèle écrit, et la mesure de cadence de chaque passage quand elle
change. Ce sont elles qu'il faudra porter sur le chart au lieu de recopier le
modèle. Substrat <b>{R.SUBSTRAT}</b>. Cliquez n'importe où pour écouter.</p>
{"".join(blocs)}
<script>{JS}</script>
</html>"""
    SORTIE.write_text(page, encoding="utf-8")
    print(f"écrit {SORTIE} ({len(page) / 1024:.0f} ko)")


if __name__ == "__main__":
    main()
