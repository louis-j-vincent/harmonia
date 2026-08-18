#!/usr/bin/env python3
"""scripts/page_du_brut_au_pli.py — du chart brut annoté au chart replié.

Louis, 2026-08-18 : « dans annotation à la main j'ai que deux sections A et B,
alors pourquoi on en ressort + quand on les écrit ? Fais moi une page
explicative de comment on va du chart brut annoté au chart replié en sections. »

La réponse tient en deux phrases, et la page les montre mesure par mesure sur
Another Day :

  * **son annotation n'entre pas dans le chart.** Le chart est découpé par le
    détecteur de la pipeline ; ses lettres à lui vivent dans un fichier à part
    et n'y entrent que s'il appuie sur « Valider les sections ».
  * **ses sections font 16 mesures, celles de la machine 8.** Son A est fait de
    deux moitiés identiques (A + A) ; son B est fait de deux moitiés
    DIFFÉRENTES, que la machine appelle B et C. D'où deux lettres chez lui et
    quatre chez elle.

    .venv/bin/python scripts/page_du_brut_au_pli.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "du_brut_au_pli.html"
COUL = {"A": "#3f6f8f", "B": "#a4462b", "C": "#3d6b47", "D": "#8a6d1f",
        "intro": "#8a8371", "?": "#d8d2c6"}


def ruban(labels, grid, titre, sous):
    """Une mesure = une case, coloree par sa lettre."""
    cases = []
    prec = None
    for b, L in enumerate(labels):
        c = COUL.get(L, "#b9b2a1")
        neuf = L != prec
        cases.append(
            f'<div class="k" style="background:{c}22;border-left:{"2.5px solid "+c if neuf else "1px solid #efe9db"}" '
            f'data-t="{grid[b]:.2f}" title="mesure {b+1} — {html.escape(L)}">'
            + (f'<b style="color:{c}">{html.escape(L)}</b>' if neuf else "")
            + f'<i>{b+1}</i></div>')
        prec = L
    return (f'<h3>{titre}<span>{sous}</span></h3>'
            f'<div class="ruban">{"".join(cases)}</div>')


def main() -> None:
    d = json.loads(Path("/tmp/expl.json").read_text())
    g = d["grid"]
    n = d["nBars"]

    # les accords, mesure par mesure
    ch = "".join(
        f'<div class="k" data-t="{g[b]:.2f}"><b>{html.escape(d["brut"][b])}</b>'
        f'<i>{b+1}</i></div>' for b in range(n))

    secs = "".join(
        f'<tr><td><b style="color:{COUL.get(s["l"],"#000")}">{html.escape(s["l"])}</b></td>'
        f'<td>{s["L"]} mesures écrites</td><td>jouée {s["r"]}×</td>'
        f'<td class="ec">{html.escape(" | ".join(s["ecrit"]))}</td></tr>'
        for s in d["secs_machine"])

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Du chart brut au chart replié</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:21px;margin:0 0 6px}}
h2{{font-size:16px;margin:26px 0 6px;border-bottom:1px solid var(--trait);
 padding-bottom:5px}}
h3{{font-size:13px;margin:14px 0 5px;display:flex;justify-content:space-between;
 align-items:baseline;gap:10px}}
h3 span{{font-weight:400;font-size:11px;color:var(--doux)}}
.chapo{{color:var(--doux);margin:0 0 10px;font-size:13.5px}}
.cle{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--bleu);
 border-radius:8px;padding:11px 13px;margin:0 0 16px;font-size:13.5px}}
.cle b{{color:var(--bleu)}}
audio{{width:100%;height:34px;margin:0 0 10px;display:block}}
.ruban{{display:flex;overflow-x:auto;-webkit-overflow-scrolling:touch;
 border:1px solid var(--trait);border-radius:6px;background:#fdfbf5}}
.k{{flex:0 0 auto;width:44px;padding:5px 3px 3px;text-align:center;cursor:pointer}}
.k b{{display:block;font:700 11px -apple-system,sans-serif;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.k i{{font-style:normal;font-size:9px;color:#b0a894}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin:6px 0}}
td{{padding:4px 8px 4px 0;border-top:1px solid #efe9db;vertical-align:top}}
td.ec{{font-variant-numeric:tabular-nums;color:var(--doux)}}
.bug{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--rouge);
 border-radius:8px;padding:11px 13px;margin:14px 0;font-size:13.5px}}
.bug b{{color:var(--rouge)}}
code{{background:#efe9db;border-radius:4px;padding:1px 5px;font-size:12px}}
</style>
<h1>Du chart brut au chart replié</h1>
<p class="chapo">Another Day, mesure par mesure. Clique n'importe quelle case
pour y placer la lecture.</p>

<div class="cle"><b>Pourquoi tu as deux lettres et le chart en a cinq&nbsp;:</b>
tes sections font <b>16 mesures</b>, celles de la machine en font <b>8</b>. Ton
A est fait de deux moitiés identiques — elle les appelle A et A. Ton B est fait
de deux moitiés <b>différentes</b> — elle les appelle B et C. Deux lettres chez
toi, quatre chez elle, sur exactement la même musique.</div>

<audio controls preload="none"></audio>

<h2>1. Le chart brut</h2>
<p class="chapo">Le décodage à plat : un accord par mesure, aucune structure.
C'est la vérité terrain, et rien de ce qui suit ne la modifie.</p>
<div class="ruban">{ch}</div>

<h2>2. Ton annotation à la main</h2>
{ruban(d["lui"], g, "deux lettres, six passages de 16 mesures",
       f"faite sur une grille de {d['n_ann']} mesures — le chart en a {n} depuis la réparation des temps")}

<h2>3. Ce que le détecteur trouve tout seul</h2>
{ruban(d["machine"], g, "cinq lettres, des passages de 8 et 4 mesures",
       "c'est CE découpage qui est dans le chart aujourd'hui")}

<h2>4. Ce que le chart écrit une fois replié</h2>
<p class="chapo">Chaque lettre est écrite <b>une seule fois</b>, avec le nombre
de fois qu'elle se joue. C'est ce qui fait tenir le morceau sur un écran.</p>
<table>{secs}</table>

<div class="bug"><b>Le bug de lecture que tu vois.</b> La lettre <code>D</code>
est écrite sur <b>4 mesures</b>, mais son premier passage en dure <b>8</b>
(mesures 42 à 49). Entre les mesures 46 et 49, l'écran rejoue donc ses quatre
cases pendant que la musique joue autre chose — <code>Fm7 C | Fm</code>, qui
n'est écrit nulle part. La tête de lecture revient en arrière au milieu du
passage, et ça se lit comme un décalage. C'est la règle « under-fold, never
over-fold » encore en dette : le repli groupe des passages de longueurs
différentes.</div>

<h2>Et ton annotation, dans tout ça ?</h2>
<p class="chapo">Elle <b>n'entre pas</b> dans le chart. Elle vit dans son
fichier à elle, et sert à l'outil sections et aux mesures. Elle ne réécrit le
chart que si tu appuies sur <b>« Valider les sections »</b> — et là, c'est
TON découpage qui remplace celui de la machine, avec l'empilement refait sur
tes sections.<br><br>
Une chose à savoir avant de le faire ici : ton annotation a été posée sur une
grille de {d['n_ann']} mesures, et le chart en a {n} depuis la réparation des
temps jumeaux. Elle est donc décalée sur la fin — à refaire avant de valider.</p>
<script>
(function(){{
  var el = document.querySelector("audio");
  el.src = "{d['audio']}";
  if (window.fetch) fetch("{d['audio']}")
    .then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  document.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
}})();
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
