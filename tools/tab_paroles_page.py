"""tools/tab_paroles_page.py — la feuille « accords sur les mots », en local.

    .venv/bin/python -m tools.tab_paroles_page "Ben E King Stand By Me" \
        ben_e_king_stand_by_me_audio SORTIE.html

Va chercher le tab le mieux noté, l'aligne sur NOTRE audio (`tab_chart.poser`),
et réécrit ses lignes avec NOS accords à la place des siens
(`tab_paroles.feuille`). La page sort en local, sur la machine de Louis ; elle
n'est pas commitée et ne passe par aucune route publique de l'app — le texte
appartient aux éditeurs, la vue est un outil de travail personnel.

CE QU'IL FAUT REGARDER SUR LA PAGE, et c'est la seule question de cette étape :
est-ce que NOS accords tombent sur la bonne syllabe ? L'accord avec le tab
n'est PAS le critère — Louis a tranché en juillet que nos charts battent les
tabs. L'accord du tab est affiché en petit à côté du nôtre pour que le
désaccord se voie, pas pour arbitrer.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from harmonia.integrations import tab_paroles as TP
from harmonia.integrations.tab_chart import poser
from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs
from harmonia.settings import SETTINGS

CSS = """
:root{--pap:#f6f3ec;--enc:#1c1c1c;--nous:#8c2f2f;--eux:#9a9287;--lig:#e2dcd0}
@media(prefers-color-scheme:dark){:root{--pap:#16150f;--enc:#ece7dc;
  --nous:#e08a7a;--eux:#7d766b;--lig:#2b2820}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--pap);color:var(--enc);
  font:16px/1.5 -apple-system,system-ui,sans-serif;max-width:52rem;margin:0 auto}
h1{font:600 22px/1.25 Georgia,serif;margin:0 0 2px}
.sous{color:var(--eux);font-size:13px;margin:0 0 18px}
.bloc{margin:0 0 26px;border-top:1px solid var(--lig);padding-top:14px}
pre{margin:0;font:14px/1.15 ui-monospace,SFMono-Regular,Menlo,monospace;
  white-space:pre;overflow-x:auto;-webkit-overflow-scrolling:touch}
.acc{color:var(--nous);font-weight:700}
.eux{color:var(--eux);font-weight:400;font-size:11px}
.mots{color:var(--enc)}
a{color:var(--nous)}
"""


def ligne_accords(accords, largeur: int, montrer_eux: bool) -> str:
    """La ligne d'accords, NOS noms posés aux colonnes du tab.

    Deux accords qui se chevauchent après substitution (le nôtre est plus long
    que le sien) sont décalés d'un cran plutôt qu'écrasés : perdre un accord en
    silence serait pire que décaler la colonne d'un caractère.
    """
    buf = []
    col = 0
    for a in sorted(accords, key=lambda x: x["colonne"]):
        cible = max(a["colonne"], col + 1 if col else 0)
        buf.append(" " * max(0, cible - col))
        col = max(col, cible)
        nom = a["notre"]
        morceau = f'<span class="acc">{html.escape(nom)}</span>'
        if montrer_eux and a["leur"] != nom:
            morceau += f'<span class="eux">{html.escape(a["leur"])}</span>'
            col += len(nom) + len(a["leur"])
        else:
            col += len(nom)
        buf.append(morceau)
    return "".join(buf)


def page(titre: str, feuille: list[dict], tab: dict, decalage: int,
         montrer_eux: bool = True) -> str:
    blocs = []
    for l in feuille:
        blocs.append('<pre>' + ligne_accords(l["accords"], 0, montrer_eux)
                     + "\n" + f'<span class="mots">{html.escape(l["texte"])}</span>'
                     + "</pre>")
    note = (f'tab source : <a href="{html.escape(tab.get("url") or "")}">'
            f'{html.escape(tab.get("titre") or "?")} — '
            f'{html.escape(tab.get("artiste") or "?")}</a> '
            f'({tab.get("note")}★, {tab.get("votes")} votes)'
            f' · transposé de {decalage:+d} demi-ton(s) pour tomber sur notre audio')
    return (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(titre)}</title><style>{CSS}</style>'
            f'<h1>{html.escape(titre)}</h1><p class="sous">{note}<br>'
            f'En rouge, <b>nos</b> accords. En gris, ceux du tab quand ils '
            f'diffèrent. La question : tombent-ils sur la bonne syllabe ?</p>'
            f'<div class="bloc">{"".join(blocs)}</div></html>')


def main(requete: str, stem: str, sortie: str) -> int:
    chart = None
    for p in SETTINGS.charts_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if Path(d.get("audio_url") or "").stem == stem:
            chart = d
            break
    if chart is None:
        print(f"aucun chart pour {stem}", file=sys.stderr)
        return 2
    pose = poser(chart, requete)
    if pose is None:
        print("aucun tab utilisable — voir le journal", file=sys.stderr)
        return 3
    res = search_tabs(requete, tab_types=("Chords",), max_results=6)
    brut = fetch_tab_chords(res[0]).raw_content
    feuille = TP.feuille(chart, pose, brut)
    if not feuille:
        print("les colonnes et les accords ne concordent pas — rien posé",
              file=sys.stderr)
        return 4
    Path(sortie).write_text(
        page(chart.get("title") or stem, feuille, pose["tab"],
             int(pose.get("decalage") or 0)), encoding="utf-8")
    n = sum(len(l["accords"]) for l in feuille)
    print(f"{sortie} — {len(feuille)} ligne(s), {n} accord(s) posé(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
