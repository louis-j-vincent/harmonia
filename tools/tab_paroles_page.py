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

import json
import sys
from pathlib import Path

from harmonia.integrations import tab_paroles as TP
from harmonia.integrations.tab_chart import poser
from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs
from harmonia.settings import SETTINGS

from harmonia.integrations.tab_paroles import page  # noqa: E402


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
    # LE MÊME TAB QUE CELUI SUR LEQUEL ON S'EST CALÉ. Ce script refaisait une
    # recherche APRÈS `poser` : il pouvait donc caler la grille sur un tab et
    # lire le texte d'un autre. `poser` expose son choix, on le reprend.
    brut = pose.get("brut") or ""
    if not brut:
        print("le tab calé n'expose pas son texte", file=sys.stderr)
        return 3
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
