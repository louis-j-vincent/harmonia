"""Sauvegarder la vérité de Louis hors du dossier jetable.

    python scripts/sync_truth.py          # state/ -> docs/ground_truth/
    python scripts/sync_truth.py --restore  # docs/ground_truth/ -> state/

Louis, 2026-08-07 : « tous les morceaux que j'ai déjà annotés, c'est bon tu les
as ? tu log leur ground truth ? »

La réponse était NON, et c'était grave. L'outil d'annotation écrit dans
`harmonia_min/state/sections/`, et `harmonia_min/.gitignore` ignore `state/` en
entier — à juste titre pour des rapports HTML de 3 Mo régénérables, mais ses
annotations, elles, ne se régénèrent pas : ce sont des heures d'écoute et le
seul étalon contre lequel tout le reste est mesuré. Seize morceaux ne tenaient
qu'à un `rm -rf state/`.

Elles sont donc recopiées dans `docs/ground_truth/sections/`, qui est suivi. Le
serveur continue de lire et d'écrire dans `state/` — on ne change pas le chemin
chaud, on en garde une copie versionnée — et `--restore` rapatrie si `state/`
est perdu.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIVE = HERE / "harmonia_min" / "state" / "sections"
KEEP = HERE / "docs" / "ground_truth" / "sections"


def main():
    restore = "--restore" in sys.argv
    src, dst = (KEEP, LIVE) if restore else (LIVE, KEEP)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            print(f"  !! illisible : {p.name}")
            continue
        if not d.get("sections"):
            continue
        shutil.copy2(p, dst / p.name)
        n += 1
        print(f"  {p.stem[:52]:<54} {len(d['sections']):>2} sections"
              f"  {'validé' if d.get('validated') else ''}")
    print(f"\n{n} morceaux {'restaurés' if restore else 'sauvegardés'} -> "
          f"{dst.relative_to(HERE)}")


if __name__ == "__main__":
    main()
