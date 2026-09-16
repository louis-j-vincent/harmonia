"""tests/test_jam_redecode.py — l'écran Jam appelle `redecode` au bon format.

Trouvé le 2026-09-16 en inventoriant les décisions de la chaîne d'accords.

`musx.redecode` rendait `(segments, latence)` tant que la recherche de latence
existait. Elle a été retirée le 2026-09-15 (`ea539de`) et la fonction rend
depuis une simple liste. Les deux appels de la chaîne d'analyse ont été mis à
jour le jour même — `pipeline.py` et `folding.py` — mais le TROISIÈME, celui du
Jam (`harmonia_min/jam.py` à l'époque, déménagé en `harmonia/jam.py` au sprint 22
du 2026-09-16 ; la route `/api/jam` de l'app vivante l'importe),
ne l'a pas été : il dépaquetait encore deux valeurs.

Conséquence : `ValueError: too many values to unpack` à chaque passe de
décodage du Jam, c'est-à-dire un écran mort. Personne ne l'a vu parce que le
Jam demande un micro et ne tourne dans aucun test.

Ce test ne joue pas de son : il vérifie la seule chose qui a cassé, que le site
d'appel accepte ce que la fonction rend vraiment. C'est la garde qui manquait
pour qu'un changement de signature fasse rougir ses trois appelants et pas
seulement deux.

    .venv/bin/python -m pytest tests/test_jam_redecode.py -v
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from harmonia import musx as musx_neuf

APPELANTS = [
    Path("harmonia/pipeline.py"),
    Path("harmonia/folding.py"),
    Path("harmonia/jam.py"),  # déménagé de harmonia_min/ au sprint 22 (2026-09-16)
]


def _cibles_des_appels(source: str) -> list[ast.AST]:
    """Les nœuds d'affectation dont la valeur est un appel à `redecode`."""
    arbre = ast.parse(source)
    out = []
    for n in ast.walk(arbre):
        if not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Call):
            continue
        f = n.value.func
        nom = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if nom == "redecode":
            out.extend(n.targets)
    return out


def test_redecode_rend_une_liste_pas_un_couple():
    """La source de vérité : ce que la fonction rend aujourd'hui."""
    ann = inspect.signature(musx_neuf.redecode).return_annotation
    assert "list" in str(ann), ann
    assert "tuple[list" not in str(ann).replace(" ", "")


def test_aucun_appelant_ne_depaquette_deux_valeurs():
    """Les trois sites d'appel, dont celui du Jam que la route /api/jam importe."""
    fautifs = []
    for p in APPELANTS:
        if not p.exists():
            continue
        for cible in _cibles_des_appels(p.read_text(encoding="utf-8")):
            if isinstance(cible, (ast.Tuple, ast.List)):
                fautifs.append(f"{p}: {ast.unparse(cible)} = redecode(...)")
    assert not fautifs, (
        "un appelant dépaquette encore le couple retiré le 2026-09-15 : "
        + " · ".join(fautifs))


def test_les_trois_appelants_existent_toujours():
    """Si un fichier est renommé, ce test doit tomber plutôt que passer à vide."""
    trouves = [p for p in APPELANTS
               if p.exists() and "redecode" in p.read_text(encoding="utf-8")]
    assert len(trouves) == 3, f"appelants trouvés : {trouves}"
