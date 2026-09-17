"""tests/test_meta_accord_recollee.py — réécrire un chart ne doit rien perdre.

Louis, 2026-09-17 : « est ce que la nouvelle facon de choper la basse et la
suggestion des top 5 accords dans le compass est bien implémentée ? j'ai
l'impression que ca a été retiré ? ».

Elle l'avait été, sur 18 morceaux sur 47 — exactement ceux dont les sections
sont validées à la main. Mesuré : `sugBass` présent sur 0 de ces 18, et sur 12
des 29 autres.

LA CAUSE. Quand un découpage de sections validé existe, le chart est réécrit
par `refold` + `soudure.sections_pour_chart`, et `soudure.accords_par_mesure`
recollait les métadonnées de chaque accord depuis une LISTE EN DUR :

    for k in ("sug", "n", "colour", "confirmed", "flag", "inflect"):

`sugBass` est arrivé dans la pipeline le 2026-09-16 à 20h01 et personne n'a
pensé à cette liste-là. Le champ disparaissait donc en silence de tout chart
passé par ce chemin — et c'est le champ que l'écran d'annotation lit pour
montrer la basse entendue.

Une liste blanche qu'il faut tenir à jour est une dette : elle ne se plaint
jamais, elle oublie. Ces tests figent la règle inverse — on garde TOUT ce que
l'accord d'origine portait, sauf ce que cette fonction recalcule elle-même —
et ils rougiront pour n'importe quel champ futur, pas seulement pour `sugBass`.

    .venv/bin/python -m pytest tests/test_meta_accord_recollee.py -v
"""
from __future__ import annotations

from harmonia.soudure import accords_par_mesure

BPB = 4
GRID = [0.0, 2.0, 4.0, 6.0]


def _chart(extra: dict):
    """Un chart minimal : une mesure, un accord, plus les champs de `extra`."""
    ecrit = {"root": 0, "q": "", "bass": -1, "nc": False, "carry": False,
             "beat": 0, "bar": 0, "t0": 0.0, "t1": 2.0, **extra}
    return {
        "barGrid": GRID, "nBars": 3, "bpb": BPB,
        "prompter": {"chords": [{"root": 0, "q": "", "bass": -1, "nc": False,
                                 "t0": 0.0, "t1": 2.0, "c": 0.9}]},
        "sections": [{"label": "A", "barRanges": [[0, 3]],
                      "bars": [[ecrit], [], []],
                      "barSpans": [[[0.0, 2.0]], [[2.0, 4.0]], [[4.0, 6.0]]]}],
    }


def _premier(chart):
    bars = accords_par_mesure(chart)
    assert bars and bars[0], "la mesure 1 doit porter un accord"
    return bars[0][0]


def test_sugbass_survit_a_la_reecriture():
    """Le champ qui manquait : la basse entendue, lue par l'écran d'annotation."""
    c = _premier(_chart({"sugBass": [{"pc": 7, "c": 0.41}, {"pc": 0, "c": 0.2}]}))
    assert c.get("sugBass") == [{"pc": 7, "c": 0.41}, {"pc": 0, "c": 0.2}]


def test_sug_survit_toujours():
    """Celui qui marchait déjà — il ne doit pas casser en réparant l'autre."""
    c = _premier(_chart({"sug": [{"root": 0, "q": "", "c": 0.8}]}))
    assert c.get("sug") == [{"root": 0, "q": "", "c": 0.8}]


def test_un_champ_inconnu_survit_aussi():
    """LA vraie garde : la règle ne doit pas être une liste à tenir à jour.

    Si quelqu'un ajoute demain un champ à un accord dans la pipeline, il doit
    traverser ce chemin sans que personne n'ait à y penser.
    """
    c = _premier(_chart({"unChampQuiNexistePasEncore": {"a": 1}}))
    assert c.get("unChampQuiNexistePasEncore") == {"a": 1}


def test_les_champs_recalcules_ne_sont_pas_ecrases():
    """`t0`, `bar`, `beat`, `carry` sont recalculés depuis la grille : la
    métadonnée de l'accord écrit ne doit pas les réimporter, sinon un chart
    replié ferait revenir des instants d'un autre passage."""
    ch = _chart({"bar": 99, "beat": 3, "t0": 123.0, "t1": 456.0, "carry": True})
    c = _premier(ch)
    assert c["bar"] == 0 and c["beat"] == 0
    assert c["t0"] == 0.0 and c["t1"] == 2.0
    assert c["carry"] is False


def test_rien_a_recoller_ne_casse_pas():
    c = _premier(_chart({}))
    assert c["root"] == 0 and "sug" not in c and "sugBass" not in c
