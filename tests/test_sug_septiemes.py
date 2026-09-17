"""tests/test_sug_septiemes.py — le compas peut proposer un maj7 et un min7.

Louis, 2026-09-17 : « typiquement l'endroit ou j'ai marqué un Emaj7, ca ne le
proposait jamais, c'est pas detecté par musx les major 7th ? ».

Mesuré sur la bibliothèque avant de répondre : `^7` apparaissait dans les
suggestions 177 fois, et l'accord ÉCRIT était déjà ce maj7 les 177 fois. Comme
ALTERNATIVE proposée : zéro. Idem pour `-7` (348 accords écrits dans la
bibliothèque, jamais proposables).

LA CAUSE. L'espace de candidats est 12 racines × CINQ familles
(`QUAL5` = maj / min / dom / hdim / dim), et `_TRIAD_SEV_TO_QUAL5` y replie les
septièmes : ("maj","maj7") -> "maj" et ("min","b7") -> "min". `Q5_TAIL` écrit
ensuite ces familles `""` et `"-"`. Un maj7 devenait donc un `E`, un min7 un
`E-`. Seul l'accord déjà écrit gardait sa queue, d'où les 177 sur 177.

CE QUI CHANGE. Le classement ne bouge PAS — c'est le même espace à 60 cases,
les mêmes probabilités, le même ordre. Seul le NOM du candidat est raffiné,
avec une information que musx produit déjà et que `acoustic_logp_musx` utilise
depuis toujours pour scorer : la tête de septième (`probs[2]`, quatre colonnes
— aucune / maj7 / b7 / bb7). On ne devine rien, on cesse de jeter.

La règle est volontairement conservatrice et sans seuil nouveau : une famille
« maj » ne devient `^7` que si la septième la plus probable EST la maj7, une
famille « min » ne devient `-7` que si c'est la b7. Les autres familles disent
déjà leur septième (`dom` -> `7`, `hdim` -> `-7b5`) et ne bougent pas.

    .venv/bin/python -m pytest tests/test_sug_septiemes.py -v
"""
from __future__ import annotations

import numpy as np

from harmonia.span_rescore import QUAL5, Q5_TAIL, queue_du_candidat

MAJ, MIN, DOM, HDIM, DIM = range(5)
NONE, MAJ7, B7, BB7 = range(4)


def _s7(i: int):
    v = np.full(4, 0.05)
    v[i] = 0.85
    return v


def test_un_majeur_avec_une_maj7_se_nomme_maj7():
    assert queue_du_candidat(MAJ, _s7(MAJ7)) == "^7"


def test_un_mineur_avec_une_b7_se_nomme_min7():
    assert queue_du_candidat(MIN, _s7(B7)) == "-7"


def test_un_majeur_sans_septieme_reste_une_triade():
    assert queue_du_candidat(MAJ, _s7(NONE)) == ""


def test_un_mineur_sans_septieme_reste_un_mineur():
    assert queue_du_candidat(MIN, _s7(NONE)) == "-"


def test_un_majeur_avec_une_b7_ne_devient_pas_maj7():
    """Cette combinaison a déjà sa famille (`dom`). La nommer `^7` serait
    contredire le repli qui a produit le classement."""
    assert queue_du_candidat(MAJ, _s7(B7)) == ""


def test_les_familles_qui_disent_deja_leur_septieme_ne_bougent_pas():
    for fam in (DOM, HDIM, DIM):
        for sev in (NONE, MAJ7, B7, BB7):
            assert queue_du_candidat(fam, _s7(sev)) == Q5_TAIL[fam], (fam, sev)


def test_sans_tete_de_septieme_on_retombe_sur_lancienne_queue():
    """Un chart d'avant, ou un appelant qui n'a pas la tête : rien ne casse."""
    for fam in range(len(QUAL5)):
        assert queue_du_candidat(fam, None) == Q5_TAIL[fam]
        assert queue_du_candidat(fam, np.zeros(4)) == Q5_TAIL[fam]
