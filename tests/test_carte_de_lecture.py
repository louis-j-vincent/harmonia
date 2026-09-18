"""La carte de lecture porte TOUTES les passes d'une section, pas une seule.

`barSpans[r]` donne l'instant de la mesure écrite `r` à chaque passage. C'est
ce que la tête de lecture consomme (`S.barTimeIndex` côté app). Le 2026-09-18,
`soudure.sections_pour_chart` n'en écrivait qu'un — celui du passage
représentatif — alors que `reps`, `spans` et `barRanges` listaient bien les N
occurrences. Résultat : la case allumée suivait le premier passage puis
s'éteignait pour tous les suivants (Louis : « la case highlighted ne suit plus
à un moment »). 27 charts sur 85 étaient touchés, tous passés par
`tools.golden --publish`, qui appelle cette fonction.

Le test est écrit ROUGE D'ABORD contre l'ancien comportement : avec l'ancienne
ligne, `fenetres == [1]*8` et la première assertion tombe.
"""
from __future__ import annotations

import pytest

from harmonia.soudure import _carte_de_lecture, sections_pour_chart

GRILLE = [i * 2.7 for i in range(67)]


def _motif(roots):
    return [[{"root": r, "q": "", "nc": False, "c": 0.9, "t0": 0.0, "t1": 1.0}]
            for r in roots]


def _bars(depart, roots, n=66):
    bars: list[list] = [[] for _ in range(n)]
    motif = _motif(roots)
    for a in depart:
        for k, bar in enumerate(motif):
            bars[a + k] = [dict(c) for c in bar]
    return bars


def test_chaque_passage_a_son_instant_sur_chaque_mesure_ecrite():
    roots = (10, 10, 3, 2, 7, 0, 5, 5)
    depart = (4, 14, 30, 46, 54)
    secs = [{"label": "A", "mesure_debut": a + 1, "mesure_fin": a + 8}
            for a in depart]
    out = sections_pour_chart({"barGrid": GRILLE, "nBars": 66, "bpb": 4},
                              secs, bars=_bars(depart, roots), fold_report={})
    assert len(out) == 1
    sec = out[0]
    assert sec["reps"] == len(depart)
    fenetres = [len(r) for r in sec["barSpans"]]
    # une fenêtre par occurrence, sur CHAQUE rangée écrite
    assert fenetres == [len(depart)] * len(sec["bars"]), fenetres
    # et aucun trou : les instants couvrent les 5 passages bout à bout
    plats = sorted(w for row in sec["barSpans"] for w in row)
    for (_, fin), (debut, _) in zip(plats, plats[1:]):
        assert debut >= fin - 1e-6


def test_aucune_mesure_jouee_n_est_perdue_meme_si_le_passage_est_plus_court():
    """Un passage plus court que le bloc écrit RÉPARTIT ses mesures : elles
    n'ont pas toutes une rangée à elles, mais aucune n'est jetée. C'est la loi
    déjà arbitrée dans `folding.minimal_fold` (carte proportionnelle), pas un
    choix refait ici — voir le test d'accord entre les deux chemins."""
    occ = [(0, 7), (20, 23)]
    rows = _carte_de_lecture(occ, 0, 7, GRILLE, 66)
    assert len(rows) == 8
    total = sum(len(r) for r in rows)
    assert total == sum(b - a + 1 for a, b in occ) == 12


def test_les_deux_chemins_qui_ecrivent_un_chart_rendent_la_meme_carte():
    """`soudure.sections_pour_chart` (découpage à la main, republication) et
    `folding.minimal_fold` (pipeline) écrivent le même chart : ils ne peuvent
    pas avoir deux cartes de lecture. C'est CE test qui garde les deux en
    phase — la divergence du 2026-09-18 est passée inaperçue parce que rien
    ne les comparait."""
    from harmonia.folding import minimal_fold
    roots = (10, 10, 3, 2, 7, 0, 5, 5)
    depart = (4, 14, 30, 46, 54)
    bars = _bars(depart, roots)
    ranges = [[a, a + 7] for a in depart]
    par_pipeline = minimal_fold(
        [{"id": "LA", "label": "A", "barRanges": ranges}], bars, GRILLE, {})
    par_soudure = sections_pour_chart(
        {"barGrid": GRILLE, "nBars": 66, "bpb": 4},
        [{"label": "A", "mesure_debut": a + 1, "mesure_fin": a + 8}
         for a in depart],
        bars=bars, fold_report={})
    assert len(par_pipeline) == len(par_soudure) == 1
    assert par_pipeline[0]["barSpans"] == par_soudure[0]["barSpans"]


def test_une_seule_occurrence_reste_une_fenetre_par_rangee():
    rows = _carte_de_lecture([(0, 3)], 0, 3, GRILLE, 66)
    assert [len(r) for r in rows] == [1, 1, 1, 1]
    assert rows[0] == [[GRILLE[0], GRILLE[1]]]


@pytest.mark.parametrize("lb,lk", [(8, 8), (8, 4), (4, 8), (2, 5), (5, 2)])
def test_toutes_les_mesures_du_passage_sont_placees(lb, lk):
    """Quelle que soit la longueur, aucune mesure jouée n'est jetée et aucune
    fenêtre n'est inventée."""
    rows = _carte_de_lecture([(0, lk - 1)], 0, lb - 1, GRILLE, 66)
    assert len(rows) == lb
    assert sum(len(r) for r in rows) == lk
