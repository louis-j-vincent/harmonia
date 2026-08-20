"""La dernière mesure d'un passage ne s'empile jamais avec le milieu.

Louis, 2026-08-20 : « attention quand tu empiles toujours pareil à ne pas
empiler les fins de sections qui sont vraiment différentes — je pense à This
Love ». Sa section B fait 8 mesures sur une pompe de 2 (`C- F- | B♭ E♭`) et se
termine par `A♭ G` : avec P=2, la mesure 8 tombait dans la pile des mesures
2, 4 et 6 — trois fois plus de membres qui disent autre chose.
"""
import numpy as np
import pytest

from harmonia_min import folding


def _sections(n_occ, longueur, depart=0):
    return [{"label": "B",
             "barRanges": [[depart + i * longueur, depart + (i + 1) * longueur - 1]]}
            for i in range(n_occ)]


def _piles(sections, P, fin_hors_pile):
    """Reproduit la construction de `pos_members` telle qu'elle est dans
    `fold_letter_groups` — le point exact que le garde-fou modifie."""
    pos = [[] for _ in range(P)]
    fins = []
    for s in sections:
        b0, b1 = s["barRanges"][0]
        sous_phrase = P < (b1 - b0 + 1)
        for b in range(b0, b1 + 1):
            if sous_phrase and b > b1 - fin_hors_pile:
                fins.append(b)
                continue
            pos[(b - b0) % P].append(b)
    return pos, fins


def test_la_derniere_mesure_sort_de_la_pile_de_sous_phrase():
    secs = _sections(n_occ=5, longueur=8)          # This Love B
    pos, fins = _piles(secs, P=2, fin_hors_pile=folding.FIN_SECTION_HORS_PILE)
    assert fins == [7, 15, 23, 31, 39]             # les cinq mesures 8
    for pile in pos:
        assert not set(pile) & set(fins)


def test_le_milieu_garde_toutes_ses_observations():
    """Le garde-fou ne doit pas appauvrir la pompe : 5 passages × 3 mesures
    paires encore empilables = 15 membres, pas 20 mais pas 5 non plus."""
    pos, _ = _piles(_sections(5, 8), P=2, fin_hors_pile=1)
    assert len(pos[0]) == 20                       # mesures 1,3,5,7 des 5 passages
    assert len(pos[1]) == 15                       # 2,4,6 — la 8 est sortie


def test_une_periode_egale_a_la_section_empile_les_fins_entre_elles():
    """Cas sain : quand la période EST la section, la cadence d'un passage
    n'est comparée qu'aux cadences des autres — on n'y touche pas."""
    pos, fins = _piles(_sections(5, 8), P=8, fin_hors_pile=1)
    assert fins == []
    assert pos[7] == [7, 15, 23, 31, 39]           # les cinq cadences ensemble


def test_le_reglage_a_deux_mesures_sort_bien_les_deux_dernieres():
    pos, fins = _piles(_sections(3, 8), P=2, fin_hors_pile=2)
    assert fins == [6, 7, 14, 15, 22, 23]


@pytest.mark.parametrize("longueur", [4, 7, 8, 11])
def test_aucune_mesure_n_est_perdue_ni_comptee_deux_fois(longueur):
    secs = _sections(4, longueur)
    pos, fins = _piles(secs, P=2, fin_hors_pile=1)
    vues = [b for pile in pos for b in pile] + fins
    assert sorted(vues) == list(range(4 * longueur))
