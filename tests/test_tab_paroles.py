"""tests/test_tab_paroles.py — le pont entre le texte d'un tab et nos accords.

L'HYPOTHÈSE PORTEUSE, et la seule raison d'être de ce fichier : `tab_chart.
poser` aligne la suite COMPRESSÉE des accords, donc l'indice de ses segments
n'est pas l'indice du jeton écrit dans le document. Se tromper là donnerait des
accords plausibles et faux — le motif d'erreur nº 1 du CLAUDE.md. On vérifie
donc la correspondance contre `compresser` lui-même, pas contre notre idée de
ce qu'il fait.
"""
from __future__ import annotations

from harmonia.integrations.tab_align import compresser, sequence_du_tab
from harmonia.integrations.tab_paroles import indices_compresses, jetons_du_tab

# Un document minuscule, écrit ici, dans la forme exacte d'un tab :
# une ligne d'accords, une ligne de texte, et un accord répété.
TAB = (
    "[Verse]\n"
    "[ch]C[/ch]       [ch]Am[/ch]\n"
    "un deux trois quatre\n"
    "[ch]Am[/ch]      [ch]F[/ch]   [ch]G[/ch]\n"
    "cinq six sept huit neuf\n"
)


def test_la_correspondance_retombe_sur_compresser():
    """LE test. Si les deux comptes divergent, un accord du tab se verra
    attribuer l'instant d'un autre."""
    seq = sequence_du_tab(TAB)
    idx = indices_compresses(seq)
    assert len(idx) == len(seq), "chaque accord écrit doit avoir un indice"
    assert max(idx) + 1 == len(compresser(sequence_du_tab(TAB))), (
        "le nombre d'indices distincts doit valoir la longueur compressée")
    assert idx == sorted(idx), "les indices ne peuvent que croître"


def test_le_Am_repete_partage_un_seul_indice():
    """Le Am de fin de première ligne et celui de début de seconde sont le
    MÊME accord tenu : `compresser` les fond, donc ils partagent l'instant."""
    idx = indices_compresses(sequence_du_tab(TAB))
    assert idx == [0, 1, 1, 2, 3], f"attendu C Am Am F G → 0 1 1 2 3, reçu {idx}"


def test_les_colonnes_sont_celles_de_la_ligne_DEBALISEE():
    """La colonne doit être celle que l'œil voit. Comptée sur le document brut,
    chaque accord dériverait de la longueur de `[ch][/ch]`."""
    jetons = jetons_du_tab(TAB)
    assert [j["texte"] for j in jetons] == ["C", "Am", "Am", "F", "G"]
    assert jetons[0]["colonne"] == 0
    assert jetons[1]["colonne"] == 8, "C + 7 espaces"
    assert jetons[3]["colonne"] == 8, "Am + 6 espaces"
    assert [j["ligne"] for j in jetons] == [1, 1, 3, 3, 3]


def test_autant_de_jetons_que_daccords_lus():
    """Les deux lectures du même document doivent compter pareil — c'est la
    garde qui fait taire `feuille` quand elles divergent."""
    assert len(jetons_du_tab(TAB)) == len(sequence_du_tab(TAB))
