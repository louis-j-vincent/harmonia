"""tests/test_tab_structure.py — la forme déduite au rasoir d'Occam.

Louis, 2026-09-18 : « avec cette vérité-là en tête et la règle qu'un pattern
en général se répète, tu peux déduire la structure finale de la grille →
rasoir d'Occam : c'est l'explication la plus simple, et donc l'écriture la
plus minimale en termes de nombres de sections différentes, qui explique le
morceau. »

Ce qui est épinglé ici, ce sont les lois. Le résultat sur les vrais morceaux
(Grenade sort `A B C C D E E F E B C C D G B C C D E E H`, où `B C C D` est le
refrain joué trois fois) demande l'audio et le réseau ; il vit dans le
docstring du module.
"""
from __future__ import annotations

import pytest

from harmonia.integrations import tab_structure as TS


def m(*mots):
    """Une suite de mesures, chacune écrite « Cm » ou « Bb Eb »."""
    return [tuple(x.split()) for x in mots]


def test_une_boucle_jouee_quatre_fois_s_ecrit_UNE_fois():
    """Le cœur du rasoir : ce qui se répète ne se paie qu'une fois."""
    f = TS.forme(m(*(["G7", "Cm", "Fm", "D°"] * 4)))
    assert TS.mot(f["sections"]) == "A A A A"
    assert f["cout"][0] == 4, "quatre mesures écrites, pas seize"
    assert f["motifs"]["A"] == m("G7", "Cm", "Fm", "D°")


def test_couplet_refrain_alternes_stricts_font_UNE_seule_lettre():
    """Et c'est juste. `couplet refrain couplet refrain` s'écrit « A A » avec
    A de huit mesures : même longueur écrite qu'« A B A B », mais une lettre
    au lieu de deux et deux sections au lieu de quatre. Le morceau EST un bloc
    de huit joué deux fois — un musicien l'écrirait pareil."""
    couplet = ["Dm", "Dm", "Am", "Am"]
    refrain = ["Bb", "C", "F", "A"]
    f = TS.forme(m(*(couplet + refrain + couplet + refrain)))
    assert TS.mot(f["sections"]) == "A A"
    assert f["cout"] == (8, 1, 2)


def test_un_refrain_joue_PLUS_que_le_couplet_force_deux_lettres():
    """Dès que l'alternance se casse, le rasoir doit séparer les deux."""
    couplet = ["Dm", "Dm", "Am", "Am"]
    refrain = ["Bb", "C", "F", "A"]
    f = TS.forme(m(*(couplet + refrain + refrain + couplet + refrain)))
    assert TS.mot(f["sections"]) == "A B B A B"
    assert f["cout"] == (8, 2, 5)


def test_une_mesure_vaut_ce_qui_y_SONNE():
    """Deux mesures qui sonnent pareil doivent s'écrire pareil. Le premier
    jet prenait les accords qui COMMENCENT dans la mesure : la même boucle
    sortait `· Cm Fm D°` puis `G7/B Cm Fm D°` selon qu'un accord démarrait une
    fraction de temps avant la barre ou après, et la répétition était perdue."""
    grille = [0.0, 4.0, 8.0]
    temps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    segs = [{"t0": -0.02, "t1": 4.0, "texte": "G7/B"},   # commence AVANT la barre
            {"t0": 4.0, "t1": 8.0, "texte": "G7/B"}]
    assert TS.mots_par_mesure(segs, grille, temps) == [("G7/B",), ("G7/B",)]


def test_deux_accords_dans_une_mesure_font_un_mot_de_deux():
    grille = [0.0, 4.0]
    temps = [0.0, 1.0, 2.0, 3.0]
    segs = [{"t0": 0.0, "t1": 2.0, "texte": "Bb"},
            {"t0": 2.0, "t1": 4.0, "texte": "Eb"}]
    assert TS.mots_par_mesure(segs, grille, temps) == [("Bb", "Eb")]


def test_la_loi_du_retour_une_section_fait_au_moins_quatre_mesures():
    """Louis, 2026-08-16. Sans ce plancher, le rasoir découperait en mesures
    isolées et « expliquerait » n'importe quoi."""
    f = TS.forme(m("Dm", "Bb", "Dm", "Bb", "Dm", "Bb", "Dm", "Bb"))
    assert all(b - a >= TS.MINI for a, b, _ in f["sections"][:-1])


def test_SEULE_la_derniere_section_a_le_droit_d_etre_courte():
    """L'exemption porte sur la SUITE, jamais sur le bloc."""
    f = TS.forme(m(*(["Dm", "Dm", "Am", "Am"] * 2 + ["Gm"])))
    assert [b - a for a, b, _ in f["sections"]] == [4, 4, 1]


def test_l_ecriture_la_plus_COURTE_gagne_et_pas_le_moins_de_lettres():
    """Le piège mesuré : compter d'abord les sections DIFFÉRENTES ne
    récompense jamais la répétition, parce que la façon la moins chère
    d'avoir peu de lettres est d'en faire de très longues. This Love sortait
    « A B C C » avec des blocs de 32 mesures — un découpage en tranches, pas
    une forme."""
    suite = ["Dm", "Dm", "Am", "Am"] * 3 + ["Gm", "Gm", "F", "F"]
    f = TS.forme(m(*suite))
    assert f["cout"][0] == 8, "4 mesures pour le couplet + 4 pour la fin"
    assert TS.mot(f["sections"]) == "A A A B"


def test_une_suite_sans_aucune_repetition_s_ecrit_en_entier():
    suite = [f"{x}" for x in ("C", "D", "E", "F", "G", "A", "B", "Db")]
    f = TS.forme(m(*suite))
    assert f["cout"][0] == len(suite), "rien ne se répète : tout se paie"


def test_les_variantes_qui_ne_different_que_par_la_FIN_sont_signalees():
    """« under-fold, never over-fold » : on ne fond pas en douce deux sections
    qui diffèrent, on les montre."""
    v = TS.variantes({"A": m("Dm", "Dm", "Am", "Am"),
                      "B": m("Dm", "Dm", "Am", "Am A7"),
                      "C": m("Gm", "Gm", "F", "F")})
    assert v == [("A", "B", 1)]


def test_une_difference_au_DEBUT_n_est_pas_une_variante():
    """Deux sections qui partent différemment sont deux sections."""
    assert TS.variantes({"A": m("Dm", "Dm", "Am", "Am"),
                         "B": m("Gm", "Dm", "Am", "Am")}) == []


def test_un_morceau_vide_ne_casse_pas():
    f = TS.forme([])
    assert f["sections"] == [] and f["motifs"] == {} and TS.mot([]) == ""
