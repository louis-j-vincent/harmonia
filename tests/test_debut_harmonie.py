"""tests/test_debut_harmonie.py — la mesure 1 se pose sur un temps fort de l'harmonie.

Louis, 2026-10-09, sur La Jupe En Laine : la basse entre en levée (2,93 s),
le Fm7 arrive au temps suivant (3,27 s) et dure 4 temps comme tout ce qui
suit — mais la mesure 1 était posée sur la levée, un temps trop tôt.

Le chiffre mesuré (temps exact : 34/41 → 35/41 sur `debuts.json`, et 2/7 → 4/7
sur ses marques posées depuis, aucun morceau perdu) vit dans le docstring de
`harmonia.debut.cale_sur_harmonie` et se refait par `python -m tools.debut_eval`.
Ici on épingle la loi elle-même, sans audio.
"""
from __future__ import annotations

import numpy as np

from harmonia.debut import cale_sur_harmonie, changements_d_accord

BEATS = [1.05 + 0.555 * i for i in range(80)]           # 108 bpm, comme La Jupe


def _runs(*runs):
    """[(t0, t1, label)] depuis des (temps0, temps1, label) en INDICES de battue."""
    return [(BEATS[a] if a else 0.0, BEATS[b] if b < len(BEATS) else BEATS[-1] + 1,
             lab) for a, b, lab in runs]


# La Jupe en battues : silence 0-4, Fm7 à 4, puis un accord par mesure de 4.
JUPE = _runs((0, 4, "N"), (4, 8, "F:min7"), (8, 12, "Bb:7"), (12, 16, "Eb:maj"),
             (16, 20, "Ab:maj"), (20, 24, "D:hdim7"), (24, 28, "G:7"),
             (28, 36, "C:min"), (36, 40, "F:min7"), (40, 44, "Bb:7"))


def test_les_changements_se_comptent_en_battues_et_ignorent_le_silence():
    assert changements_d_accord(JUPE, BEATS)[:3] == [4, 8, 12]


def test_une_basse_en_levee_est_poussee_sur_le_temps_fort_qui_suit():
    """La basse à 2,93 s : la battue la plus proche est 2,70 (indice 3), une
    levée. L'harmonie change sur les indices pairs : la mesure 1 va à 3,27."""
    t, info = cale_sur_harmonie(2.93, BEATS, JUPE, 4)
    assert t == BEATS[4]
    assert info["decale"] == 1


def test_une_basse_deja_sur_un_temps_fort_ne_bouge_pas():
    t, info = cale_sur_harmonie(BEATS[4], BEATS, JUPE, 4)
    assert t == BEATS[4] and info["decale"] == 0


def test_lharmonie_ne_choisit_jamais_entre_le_1_et_le_3():
    """Une harmonie à la demi-mesure (Sam Smith, nDUVEUjOKMw : changements à
    égalité sur les temps 2 et 4) ne sait pas lequel est le 1. C'est la basse
    qui le dit : un départ sur un temps fort reste où il est, même si c'est
    le 3 de la grille des accords."""
    demi = _runs((0, 5, "N"), *[(5 + 2 * i, 7 + 2 * i, "C:maj" if i % 2 else "G:maj")
                                for i in range(20)])
    t, info = cale_sur_harmonie(BEATS[7], BEATS, demi, 4)
    assert t == BEATS[7] and info["decale"] == 0


def test_sans_preuve_harmonique_on_garde_la_basse_et_on_le_dit():
    """Chain of Fools : un seul accord tout le morceau. Pas de grille à lire."""
    vamp = _runs((0, 4, "N"), (4, 80, "C:7"))
    t, info = cale_sur_harmonie(2.93, BEATS, vamp, 4)
    assert t == 2.93 and info["raison"] == "pas de preuve"


def test_une_egalite_entre_temps_forts_et_faibles_ne_tranche_rien():
    """Un accord par temps : autant de changements sur les temps forts que
    sur les faibles (15 contre 15 dans la fenêtre) — rien à lire."""
    egal = _runs((0, 4, "N"), *[(4 + i, 5 + i, "C:maj" if i % 2 else "G:maj")
                                for i in range(40)])
    t, info = cale_sur_harmonie(BEATS[2] + 0.1, BEATS, egal, 4)
    assert t == BEATS[2] + 0.1 and info["raison"] == "pas de preuve"


def test_une_mesure_a_trois_temps_lit_la_mesure_entiere():
    """En 3/4 il n'y a pas de demi-mesure : la grille est la mesure."""
    valse = _runs((0, 3, "N"), *[(3 + 3 * i, 6 + 3 * i, "C:maj" if i % 2 else "F:maj")
                                 for i in range(20)])
    t, _ = cale_sur_harmonie(BEATS[4], BEATS, valse, 3)
    assert t == BEATS[6]


def test_pas_de_detection_rend_none():
    assert cale_sur_harmonie(None, BEATS, JUPE, 4) == (None, {"raison": "pas de début"})
