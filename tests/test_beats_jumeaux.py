"""tests/test_beats_jumeaux.py — deux marques pour un seul temps.

Louis, 2026-08-18, sur Another Day de Jamie Lidell : « les dernières sections
devraient toutes être un A et ils décalent, comment ça se fait ? ».

Le morceau tient 136,4 BPM du début à la fin, mais Beat This! pose quinze temps
de trop après 149,5 s, chacun à exactement 0,080 s du précédent — 0,18 fois le
temps médian. Onze des vingt-cinq mesures de la dernière section font alors 20
à 50 % de moins que les autres, dont quatre une DEMI-mesure, et la boucle de
quatre accords glisse d'une mesure toutes les huit.

Ce n'est pas l'erreur d'octave de `drop_inserted_beats` : un temps inséré tombe
à la MOITIÉ du temps, un jumeau à 0,18.
"""
from __future__ import annotations

import numpy as np

from harmonia_min.beats import (drop_duplicate_beats, drop_inserted_beats)

PER = 0.44          # 136,4 BPM, le tempo d'Another Day


def _grille(n=40, per=PER):
    return [round(i * per, 3) for i in range(n)]


def test_un_jumeau_est_retire():
    b = _grille()
    b.insert(20, round(b[19] + 0.08, 3))          # une marque en trop
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b) - 1
    d = np.diff(out)
    assert d.min() > 0.8 * PER, f"la grille reste irrégulière : {d.min():.3f}"


def test_on_garde_celui_qui_tombe_le_mieux():
    """Des deux jumeaux, on garde celui le plus proche de l'attendu — pas
    systématiquement le premier."""
    b = _grille()
    # le VRAI temps est a 20*PER ; le traceur le double 0,08 s AVANT
    b.insert(20, round(20 * PER - 0.08, 3))
    out, _ = drop_duplicate_beats(b, [])
    assert round(20 * PER, 3) in out, "on a gardé le mauvais des deux"


def test_un_demi_temps_nest_pas_un_jumeau():
    """L'erreur d'octave reste le domaine de drop_inserted_beats."""
    b = _grille()
    b.insert(20, round(b[19] + PER / 2, 3))
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b), "drop_duplicate_beats a mangé un demi-temps"
    out2, _ = drop_inserted_beats(b, [])
    assert len(out2) == len(b) - 1, "drop_inserted_beats aurait dû le voir"


def test_une_grille_saine_nest_pas_touchee():
    b = _grille()
    out, db = drop_duplicate_beats(b, [b[0], b[4]])
    assert out == b and db == [b[0], b[4]]


def test_les_downbeats_suivent():
    b = _grille()
    jum = round(b[19] + 0.08, 3)
    b.insert(20, jum)
    out, db = drop_duplicate_beats(b, [b[0], jum])
    assert jum not in db or jum in out, "un downbeat pointe sur un temps retiré"
