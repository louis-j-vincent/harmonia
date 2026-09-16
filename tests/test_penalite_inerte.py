"""tests/test_penalite_inerte.py — `DEFAULT_PENALTY` ne décide rien, et pourquoi.

Trouvé le 2026-09-16 en cherchant à ILLUSTRER ce que chaque étage de la chaîne
change (Louis : « une illustration à chaque étage […] avec un cas concret »).
Le plan était de faire varier la pénalité et de montrer le chart bouger. Il
n'a pas bougé : de 5 à 200, Ready rend exactement les mêmes 86 accords.

La raison est structurelle, pas une histoire de valeur. Le décodeur vendu
n'applique `diff_trans_penalty` que là où `beat_arr[t] == 1`
(`third_party/musx_ismir2019/extractors/xhmm_ismir.py:120`), et
`musx.make_beat_arr` ne laisse la valeur 1 que sur les images situées HORS de
la grille de temps — avant le premier temps et après le dernier. Partout
ailleurs elle pose 0 (aucun changement permis), ou 2/3/4, qui vont chercher
`beat_trans_penalty`. La densité d'accords est donc décidée par ce triplet-là,
pas par la pénalité que le commentaire mettait en avant.

Ce test ne joue pas de son et ne charge aucun modèle : il vérifie la propriété
de la grille, qui est la CAUSE. Si un jour `make_beat_arr` pose des 1 au milieu
d'un morceau, la pénalité redeviendra vivante — et ce test rougira pour le
dire, au lieu de laisser réapparaître en silence un curseur qu'on croyait mort.

    .venv/bin/python -m pytest tests/test_penalite_inerte.py -v
"""
from __future__ import annotations

import numpy as np

from harmonia.musx import FRAME_DT, make_beat_arr

PER = 0.5
N_BEATS = 64
N_FRAME = int(round(N_BEATS * PER / FRAME_DT)) + 40


def _grille():
    bt = [round(i * PER + 1.0, 4) for i in range(N_BEATS)]
    db = bt[::4]
    return bt, db


def test_la_valeur_1_n_apparait_qu_aux_extremites():
    """La valeur 1 est celle qui déclenche `diff_trans_penalty`."""
    bt, db = _grille()
    arr = make_beat_arr(N_FRAME, bt, db, beats_per_bar=4, quarter_beats="all")
    uns = np.flatnonzero(arr == 1)
    assert len(uns), "grille dégénérée : le test ne teste rien"
    premier = int(round(bt[0] / FRAME_DT))
    dernier = int(round(bt[-1] / FRAME_DT))
    dedans = uns[(uns > premier) & (uns < dernier)]
    assert not len(dedans), (
        f"{len(dedans)} image(s) à 1 DANS le morceau : la pénalité redevient "
        f"vivante, il faut la re-mesurer et corriger sa docstring")


def test_entre_le_premier_et_le_dernier_temps_tout_est_0_2_3_ou_4():
    bt, db = _grille()
    arr = make_beat_arr(N_FRAME, bt, db, beats_per_bar=4, quarter_beats="all")
    a = int(round(bt[0] / FRAME_DT)) + 1
    b = int(round(bt[-1] / FRAME_DT))
    assert set(np.unique(arr[a:b]).tolist()) <= {0, 2, 3, 4}


def test_les_temps_portent_bien_les_trois_grades():
    """Temps fort 2, mi-mesure 3, autre temps 4 — les coûts qui décident vraiment."""
    bt, db = _grille()
    arr = make_beat_arr(N_FRAME, bt, db, beats_per_bar=4, quarter_beats="all")
    fr = [int(round(t / FRAME_DT)) for t in bt]
    grades = [int(arr[f]) for f in fr]
    assert grades[0] == 2 and grades[4] == 2, "les temps forts doivent être 2"
    assert grades[2] == 3, "le mi-mesure doit être 3"
    assert grades[1] == 4 and grades[3] == 4, "les autres temps doivent être 4"


def test_sans_quarter_beats_les_autres_temps_sont_interdits():
    """Le mode historique demi-mesure : grade 4 remis à 0."""
    bt, db = _grille()
    arr = make_beat_arr(N_FRAME, bt, db, beats_per_bar=4, quarter_beats=None)
    fr = [int(round(t / FRAME_DT)) for t in bt]
    assert int(arr[fr[1]]) == 0 and int(arr[fr[3]]) == 0
    assert int(arr[fr[2]]) == 3 and int(arr[fr[0]]) == 2
