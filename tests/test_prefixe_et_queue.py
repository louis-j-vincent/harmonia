"""tests/test_prefixe_et_queue.py — un bloc qui joue B puis continue est B + queue.

Louis, 2026-09-30, sur « Julia » (Don West) : « On détecte une section C qui
est clairement un B avec une queue, pourquoi est-ce que ça n'a pas été
classifié en tant que tel ? »

Dans « Valider les sections », les zones libres sont nommées par les quatre
mots (`simulation.remplir_par_algo`). Deux blocs libres n'avaient la même
lettre que si leurs MOTS étaient identiques (`grouper_restes` groupe sur le
type exact, `nommer` n'accorde le prime qu'à longueur égale) : B = `cdada`,
le bloc suivant = `cdadaa` → une lettre neuve.

La règle : un bloc libre dont le mot COMMENCE par le mot entier d'un autre
bloc, sur au moins 4 bi-mesures et 2 lettres distinctes (le garde-fou mesuré
le 2026-09-17, `docs/regle_prefixe_2026-09-17.md` : sans lui, les mots pauvres
de Let It Be / Yesterday cassent), avec une queue plus courte que ce préfixe,
est coupé en deux : la lettre de l'autre bloc (mêmes mesures, même longueur,
donc replié avec lui) + une « queue » écrite à part — sous-replier, jamais
sur-replier.
"""
from __future__ import annotations

import pytest

from harmonia import soudure
from harmonia.sections import simulation as sim_mod

# Le mot de Julia, tel que `mot_sur_traits` le rend sur la grille calée sur
# les quatre A (bi-mesures, la dernière de 3 mesures).
JULIA_BORNES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
                66, 68, 71]
JULIA_MOT = "abbbabbbacdadabbbabbbacdadaabbbbbbb"


def _chart(n):
    grid = [round(i * 2.66, 3) for i in range(n + 1)]
    return {"title": "bidon", "barGrid": grid, "nBars": n,
            "prompter": {"chords": []}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": []}


def _occ(*plages):
    return [{"m0": a, "m1": b, "label": lab, "score": 1.0, "rot": 0,
             "trace": True} for a, b, lab in plages]


@pytest.fixture
def mot(monkeypatch):
    def poser(bornes, m):
        monkeypatch.setattr(soudure, "mot_sur_traits",
                            lambda chart, traits, audio_dir=None: (bornes, m, "test"))
    return poser


def _plages(blocs):
    return [(b["m0"], b["m1"], b["label"]) for b in blocs]


def test_julia_le_C_est_un_B_avec_une_queue(mot, tmp_path):
    mot(JULIA_BORNES, JULIA_MOT)
    occ = _occ((2, 9, "A"), (10, 17, "A"), (28, 35, "A"), (36, 43, "A"))
    blocs = sim_mod.remplir_par_algo(_chart(71), occ, tmp_path)
    pl = _plages(blocs)
    assert (18, 27, "B") in pl, pl
    # le « C » de 12 mesures : les 10 premières sont B, les 2 dernières une queue
    assert (44, 53, "B") in pl, pl
    assert (54, 55, "queue") in pl, pl
    # la boucle de fin prend la première lettre libre (C), pas D
    assert (56, 70, "C") in pl, pl
    assert not any(lab == "D" for _a, _b, lab in pl), pl


def test_le_prefixe_pauvre_ne_declenche_pas(mot, tmp_path):
    """`aaaa` ⊂ `aaaab` : une seule lettre dans le préfixe — le mot pauvre qui
    a cassé Let It Be et Yesterday le 2026-09-17. Rien ne change."""
    bornes = list(range(0, 42, 2))            # 20 bi-mesures, 40 mesures
    m = "cc" + "aaaa" + "ddd" + "aaaab" + "cc" + "eeee"
    assert len(m) == 20
    mot(bornes, m)
    occ = _occ((0, 3, "A"), (24, 27, "A"))
    blocs = sim_mod.remplir_par_algo(_chart(40), occ, tmp_path)
    assert not any(b["label"] == "queue" for b in blocs), _plages(blocs)


def test_un_bloc_different_garde_sa_lettre(mot, tmp_path):
    """Aucun préfixe commun : deux lettres, comme avant."""
    bornes = list(range(0, 42, 2))
    m = "xx" + "cdada" + "xx" + "efgfg" + "xx" + "hhhh"
    assert len(m) == 20
    mot(bornes, m)
    occ = _occ((0, 3, "A"), (14, 17, "A"), (28, 31, "A"))
    blocs = sim_mod.remplir_par_algo(_chart(40), occ, tmp_path)
    labs = {(b["m0"], b["m1"]): b["label"] for b in blocs}
    assert labs[(4, 13)] != labs[(18, 27)], _plages(blocs)
    assert not any(b["label"] == "queue" for b in blocs), _plages(blocs)
