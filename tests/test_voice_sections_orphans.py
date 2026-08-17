"""tests/test_voice_sections_orphans.py — deux orphelins ne sont pas la même
section, red-first.

Louis, 2026-08-17, sur Easy On Me (Adele) : « les sections sont un peu mal
branlées, le D est mal utilisé ». Il avait raison, et le coupable n'est pas la
fusion de lettres : c'est l'écriture des mesures que PERSONNE n'a réclamées.

Ce que le détecteur produisait sur ce morceau :

    D  mes. 25-26   (2 mesures — la liaison vers le couplet 2)
    D  mes. 43-50   (8 mesures — un vrai pont)
    D  mes. 63-63   (1 mesure  — l'accord final)

Trois musiques différentes, trois longueurs différentes, une seule lettre.
Mécanisme : `owner` vaut **-1** pour toute mesure non réclamée, partout dans le
morceau ; le re-lettrage traduisait donc l'entier -1 en UNE lettre, la même
pour tous les orphelins, où qu'ils soient. Le commentaire du code disait
pourtant l'intention juste — « ce qui n'est réclamé par personne garde une
lettre à lui » — c'est l'implémentation qui les mettait en commun.

Le contrat épinglé ici : chaque passage orphelin reçoit sa PROPRE lettre. S'ils
sont réellement la même musique, c'est `merge_letters` qui les recolle — et
elle a un plancher de longueur (`MERGE_MIN`) qui empêche justement de coller
une liaison de 2 mesures sur un pont de 8. Séparer d'abord, fusionner ensuite :
l'inverse ne peut pas se rattraper.

CE QUE ÇA NE RÉSOUT PAS : deux orphelins qui SONT la même musique et font moins
de `MERGE_MIN` mesures restent deux lettres (la liaison de 2 mesures et
l'accord final d'Easy On Me, par exemple). C'est le côté prudent de
« under-fold, never over-fold », et c'est délibéré.
"""
from __future__ import annotations

import numpy as np

from harmonia_min.voice_sections import _lettres


def test_orphelins_separes_recoivent_des_lettres_differentes():
    """La forme exacte d'Easy On Me : trois orphelins, trois lettres."""
    out = [{"b0": 0, "b1": 3, "label": "intro"},
           {"b0": 4, "b1": 11, "label": 0},        # A
           {"b0": 12, "b1": 19, "label": 1},       # B
           {"b0": 20, "b1": 23, "label": 2},       # C
           {"b0": 24, "b1": 25, "label": -1},      # orphelin : la liaison
           {"b0": 26, "b1": 33, "label": 0},       # A
           {"b0": 34, "b1": 41, "label": 3},
           {"b0": 42, "b1": 49, "label": -1},      # orphelin : le pont
           {"b0": 50, "b1": 57, "label": 1},       # B
           {"b0": 58, "b1": 61, "label": 2},       # C
           {"b0": 62, "b1": 62, "label": -1}]      # orphelin : l'accord final
    _lettres(out)
    lab = [s["label"] for s in out]
    assert lab[0] == "intro"
    # les vraies reprises gardent UNE lettre chacune
    assert lab[1] == lab[5] and lab[2] == lab[8] and lab[3] == lab[9]
    # les trois orphelins ne se confondent plus
    orph = [lab[4], lab[7], lab[10]]
    assert len(set(orph)) == 3, f"orphelins encore confondus : {orph}"
    # …ni avec les sections réclamées
    assert not (set(orph) & {lab[1], lab[2], lab[3]})


def test_les_lettres_restent_dans_lordre_dapparition():
    out = [{"b0": 0, "b1": 3, "label": 5},
           {"b0": 4, "b1": 7, "label": 2},
           {"b0": 8, "b1": 11, "label": 5}]
    _lettres(out)
    assert [s["label"] for s in out] == ["A", "B", "A"]


def test_numpy_int_est_traite_comme_un_entier():
    """`owner` est un tableau numpy : ses labels sont des np.int64."""
    out = [{"b0": 0, "b1": 3, "label": np.int64(-1)},
           {"b0": 4, "b1": 7, "label": np.int64(0)},
           {"b0": 8, "b1": 11, "label": np.int64(-1)}]
    _lettres(out)
    lab = [s["label"] for s in out]
    assert len(set(lab)) == 3, f"np.int64 mal reconnu : {lab}"


def test_un_seul_orphelin_ne_change_rien():
    out = [{"b0": 0, "b1": 7, "label": 0},
           {"b0": 8, "b1": 9, "label": -1},
           {"b0": 10, "b1": 17, "label": 0}]
    _lettres(out)
    assert [s["label"] for s in out] == ["A", "B", "A"]
