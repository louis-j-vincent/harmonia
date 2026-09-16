"""tests/test_basse_musx.py — la basse vient de musx, et la règle n'a plus de plancher.

Louis, 2026-09-16, en regardant `/plots/probas16.html` : « il faut utiliser la
tete de basse de musx qui est parfaite enfaite ». Mesuré avant de le croire,
sur ses 16 arbitrages du 2026-09-15, dont 12 localisables avec certitude :

    chroma NNLS, attaque de 150 ms (l'ancienne source)  ...  6/12
    tête basse musx, même attaque de 150 ms             ...  8/12
    tête basse musx, moyennée sur TOUT l'accord         ... 11/12

Ce fichier fige les deux choses qui pourraient casser en silence.

1. LA LECTURE DES COLONNES. La tête basse a 13 colonnes : la 0 est « pas de
   basse » et les 1 à 12 sont les douze hauteurs À PARTIR DE DO. Se tromper
   d'une case décale toute la lecture d'un demi-ton, et se tromper d'origine
   (l'autre convention du projet, la chroma NNLS, commence au LA) la décale
   d'un triton — une erreur parfaitement plausible à l'œil, et c'est exactement
   celle que j'ai commise en dessinant la carte avant de la corriger. La
   convention est vérifiable dans le clone vendu : `xhmm_ismir.py` fait
   `result_array[:,1] += 1` sur une valeur de basse qui vaut -1 quand il n'y en
   a pas, et `complex_chord.NUM_TO_ABS_SCALE[0] == 'C'`.

2. L'ABSENCE DE PLANCHER. La part de la fondamentale sépare les deux classes
   sans recouvrement (5,8 à 42,1 % quand il y a un slash, 80,5 à 97,2 % quand
   il n'y en a pas), mais dans les SEPT cas sans slash l'argmax de musx est
   déjà la fondamentale : un plancher entre 40 et 80 % donne exactement le même
   résultat que pas de plancher. On n'en met donc pas — un nombre qui ne décide
   rien est une dette, pas une sécurité.

    .venv/bin/python -m pytest tests/test_basse_musx.py -v
"""
from __future__ import annotations

import numpy as np

from harmonia.bass_rules import PLAUSIBLE
from harmonia.pipeline import _write_sounding_bass

NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()
DT = 0.02322


def _probs(n_frames: int, colonnes: dict[int, float]):
    """Une tête basse constante : {hauteur (0=DO): masse}. Le reste se partage."""
    head = np.zeros((n_frames, 13), dtype=float)
    reste = max(0.0, 1.0 - sum(colonnes.values()))
    head[:, 1:13] = reste / 12.0
    for pc, v in colonnes.items():
        head[:, 1 + pc] = v
    return [None, head]


def _bar(root_pc: int, t0=0.0, t1=2.0):
    return [[{"root": root_pc, "q": "", "t0": t0, "t1": t1, "nc": False}]]


def test_la_colonne_zero_nest_pas_une_hauteur():
    """Si on lisait les colonnes 0..11, tout serait décalé d'un demi-ton."""
    bars = _bar(0)                                   # un accord de DO
    head = np.zeros((90, 13), dtype=float)
    head[:, 0] = 0.9                                 # « pas de basse » écrasant
    head[:, 1 + 7] = 0.1                             # un SOL discret
    _write_sounding_bass(bars, [None, head], None, None)
    assert bars[0][0]["bass"] == 7, "la colonne 0 a été comptée comme une hauteur"


def test_do_est_en_colonne_un():
    bars = _bar(5)                                   # un accord de FA
    _write_sounding_bass(bars, _probs(90, {0: 0.8}), None, None)   # basse DO
    assert bars[0][0]["bass"] == 0, "l'origine des hauteurs n'est pas DO"


def test_la_fondamentale_en_tete_ne_donne_pas_de_slash():
    bars = _bar(3)                                   # MIb
    _write_sounding_bass(bars, _probs(90, {3: 0.9}), None, None)
    assert bars[0][0]["bass"] == -1


def test_un_intervalle_jouable_donne_un_slash():
    """+7, la quinte : l'un des cinq intervalles que Louis a entendus."""
    bars = _bar(0)
    _write_sounding_bass(bars, _probs(90, {7: 0.7}), None, None)
    assert bars[0][0]["bass"] == 7


def test_un_intervalle_ecarte_ne_sexecrit_pas():
    """+1, la b9 : refusée à l'oreille le 2026-09-15. Défaut = accord nu."""
    assert 1 not in PLAUSIBLE
    bars = _bar(0)
    _write_sounding_bass(bars, _probs(90, {1: 0.9}), None, None)
    assert bars[0][0]["bass"] == -1


def test_aucun_plancher_de_confiance():
    """Une basse FAIBLE mais gagnante s'écrit quand même.

    C'est la conclusion de la re-dérivation : le plancher ne séparait rien que
    l'argmax ne séparait déjà. Un cas comme r03 (fondamentale à 5,8 %, SOL à
    86,9 %) doit passer, et un cas serré aussi.
    """
    bars = _bar(0)
    # DO à 20 %, SOL à 24 % : personne n'est confiant, mais SOL gagne
    _write_sounding_bass(bars, _probs(90, {0: 0.20, 7: 0.24}), None, None)
    assert bars[0][0]["bass"] == 7


def test_la_moyenne_porte_sur_toute_la_duree_de_laccord():
    """Et non sur l'attaque : c'est la mesure qui l'a tranché (11/12 contre 8/12).

    Une basse qui ne domine QUE le début ne doit pas gagner tout l'accord.
    """
    n = 200
    head = np.zeros((n, 13), dtype=float)
    head[:, 1:13] = 0.01
    k = int(0.15 / DT)                               # les 150 premières ms
    head[:k, 1 + 7] = 0.9                            # SOL à l'attaque seulement
    head[k:, 1 + 0] = 0.9                            # DO tout le reste
    bars = _bar(0, t0=0.0, t1=n * DT)
    _write_sounding_bass(bars, [None, head], None, None)
    assert bars[0][0]["bass"] == -1, "la fenêtre d'attaque est revenue"


def test_les_accords_tenus_et_les_silences_sont_laisses_tranquilles():
    bars = [[{"root": 0, "q": "", "t0": 0.0, "t1": 2.0, "nc": True},
             {"root": 0, "q": "", "t0": 0.0, "t1": 2.0, "carry": True, "nc": False}]]
    n = _write_sounding_bass(bars, _probs(90, {7: 0.9}), None, None)
    assert n == 0
    assert "bass" not in bars[0][0] and "bass" not in bars[0][1]
