"""tests/test_soudure_bar1.py — la bande doit partir de la mesure 1 de Louis.

Louis, 2026-08-15 : « lorsque je reset la mesure 1, le recalcul des sections
est bloqué à l'ancienne version, il ne s'update pas, les sections commencent
donc sur la mauvaise 1 ».

LE SYMPTÔME, mesuré sur son chart (min_Oextk-If8HQ, `bar1` à 3,358 s → mesure
1 = la barre d'index 1, l'index 0 étant l'intro). La bande sortait ses jetons
en `[0, 2, 4, 6, …]`, c'est-à-dire toujours calés sur la barre 0, alors que le
chart, lui, avait bien suivi la marque : `intro[0,0] A[1,8] A[9,16] …`. Chaque
plaque de la bande enjambait donc la frontière — une section ne pouvait
commencer que sur une mesure PAIRE, jamais sur sa mesure 1. Remarquer la
mesure 1 ne changeait rien à la bande : le treillis ne bougeait pas.

Ce que ce test épingle : la marque de Louis est l'ancre du treillis de
bi-mesures, comme elle est déjà l'ancre de la phase des barres et des
frontières de sections (`pipeline._force_bar1_sections`). Les mesures d'avant
la marque forment leur propre tête de bande, la mesure orpheline étant
absorbée par le PREMIER jeton (l'intro), jamais par la structure qui suit.

CE QUE ÇA NE RÉSOUT PAS : un chart dont l'intro a été DÉTECTÉE sans que Louis
ait posé sa marque garde le treillis calé sur la barre 0 — `bar1` est la seule
ancre que ce module lit, et c'est délibéré (sa marque prime sur tout, le
détecteur n'a pas ce statut).
"""
from __future__ import annotations

from harmonia_min.soudure import _jetons, _mesure1, _tete, mot_du_chart


def _chart(n_mesures: int, bar1=None) -> dict:
    """Un chart minimal : une grille régulière d'une seconde par mesure, un
    accord par mesure, et la marque de Louis si on en veut une."""
    grid = [float(i) for i in range(n_mesures + 1)]
    chords = [{"t0": float(i), "t1": float(i + 1),
               "root": i % 3, "bass": -1, "nc": False}
              for i in range(n_mesures)]
    return {"barGrid": grid, "nBars": n_mesures, "bar1": bar1,
            "prompter": {"chords": chords}}


def test_mesure1_lit_la_marque_sur_la_grille():
    assert _mesure1(_chart(20)) == 0                  # pas de marque
    assert _mesure1(_chart(20, bar1=1.0)) == 1
    assert _mesure1(_chart(20, bar1=4.4)) == 4        # se recale au plus proche
    assert _mesure1(_chart(20, bar1="bof")) == 0      # marque illisible


def test_tete_absorbe_la_mesure_orpheline_dans_lintro():
    assert _tete(0) == []
    assert _tete(1) == [0]                            # un jeton d'une mesure
    assert _tete(2) == [0]
    assert _tete(3) == [0]                            # trois mesures d'un coup
    assert _tete(4) == [0, 2]
    assert _tete(5) == [0, 3]                         # 3 + 2, pas 2 + 3


def test_jetons_partent_de_la_marque():
    # sans marque : le comportement d'origine, inchangé
    assert _jetons(9) == [0, 2, 4, 6, 9]
    # avec la marque en mesure 1 : la tête fait une mesure, puis 2 par 2 À
    # PARTIR d'elle — c'est ce qui permet à une section de commencer là
    assert _jetons(9, 1) == [0, 1, 3, 5, 7, 9]
    assert 1 in _jetons(9, 1)


def test_la_bande_du_chart_suit_la_marque():
    sans = mot_du_chart(_chart(21))
    avec = mot_du_chart(_chart(21, bar1=1.0))
    assert sans["jetons"][:4] == [0, 2, 4, 6]
    # la marque déplace TOUT le treillis, pas seulement la première plaque
    assert avec["jetons"][:4] == [0, 1, 3, 5]
    assert len(avec["mot"]) == len(avec["jetons"]) - 1


def test_une_marque_absurde_ne_casse_pas_la_bande():
    # une marque au ras de la fin ne laisse pas de quoi faire une bande :
    # on retombe sur le treillis d'origine plutôt que de rendre 2 plaques
    tard = mot_du_chart(_chart(12, bar1=11.0))
    assert tard["jetons"][0] == 0
    assert tard["jetons"][-1] == 12
    assert len(tard["mot"]) == len(tard["jetons"]) - 1
