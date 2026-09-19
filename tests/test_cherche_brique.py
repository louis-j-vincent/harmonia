"""tests/test_cherche_brique.py — chercher les reprises d'une brique de Louis.

Louis, 2026-09-17, sur Cry Me A River : « l'outil d'annotation avec les briques
faites par l'humain ne marche pas, j'ai annoté, l'outil a complètement pas
respecté ce que j'avais écrit ».

Deux défauts, tous deux épinglés ici.

1. **Le seuil était ABSOLU** (0,90) alors que la distribution des scores ne
   l'est pas. Mesuré sur les deux morceaux :

       Don't Want My Love   médiane 0,692 · 1,000 0,996 0,992 0,990 puis une
                            CHUTE de 0,135 — une falaise nette
       Cry Me A River       médiane 0,896 · 1,000 0,963 0,963 0,961 0,952…
                            des chutes de 0,000 à 0,009 — aucune falaise

   Sur le second, 38 départs sur 80 passaient 0,90 : la brique de Louis était
   recopiée sur la moitié du morceau. Le seuil est maintenant relatif —
   `médiane + 1,5 σ`, borné — donc un morceau sans contraste s'abstient.

2. **Une intro était rejouée.** « Une intro ne se rejoue pas plus tard »
   (Louis, 2026-09-16). La règle existait dans l'ancienne route et avait été
   perdue en la réécrivant ; sur Cry Me A River « intro » était recopié SEPT
   fois.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from harmonia.sections import simulation as sim


@pytest.fixture()
def morceau(tmp_path, monkeypatch):
    """Un chart de 40 mesures dont on pilote la matrice de ressemblance."""
    (tmp_path / "essai.m4a").write_bytes(b"x")
    chart = {"audio_url": "/audio/essai.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}

    def pose(scores):
        """`scores[b]` = à quel point la brique se rejoue en partant de b."""
        from harmonia.sections import similarity as vs
        monkeypatch.setattr(sim, "Path", sim.Path)
        monkeypatch.setattr(
            "harmonia.musx.frame_posteriors",
            lambda *a, **k: [np.zeros((10, 73), dtype=np.float32)])
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: np.zeros((40, 40)))
        monkeypatch.setattr(vs, "_slide",
                            lambda M, a0, L, n: np.asarray(scores, dtype=float))
        return chart, tmp_path

    return pose


def test_une_intro_ne_cherche_aucune_reprise(morceau):
    """La règle de Louis. Sur Cry Me A River « intro » repartait sept fois."""
    chart, audio = morceau([0.99] * 33)          # tout se ressemble : peu importe
    for role in ("intro", "Intro", " OUTRO ", "silence"):
        occ, info = sim.cherche_brique(chart, 0, 7, audio, label=role)
        assert occ == [(0, 7, 1.0, 0)], f"« {role} » ne doit pas se propager"
        assert info["raison"] == "role"


def test_un_morceau_SANS_contraste_s_abstient(morceau):
    """Cry Me A River : médiane 0,896, aucune falaise. Un seuil fixe laissait
    passer la moitié du morceau ; le seuil relatif ne laisse passer que la
    brique."""
    plat = [0.90 + 0.02 * ((i * 7) % 5) / 5 for i in range(33)]
    plat[8] = 1.0                                 # la brique elle-même
    chart, audio = morceau(plat)
    occ, info = sim.cherche_brique(chart, 8, 15, audio, label="A")
    assert occ == [(8, 15, 1.0, 0)], "rien ne se détache : on n'ajoute rien"
    # et on le DIT, au lieu de rendre une liste vide muette
    assert info["raison"] == "sans_contraste" and info["n_reprises"] == 0


def test_une_falaise_nette_donne_ses_reprises(morceau):
    """Don't Want My Love : quatre scores au-dessus, puis une chute de 0,135."""
    sc = [0.65] * 33
    for b in (0, 8, 24):                          # les vraies reprises
        sc[b] = 0.99
    chart, audio = morceau(sc)
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, _s, _r in occ] == [(0, 7), (8, 15), (24, 31)]
    assert info["raison"] is None and info["n_reprises"] == 2


def test_deux_reprises_ne_se_recouvrent_jamais(morceau):
    """Une phrase de 4 mesures répétée fait scorer aussi le décalage de 4 :
    le second pic tombe DANS le premier bloc et doit être refusé."""
    sc = [0.60] * 33
    sc[0] = 1.00
    sc[4] = 0.99                                  # décalage d'une demi-brique
    sc[16] = 0.98
    chart, audio = morceau(sc)
    occ, _info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, _s, _r in occ] == [(0, 7), (16, 23)]


def test_la_brique_est_TOUJOURS_rendue(morceau):
    """Même quand le seuil relatif monte au-dessus de tout — c'était le cas de
    Let It Be, où `médiane + 1,5 σ` valait 1,049 et où la brique elle-même
    disparaissait, donc le trait de Louis avec elle."""
    chart, audio = morceau([0.5] * 33)
    occ, _info = sim.cherche_brique(chart, 3, 10, audio, label="A")
    assert occ[0] == (3, 10, 1.0, 0) and len(occ) == 1


def test_sans_audio_la_brique_reste_seule(tmp_path):
    """Pas de matrice possible : on n'ajoute rien, et on ne plante pas."""
    chart = {"audio_url": "/audio/absent.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}
    occ, info = sim.cherche_brique(chart, 0, 7, tmp_path, label="A")
    assert occ == [(0, 7, 1.0, 0)] and info["raison"] == "pas_d_audio"


# ── les reprises transposées (2026-09-19) ────────────────────────────────────

@pytest.fixture()
def morceau_tourne(tmp_path, monkeypatch):
    """Un chart dont on pilote la matrice DROITE et les matrices TOURNÉES
    séparément — c'est la seule façon de tester qu'une reprise modulée est
    trouvée là où la droite ne voit rien."""
    (tmp_path / "essai.m4a").write_bytes(b"x")
    chart = {"audio_url": "/audio/essai.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}

    def pose(droite, par_rotation):
        from harmonia.sections import similarity as vs
        monkeypatch.setattr("harmonia.musx.frame_posteriors",
                            lambda *a, **k: [np.zeros((10, 73), dtype=np.float32)])
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: np.zeros((40, 40)))
        # `harmonic_vectors` rend V ; `cherche_brique` calcule V @ roll(V,k).T.
        # On met une colonne par demi-ton pour que la matrice tournée de rang k
        # soit reconnaissable, et on intercepte `_slide` pour rendre la courbe
        # demandée selon la matrice qu'on lui passe.
        V = np.eye(40, 12)
        monkeypatch.setattr(vs, "harmonic_vectors", lambda *a, **k: V)

        def slide(M, a0, L, n):
            for k, courbe in par_rotation.items():
                if np.allclose(M, V @ np.roll(V, k, axis=1).T):
                    return np.asarray(courbe, dtype=float)
            return np.asarray(droite, dtype=float)
        monkeypatch.setattr(vs, "_slide", slide)
        return chart, tmp_path

    return pose


def test_une_reprise_MONTEE_dun_ton_est_retrouvee(morceau_tourne):
    """Le trou écrit de cette fonction jusqu'au 2026-09-19 : le dernier refrain
    monté d'un ton. La matrice droite ne voit rien en mesure 25 ; la matrice
    tournée de 2 demi-tons, si."""
    droite = [0.60] * 33
    droite[0] = 1.00
    tournee = [0.60] * 33
    tournee[24] = 0.99
    chart, audio = morceau_tourne(droite, {2: tournee})
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b, r) for a, b, _s, r in occ] == [(0, 7, 0), (24, 31, 2)]
    assert info["n_reprises"] == 1


def test_une_brique_TROP_COURTE_ne_cherche_aucune_transposition(morceau_tourne):
    """Bein Green : un motif de 2 mesures tourné collait partout — dix reprises,
    toutes fausses. Sous `MIN_MESURES_TRANSPOSE` on ne regarde même pas."""
    droite = [0.60] * 39
    droite[0] = 1.00
    tournee = [0.99] * 39                     # la tournée colle PARTOUT
    chart, audio = morceau_tourne(droite, {k: tournee for k in range(1, 12)})
    occ, _info = sim.cherche_brique(chart, 0, 1, audio, label="A")
    assert occ == [(0, 1, 1.0, 0)], "une brique de 2 mesures ne se transpose pas"


def test_la_reprise_DROITE_se_sert_avant_la_tournee(morceau_tourne):
    """Au même endroit, une reprise à l'identique vaut toujours mieux qu'une
    reprise modulée : elle prend la place, la tournée est refusée."""
    droite = [0.60] * 33
    droite[0] = 1.00
    droite[16] = 0.99                          # vraie reprise, droite
    tournee = [0.60] * 33
    tournee[16] = 1.00                         # la même, en mieux notée
    chart, audio = morceau_tourne(droite, {5: tournee})
    occ, _info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b, r) for a, b, _s, r in occ] == [(0, 7, 0), (16, 23, 0)]
