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
        occ = sim.cherche_brique(chart, 0, 7, audio, label=role)
        assert occ == [(0, 7, 1.0)], f"« {role} » ne doit pas se propager"


def test_un_morceau_SANS_contraste_s_abstient(morceau):
    """Cry Me A River : médiane 0,896, aucune falaise. Un seuil fixe laissait
    passer la moitié du morceau ; le seuil relatif ne laisse passer que la
    brique."""
    plat = [0.90 + 0.02 * ((i * 7) % 5) / 5 for i in range(33)]
    plat[8] = 1.0                                 # la brique elle-même
    chart, audio = morceau(plat)
    occ = sim.cherche_brique(chart, 8, 15, audio, label="A")
    assert occ == [(8, 15, 1.0)], "rien ne se détache : on n'ajoute rien"


def test_une_falaise_nette_donne_ses_reprises(morceau):
    """Don't Want My Love : quatre scores au-dessus, puis une chute de 0,135."""
    sc = [0.65] * 33
    for b in (0, 8, 24):                          # les vraies reprises
        sc[b] = 0.99
    chart, audio = morceau(sc)
    occ = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, _ in occ] == [(0, 7), (8, 15), (24, 31)]


def test_deux_reprises_ne_se_recouvrent_jamais(morceau):
    """Une phrase de 4 mesures répétée fait scorer aussi le décalage de 4 :
    le second pic tombe DANS le premier bloc et doit être refusé."""
    sc = [0.60] * 33
    sc[0] = 1.00
    sc[4] = 0.99                                  # décalage d'une demi-brique
    sc[16] = 0.98
    chart, audio = morceau(sc)
    occ = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, _ in occ] == [(0, 7), (16, 23)]


def test_la_brique_est_TOUJOURS_rendue(morceau):
    """Même quand le seuil relatif monte au-dessus de tout — c'était le cas de
    Let It Be, où `médiane + 1,5 σ` valait 1,049 et où la brique elle-même
    disparaissait, donc le trait de Louis avec elle."""
    chart, audio = morceau([0.5] * 33)
    occ = sim.cherche_brique(chart, 3, 10, audio, label="A")
    assert occ[0] == (3, 10, 1.0) and len(occ) == 1


def test_sans_audio_la_brique_reste_seule(tmp_path):
    """Pas de matrice possible : on n'ajoute rien, et on ne plante pas."""
    chart = {"audio_url": "/audio/absent.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}
    assert sim.cherche_brique(chart, 0, 7, tmp_path, label="A") == [(0, 7, 1.0)]
