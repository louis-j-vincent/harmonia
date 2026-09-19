"""Le détecteur de Foote repêché — `harmonia/sections/nouveaute.py`.

Ces tests sont AUTONOMES : aucune audio, aucun modèle, aucun cache. Ils
construisent des matrices de ressemblance à la main, parce que la propriété
qu'on veut verrouiller est celle du damier, pas celle de musx.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.sections import nouveaute as nv


# ── de quoi fabriquer une SSM jouet ────────────────────────────────────────
def ssm_blocs(longueurs: list[int], *, dedans: float = 1.0,
              dehors: float = 0.0) -> np.ndarray:
    """Une SSM en blocs : `dedans` à l'intérieur d'un bloc, `dehors` entre.

    C'est la forme d'un morceau idéal : chaque section se ressemble à
    elle-même et ne ressemble à rien d'autre.
    """
    n = sum(longueurs)
    S = np.full((n, n), float(dehors))
    d = 0
    for L in longueurs:
        S[d:d + L, d:d + L] = float(dedans)
        d += L
    np.fill_diagonal(S, 1.0)
    return S


def frontieres_vraies(longueurs: list[int]) -> list[int]:
    out, d = [], 0
    for L in longueurs[:-1]:
        d += L
        out.append(d)
    return out


# ── la courbe ──────────────────────────────────────────────────────────────
def test_la_courbe_a_une_valeur_par_mesure():
    for n in (1, 5, 16, 33, 64):
        S = np.eye(n)
        assert nv.courbe_de_nouveaute(S).shape == (n,)


def test_la_courbe_refuse_une_matrice_non_carree():
    with pytest.raises(ValueError):
        nv.courbe_de_nouveaute(np.zeros((4, 7)))


def test_deux_blocs_de_8_le_pic_tombe_sur_la_vraie_frontiere():
    """Le cas d'école : 8 mesures, puis 8 autres, sans rien de commun.

    Le maximum de la courbe doit tomber EXACTEMENT sur la mesure 8, la
    première du second bloc.
    """
    S = ssm_blocs([8, 8])
    nov = nv.courbe_de_nouveaute(S)
    assert int(nov.argmax()) == 8


@pytest.mark.parametrize("longueurs", [[8, 8], [16, 16], [12, 20],
                                       [16, 16, 16], [24, 16, 24]])
def test_le_pic_tombe_sur_la_frontiere_quelle_que_soit_la_forme(longueurs):
    S = ssm_blocs(longueurs)
    nov = nv.courbe_de_nouveaute(S)
    vraies = frontieres_vraies(longueurs)
    assert int(nov.argmax()) in vraies


def test_la_courbe_vaut_le_contraste_en_points_de_cosinus():
    """Sur deux blocs parfaits, « dedans − à travers » vaut 1,0 au pic.

    C'est ce qui rend `CONTRASTE_MIN` lisible : l'unité est le cosinus, pas
    une somme de noyau en unités arbitraires (l'ancien code ne normalisait
    pas, voir le point 3 du docstring du module).
    """
    nov = nv.courbe_de_nouveaute(ssm_blocs([16, 16], dedans=1.0, dehors=0.0))
    assert nov.max() == pytest.approx(1.0, abs=0.05)
    nov = nv.courbe_de_nouveaute(ssm_blocs([16, 16], dedans=1.0, dehors=0.7))
    assert nov.max() == pytest.approx(0.3, abs=0.05)


# ── les frontières ─────────────────────────────────────────────────────────
def test_une_ssm_plate_ne_rend_aucune_frontiere():
    """Pas de contraste, pas de coupure — surtout pas du bruit numérique."""
    assert nv.frontieres(np.ones((64, 64))) == []
    assert nv.frontieres(np.zeros((64, 64))) == []


def test_une_ssm_presque_plate_ne_rend_aucune_frontiere():
    """Même chose avec du bruit : sous `CONTRASTE_MIN`, on s'abstient."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1e-3, (64, 64))
    S = 0.9 + (A + A.T) / 2.0
    np.fill_diagonal(S, 1.0)
    assert float(nv.courbe_de_nouveaute(S).max()) < nv.CONTRASTE_MIN
    assert nv.frontieres(S) == []


def test_trois_blocs_les_frontieres_sont_retrouvees():
    longueurs = [16, 16, 16]
    got = nv.frontieres(ssm_blocs(longueurs))
    assert got == frontieres_vraies(longueurs)


def test_la_mesure_zero_n_est_jamais_rendue():
    got = nv.frontieres(ssm_blocs([16, 16, 16]))
    assert 0 not in got


def test_min_bars_espace_les_frontieres():
    S = ssm_blocs([16, 16, 16, 16])
    for mb in (2, 4, 8):
        got = nv.frontieres(S, min_bars=mb)
        assert all(b - a >= mb for a, b in zip(got, got[1:]))


# ── `dans=` : les briques de Louis ne sont JAMAIS touchées ─────────────────
def test_dans_ne_rend_rien_hors_des_trous():
    """Toute frontière rendue est strictement à l'intérieur d'un trou."""
    longueurs = [16, 16, 16, 16]
    S = ssm_blocs(longueurs)
    trous = [(20, 44)]
    got = nv.frontieres(S, dans=trous)
    assert got, "le trou contient une vraie frontière, on attend au moins une"
    for c in got:
        assert any(b0 + nv.MIN_SEG_BARS <= c <= b1 + 1 - nv.MIN_SEG_BARS
                   for b0, b1 in trous), c


def test_dans_est_un_sous_ensemble_du_sans_dans():
    """Restreindre ne peut que retirer : `dans=` ne fabrique rien."""
    S = ssm_blocs([16, 16, 16, 16])
    tout = set(nv.frontieres(S))
    for trous in ([(0, 31)], [(32, 63)], [(10, 25), (40, 60)]):
        assert set(nv.frontieres(S, dans=trous)) <= tout


def test_dans_avec_un_trou_trop_petit_ne_rend_rien():
    """Un trou de 3 mesures ne peut pas se couper en deux moitiés de 2."""
    S = ssm_blocs([16, 16, 16, 16])
    assert nv.frontieres(S, dans=[(30, 32)]) == []
    # le même trou, à peine plus grand, se coupe : 4 mesures = 2 + 2
    assert nv.frontieres(S, dans=[(30, 33)]) == [32]


def test_dans_vide_ne_rend_rien():
    assert nv.frontieres(ssm_blocs([16, 16, 16]), dans=[]) == []


def test_dans_ne_propose_jamais_le_bord_d_une_brique():
    """Le début d'un trou est DÉJÀ une frontière (la fin d'une brique) — on
    ne le rend pas, et on ne rend rien à moins de `min_bars` de lui."""
    S = ssm_blocs([16, 16, 16, 16])
    trou = (32, 63)
    for c in nv.frontieres(S, dans=[trou], min_bars=4):
        assert c - trou[0] >= 4 and (trou[1] + 1) - c >= 4


# ── les constantes repêchées ───────────────────────────────────────────────
def test_les_constantes_gardent_leur_valeur_d_origine():
    """`8d5137b:harmonia_min/sections.py` — une régression ici est
    silencieuse et coûteuse, elle mérite un test."""
    assert nv.KERNEL_HB == 16
    assert nv.BLUR_SIGMA == 0.0
    assert nv.PEAK_FRAC == 0.5
    assert nv.MIN_SEG_BARS == 2
    assert nv.TILE_QUANTILE == 0.95
    assert nv.TILE_LAG_MIN == 2
    assert nv.RUN_COVERAGE_MIN == 0.5
    # la conversion d'unité : 16 DEMI-mesures = 8 mesures
    assert nv.NOYAU_MESURES == 8


def test_le_flou_a_zero_ne_touche_a_rien():
    S = ssm_blocs([8, 8])
    assert np.array_equal(nv._flou(S, 0.0), S)


# ── le tuilage ─────────────────────────────────────────────────────────────
def test_le_seuil_de_tuilage_se_tait_sur_un_morceau_trop_court():
    assert nv.seuil_de_tuilage(np.ones((3, 3))) == 1.0


def test_le_tuilage_trouve_une_boucle_de_4_mesures():
    """Une vraie boucle : la mesure b ressemble à b±4, et à rien d'autre."""
    n = 32
    motif = np.eye(4)[np.arange(n) % 4]        # (n, 4), une classe par temps
    S = motif @ motif.T
    runs = nv.tuilages(S)
    assert runs, "une boucle parfaite de 4 mesures doit être vue"
    assert {r["period"] for r in runs} <= {2, 4, 8}
    couverture = sum(r["b1"] - r["b0"] + 1 for r in runs) / n
    assert couverture > nv.RUN_COVERAGE_MIN


def test_le_tuilage_ne_plante_pas_sur_un_morceau_minuscule():
    for n in (0, 1, 2, 3, 5):
        assert nv.tuilages(np.eye(n)) == []


def test_tuilage_false_ne_garde_que_les_pics():
    S = ssm_blocs([16, 16, 16])
    assert nv.frontieres(S, tuilage=False) == frontieres_vraies([16, 16, 16])


# ── le module ne touche à rien ─────────────────────────────────────────────
def test_le_module_n_importe_ni_torch_ni_audio():
    """Pur numpy/scipy : pas de modèle, pas de décodage audio, et pas même
    un import du reste de `harmonia` — ce module doit rester importable et
    testable tout seul."""
    import ast

    with open(nv.__file__, encoding="utf-8") as fh:
        arbre = ast.parse(fh.read())
    modules = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            modules |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            modules.add(n.module.split(".")[0])
    assert modules <= {"__future__", "numpy", "scipy"}, modules
