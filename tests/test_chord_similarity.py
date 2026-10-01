"""Propriétés de `harmonia.chord_similarity` (docs/similarite_accords.md).

Ce que ces tests verrouillent : symétrie, invariance par transposition,
bornes, et l'ordre proche/loin du panneau là où il n'est pas ambigu. Ils ne
verrouillent PAS les seuils (Louis les arbitre sur la page).
"""
from __future__ import annotations

import itertools

import pytest

from harmonia import chord_similarity as cs

QS = ["", "-", "7", "^7", "-7", "6", "h7", "o", "o7", "+", "sus4", "7sus4", "9", "13", "-9"]


def C(r, q="", b=-1):
    return {"root": r % 12, "q": q, "bass": -1 if b < 0 else b % 12, "nc": False}


def _paires():
    base = [C(0, q) for q in QS] + [C(10, "", 0), C(0, "", 4), C(2, "-7", 7)]
    autres = [C(r, q) for r in (0, 2, 5, 7, 9, 11) for q in ("", "-7", "7", "7sus4")]
    return list(itertools.product(base, autres))


@pytest.mark.parametrize("k", list(cs.CANDIDATES))
def test_symetrie_et_bornes(k):
    ctx = {"cle": (0, "major")}
    for a, b in _paires():
        x, y = cs.CANDIDATES[k](a, b, ctx), cs.CANDIDATES[k](b, a, ctx)
        if x is None:
            assert y is None
            continue
        assert x == pytest.approx(y, abs=1e-9), (k, a, b)
        assert -1e-9 <= x <= 1 + 1e-9


@pytest.mark.parametrize("k", list(cs.CANDIDATES))
def test_invariance_par_transposition(k):
    def tr(c, d):
        return C(c["root"] + d, c["q"], c["bass"] + d if c["bass"] >= 0 else -1)
    for a, b in _paires()[::3]:
        ref = cs.CANDIDATES[k](a, b, {"cle": (0, "major")})
        for d in (1, 5, 6, 11):
            got = cs.CANDIDATES[k](tr(a, d), tr(b, d), {"cle": (d, "major")})
            if ref is None:
                assert got is None
            else:
                assert got == pytest.approx(ref, abs=1e-6), (k, a, b, d)


@pytest.mark.parametrize("k", list(cs.CANDIDATES))
def test_identite(k):
    for q in QS:
        v = cs.CANDIDATES[k](C(4, q), C(4, q), {"cle": (4, "major")})
        if v is None:                 # `fonction` : accord hors tonalité, indéfini
            assert k == "fonction"
            continue
        assert v == pytest.approx(1.0)


def test_nc():
    nc = {"root": 0, "q": "", "bass": -1, "nc": True}
    assert cs.similarite(nc, nc) == 1.0
    assert cs.similarite(nc, C(0)) == 0.0


# ── le panneau, là où il n'est pas ambigu ──────────────────────────────────
PROCHES = [(C(2, "-7"), C(7, "7sus4")),      # D-7 / G7sus4 (Lost Without U)
           (C(10, "", 0), C(0, "7sus4")),    # Bb/C / C7sus4 (Claustrophobic)
           (C(0, "7"), C(0, "9")), (C(0), C(0, "^7")), (C(7, "7"), C(7))]
LOINS = [(C(0), C(5)), (C(0), C(6)), (C(7, "7"), C(0)),
         (C(5, "-"), C(5, "7")),              # This Love B mes. 7 : Fm / F7
         (C(8), C(10)), (C(7), C(3))]         # This Love B mes. 8 : Ab / Bb, G / Eb


def test_recommandee_separe_proches_et_loins():
    """Toutes les paires que Louis a déclarées équivalentes passent au-dessus
    de toutes celles qu'il veut différentes, avec de la marge."""
    s = cs.similarite
    assert min(s(a, b) for a, b in PROCHES) > max(s(a, b) for a, b in LOINS) + 0.1


def test_recommandee_au_cran_par_defaut():
    for a, b in PROCHES:
        assert cs.cran(a, b) <= 1
    for a, b in LOINS:
        assert cs.cran(a, b) >= 3


def test_bb_plus_pres_de_gm_que_de_f():
    """La règle verrouillée de Louis (mémoire feedback_ssm_must_be_chordtone)."""
    for k in ("notes", "notes7", "tiv"):
        f = cs.CANDIDATES[k]
        assert f(C(10), C(7, "-")) > f(C(10), C(5))


def test_d7_g7sus4_plus_pres_que_c_f_pour_toutes():
    ctx = {"cle": (0, "major")}
    for k, f in cs.CANDIDATES.items():
        assert f(C(2, "-7"), C(7, "7sus4"), ctx) > f(C(0), C(5), ctx), k


def test_couleurs_retirees():
    assert cs.notes7(C(0, "7"), C(0, "13")) == pytest.approx(1.0)
    assert cs.notes7(C(2, "-7"), C(2, "-11")) == pytest.approx(1.0)
    assert cs.notes7(C(0, "7sus4"), C(0, "9sus4")) == pytest.approx(1.0)
    # la sus n'est pas une couleur, la 6 non plus
    assert cs.notes7(C(0, "7"), C(0, "7sus4")) < 1.0
    assert cs.notes7(C(0), C(0, "6")) < 1.0


def test_crans_tombent_dans_des_trous():
    """Aucun couple d'accords usuels (sans basse étrangère) ne tombe à moins
    de 0,02 d'un seuil de cran : un arrondi ne peut pas faire basculer."""
    vals = {round(cs.notes7(C(0, qa), C(d, qb)), 4)
            for qa in QS for qb in QS for d in range(12)}
    for t in cs.CRANS.values():
        assert all(abs(v - t) >= 0.02 for v in vals), t


def test_table_musx():
    t = cs.table_musx()
    Q, S = t["Q"], t["S"]
    assert len(Q) == 25 and len(S) == 25 and len(S[0][0]) == 12
    for i in range(25):
        assert S[i][i][0] == pytest.approx(1.0, abs=1e-3)
        for j in range(25):
            for d in range(12):
                assert S[i][j][d] == pytest.approx(S[j][i][(-d) % 12], abs=1e-3)
    assert t["diagnostics"]["split_half_r"] > 0.9


def test_projection_musx_exacte_sur_son_vocabulaire():
    assert cs.nom_musx(C(2, "-7")) == "D:min7"
    assert cs.nom_musx(C(7, "7sus4")) == "G:sus4(b7)"
    assert cs.nom_musx(C(10, "", 0)) == "Bb:maj/2"
    assert cs.nom_musx(C(0, "", 4)) == "C:maj/3"
