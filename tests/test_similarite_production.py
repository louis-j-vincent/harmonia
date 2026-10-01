"""La mesure de ressemblance branchée dans `roles.meme_lecture` (grain
"famille") et dans `folding.meme_harmonie` — verdicts de Louis du 2026-10-01
sur `docs/plots/similarite_accords.html` :

  1. « G7 différent de Gm7, ce sont des familles différentes, il faut faire
     peser la tierce. Un sus4 est ambigu entre mineur et majeur par essence,
     donc proche des deux. »
  2. « This Love, les mesures 4 et 8 sont la même chose. »
  3. « Dm7 tenu » contre « Dm7 | G7 » : « c'est au cas par cas […]
     l'utilisateur peut choisir » — le curseur « Section merging ».

Écrits AVANT le branchement (rouges contre d0222a8).
"""
from __future__ import annotations

import json

import pytest

from harmonia import chord_similarity as cs
from harmonia.roles import cran_de_fusion, meme_lecture
from harmonia.settings import SETTINGS


def C(r, q="", b=-1, beat=0):
    return {"root": r % 12, "q": q, "bass": -1 if b < 0 else b % 12, "nc": False, "beat": beat}


G7, Gm7, G7sus4, Dm7 = C(7, "7"), C(7, "-7"), C(7, "7sus4"), C(2, "-7")


# ── 1. la tierce ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [(G7, Gm7), (C(0), C(0, "-")), (C(9, "7"), C(9, "-7")),
                                 (C(5, "-"), C(5, "7"))])
def test_tierces_contraires_sont_loin(a, b):
    assert not meme_lecture(a, b)
    assert cs.similarite(a, b) < cs.CRANS_V2[3][0]       # loin même au cran 3


@pytest.mark.parametrize("autre", [G7, Gm7, C(7), C(7, "-")])
def test_sus4_proche_des_deux(autre):
    assert meme_lecture(G7sus4, autre)
    assert meme_lecture(C(7, "sus4"), autre)


def test_d7_g7sus4_et_bb_c():
    assert meme_lecture(Dm7, G7sus4)
    assert meme_lecture(C(10, "", 0), C(0, "7sus4"))
    assert meme_lecture(C(0, "7"), C(0, "7sus4"))


def test_c_f_et_this_love_restent_loin():
    for a, b in [(C(0), C(5)), (C(8), C(10)), (C(7), C(3))]:
        for n in range(4):
            with cran_de_fusion(n):
                assert not meme_lecture(a, b), (a, b, n)


def test_couleurs_toujours_egales():
    for a, b in [(C(0, "7"), C(0, "9")), (C(0), C(0, "^7")), (C(7, "7"), C(7)),
                 (C(2, "-"), C(2, "-7")), (C(0), C(0, "", 4))]:
        assert meme_lecture(a, b)


# ── 2. This Love, mes. 4 ≡ mes. 8 de l'intro ────────────────────────────────

def test_this_love_mes4_egale_mes8():
    from harmonia.folding import meme_harmonie
    mes4 = [C(2, "o", beat=0), C(5, "-", 8, beat=2)]       # D° puis Fm/A♭
    mes8 = [C(2, "h7", beat=0)]                            # Dø7
    assert meme_harmonie(mes8, mes4)


# ── 3. le rythme harmonique suit le curseur ────────────────────────────────

def test_dm7_tenu_contre_dm7_g7_selon_le_cran():
    from harmonia.folding import meme_harmonie
    tenu, ii_v = [C(2, "-7")], [C(2, "-7", beat=0), C(7, "7", beat=2)]
    with cran_de_fusion(1):
        assert not meme_harmonie(tenu, ii_v)
        assert not meme_harmonie([Dm7], [G7])
    with cran_de_fusion(3):
        assert meme_harmonie([Dm7], [G7])


def test_crans_monotones():
    seuils = [cs.CRANS_V2[n] for n in sorted(cs.CRANS_V2)]
    for (t0, d0), (t1, d1) in zip(seuils, seuils[1:]):
        assert t1 <= t0 and d1 >= d0


def _lwu():
    p = SETTINGS.repo / "state" / "cache" / "charts" / "min_0DdCoNbbRvQ.json"
    h = SETTINGS.repo / "state" / "human" / "sections" / "0DdCoNbbRvQ.json"
    if not p.exists() or not h.exists():
        pytest.skip("Lost Without U absent")
    return json.loads(p.read_text()), json.loads(h.read_text())


def test_lost_without_u_fin_rejoint_A_au_cran_haut_seulement():
    from harmonia.soudure import sections_pour_chart
    chart, human = _lwu()
    secs = [{"label": s["label"], "mesure_debut": s["b0"] + 1, "mesure_fin": s["b1"] + 1}
            for s in human["sections"]]
    labels = {n: [s["label"] for s in sections_pour_chart(chart, secs, fusion=n)]
              for n in (1, 3)}
    assert "outro" in labels[1]
    assert "outro" not in labels[3]


# ── garde-fous trouvés par le rapport d'or (2026-10-01) ─────────────────────

@pytest.mark.parametrize("a,b", [(C(2, "^7"), C(6, "-7")),     # San Francisco Street : I^7 / iii-7
                                 (C(5, "7"), C(2, "-7")),      # Don't Know Why : F7 / D-7
                                 (C(0, "^7"), C(9, "-7"))])    # tonique / relatif
def test_deux_tierces_sur_deux_fondamentales_restent_differentes_au_defaut(a, b):
    assert not meme_lecture(a, b)


def test_seuils_v2_loin_des_valeurs():
    qs = ["", "-", "7", "^7", "-7", "6", "h7", "o", "o7", "+", "sus4", "7sus4", "9", "13"]
    vals = {round(cs.v2(C(0, qa), C(d, qb)), 4) for qa in qs for qb in qs for d in range(12)}
    for t, _d in cs.CRANS_V2.values():
        assert all(abs(v - t) >= 0.015 for v in vals), (t, sorted(v for v in vals if abs(v - t) < .015))
