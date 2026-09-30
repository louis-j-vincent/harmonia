"""tests/test_candidats_en_suspens.py — règle 5b + candidats en suspens.

Louis, 2026-09-30, sur `docs/plots/occurrences_brique_v4_2026-09-30.html` :

    « Oui pour la 5b très bien, mais alors il faut trancher par rapport à la
    cohérence globale du morceau, en cherchant d'autres sections et en gardant
    ces 2 candidats potentiels sans en acter un. Puis on voit ce qui reste
    quand les autres sections ont mangé ce qu'elles ont à manger. »

1. 5b : deux départs voisins sont à égalité tant que leur écart reste sous la
   TOLÉRANCE DU MORCEAU de la règle 4 (et plus sous 0,01).
2. Deux calages rivaux à égalité (même nombre de boucles) ne sont pas actés :
   ils restent « en suspens », exclusifs l'un de l'autre.
3. On tranche quand toutes les sections connues sont posées : une alternative
   mangée par une autre section perd ; puis celle qui est dans la phase des
   autres occurrences de la brique (collée à elles) ; puis celle qui ne laisse
   pas de fragment orphelin (< 4 mesures) ; le score seulement en dernier.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.sections import simulation as sim


@pytest.fixture()
def courbe(tmp_path, monkeypatch):
    """La courbe de reprise pilotée directement ; matrice nulle, donc aucune
    boucle interne (boucle cherchée = la brique, 8 mesures)."""
    (tmp_path / "essai.m4a").write_bytes(b"x")
    chart = {"audio_url": "/audio/essai.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}

    def pose(scores):
        from harmonia.sections import similarity as vs
        monkeypatch.setattr("harmonia.musx.frame_posteriors",
                            lambda *a, **k: [np.zeros((10, 73), np.float32)])
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: np.zeros((40, 40)))
        monkeypatch.setattr(vs, "harmonic_vectors",
                            lambda *a, **k: np.zeros((40, 12)))
        monkeypatch.setattr(vs, "_slide",
                            lambda M, a0, L, n: np.asarray(scores, float))
        return chart, tmp_path

    return pose


# ── 5b : l'égalité, c'est la tolérance du morceau ────────────────────────────

def test_5b_egalite_a_la_tolerance_du_morceau():
    """Lazy Song : le bon départ (0,963) est un peu plus bas que le décalé
    (0,991). À 0,01 près, le décalé l'efface ; à la tolérance du morceau
    (0,073), les deux restent."""
    c = [0.5] * 20
    c[4], c[6] = 0.963, 0.991
    assert sim.maxima_locaux(c, 4, {4, 6}) == {6}
    assert sim.maxima_locaux(c, 4, {4, 6}, tol=0.073) == {4, 6}


def test_5b_deux_calages_a_egalite_restent_en_suspens(courbe):
    """Deux départs rivaux (même reprise, calée à 1 mesure près), écart 0,012 :
    sous la tolérance du morceau, aucun n'efface l'autre, et chacun ne case
    qu'UNE reprise — rien ne les départage dans la brique seule. Ils sont
    rendus comme un groupe « en suspens » à deux alternatives."""
    sc = [0.55] * 33
    sc[0] = 1.0
    sc[20] = 0.99                       # une reprise sans rival
    sc[10], sc[11] = 0.975, 0.987       # deux calages rivaux
    chart, audio = courbe(sc)
    _occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    groupes = info["suspens"]
    assert len(groupes) == 1
    assert sorted(alt[0][0] for alt in groupes[0]["alternatives"]) == [10, 11]


# ── la résolution par la cohérence globale ───────────────────────────────────

def test_the_walk_la_phase_des_autres_occurrences_tranche():
    """The Walk : A–Bm–A–Bm, deux calages à égalité en 53 et 55. Les boucles
    confirmées de A finissent en 52 ; le B de Louis commence en 59. 53 colle
    à la dernière boucle de A (même phase) ; 55 laisse 53–54 orphelines. Le
    score (55 un peu plus haut) ne compte pas."""
    g = {"label": "A", "boucle": 4,
         "alternatives": [[(53, 56, 0.975, 0)], [(55, 58, 0.990, 0)]]}
    placees = [{"m0": 45, "m1": 48, "label": "A"}, {"m0": 49, "m1": 52, "label": "A"},
               {"m0": 59, "m1": 70, "label": "B"}]
    choisis = sim.resoudre_suspens([g], placees, 80)
    assert [o for _l, o in choisis] == [(53, 56, 0.975, 0)]
    assert g["raison"] == "phase"


def test_lazy_song_une_autre_section_mange_une_alternative():
    """Deux calages de A à égalité (58 et 60) ; la reprise de C (posée par
    une autre brique) occupe 62–69 : 60–63 est mangée, 58 reste — même si
    60 avait le meilleur score."""
    g = {"label": "A", "boucle": 4,
         "alternatives": [[(58, 61, 0.963, 0)], [(60, 63, 0.991, 0)]]}
    placees = [{"m0": 62, "m1": 69, "label": "C"}]
    choisis = sim.resoudre_suspens([g], placees, 90)
    assert [o for _l, o in choisis] == [(58, 61, 0.963, 0)]
    assert g["raison"] == "mangee"


def test_encore_a_egalite_a_la_fin_le_score_tranche():
    """Rien autour : ni section qui mange, ni occurrence de la brique à qui
    coller, aucun fragment d'un côté plus que de l'autre. Seul le score
    reste : le plus haut gagne."""
    g = {"label": "A", "boucle": 4,
         "alternatives": [[(20, 23, 0.975, 0)], [(22, 25, 0.990, 0)]]}
    choisis = sim.resoudre_suspens([g], [], 60)
    assert [o for _l, o in choisis] == [(22, 25, 0.990, 0)]
    assert g["raison"] == "score"


def test_occurrences_de_tous_tranche_avec_les_autres_briques(monkeypatch):
    """Le flux complet : la brique A rend deux calages en suspens (26 ou 28,
    localement 28 gagnait au score) ; la brique B, posée APRÈS dans la liste
    de Louis, a une reprise en 30–37 qui mange 28–31. On ne sait qu'une fois
    B posée : A garde 26, et la reprise est marquée `suspens` (une
    proposition, pas une contrainte)."""
    def faux(chart, b0, b1, audio_dir, label=""):
        if label == "A":
            alts = [[(26, 29, 0.975, 0)], [(28, 31, 0.99, 0)]]
            return ([(0, 3, 1.0, 0), (28, 31, 0.99, 0)],
                    {"suspens": [{"label": None, "boucle": 4, "alternatives": alts}]})
        return [(10, 17, 1.0, 0), (30, 37, 0.97, 0)], {}
    monkeypatch.setattr(sim, "cherche_brique", faux)
    chart = {"barGrid": list(range(61))}
    T = lambda a, b, lbl: {"mesure_debut": a + 1, "mesure_fin": b + 1, "label": lbl}  # noqa: E731
    occ, _, _ = sim.occurrences_de_tous(chart, [T(0, 3, "A"), T(10, 17, "B")], None)
    plages = {(o["m0"], o["m1"], o["label"]): o for o in occ}
    assert (28, 31, "A") not in plages
    assert (26, 29, "A") in plages and plages[(26, 29, "A")]["suspens"]
    assert (30, 37, "B") in plages
