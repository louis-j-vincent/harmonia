"""harmonia.span_rescore.delta_candidates — classer les candidats d'un créneau
par ce qu'ils GAGNENT sur le précédent, au lieu de leur probabilité brute.

Louis, 2026-09-16 : « la résonance et la pédale font que l'accord d'avant
continue de bien scorer — le delta l'annule, puisqu'il était déjà haut. »

Les postérieures sont synthétiques : ce test pose une pédale à la main (un
accord qui reste haut parce qu'il résonne) et vérifie que le classement par
delta le fait descendre. Le cas réel — Ready mes. 3 et 4, les deux accords
que Louis a corrigés à l'oreille — est reproduit dans
`docs/known_issues.md` avec ses rangs mesurés ; il demande l'audio et les
postérieures en cache, donc il ne tourne pas ici.
"""
import numpy as np
import pytest

from harmonia.musx import FRAME_DT
from harmonia.span_rescore import (N_CANDIDATES, delta_candidates, idx_of,
                                   token_of)

N_FRAMES = 200


def _probs(plan):
    """`plan` : [(frame0, frame1, {colonne triade: masse})] → postérieures musx.

    Seuls `triad` (73) et `s7` (4) sont lus par le pooling ; le reste est posé
    à une forme valide et ignoré.
    """
    tri = np.zeros((N_FRAMES, 73), dtype=np.float32)
    tri[:, 0] = 1.0                       # N par défaut
    for f0, f1, cols in plan:
        tri[f0:f1, :] = 0.0
        for col, mass in cols.items():
            tri[f0:f1, col] = mass
    s7 = np.zeros((N_FRAMES, 4), dtype=np.float32)
    s7[:, 0] = 1.0                        # pas de septième
    petit = lambda k: np.full((N_FRAMES, k), 1.0 / k, dtype=np.float32)  # noqa: E731
    return [tri, np.zeros((N_FRAMES, 13), dtype=np.float32), s7,
            petit(4), petit(3), petit(3)]


#: colonne triade de musx : 1 + racine + 12 × type ; type 0 = majeur, 1 = mineur
def _col(root, typ=0):
    return 1 + root % 12 + 12 * typ


def _span(f0, f1):
    return (f0 * FRAME_DT, f1 * FRAME_DT)


def test_la_pedale_descend_quand_on_classe_par_gain():
    """Le cas de Louis, en laboratoire : l'accord d'avant reste haut par
    résonance, le vrai nouvel accord monte. La probabilité brute donne le
    premier ; le gain donne le second."""
    C, Ab = _col(0), _col(8)
    probs = _probs([(0, 100, {C: 0.95, Ab: 0.02}),        # créneau 1 : C net
                    (100, 200, {C: 0.60, Ab: 0.30})])     # créneau 2 : C résonne
    prev, cur = _span(0, 100), _span(100, 200)

    brut = delta_candidates(probs, cur, None)
    assert brut["source"] == "posterior"
    assert (brut["sug"][0]["root"], brut["sug"][0]["q"]) == (0, "")   # C en tête

    par_gain = delta_candidates(probs, cur, prev)
    assert par_gain["source"] == "delta"
    assert (par_gain["sug"][0]["root"], par_gain["sug"][0]["q"]) == (8, "")  # Ab
    # `c` reste la probabilité DU CRÉNEAU, pas le gain : les deux se lisent,
    # et celui que le gain met en tête vaut MOINS que celui de la proba brute.
    assert par_gain["sug"][0]["c"] < brut["sug"][0]["c"]
    assert par_gain["sug"][0]["gain"] > 0
    # La pédale est bien descendue, sans disparaître : elle reste un candidat.
    assert (0, "") in {(c["root"], c["q"]) for c in par_gain["sug"]}


def test_le_plancher_s_applique_avant_le_classement():
    """Un candidat qui passe de presque rien à presque rien a « gagné »
    proportionnellement beaucoup — il ne doit pas doubler un vrai accord.
    C'est l'écart assumé par rapport à la description de Louis."""
    C, Ab, B = _col(0), _col(8), _col(11)
    probs = _probs([(0, 100, {C: 0.97, Ab: 0.02, B: 0.001}),
                    (100, 200, {C: 0.50, Ab: 0.40, B: 0.005})])
    out = delta_candidates(probs, _span(100, 200), _span(0, 100))
    montres = {(c["root"], c["q"]) for c in out["sug"]}
    assert (8, "") in montres                 # le vrai gagnant est là
    assert (11, "") not in montres            # le bruit qui « double » ne l'est pas


def test_sans_creneau_precedent_on_le_dit():
    """Premier accord d'un morceau : pas de delta possible. On rend le
    classement brut, et `source` l'annonce — jamais un repli muet."""
    probs = _probs([(0, 100, {_col(0): 0.9})])
    out = delta_candidates(probs, _span(0, 100), None)
    assert out["source"] == "posterior"
    assert out["sug"] and all("gain" not in c for c in out["sug"])


def test_forme_et_plafond():
    probs = _probs([(0, 100, {_col(0): 0.5, _col(5): 0.3}),
                    (100, 200, {_col(7): 0.4, _col(2): 0.3, _col(4): 0.2})])
    out = delta_candidates(probs, _span(100, 200), _span(0, 100), top_k=3)
    assert len(out["sug"]) <= 3
    for c in out["sug"]:
        assert set(c) >= {"root", "q", "c"}
        assert 0 <= c["root"] < 12 and 0.0 <= c["c"] <= 1.0


def test_un_creneau_muet_rend_quand_meme_un_candidat():
    """Un créneau quasi silencieux : on ne rend JAMAIS une liste vide, qui
    ferait dire à l'éditeur « pas de classement » alors qu'on en a un.

    (Le nombre exact dépend du repli des 6 types de musx sur les 5 familles —
    une case QUAL5 peut recevoir deux colonnes et passer le plancher là où
    l'uniforme ne le passerait pas. C'est le « non vide » qui est le
    contrat, pas le compte.)"""
    tri = np.zeros((N_FRAMES, 73), dtype=np.float32)
    tri[:, 0] = 0.99                      # quasi tout en N
    for j in range(1, 73):
        tri[:, j] = 0.01 / 72
    s7 = np.zeros((N_FRAMES, 4), dtype=np.float32); s7[:, 0] = 1.0
    petit = lambda k: np.full((N_FRAMES, k), 1.0 / k, dtype=np.float32)  # noqa: E731
    probs = [tri, np.zeros((N_FRAMES, 13), dtype=np.float32), s7,
             petit(4), petit(3), petit(3)]
    out = delta_candidates(probs, _span(100, 200), _span(0, 100))
    assert out["sug"]


@pytest.mark.parametrize("i", [0, 7, N_CANDIDATES - 1])
def test_aller_retour_des_index(i):
    r, q5 = token_of(i)
    assert idx_of(r, q5) == i
