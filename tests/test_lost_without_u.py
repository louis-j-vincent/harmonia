"""tests/test_lost_without_u.py — « y'a juste un A et un B, pas de A′ ».

Louis, 2026-09-30, sur Robin Thicke « Lost Without U » (0DdCoNbbRvQ). Son
découpage validé ne contient que des A, des B et une outro ; le chart
imprimait A, A′, B, B′. Deux causes dans `folding._ireal_cascade` :

1. A′ : un A de 4 mesures `D-7 | G7 | C^7 | F^7` ne rejoignait pas la boucle
   A écrite `D-7 | G7 | C^7 | F^7 G-7` — `_merge_boucle` exigeait la même
   mesure accord pour accord, et la 4e mesure du passage n'a pas le G-7 de
   relance. C'est la même musique, sans l'accord de passage.
2. B′ : la mesure 24 (un N.C. seul, étiqueté B) était choisie comme HÔTE de
   la lettre B (à égalité de passages, la première dans le morceau) ; le vrai
   B de 4 mesures, plus long, ne pouvait pas s'y fondre et prenait un prime.
   Une plage muette n'est jamais l'hôte d'une lettre : elle la rejoint.

Garde-fou « under-fold, never over-fold » : un passage qui joue AUTRE CHOSE
garde son prime.
"""
from __future__ import annotations

from harmonia.soudure import sections_pour_chart

GRILLE = [round(i * 2.8, 3) for i in range(60)]


def _c(root, q=""):
    return {"root": root, "q": q, "nc": False, "c": 0.9, "bass": -1,
            "t0": 0.0, "t1": 1.0}


NC = {"root": 0, "q": "", "nc": True, "c": 0.5, "bass": -1, "t0": 0.0, "t1": 1.0}

#: la boucle A de Lost Without U : Dm7 | G7 | Cmaj7 | Fmaj7 Gm7
BOUCLE = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")], [_c(5, "^7"), _c(7, "-7")]]


def _morceau(passage_court, avec_b_muet=True):
    """A(8) A(8) [B muet 1] A-court(4) B(4) A(8)."""
    bars, secs = [], []

    def ajoute(lab, contenu):
        b0 = len(bars)
        bars.extend([dict(c) for c in bar] for bar in contenu)
        secs.append({"label": lab, "mesure_debut": b0 + 1,
                     "mesure_fin": len(bars)})
    ajoute("A", BOUCLE * 2)
    ajoute("A", BOUCLE * 2)
    if avec_b_muet:
        ajoute("B", [[dict(NC)]])
    ajoute("A", passage_court)
    ajoute("B", [[_c(5, "^7")], [_c(2, "-7")], [_c(9, "-7")], [_c(9, "-7")]])
    ajoute("A", BOUCLE * 2)
    n = len(bars)
    return sections_pour_chart({"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4},
                               secs, bars=bars, fold_report={}), n


def test_un_A_court_sans_accord_de_passage_rejoint_la_boucle():
    court = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")], [_c(5, "^7")]]
    out, _n = _morceau(court, avec_b_muet=False)
    labels = [s["label"] for s in out]
    assert labels == ["A", "B"], labels
    a = out[0]
    assert a["boucle"]["cellule"] == 4
    assert [16, 19] in a["barRanges"]
    # la carte de lecture allume les 4 rangées pendant le passage court
    for r in range(4):
        assert [GRILLE[16 + r], GRILLE[17 + r]] in a["barSpans"][r]


def test_une_mesure_muette_n_est_jamais_l_hote_de_sa_lettre():
    court = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")], [_c(5, "^7")]]
    out, _n = _morceau(court, avec_b_muet=True)
    labels = [s["label"] for s in out]
    assert labels == ["A", "B"], labels
    b = [s for s in out if s["label"] == "B"][0]
    # le B écrit est le vrai B de 4 mesures, pas le N.C.
    assert [c["root"] for bar in b["bars"] for c in bar] == [5, 2, 9, 9]
    # la mesure muette reste dans la forme (la bande la montre à sa place)…
    assert [16, 16] in b["barRanges"]
    # …mais n'allume aucune case : il n'y a rien à lire pendant un N.C.
    allumees = [w for row in b["barSpans"] for w in row]
    assert [GRILLE[16], GRILLE[17]] not in allumees


def test_un_A_court_qui_joue_autre_chose_garde_son_prime():
    """Under-fold : `E-7 | A7 | D-7 | G7` n'est pas la boucle A."""
    autre = [[_c(4, "-7")], [_c(9, "7")], [_c(2, "-7")], [_c(7, "7")]]
    out, _n = _morceau(autre, avec_b_muet=False)
    labels = [s["label"] for s in out]
    assert "A′" in labels, labels
