"""tests/test_dernier_tour_du_a.py — la règle 2 de Louis du 2026-10-01.

« quand une section est la dernière du morceau, qu'elle joue le début du A
puis tient la tonique et s'éteint, la traiter comme le dernier tour du A »

Cas qui l'a déclenchée : Robin Thicke « Lost Without U » (0DdCoNbbRvQ),
mesures 83-89 : G7sus4 | G7 | C | C G-7 | G-7 D-7 | N.C. | N.C. contre la
boucle A D-7 | G7 | C^7 | F^7 G-7 | D-7 | D-7 G7 | C^7 | F^7 G-7.

La 1re mesure (G7sus4 pour D-7) ne colle pas tant que la mesure de
similarité apprise ne sait pas que D-7 ≈ G7sus4 : l'ouverture tolère une
1re mesure différente, à condition que les DEUX suivantes collent.
"""
from __future__ import annotations

from harmonia.folding import dernier_tour_partiel
from harmonia.sections.simulation import nommer_dernier_tour
from harmonia.soudure import sections_pour_chart

GRILLE = [round(i * 2.8, 3) for i in range(80)]
DO = (0, "major")


def _c(root, q="", beat=0):
    return {"root": root, "q": q, "nc": False, "c": 0.9, "bass": -1,
            "beat": beat, "t0": 0.0, "t1": 1.0}


NC = {"root": 0, "q": "", "nc": True, "c": 0.5, "bass": -1, "beat": 0,
      "t0": 0.0, "t1": 1.0}

A = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")], [_c(5, "^7"), _c(7, "-7", 2)],
     [_c(2, "-7")], [_c(2, "-7"), _c(7, "7", 2)], [_c(0, "^7")],
     [_c(5, "^7"), _c(7, "-7", 2)]]
FIN_LWU = [[_c(7, "7sus4")], [_c(7, "7")], [_c(0, "")], [_c(0, ""), _c(7, "-7", 2)],
           [_c(7, "-7"), _c(2, "-7", 2)], [dict(NC)], [dict(NC)]]


# ── le détecteur ──────────────────────────────────────────────────────────
def test_lost_without_u_est_le_dernier_tour_du_A():
    r = dernier_tour_partiel(FIN_LWU, A, DO)
    assert r is not None
    assert r["ouverture"] == 3 and r["premiere_toleree"]
    assert r["tenue"] == 2 and r["fondu"] == 2


def test_une_seule_mesure_qui_colle_ne_suffit_pas():
    """1re mesure tolérée + UNE seule qui colle : pas assez de preuve."""
    fin = [[_c(7, "7sus4")], [_c(7, "7")], [_c(9, "-7")], [_c(9, "-7")], [dict(NC)]]
    assert dernier_tour_partiel(fin, A, DO) is None


def test_un_debut_de_A_qui_part_ailleurs_n_est_pas_un_dernier_tour():
    """Après l'ouverture, autre chose que la tonique tenue : une autre
    section (under-fold, never over-fold)."""
    fin = [[_c(2, "-7")], [_c(7, "7")], [_c(9, "-7")], [_c(4, "7")], [dict(NC)]]
    assert dernier_tour_partiel(fin, A, DO) is None


def test_la_tenue_doit_etre_la_tonique():
    fin = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")], [_c(0, "^7")], [dict(NC)]]
    assert dernier_tour_partiel(fin, A, DO) is not None
    assert dernier_tour_partiel(fin, A, (7, "major")) is None


def test_une_fin_aussi_longue_que_le_A_n_est_pas_partielle():
    assert dernier_tour_partiel(A + [[dict(NC)]], A, DO) is None


def test_une_tenue_plus_longue_que_l_ouverture_est_autre_chose():
    """3 mesures d'ouverture, puis 4 de tonique : plus une fin, une section."""
    fin = [[_c(2, "-7")], [_c(7, "7")], [_c(0, "^7")]] + [[_c(0, "")]] * 4
    assert dernier_tour_partiel(fin, A, DO) is None
    assert dernier_tour_partiel(fin[:-1], A, DO) is not None


# ── le nommage des zones libres (remplir_par_algo) ────────────────────────
def _blocs(dernier_label="C", source="algo"):
    bars = A + A + FIN_LWU
    out = [{"m0": 0, "m1": 7, "label": "A", "source": "algo"},
           {"m0": 8, "m1": 15, "label": "A", "source": "algo"},
           {"m0": 16, "m1": 22, "label": dernier_label, "source": source}]
    return out, bars


def test_une_zone_libre_finale_prend_la_lettre_du_A():
    out, bars = _blocs()
    nommer_dernier_tour(out, bars, DO)
    assert out[-1]["label"] == "A"


def test_une_section_posee_par_louis_n_est_jamais_renommee():
    out, bars = _blocs("outro", source="brique")
    nommer_dernier_tour(out, bars, DO)
    assert out[-1]["label"] == "outro"


# ── l'écriture (sections_pour_chart → cascade) ────────────────────────────
def _chart(secs, bars):
    n = len(bars)
    return sections_pour_chart({"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4,
                                "keyName": "C major"},
                               secs, bars=bars, fold_report={})


def _secs(dernier):
    return [{"label": "A", "mesure_debut": 1, "mesure_fin": 8},
            {"label": "A", "mesure_debut": 9, "mesure_fin": 16},
            {"label": dernier, "mesure_debut": 17, "mesure_fin": 23}]


def test_le_dernier_tour_rejoint_le_A_avec_la_tonique_en_petite_lecture():
    bars = [[dict(c) for c in b] for b in A + A + FIN_LWU]
    out = _chart(_secs("A"), bars)
    assert [s["label"] for s in out] == ["A"]
    a = out[0]
    assert a["reps"] == 3
    # les 5 mesures qui jouent s'allument, le fondu muet n'allume rien
    allumees = [r for r, spans in enumerate(a["barSpans"]) if len(spans) == 3]
    assert allumees == [0, 1, 2, 3, 4]
    # la tonique tenue (mes. 4 du tour) s'écrit en petite lecture sur F^7
    var = a["bars"][3][0].get("var")
    assert var and var["root"] == 0
    # et le bloc écrit des autres passes n'est pas touché ailleurs
    assert bars[3][0].get("var") is None


def test_une_outro_de_louis_reste_une_outro():
    bars = [[dict(c) for c in b] for b in A + A + FIN_LWU]
    out = _chart(_secs("outro"), bars)
    assert [s["label"] for s in out] == ["A", "outro"]
