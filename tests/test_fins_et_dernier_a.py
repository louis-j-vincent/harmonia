"""tests/test_fins_et_dernier_a.py — bien gérer les queues (Louis, 2026-10-12).

« il faudrait pouvoir toujours bien gérer les queues, je vois que ce n'est pas
forcément le cas ici, sur les parties A » — Evening Star (`min_CHVWPIFAT5A`),
en Fa : A 13–20 et A 21–28 ne diffèrent que par leurs deux dernières mesures
(`F C | G-7 C7` contre `F | F B♭`), plus deux couleurs ailleurs (`C7` contre
`C/G C7`, `F C` contre `F`). Et le morceau finit par le début du A (37–43)
collé au bout du B que l'algorithme a nommé.

Deux règles, choisies sur maquettes :
  1. « Un A, fins 1. et 2. » : une différence confinée au dernier quart de la
     section (2 mesures au plus) ne refuse ni la pile ni le bloc ; ces mesures
     deviennent les fins 1./2.
  2. « Un dernier A » : la zone finale nommée par l'algorithme qui rejoue le
     DÉBUT d'une lettre connue jusqu'à la fin du morceau est coupée là.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from harmonia.folding import _ireal_cascade, _ireal_endings, tronc_et_queue
from harmonia.sections.simulation import nommer_dernier_tour
from harmonia.soudure import sections_pour_chart

GRILLE = [round(i * 2.0, 3) for i in range(120)]
FA = (5, "major")


def _c(root, q="", beat=0, bass=-1):
    return {"root": root, "q": q, "nc": False, "c": 0.9, "bass": bass,
            "beat": beat, "t0": 0.0, "t1": 1.0}


F, C7, Gm, G7, Bb = 5, 0, 7, 7, 10
# Evening Star, le décodage brut (soudure.accords_par_mesure), mes. 13–20 et 21–28
A1 = [[_c(F)], [_c(C7, "7")], [_c(F)], [_c(Gm, "-")], [_c(F, bass=0)],
      [_c(G7, "7"), _c(C7, "7", 2)], [_c(F), _c(0, "", 2)],
      [_c(Gm, "-7"), _c(C7, "7", 2)]]
A2 = [[_c(F)], [_c(0, "", bass=7), _c(C7, "7", 2)], [_c(F)], [_c(Gm, "-")],
      [_c(F, bass=0)], [_c(G7, "7"), _c(C7, "7", 2)], [_c(F)],
      [_c(F), _c(Bb, "", 2)]]
# B 29–36 puis le dernier A 37–43 (F^7 C7 | C7 | F | G- | F/C | G7 C7 | F D♭7)
B8 = [[_c(4, "-"), _c(9, "7", 2)], [_c(9, "7")], [_c(2, "7")], [_c(2, "7")],
      [_c(2, "-7"), _c(7, "7", 2)], [_c(7, "7")], [_c(Gm, "-7"), _c(C7, "7", 2)],
      [_c(Gm, "-7"), _c(C7, "7", 2)]]
FIN = [[_c(F, "^7"), _c(C7, "7", 2)], [_c(C7, "7")], [_c(F)], [_c(Gm, "-")],
       [_c(F, bass=0)], [_c(G7, "7"), _c(C7, "7", 2)], [_c(F), _c(1, "7", 2)]]


def _bars(*blocs):
    return [[dict(c) for c in b] for blk in blocs for b in blk]


# ── règle 1 : le tronc et la queue ────────────────────────────────────────
def test_queue_de_deux_mesures_malgre_deux_couleurs_dans_le_tronc():
    assert tronc_et_queue([A1, A2]) == 6


def test_une_vraie_difference_au_milieu_refuse_les_fins():
    """Une autre harmonie en mesure 4 : ce n'est pas une queue."""
    autre = copy.deepcopy(A2)
    autre[3] = [_c(2, "7")]
    assert tronc_et_queue([A1, autre]) is None


def test_une_difference_plus_longue_que_le_dernier_quart_refuse_les_fins():
    """Mesures 6, 7, 8 différentes sur 8 : trois mesures, plus d'un quart."""
    autre = copy.deepcopy(A2)
    autre[5] = [_c(2, "7")]
    assert tronc_et_queue([A1, autre]) is None


def test_la_regle_exacte_d_avant_reste_la_meme():
    """Tronc identique, queue de 2 sur 4 mesures (Sunny, The Lazy Song) : la
    règle d'avant l'acceptait, elle l'accepte toujours."""
    a = [[_c(0)], [_c(5)], [_c(7)], [_c(0)]]
    b = [[_c(0)], [_c(5)], [_c(2, "-")], [_c(9, "-")]]
    assert tronc_et_queue([a, b]) == 2


def test_evening_star_ecrit_un_A_avec_fins_1_et_2():
    bars = _bars(A1, A2)
    blk = {"label": "A", "reps": 2, "barRanges": [[0, 7], [8, 15]],
           "spans": [[0, 16], [16, 32]], "bars": bars[8:16],
           "barSpans": [[[GRILLE[r], GRILLE[r + 1]], [GRILLE[8 + r], GRILLE[9 + r]]]
                        for r in range(8)]}
    _ireal_endings(blk, bars, GRILLE)
    e = blk.get("endings")
    assert e is not None and e["tail"] == 2
    assert [v["passes"] for v in e["variants"]] == [[0], [1]]
    # la 1re fin est celle du 1er passage : F C | G-7 C7 (elle était PERDUE)
    assert [c["root"] for c in e["variants"][0]["bars"][1]] == [Gm, C7]
    assert [c["root"] for c in e["variants"][1]["bars"][1]] == [F, Bb]
    # 6 rangées de tronc (2 passes chacune) + 2 × 2 rangées de fin (1 passe)
    assert [len(r) for r in blk["barSpans"]] == [2] * 6 + [1] * 4


# ── règle 2 : le dernier A ────────────────────────────────────────────────
def _blocs(source_fin="algo"):
    bars = _bars(A1, A2, B8, FIN)
    out = [{"m0": 0, "m1": 7, "label": "A", "source": "brique"},
           {"m0": 8, "m1": 15, "label": "A", "source": "brique"},
           {"m0": 16, "m1": 30, "label": "B", "source": source_fin}]
    return out, bars


def test_la_zone_finale_de_l_algo_est_coupee_au_debut_du_A():
    out, bars = _blocs()
    nommer_dernier_tour(out, bars, FA)
    assert [(o["label"], o["m0"], o["m1"]) for o in out] == [
        ("A", 0, 7), ("A", 8, 15), ("B", 16, 23), ("A", 24, 30)]
    assert out[-1]["source"] == "algo" and out[-1]["regle"] == "dernier tour"


def test_un_bloc_trace_par_louis_n_est_jamais_coupe():
    out, bars = _blocs(source_fin="brique")
    nommer_dernier_tour(out, bars, FA)
    assert [(o["label"], o["m0"], o["m1"]) for o in out][-1] == ("B", 16, 30)


def test_une_seule_mesure_de_A_a_la_fin_ne_coupe_rien():
    """Moins de la moitié du A : c'est la fin du B, pas un tour de A."""
    bars = _bars(A1, A2, B8, [[_c(F)]])
    out = [{"m0": 0, "m1": 7, "label": "A", "source": "brique"},
           {"m0": 8, "m1": 15, "label": "A", "source": "brique"},
           {"m0": 16, "m1": 24, "label": "B", "source": "algo"}]
    nommer_dernier_tour(out, bars, FA)
    assert len(out) == 3 and out[-1]["label"] == "B"


# ── l'écriture : un A avec ses fins + son dernier tour, rien de perdu ────────
def test_le_chart_ecrit_un_seul_A_fins_1_et_2_et_le_dernier_tour():
    bars = _bars(A1, A2, B8, FIN)
    n = len(bars)
    secs = [{"label": "A", "mesure_debut": 1, "mesure_fin": 8},
            {"label": "A", "mesure_debut": 9, "mesure_fin": 16},
            {"label": "B", "mesure_debut": 17, "mesure_fin": 24},
            {"label": "A", "mesure_debut": 25, "mesure_fin": 31}]
    out = sections_pour_chart({"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4,
                               "keyName": "F major"}, secs, bars=bars,
                              fold_report={})
    assert [s["label"] for s in out] == ["A", "B"]
    a = out[0]
    assert a["reps"] == 3 and a["endings"]["tail"] == 2
    # le dernier tour allume les 6 rangées du tronc, puis la 1re mesure de
    # la fin qu'il joue (F, la 2e fin — pas F C de la 1re)
    t = GRILLE[24]
    rangs = [r for r, spans in enumerate(a["barSpans"]) if any(
        GRILLE[24] <= sp[0] < GRILLE[31] for sp in spans)]
    assert rangs == [0, 1, 2, 3, 4, 5, 8], (t, rangs)


# ── le vrai morceau ───────────────────────────────────────────────────────
def _evening_star():
    from harmonia.settings import SETTINGS
    p = SETTINGS.charts_dir / "min_CHVWPIFAT5A.json"
    if not p.exists():
        return None
    ch = json.loads(p.read_text(encoding="utf-8"))
    st = Path(ch.get("audio_url") or "").stem
    if not (SETTINGS.audio_dir / f"{st}.m4a").exists():
        return None
    return ch


def test_evening_star_de_bout_en_bout():
    ch = _evening_star()
    if ch is None:
        pytest.skip("Evening Star absent de cette machine")
    from harmonia.refold import refold
    from harmonia.sections.explication import combler_trous, reponse_inferer
    from harmonia.sections.simulation import occurrences_de_tous, remplir_par_algo
    from harmonia.settings import SETTINGS
    from harmonia.soudure import scinder_multiples, sections_et_fusions
    ad = SETTINGS.audio_dir
    humain = [{"label": "A", "mesure_debut": 13, "mesure_fin": 20}]
    occ, perdus, diag = occurrences_de_tous(ch, humain, ad)
    qp: list = []
    blocs = remplir_par_algo(ch, occ, ad, propositions=qp)
    rep = reponse_inferer(occ, blocs, qp, perdus, diag)
    secs = [(s["label"], s["mesure_debut"], s["mesure_fin"]) for s in rep["sections"]]
    assert ("B", 29, 36) in secs and ("A", 37, 43) in secs, secs
    n = len(ch["barGrid"]) - 1
    sc = scinder_multiples(combler_trous(rep["sections"], n))
    bars, rap = refold(copy.deepcopy(ch), sc, ad)
    ra = rap["rapport"]["A"]
    assert not ra.get("reason"), ra          # la pile du A n'est plus refusée
    assert {18, 19, 26, 27} <= set(ra["variants"])   # les fins hors pile
    neuves, _ = sections_et_fusions(copy.deepcopy(ch), sc, bars=bars,
                                    fold_report=rap["rapport"])
    a = [s for s in neuves if s["label"] == "A"]
    assert len(a) == 1 and a[0]["reps"] == 3
    assert a[0]["endings"]["tail"] == 2
    assert not any(s["label"].startswith("A′") for s in neuves)
