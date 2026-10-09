"""tests/test_queue_optionnelle.py — une queue est un ajout optionnel en fin de section.

Louis, 2026-10-09, sur La Jupe En Laine : « La mesure pleine de Cm est un
accord de fin de A qui se répète souvent mais pas toujours, c'est la
définition d'une queue (un ajout optionnel à la fin d'une section, qui
n'arrive pas toujours) ».

Dans ce morceau B (8 mesures) finit par un turnaround. Trois fois il se
résout sur UNE mesure pleine de Cm avant A ; trois fois il enchaîne
directement sur F-7 (B suivant). Le Cm est donc une queue de B, pas la
première mesure de A : A jouait 9 mesures (Cm + 8) deux fois et 8 la
troisième — l'odeur d'une mesure forcée dans A.
"""
from __future__ import annotations

import pytest

from harmonia import soudure
from harmonia.sections import simulation as sim

NOTES = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6, "G": 7,
         "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def _acc(nom):
    """'F-7' -> (5, '-7'), 'N' -> N.C."""
    if nom == "N":
        return {"root": 0, "q": "", "bass": -1, "nc": True}
    r = nom[:2] if len(nom) > 1 and nom[1] == "b" else nom[:1]
    return {"root": NOTES[r], "q": nom[len(r):], "bass": -1, "nc": False}


def _chart(barres):
    """Un chart minimal : une chaîne d'accords par mesure ('A B' = 2 accords)."""
    n = len(barres)
    grid = [round(i * 2.0, 3) for i in range(n + 1)]
    chords = []
    for b, txt in enumerate(barres):
        accs = txt.split()
        for i, a in enumerate(accs):
            d = dict(_acc(a), c=0.9)
            d["t0"] = grid[b] + 2.0 * i / len(accs)
            d["t1"] = grid[b] + 2.0 * (i + 1) / len(accs)
            chords.append(d)
    return {"title": "bidon", "barGrid": grid, "nBars": n, "bpb": 4,
            "prompter": {"chords": chords}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": []}


def _blocs(*plages, src="algo"):
    return [{"m0": a, "m1": b, "label": lab, "source": src}
            for a, b, lab in plages]


def _plages(blocs):
    return [(b["m0"], b["m1"], b["label"]) for b in blocs]


# --- le cas réel : La Jupe En Laine, les accords mesure par mesure ---------
B8 = ["F-7", "Bb7", "Bb-/Db", "C7", "F-7", "Dh7", "G7", "Eb+ G/D"]
A8 = ["F-7", "Bb7", "Eb", "Ab", "Dh7", "G7", "C-", "C7"]
B8b = ["F-7", "Bb7", "Db", "C7", "F-7", "Dh7", "G7", "Eb G/D"]
JUPE = (["N"] + (A8[:7] + ["C-"]) + (A8) + B8
        + ["C-"] + A8                      # 25-33 : Cm + A
        + B8b + B8b                        # 34-41, 42-49
        + ["C-"] + A8                      # 50-58
        + B8b + B8b + B8b                  # 59-82
        + ["C-"] + A8[:7])                 # 83-90 : Cm + A sans son C7 final
JUPE_BLOCS = [(0, 16, "intro"), (17, 24, "B"), (25, 33, "A"), (34, 41, "B"),
              (42, 49, "B"), (50, 58, "A"), (59, 66, "B"), (67, 74, "B"),
              (75, 82, "B"), (83, 90, "A")]


def test_la_jupe_le_Cm_est_une_queue_de_B():
    assert len(JUPE) == 91
    mes = soudure.accords_par_mesure(_chart(JUPE))
    out = sim.queue_optionnelle(_blocs(*JUPE_BLOCS), mes)
    pl = _plages(out)
    # les trois Cm sortent, seuls, sous un même nom
    assert (25, 25, "queue") in pl and (50, 50, "queue") in pl \
        and (83, 83, "queue") in pl, pl
    # A reprend ses huit mesures F-7 ... C7 (la dernière coupée par la fin)
    assert (26, 33, "A") in pl and (51, 58, "A") in pl and (84, 90, "A") in pl, pl
    # B n'a pas bougé d'une mesure
    assert [p for p in pl if p[2] == "B"] == \
        [p for p in JUPE_BLOCS if p[2] == "B"]
    # le chart est toujours couvert sans trou ni recouvrement
    ordre = sorted(out, key=lambda b: b["m0"])
    assert ordre[0]["m0"] == 0 and ordre[-1]["m1"] == 90
    assert all(a["m1"] + 1 == b["m0"] for a, b in zip(ordre, ordre[1:]))


def test_la_jupe_les_traits_de_louis_sont_souverains():
    """Si Louis a tracé A en 25-33 lui-même, on ne bouge pas sa frontière :
    la queue reste une PROPOSITION."""
    mes = soudure.accords_par_mesure(_chart(JUPE))
    props = []
    out = sim.queue_optionnelle(
        _blocs(*JUPE_BLOCS), mes, souverains={(25, 33), (50, 58), (83, 90)},
        propositions=props)
    assert _plages(out) == sorted(JUPE_BLOCS)
    assert sorted((p["m0"], p["m1"]) for p in props) == [(25, 25), (50, 50), (83, 83)]


def test_la_jupe_de_bout_en_bout_depuis_une_seule_brique(monkeypatch, tmp_path):
    """Louis ne trace que B (18-25) : l'algo remplit A, et la queue sort."""
    chart = _chart(JUPE)
    chart["file"] = "min_jupe"
    # le mot tel que les traits le découpent : une lettre par bloc de la forme
    bornes = [0, 17, 25, 34, 42, 50, 59, 67, 75, 83, 91]
    monkeypatch.setattr(soudure, "mot_sur_traits",
                        lambda c, t, audio_dir=None: (bornes, "abcbbcbbbc", "t"))
    occ = [{"m0": a, "m1": b, "label": "B", "score": 1.0, "rot": 0,
            "trace": a == 17} for a, b in
           [(17, 24), (34, 41), (42, 49), (59, 66), (67, 74), (75, 82)]]
    blocs = sim.remplir_par_algo(chart, occ, tmp_path)
    queues = [(b["m0"], b["m1"]) for b in blocs if str(b["label"]).startswith("queue")]
    assert queues == [(25, 25), (50, 50), (83, 83)], _plages(blocs)


# --- synthétique : la règle seule ------------------------------------------
def _forme(*morceaux):
    out = []
    for m in morceaux:
        out += m
    return out


S = ["C", "F", "G", "C"]                # section de 4 mesures
T = ["Dm"]                              # queue d'une mesure
X = ["Am", "Em", "F", "G", "Am", "Em", "F", "G"]   # une autre section
BLOCS_SYN = [(0, 3, "S"), (4, 4, "Z"), (5, 12, "X")]


def test_synthetique_queue_quand_certaines_occurrences_seulement():
    # S S+T S S+T S : deux S sont suivis de Dm puis de la même suite (S),
    # deux autres vont directement à S.
    barres = _forme(S, S, T, S, S, T, S)
    mes = soudure.accords_par_mesure(_chart(barres))
    blocs = _blocs((0, 3, "S"), (4, 7, "S"), (8, 8, "Z"), (9, 12, "S"),
                   (13, 16, "S"), (17, 17, "Z"), (18, 21, "S"))
    out = sim.queue_optionnelle(blocs, mes)
    pl = _plages(out)
    assert (8, 8, "queue") in pl and (17, 17, "queue") in pl, pl


def test_synthetique_une_queue_toujours_presente_n_est_pas_optionnelle():
    # S+T S+T S+T : le Dm est toujours là, rien ne dit qu'il est optionnel.
    barres = _forme(S, T, S, T, S, T, X)
    mes = soudure.accords_par_mesure(_chart(barres))
    blocs = _blocs((0, 3, "S"), (4, 4, "Z"), (5, 8, "S"), (9, 9, "Z"),
                   (10, 13, "S"), (14, 14, "Z"), (15, 22, "X"))
    out = sim.queue_optionnelle(blocs, mes)
    assert _plages(out) == sorted(_plages(blocs))


def test_synthetique_deux_queues_differentes_ne_se_replient_pas():
    """Deux queues de contenu différent (Dm / Ab-) et de même longueur : deux
    noms, sinon `sections_pour_chart` (qui groupe sur label + longueur) les
    écrirait comme une seule section jouée quatre fois."""
    barres = _forme(S, S, ["Dm"], S, S, ["Dm"], S, S, ["Ab-"], S, S, ["Ab-"], S)
    mes = soudure.accords_par_mesure(_chart(barres))
    blocs, m = [], 0
    for x in barres_blocs(barres):
        blocs.append(x)
    out = sim.queue_optionnelle(_blocs(*blocs), mes)
    par_nom = {}
    for b in out:
        if str(b["label"]).startswith("queue"):
            par_nom.setdefault(b["label"], []).append(barres[b["m0"]])
    assert sorted(map(sorted, par_nom.values())) == [["Ab-", "Ab-"], ["Dm", "Dm"]], par_nom


def barres_blocs(barres):
    """Découpe l'aide : S (4 mesures) ou queue d'une mesure (Dm/Gm -> 'Z')."""
    out, i = [], 0
    while i < len(barres):
        if barres[i] in ("Dm", "Ab-"):
            out.append((i, i, "Z")); i += 1
        else:
            out.append((i, i + 3, "S")); i += 4
    return out
