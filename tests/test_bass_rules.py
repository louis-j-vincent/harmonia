"""Les arbitrages de Louis rejoués contre `harmonia.bass_rules`.

Ce fichier n'est pas un test unitaire ordinaire : c'est la vérité terrain du
2026-09-15 (33 arbitrages à l'oreille, `state/human/bass_verdicts.json`)
transformée en garde-fou. Un changement des tables de `bass_rules` qui
contredit une de ces écoutes doit faire rougir ce fichier — c'est tout son
objet. Si une règle doit bouger, il faut de NOUVEAUX arbitrages, pas un
ajustement du test.

    .venv/bin/python -m pytest tests/test_bass_rules.py -v
"""
import json

import pytest

from harmonia.bass_rules import (FLOOR, IMPOSSIBLE, PLAUSIBLE, decide_bass,
                                 is_decoration)
from harmonia.settings import SETTINGS

NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()
PC = {n: i for i, n in enumerate(NAMES)}
VERDICTS = SETTINGS.repo / "state" / "human" / "bass_verdicts.json"


def _cases(round_key, decided):
    d = json.loads(VERDICTS.read_text(encoding="utf-8"))
    return [c for c in d[round_key]["cases"] if c["verdict"] in decided]


# ── les tables elles-mêmes ───────────────────────────────────────────────

def test_la_neuvieme_a_la_basse_est_jouable():
    """F/G, Db/Eb, E/Gb : trois vraies basses à +2, arbitrées au 2e tour.
    Les classer « douteuses » (mon a priori) jetait les trois."""
    assert 2 in PLAUSIBLE


def test_la_b9_a_la_basse_est_impossible():
    """F-7/Gb : le seul rejet juste du groupe C, et c'était le plus CONFIANT
    des quatre (32,4 %). L'intervalle décide, pas la confiance."""
    assert 1 in IMPOSSIBLE


def test_les_intervalles_de_slash_courants_sont_jouables():
    for iv in (3, 4, 5, 7, 10, 11):       # m3, M3, 4te/11, 5te, b7, 7M
        assert iv in PLAUSIBLE, iv


def test_plausible_et_impossible_partitionnent_les_douze():
    assert PLAUSIBLE | IMPOSSIBLE == frozenset(range(12))
    assert not (PLAUSIBLE & IMPOSSIBLE)


# ── la quinte est asymétrique ────────────────────────────────────────────

def test_la_quinte_au_dessus_decore():
    """Ab -> Eb sous un Ab : Eb est la quinte d'Ab, Ab porte."""
    assert is_decoration(PC["Ab"], PC["Eb"], returns=False) is True


def test_la_quarte_au_dessus_porte():
    """Eb -> Ab : Eb est la quinte d'Ab, donc c'est Ab qui porte."""
    assert is_decoration(PC["Eb"], PC["Ab"], returns=False) is False


def test_le_retour_ne_demonte_pas_une_quarte():
    """Comportement conservé, mais son socle a fondu (2026-09-15 au soir) :
    voir `test_les_quartes_avec_retour_sont_deja_expliquees_par_la_regle_3`."""
    assert is_decoration(PC["Bb"], PC["Eb"], returns=True) is False


def test_les_quartes_avec_retour_sont_deja_expliquees_par_la_regle_3():
    """Les 4 arbitrages qui fondaient cette exception ne la fondent plus.

    Trois (c04, c05, c12) ont le candidat ÉGAL à la fondamentale de l'accord
    — la branche 1 de `decide_bass` les explique sans rien savoir du retour.
    Le quatrième (c11) s'est révélé être un changement d'accord manqué.
    Ce test le fige : si un futur cas vient VRAIMENT soutenir l'exception,
    il faudra l'ajouter aux arbitrages, et ce test devra être revu.
    """
    d = json.loads(VERDICTS.read_text(encoding="utf-8"))
    quartes = [c for c in d["round1"]["cases"]
               if c["interval"] == 5 and c["returns"] and c["verdict"] == "bass"]
    assert {c["id"] for c in quartes} == {"c04", "c05", "c11", "c12"}
    for c in quartes:
        if c["id"] == "c11":
            assert c.get("reclassified"), "c11 doit rester marqué comme mal segmenté"
        else:
            assert c["candIsChordRoot"], (
                f"{c['id']} ne soutient l'exception que s'il n'est PAS déjà "
                f"expliqué par la fondamentale de l'accord")


def test_le_retour_demonte_le_reste():
    assert is_decoration(PC["Bb"], PC["Ab"], returns=True) is True


# ── le 1er tour : basse ou note de passage ───────────────────────────────

@pytest.mark.parametrize("case", _cases("round1", {"bass", "pass"}),
                         ids=lambda c: f"{c['id']}-{c['song'][:9]}")
def test_premier_tour(case):
    """« candidat == fondamentale de l'accord, et au-dessus du plancher » :
    16/17 des arbitrages du 1er tour. La seule exception assumée est c11
    (Bb-7/Eb, la 11e à la basse — un Eb sus), que la 2e branche de
    `decide_bass` rattrape."""
    root = PC[case["chord"][:2].rstrip("^-7 ") if case["chord"][:2] in PC
              else case["chord"][0]]
    cand_is_root = case["candIsChordRoot"]
    predicted_bass = cand_is_root and case["meanShare"] >= FLOOR
    truth = case["verdict"] == "bass"
    if case["id"] == "c11":
        pytest.xfail("Ready mes. 49 : ce n'est pas une basse de slash mais un "
                     "changement d'accord manqué (Bb-7 puis Eb6b9, Louis "
                     "2026-09-15) — hors du domaine de cette règle")
    assert predicted_bass == truth, (
        f"{case['id']} {case['chord']} {case['onset']}->{case['cand']} "
        f"part {case['meanShare']}% : la règle dit "
        f"{'basse' if predicted_bass else 'passage'}, Louis dit {case['verdict']}")


# ── le 2e tour : la ligne écrite est-elle juste ──────────────────────────

@pytest.mark.parametrize("case", _cases("round2", {"ok", "ko"}),
                         ids=lambda c: f"{c['id']}-{c['group']}")
def test_second_tour(case):
    """Rejoue la décision sur les lectures enregistrées. Les `ko` sont les
    cas où la version testée ce jour-là se trompait : la règle actuelle doit
    maintenant écrire AUTRE CHOSE que ce qu'elle écrivait alors."""
    root_name = case["oldLabel"].split("/")[0]
    for tail in ("^7", "-7", "7", "-", "^"):
        if root_name.endswith(tail):
            root_name = root_name[: -len(tail)]
            break
    root = PC[root_name]
    onset = PC[case["onsetBass"]]
    ev = PC[case["evBass"]]

    beats = [{"pc": onset, "share": case["onsetShare"]}]
    if ev != onset:
        beats.append({"pc": ev, "share": case["evShare"]})
    got_pc, _why = decide_bass(root, beats)
    wrote_slash = got_pc != root

    expected_slash = "/" in case["newLabel"]
    if case["verdict"] == "ok":
        assert wrote_slash == expected_slash, (
            f"{case['id']} : la règle écrivait {case['newLabel']} et Louis a dit juste, "
            f"elle écrit maintenant {'un slash' if wrote_slash else 'l’accord nu'}")
    else:
        assert wrote_slash != expected_slash, (
            f"{case['id']} {case['oldLabel']} : Louis a dit que "
            f"{case['newLabel']} était FAUX, la règle l'écrit toujours")
