"""L'invariant iReal : une lettre = un seul bloc écrit.

Spec : docs/spec_affichage_sections.md. Louis, 2026-09-14, devant un chart qui
affichait deux « A » : « tu ne peux pas afficher un A deux fois ».
"""
from __future__ import annotations

from harmonia.folding import _ireal_cascade, _ireal_endings

GRID = [float(i) for i in range(200)]


def _bar(root, q="", nc=False):
    return [{"root": root, "q": q, "nc": nc}]


def _blk(label, bar_roots, ranges, reps=None):
    """Un bloc rendu minimal : ce que `minimal_fold` met dans `sections`."""
    return {
        "id": "L" + label, "label": label, "tag": "",
        "reps": reps if reps is not None else len(ranges),
        "spans": [[float(c0), float(c1 + 1)] for c0, c1 in ranges],
        "barRanges": [list(r) for r in ranges],
        "bars": [_bar(r) for r in bar_roots],
        "barSpans": [[[float(c0 + i), float(c0 + i + 1)] for c0, _ in ranges]
                     for i in range(len(bar_roots))],
    }


def test_deux_blocs_meme_lettre_impossible():
    """Règle 5 : ce que la cascade ne réduit pas prend un prime."""
    a = _blk("A", [0, 5, 7, 2], [(0, 3), (8, 11)])
    a2 = _blk("A", [9, 9, 9], [(20, 22)])          # autre musique
    out = _ireal_cascade([a, a2], None, GRID, {})
    labels = [b["label"] for b in out]
    assert len(labels) == len(set(labels)), labels
    assert labels == ["A", "A′"]


def test_passage_coupe_rejoint_le_bloc():
    """Règle 3 : un passage qui s'arrête en cours de boucle garde la lettre."""
    cell = [0, 5, 7, 2]
    host = _blk("A", cell + cell, [(0, 7), (16, 23)])
    court = _blk("A", cell + cell[:2], [(40, 45)])
    out = _ireal_cascade([host, court], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A"]
    assert out[0]["reps"] == 3 and len(out[0]["barRanges"]) == 3
    # le passage court n'allume QUE ses 6 mesures
    assert [len(rows) for rows in out[0]["barSpans"]] == [3] * 6 + [2, 2]


def test_passage_coupe_qui_ne_suit_pas_la_cellule_prend_un_prime():
    host = _blk("A", [0, 5, 7, 2, 0, 5, 7, 2], [(0, 7), (16, 23)])
    court = _blk("A", [3, 3, 3], [(40, 42)])
    out = _ireal_cascade([host, court], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A", "A′"]


def test_passage_coupe_qui_tient_le_dernier_accord_rejoint_aussi():
    """La phase du raccord n'est plus figée à 0 (Louis, 2026-09-25, San
    Francisco Street : mesure 40 tient le DERNIER accord de la cellule B
    -- 0,97 de ressemblance contre la dernière position, 0,61 seulement
    contre la première). Un passage d'UNE mesure qui prolonge la fin de la
    cellule doit rejoindre le bloc, pas prendre un prime."""
    cell = [0, 5, 7, 2]
    host = _blk("A", cell + cell + cell, [(0, 3), (8, 11), (16, 19)])
    tenu = _blk("A", [2], [(40, 40)])           # tient la DERNIÈRE position (2)
    out = _ireal_cascade([host, tenu], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A"]
    assert out[0]["reps"] == 4 and len(out[0]["barRanges"]) == 4
    # rejoint en phase 3 (la dernière position), pas en phase 0
    assert len(out[0]["barSpans"][3]) == 4 and len(out[0]["barSpans"][0]) == 3


def test_passage_coupe_phase_0_toujours_prioritaire():
    """Une phase 0 valide continue de gagner (compat exacte de l'ancien
    comportement) même quand une autre phase matcherait aussi."""
    cell = [0, 0, 0, 0]                          # toutes les phases matchent
    host = _blk("A", cell + cell, [(0, 3), (8, 11)])
    court = _blk("A", [0], [(40, 40)])
    out = _ireal_cascade([host, court], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A"]
    assert len(out[0]["barSpans"][0]) == 3 and len(out[0]["barSpans"][1]) == 2


def test_passage_coupe_rejoint_un_host_a_fin_alternative():
    """Un host qui porte déjà un 1st/2nd-ending n'est plus écarté d'office
    (Louis, 2026-09-25, San Francisco Street : B a une fin alternative ET
    une mesure orpheline sur le même B -- les deux doivent coexister). Le
    raccord doit tomber dans la rangée de la variante MAJORITAIRE, jamais
    dans celle de la variante alternative ni ailleurs."""
    bars = {}
    for c0 in (0, 16):
        for i, r in enumerate([0, 5, 7, 2]):
            bars[c0 + i] = _bar(r)
    for i, r in enumerate([0, 5, 7, 4]):        # 3e passage: fin alternative
        bars[32 + i] = _bar(r)
    blk = _blk("B", [0, 5, 7, 2], [(0, 3), (16, 19), (32, 35)])
    _ireal_endings(blk, bars, GRID)
    assert blk["endings"]["tail"] == 1

    court = _blk("B", [2], [(50, 50)])          # tient la fin MAJORITAIRE (2)
    out = _ireal_cascade([blk, court], bars, GRID, {"B": {"period": 4}})
    assert [b["label"] for b in out] == ["B"]   # pas de B′
    assert out[0]["reps"] == 4
    assert [50, 50] in out[0]["barRanges"]
    # rejoint la rangée de la variante majoritaire (index 3), pas celle de
    # l'alternative (index 4, inchangée à 1 seule mesure)
    assert len(out[0]["barSpans"][3]) == 3 and len(out[0]["barSpans"][4]) == 1


def test_passage_coupe_matche_par_famille_pas_par_qualite_exacte():
    """Louis, meme echange que ci-dessus, sur la toute fin de San Francisco
    Street (D, sans 7e) contre la cellule ecrite (Dmaj7) : une triade nue et
    sa version avec 7e sont le MEME moment musical (`famille_de`), le bruit
    de decodage sur la 7e ne doit pas empecher le raccord -- exactement le
    cas Fallin' documente par `_barsig_famille` mais jamais branche jusqu'ici."""
    def _b(root, q):
        return [{"root": root, "q": q, "nc": False}]

    host = {
        "id": "LB", "label": "B", "tag": "", "reps": 2,
        "spans": [[0.0, 4.0], [16.0, 20.0]],
        "barRanges": [[0, 3], [16, 19]],
        "bars": [_b(2, "^7"), _b(7, "^7"), _b(4, "-7"), _b(9, "7")],
        "barSpans": [[[float(c0 + i), float(c0 + i + 1)] for c0, _ in
                     [(0, 3), (16, 19)]] for i in range(4)],
    }
    fin = {  # une triade nue, sans 7e -- meme famille que "^7" (majeure)
        "id": "LBx", "label": "B", "tag": "", "reps": 1,
        "spans": [[117.0, 118.0], [118.0, 119.0], [119.0, 120.0]],
        "barRanges": [[117, 119]],
        "bars": [_b(2, ""), _b(7, "^7"), _b(4, "-7")],
        "barSpans": [[[117.0, 118.0]], [[118.0, 119.0]], [[119.0, 120.0]]],
    }
    out = _ireal_cascade([host, fin], None, GRID, {"B": {"period": 4}})
    assert [b["label"] for b in out] == ["B"]
    assert [117, 119] in out[0]["barRanges"]


def test_sans_cellule_connue_le_test_devient_le_prefixe():
    """Repli refusé, ou découpage à la main (chemin Soudure, aucun rapport de
    repli) : on teste « est-ce le début du bloc long ? »."""
    for report in ({"A": {"period": 4, "reason": "stack incoherent (0.61)"}},
                   None, {}):
        host = _blk("A", [0, 5, 7, 2, 0, 5, 7, 2], [(0, 7), (16, 23)])
        court = _blk("A", [0, 5, 7, 2, 0, 5], [(40, 45)])
        out = _ireal_cascade([host, court], None, GRID, report)
        assert [b["label"] for b in out] == ["A"], report
        assert out[0]["reps"] == 3


def test_un_repli_refuse_ne_sert_pas_de_cellule_modulo():
    """La cellule de 4 ferait coller le passage court, mais le bloc long ne
    répète pas vraiment cette cellule (mesure 5 différente) : préfixe, donc
    prime. Un repli refusé ne doit jamais piloter l'affichage."""
    host = _blk("A", [0, 5, 7, 2, 9, 5, 7, 2], [(0, 7), (16, 23)])
    court = _blk("A", [0, 5, 7, 2, 0, 5], [(40, 45)])
    report = {"A": {"period": 4, "reason": "stack incoherent (0.61)"}}
    out = _ireal_cascade([host, court], None, GRID, report)
    assert [b["label"] for b in out] == ["A", "A′"]


def test_fins_1_2_quand_seule_la_queue_differe():
    """Règle 2 : deux passages identiques sauf les 2 dernières mesures."""
    bars = {i: _bar(r) for i, r in enumerate([0, 5, 7, 2, 0, 5, 9, 4])}
    bars.update({16 + i: _bar(r)
                 for i, r in enumerate([0, 5, 7, 2, 0, 5, 11, 3])})
    blk = _blk("B", [0, 5, 7, 2, 0, 5, 9, 4], [(0, 7), (16, 23)])
    _ireal_endings(blk, bars, GRID)
    assert blk["endings"]["tail"] == 2
    assert [v["passes"] for v in blk["endings"]["variants"]] == [[0], [1]]
    # barSpans = 6 mesures de tronc commun, puis 2 + 2 mesures de queue
    assert [len(r) for r in blk["barSpans"]] == [2] * 6 + [1] * 4


def test_pas_de_fins_pour_un_simple_changement_de_couleur():
    """Une vraie 2e fin change d'ACCORD. `C7` contre `Cm7` sur la dernière
    mesure, c'est du bruit de décodage (Sunny Afternoon, 2026-09-14)."""
    bars = {i: _bar(r, q) for i, (r, q) in
            enumerate([(0, ""), (5, ""), (7, ""), (2, "7")])}
    bars.update({16 + i: _bar(r, q) for i, (r, q) in
                 enumerate([(0, ""), (5, ""), (7, ""), (2, "-7")])})
    blk = _blk("B", [0, 5, 7, 2], [(0, 3), (16, 19)])
    _ireal_endings(blk, bars, GRID)
    assert "endings" not in blk


def test_pas_de_fins_sur_un_bloc_deja_replie_en_cellule():
    """Un bloc réduit à sa cellule n'a pas la place de suspendre une queue."""
    bars = {i: _bar(r) for i, r in enumerate([0, 5, 7, 2])}
    bars.update({16 + i: _bar(r) for i, r in enumerate([0, 5, 7, 9])})
    blk = _blk("B", [0, 5, 7, 2], [(0, 7), (16, 23)])     # 4 écrites, 8 jouées
    _ireal_endings(blk, bars, GRID)
    assert "endings" not in blk
