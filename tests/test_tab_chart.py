"""tests/test_tab_chart.py — le tab comme source de chart et comme deuxième avis.

Louis, 2026-09-18 : « branche-nous ça comme façon alternative de choper des
charts, et tu vas t'en servir pour flagger si on a fait des détections
d'accords douteux ».

Ce qui est épinglé : ce qu'on refuse de marquer, et ce qu'un chart importé
doit garder du chart source. Le reste (aller chercher un tab, poser la grille)
demande le réseau et l'audio.
"""
from __future__ import annotations

import pytest

from harmonia.integrations import tab_chart as TC


def _chart(**kw):
    """Deux mesures : Dm puis Am, sur deux passages chacune."""
    base = {
        "audio_url": "/audio/essai.m4a", "title": "Essai", "bpb": 4,
        "barGrid": [0.0, 2.0, 4.0, 6.0, 8.0],
        "beatTimes": [i * 0.5 for i in range(17)],
        "key": {"tonic": 2, "mode": "minor"}, "keyName": "D minor",
        "sections": [{
            "label": "A", "id": "A", "reps": 2,
            "barRanges": [[0, 1], [2, 3]],
            "spans": [[0.0, 4.0], [4.0, 8.0]],
            "barSpans": [[[0.0, 2.0], [4.0, 6.0]], [[2.0, 4.0], [6.0, 8.0]]],
            "bars": [[{"root": 2, "q": "-", "bass": -1, "beat": 0, "bar": 0}],
                     [{"root": 9, "q": "-", "bass": -1, "beat": 0, "bar": 1}]],
        }],
    }
    base.update(kw)
    return base


def _pose(*accords, **kw):
    """Une pose factice. Chaque accord est `("Bb", t0, t1)`, ou juste `"Bb"`
    pour tenir toute la plage."""
    from harmonia.integrations.tab_align import lire_accord
    c = _chart()
    segs = []
    for i, x in enumerate(accords):
        nom, t0, t1 = x if isinstance(x, tuple) else (x, 0.0, 8.0)
        a = lire_accord(nom)
        segs.append({"i": i, "t0": t0, "t1": t1, "root": a["root"],
                     "q5": a["q5"], "bass": a["bass"], "texte": nom,
                     "section": "Verse"})
    return {"segments": segs, "temps": c["beatTimes"],
            "grille": c["barGrid"], "bpb": 4, **kw}


def test_un_accord_CONFIRME_n_est_jamais_marque():
    """Ce que Louis a validé à l'oreille est la vérité terrain — un tab n'a
    pas voix au chapitre là-dessus."""
    c = _chart()
    c["sections"][0]["bars"][0][0]["confirmed"] = True
    d = TC.doutes(c, _pose("Bb"))
    assert [x for x in d if x["bar"] == 0] == []
    assert [x for x in d if x["bar"] == 1], "l'autre mesure reste jugeable"


def test_un_N_C_n_est_jamais_marque():
    """« pas d'accord ici » n'est pas un accord qu'on peut contredire."""
    c = _chart()
    c["sections"][0]["bars"][0][0]["nc"] = True
    d = TC.doutes(c, _pose("Bb"))
    assert all(x["bar"] != 0 for x in d)


def test_une_fondamentale_differente_est_GRAVE():
    d = TC.doutes(_chart(), _pose("Bb"))
    assert d and all(x["gravite"] == "fondamentale" for x in d)
    assert d[0]["tab"] == "Bb"


def test_une_simple_COULEUR_differente_est_legere():
    """Le tab écrit souvent `D°` là où on écrit `Dø` : c'est de l'orthographe,
    et ça ne vaut pas un point d'exclamation. Ici : notre `D-` contre son
    `D` — même fondamentale, même basse, autre famille."""
    d = TC.doutes(_chart(), _pose("D"))
    m0 = [x for x in d if x["bar"] == 0]
    assert m0 and m0[0]["gravite"] == "couleur"


def test_la_BASSE_compte_comme_la_fondamentale():
    """La cible du projet est la basse qui SONNE (2026-07-16), donc `Dm` et
    `Dm/F` ne sont pas le même accord."""
    d = TC.doutes(_chart(), _pose("Dm/F"))
    m0 = [x for x in d if x["bar"] == 0]
    assert m0 and m0[0]["gravite"] == "fondamentale"


def test_contester_UN_passage_sur_DEUX_ne_suffit_pas():
    """Un désaccord isolé est du bruit d'alignement, pas un désaccord
    d'accord. La mesure 0 est jouée deux fois (0-2 s et 4-6 s) : le tab dit
    comme nous la première fois et autre chose la seconde."""
    pose = _pose(("Dm", 0.0, 4.0), ("Bb", 4.0, 8.0))
    d = [x for x in TC.doutes(_chart(), pose, part_mini=0.75) if x["bar"] == 0]
    assert d == [], "1 passage contesté sur 2, sous le seuil de 0,75"
    d = [x for x in TC.doutes(_chart(), pose, part_mini=0.5) if x["bar"] == 0]
    assert d and d[0]["contestes"] == 1 and d[0]["passages"] == 2


def test_un_passage_que_le_tab_ne_couvre_PAS_ne_compte_pas():
    """On ne juge pas ce qu'on n'a pas entendu : un passage sans accord du tab
    est sauté, ni pour ni contre."""
    pose = _pose(("Bb", 0.0, 2.0))          # seul le 1er passage est couvert
    d = [x for x in TC.doutes(_chart(), pose) if x["bar"] == 0]
    assert d and d[0]["passages"] == 1


def test_un_accord_identique_ne_remonte_pas():
    d = TC.doutes(_chart(), _pose(("Dm", 0.0, 2.0)))
    assert [x for x in d if x["bar"] == 0] == []


def test_le_chart_importe_GARDE_tout_ce_qui_vient_de_l_audio():
    """Grille, temps, tonalité, départ : rien de tout ça ne dépend de qui
    fournit les accords, et le recalculer donnerait deux vérités."""
    src = _chart(bar1=0.25, video_id="abc")
    pose = {**_pose("Bb"), "decalage": 3, "marge": 0.2,
            "tab": {"titre": "x", "artiste": "y", "note": 4.9, "votes": 10,
                    "url": "u"},
            "forme": {"sections": [(0, 4, "A")], "motifs": {"A": []},
                      "cout": (4, 1, 1)}}
    m = TC.chart(src, pose, stem="tab_essai")
    assert m["barGrid"] == src["barGrid"] and m["beatTimes"] == src["beatTimes"]
    assert m["key"] == src["key"] and m["bar1"] == 0.25
    assert m["video_id"] == "abc" and m["audio_url"] == src["audio_url"]
    assert m["file"] == "tab_essai" and m["meta"]["engine"] == "tab"
    assert m["meta"]["transposition"] == 3


def test_le_chart_importe_couvre_TOUTES_les_mesures():
    """Une mesure sans section serait une mesure qu'on ne peut pas lire."""
    src = _chart()
    pose = {**_pose("Bb"), "decalage": 0, "marge": 0.2,
            "tab": {"titre": "x", "artiste": "y", "note": 4.9, "votes": 1,
                    "url": "u"},
            "forme": {"sections": [(0, 2, "A"), (2, 4, "A")],
                      "motifs": {"A": []}, "cout": (2, 1, 2)}}
    m = TC.chart(src, pose)
    vues = {b for s in m["sections"] for a, z in s["barRanges"]
            for b in range(a, z + 1)}
    assert vues == set(range(m["nBars"]))


def test_les_familles_repassent_dans_le_vocabulaire_d_un_chart():
    assert TC.VERS_CHART == {0: "", 1: "-", 2: "7", 3: "h7", 4: "o"}
    assert TC.lire_du_chart({"root": 2, "q": "h7", "bass": -1}) == (2, 3, None)
    assert TC.lire_du_chart({"root": 0, "q": "^7", "bass": 7}) == (0, 0, 7)
    assert TC.lire_du_chart({"root": 0, "q": "", "nc": True}) is None
