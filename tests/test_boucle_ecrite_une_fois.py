"""tests/test_boucle_ecrite_une_fois.py — une section qui n'est qu'une petite
boucle répétée s'écrit sur UNE rangée.

Louis, 2026-09-30, sur « Julia » : « Sur le D, c'est une petite boucle de 2
barres qui se répète, pas la peine d'écrire sur 3 lignes si on peut l'écrire
sur une ligne comme quelque chose qui se répète (et tant pis si ça finit pas
pile, il y a toujours des fade in […], c'est une section qui se finit plus
tôt). »

`soudure.sections_pour_chart` (le chemin de « Valider les sections ») écrivait
chaque passage en entier, sans jamais regarder sa boucle interne : D = 15
mesures `Ab^7 | Bb-7` ×7 + un N.C. final → 15 mesures écrites. Désormais : si
TOUTES les mesures de tous les passages d'une lettre suivent une cellule de 1,
2 ou 4 mesures (même accord à la famille près), N.C. de fin et dernier tour
incomplet tolérés, et qu'il y a au moins deux rangées de 4 à économiser, on
écrit une rangée de 4 mesures, et la carte de lecture allume la rangée à
chaque tour — dernier tour incomplet compris.
"""
from __future__ import annotations

from harmonia.soudure import sections_pour_chart

GRILLE = [round(i * 2.66, 3) for i in range(80)]


def _c(root, q=""):
    return {"root": root, "q": q, "nc": False, "c": 0.9, "bass": -1,
            "t0": 0.0, "t1": 1.0}


NC = {"root": 0, "q": "", "nc": True, "c": 0.5, "bass": -1, "t0": 0.0, "t1": 1.0}


def _bars(n, contenu: dict):
    bars = [[_c(0, "-7")] for _ in range(n)]
    for b, bar in contenu.items():
        bars[b] = bar
    return bars


def _sec(out, lab):
    s = [x for x in out if x["label"] == lab]
    assert len(s) == 1, [x["label"] for x in out]
    return s[0]


def test_julia_D_boucle_de_2_ecrite_sur_une_rangee():
    n = 71
    contenu = {}
    for b in range(56, 70):
        contenu[b] = [_c(8, "^7")] if (b - 56) % 2 == 0 else [_c(10, "-7")]
    contenu[70] = [dict(NC)]
    bars = _bars(n, contenu)
    secs = [{"label": "A", "mesure_debut": 1, "mesure_fin": 56},
            {"label": "D", "mesure_debut": 57, "mesure_fin": 71}]
    out = sections_pour_chart({"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4},
                              secs, bars=bars, fold_report={})
    d = _sec(out, "D")
    assert len(d["bars"]) == 4, len(d["bars"])
    assert [c["root"] for bar in d["bars"] for c in bar] == [8, 10, 8, 10]
    assert d["barRanges"] == [[56, 70]]
    # la carte de lecture : chaque mesure jouée allume une rangée, et la
    # rangée r est allumée par les mesures 56+r, 60+r, 64+r, 68+r (dernier
    # tour incomplet compris)
    assert len(d["barSpans"]) == 4
    assert sum(len(r) for r in d["barSpans"]) == 15
    for r, fen in enumerate(d["barSpans"]):
        attendu = [[GRILLE[b], GRILLE[b + 1]] for b in range(56 + r, 71, 4)]
        assert fen == attendu, (r, fen)
    b = d["boucle"]
    assert (b["rangee"], b["cellule"], b["tours"], b["tours_pleins"]) == (4, 2, 3.75, 3)
    assert b["conflations"] == []


def _this_love(n=24, fin=(7, "7")):
    motif = [(0, "-7"), (5, "-7"), (10, ""), (3, "")] + [(0, "-7"), (5, "-7"), (10, ""), fin]
    contenu = {8 + i: [_c(r, q)] for i, (r, q) in enumerate(motif)}
    contenu.update({b: [_c(2, "")] for b in range(8)})
    contenu.update({b: [_c(4, "-7")] if b % 3 else [_c(9, "")] for b in range(16, 24)})
    secs = [{"label": "intro", "mesure_debut": 1, "mesure_fin": 8},
            {"label": "A", "mesure_debut": 9, "mesure_fin": 16},
            {"label": "B", "mesure_debut": 17, "mesure_fin": 24}]
    return secs, _bars(n, contenu), {"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4}


def test_un_vrai_changement_d_harmonie_garde_la_section_entiere():
    """Louis, 2026-09-30, sur This Love : « tu m'as viré des deuxièmes
    parties de sections qui étaient vraiment différentes en les mettant comme
    accords suggérés ». Au défaut, Eb au 1er tour contre G7 au 2e = une autre
    fondamentale : la section s'écrit en entier, sans boucle."""
    secs, bars, chart = _this_love()
    out = sections_pour_chart(chart, secs, bars=bars, fold_report={})
    a = _sec(out, "A")
    assert len(a["bars"]) == 8 and "boucle" not in a
    # un accord tenu huit mesures n'est pas une boucle
    assert len(_sec(out, "intro")["bars"]) == 8


def test_une_simple_couleur_se_factorise_au_defaut():
    """Eb contre Eb7 : même fondamentale, même famille — une couleur. La
    boucle s'écrit (4 mesures ×2)."""
    secs, bars, chart = _this_love(fin=(3, "7"))
    a = _sec(sections_pour_chart(chart, secs, bars=bars, fold_report={}), "A")
    assert len(a["bars"]) == 4 and a["boucle"]["tours_pleins"] == 2


def test_le_curseur_au_maximum_factorise_et_montre_la_difference_en_var():
    secs, bars, chart = _this_love()
    out = sections_pour_chart(chart, secs, bars=bars, fold_report={}, fusion=4)
    a = _sec(out, "A")
    assert [c["root"] for bar in a["bars"] for c in bar] == [0, 5, 10, 3]
    assert a["bars"][3][0]["var"]["root"] == 7
    assert [(x["rang"], x["mesures"]) for x in a["boucle"]["conflations"]] == [(3, [15])]
    assert [len(r) for r in a["barSpans"]] == [2, 2, 2, 2]


def test_le_curseur_a_zero_n_ecrit_jamais_de_boucle():
    secs, bars, chart = _this_love(fin=(3, "7"))
    a = _sec(sections_pour_chart(chart, secs, bars=bars, fold_report={}, fusion=0), "A")
    assert len(a["bars"]) == 8 and "boucle" not in a


def test_une_section_courte_n_est_pas_touchee():
    """Moins de deux rangées de 4 : rien à gagner, on écrit tel quel."""
    n = 12
    contenu = {b: [_c(8, "^7")] if b % 2 == 0 else [_c(10, "-7")] for b in range(6)}
    secs = [{"label": "A", "mesure_debut": 1, "mesure_fin": 6},
            {"label": "B", "mesure_debut": 7, "mesure_fin": 12}]
    out = sections_pour_chart({"barGrid": GRILLE[:n + 1], "nBars": n, "bpb": 4},
                              secs, bars=_bars(n, contenu), fold_report={})
    assert len(_sec(out, "A")["bars"]) == 6
