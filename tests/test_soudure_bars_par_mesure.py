"""tests/test_soudure_bars_par_mesure.py — une mesure porte SES accords.

Louis, 2026-08-17, sur Easy On Me : « les barres 25 et 26 sont détectées comme
A- et Bb, mais affichées dans le chart de la modification des sections comme Bb
et F, c'est quoi ce bug ????? »

A- et Bb sont bien ce que le modèle décode à ces instants-là. Bb et F sont les
accords des mesures 43 et 44. La section D du chart repliait TROIS passages de
longueurs différentes — [24,25] (2 mesures), [42,49] (8) et [62,62] (1) — sous
un seul motif de 8 mesures écrit d'après le passage long, et la reconstruction
pavait ce motif modulo sa longueur sur chaque passage :

    par_mesure[b] = bars[(b - b0) % len(bars)]

La mesure 24 recevait donc `bars[0]` (l'accord de la mesure 42) et la 25
`bars[1]` (celui de la 43). Le contenu réel du passage court n'existait plus
dans les sections — mais il est intact dans `prompter.chords`, la liste à plat
du décodage, qui n'a jamais été repliée.

Le chart ci-dessous est le squelette exact de ce cas.
"""
from __future__ import annotations

from harmonia_min.soudure import accords_par_mesure

BPB = 4
GRID = [round(i * 4.0, 3) for i in range(9)]      # 8 mesures de 4 s


def _acc(t0, t1, root, q=""):
    return {"t0": t0, "t1": t1, "root": root, "q": q, "bass": -1, "nc": False,
            "c": 0.8}


# mesures 0..7 : F  D-7  A-  Bb  |  Bb  F  D-7  A-
PLAT = [_acc(0, 4, 5), _acc(4, 8, 2, "-7"), _acc(8, 12, 9, "-"), _acc(12, 16, 10),
        _acc(16, 20, 10), _acc(20, 24, 5), _acc(24, 28, 2, "-7"), _acc(28, 32, 9, "-")]

# La section D replie un passage COURT [2,3] et un passage LONG [4,7] : son
# motif écrit vient du long (Bb F D-7 A-), et le pavage le recopie sur le court.
CHART = {
    "nBars": 8, "bpb": BPB, "barGrid": GRID,
    "prompter": {"chords": PLAT},
    "sections": [
        {"id": "LD", "label": "D", "reps": 2,
         "barRanges": [[2, 3], [4, 7]],
         "bars": [[dict(_acc(16, 20, 10), bar=4, beat=0, sug=[{"root": 10, "q": "", "c": .9}])],
                  [dict(_acc(20, 24, 5), bar=5, beat=0)],
                  [dict(_acc(24, 28, 2, "-7"), bar=6, beat=0)],
                  [dict(_acc(28, 32, 9, "-"), bar=7, beat=0)]],
         "barSpans": [[[16, 20]], [[20, 24]], [[24, 28]], [[28, 32]]]},
        {"id": "LA", "label": "A", "reps": 1, "barRanges": [[0, 1]],
         "bars": [[dict(_acc(0, 4, 5), bar=0, beat=0)],
                  [dict(_acc(4, 8, 2, "-7"), bar=1, beat=0)]],
         "barSpans": [[[0, 4]], [[4, 8]]]},
    ],
}


def _noms(bars):
    return [" ".join(str(c["root"]) + (c["q"] or "") for c in bar) or "-"
            for bar in bars]


def test_le_passage_court_garde_ses_propres_accords():
    bars = accords_par_mesure(CHART)
    assert _noms(bars) == ["5", "2-7", "9-", "10", "10", "5", "2-7", "9-"], (
        "les mesures repliées sous un passage plus long ont reçu les accords "
        f"de l'autre passage : {_noms(bars)}")


def test_la_mesure_2_nest_pas_la_mesure_4():
    """Le cas de Louis, réduit à une ligne : mesure 2 = A-, pas Bb."""
    bars = accords_par_mesure(CHART)
    assert bars[2][0]["root"] == 9 and bars[2][0]["q"] == "-"
    assert bars[3][0]["root"] == 10 and bars[3][0]["q"] == ""


def test_les_candidats_du_modele_suivent_leur_accord():
    """`sug` est recollé par le TEMPS : il reste sur la mesure 4, il ne part
    pas sur la mesure 2 avec le motif pavé."""
    bars = accords_par_mesure(CHART)
    assert "sug" in bars[4][0], "la mesure qui porte vraiment l'accord écrit "\
                                "a perdu ses candidats"
    assert "sug" not in bars[2][0], "les candidats d'une mesure ont été "\
                                    "recopiés sur une autre"


def test_sans_liste_a_plat_on_garde_lancien_pavage():
    """Un vieux chart sans `prompter` ne doit pas sortir vide."""
    sans = {k: v for k, v in CHART.items() if k != "prompter"}
    bars = accords_par_mesure(sans)
    assert all(bars[b] for b in range(8)), f"mesures vides : {_noms(bars)}"
