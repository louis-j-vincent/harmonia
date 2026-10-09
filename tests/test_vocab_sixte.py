"""tests/test_vocab_sixte.py — le décodeur peut écrire un 6 (2026-10-09).

Louis entend Eb6/Db6 sur La Jupe En Laine là où le chart écrivait Eb+/Db. Le
vocabulaire `submission_plus6` ajoute maj(6)/min(6) ; ce test épingle que la
chaîne entière les accepte (décodeur → étiquette → champs de l'UI → tables).

CE QUE ÇA NE RÉSOUT PAS : brancher le vocabulaire ne change AUCUN accord
(mesuré : 0 s sur 64 morceaux). Le plan 13e/6te du réseau ne dit « 6te
présente » plus fort que « aucune » que sur 0,04 % des trames — un 6 ne peut
donc presque jamais battre la triade nue. Voir docs/known_issues.md.
"""
from __future__ import annotations

from harmonia import musx
from harmonia.chord_similarity import MUSX_PCS
from harmonia.labels import chord_pcs, to_chord
from harmonia.settings import SETTINGS


def test_les_etiquettes_six_deviennent_les_queues_ireal():
    assert to_chord("Db:maj(6)") == {"root": 1, "q": "6", "bass": -1}
    assert to_chord("A:min(6)") == {"root": 9, "q": "-6", "bass": -1}
    assert set(chord_pcs(3, "6")) == {3, 7, 10, 0}          # Eb G Bb C
    assert set(chord_pcs(9, "-6")) == {9, 0, 4, 6}          # A C E F#


def test_le_decodeur_de_production_porte_les_24_etats_six():
    assert SETTINGS.chord_dict == "submission_plus6"
    with musx._InMusxDir():
        names = [n for _a, n in musx._decoder(musx.DEFAULT_PENALTY).known_chord_array]
    six = [n for n in names if n.endswith("(6)")]
    assert len(six) == 24
    for n in names:
        if n != "N":
            to_chord(n)                                     # aucune qualité inconnue
            q = n.split(":", 1)[1]
            assert q in musx.TRIAD_FAMILY and q in MUSX_PCS, q
