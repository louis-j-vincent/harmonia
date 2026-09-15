"""tests/test_beats_jumeaux_larges.py — le jumeau qui passe sous le seuil.

Louis, 2026-09-15, sur Ready de PJ Morton : « entre la mesure 8 et la mesure 9
il y a un petit temps de pause qui devrait être détecté, puis après ça reprend
mais vu qu'on ne le prend pas en compte on décale tous les accords d'un quart
de barre ». Il avait raison sur le symptôme et la cause était en amont.

Mesure : Beat This! pose deux marques à **0,180 s** l'une de l'autre (13,900 et
14,080) alors que le temps médian est de 0,640 s. Le rapport vaut 0,281 — et le
seuil de `drop_duplicate_beats` coupe à 0,25. La paire passe donc au travers
d'un cheveu, la liste garde un temps de trop, et **toutes les barres après 14 s
tombent un temps trop tôt** : mesure 9 à 20,900 au lieu de 21,520, mesure 10 à
23,540 au lieu de 24,100. Beat This! plaçait pourtant ses temps forts à 21,520
et 24,100, et Louis a posé les accords exactement là à l'oreille.

Le seuil ne peut pas monter : sur les 199 morceaux en cache, les intervalles
courts ne font PAS les trois paquets nets que promettait la docstring d'origine
— la distribution est continue de 0,10 à 0,40 et le gros paquet des triolets
(0,34-0,36, 130 cas) est trop proche.

Ce qui sépare proprement, c'est ce que la paire COUVRE : du temps d'avant au
temps d'après, la distribution est franchement bimodale — 1,0 période (deux
marques pour un seul temps) ou 2,0 périodes (deux vrais temps). Le critère est
donc « la paire couvre une seule période », sans seuil à régler.

    .venv/bin/python -m pytest tests/test_beats_jumeaux_larges.py -v
"""
from __future__ import annotations

import json

import numpy as np

from harmonia.beats import (DUP_TOL, drop_duplicate_beats,
                            drop_inserted_beats)
from harmonia.settings import SETTINGS

PER = 0.64          # le temps de Ready, 93,75 BPM


def _grille(n=48, per=PER):
    return [round(i * per, 3) for i in range(n)]


def test_le_jumeau_de_ready_est_retire():
    """0,180 s sur un temps de 0,640 : 0,28 fois le temps, au-dessus du seuil.

    La paire couvre une seule période (20 -> 22 = 0,640 s), donc c'est bien
    une marque en trop.
    """
    b = _grille()
    b.insert(21, round(b[20] + 0.18, 3))
    assert 0.18 / PER > DUP_TOL, "sinon le seuil suffirait et ce test ne teste rien"
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b) - 1, f"{len(b) - len(out)} marque(s) retirée(s), il en fallait 1"
    assert 21.0 - 0.18 not in out


def test_un_vrai_temps_court_reste():
    """Deux marques qui couvrent DEUX périodes sont deux vrais temps.

    C'est le cas d'un ralentissement ou d'une subdivision : l'intervalle est
    court, mais retirer une marque ferait un trou.
    """
    b = _grille()
    b[24] = round(b[23] + 0.24, 3)              # 0,38 x le temps, mais…
    b[25] = round(b[23] + 2 * PER, 3)           # …la paire couvre 2 périodes
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b), "on a retiré un temps qui portait vraiment"


def test_une_grille_saine_nest_pas_touchee():
    b = _grille()
    out, _ = drop_duplicate_beats(b, [])
    assert out == b


def test_on_ne_fait_que_supprimer():
    """La doctrine du module : jamais une marque inventée."""
    b = _grille()
    b.insert(21, round(b[20] + 0.18, 3))
    out, _ = drop_duplicate_beats(b, [])
    assert set(out) <= set(b)


def test_les_downbeats_suivent_le_jumeau_retire():
    b = _grille()
    b.insert(21, round(b[20] + 0.18, 3))
    dbs = [b[0], b[21], b[25]]                  # un temps fort SUR le jumeau
    out, odb = drop_duplicate_beats(b, dbs)
    assert set(odb) <= set(out), "un temps fort pointe un temps qui n'existe plus"


def test_ready_na_plus_de_paire_qui_couvre_un_seul_temps():
    """Le morceau réel, sur les temps que Beat This! a réellement produits."""
    cache = SETTINGS.repo / "state/cache/beats/191P7nIeECo__2505600.json"
    if not cache.exists():                      # cache regénérable, pas suivi par git
        import pytest
        pytest.skip("temps de Ready pas en cache")
    d = json.loads(cache.read_text())
    b, db = drop_duplicate_beats(d["beats"], d["downbeats"])
    b, db = drop_inserted_beats(b, db)
    b = np.asarray(b, float)
    dd = np.diff(b)
    med = float(np.median(dd))
    restes = []
    for i in np.where(dd / med < 0.45)[0]:
        if i == 0 or i + 2 >= len(b):
            continue
        loc = float(np.median(np.diff(b[max(0, i - 6):i + 7])))
        if 0.88 < (b[i + 1] - b[i - 1]) / loc < 1.12:
            restes.append(round(float(b[i]), 3))
    assert not restes, f"marques en trop encore présentes : {restes}"


def test_ready_retombe_sur_les_temps_forts_du_traceur():
    """Le juge extérieur : les temps forts de Beat This! lui-même.

    Une fois le jumeau retiré, la grille de mesures reconstruite depuis la
    marque « bar 1 » de Louis doit tomber sur les temps forts que le traceur
    a posés — 18,960 / 21,520 / 24,100 — et non un temps avant.
    """
    cache = SETTINGS.repo / "state/cache/beats/191P7nIeECo__2505600.json"
    if not cache.exists():
        import pytest
        pytest.skip("temps de Ready pas en cache")
    d = json.loads(cache.read_text())
    b, dbs = drop_duplicate_beats(d["beats"], d["downbeats"])
    b, dbs = drop_inserted_beats(b, dbs)
    b = np.asarray(b, float)
    k = int(np.abs(b - 1.355).argmin())          # la marque de Louis
    lignes = b[k % 4::4]
    for attendu in (18.960, 21.520, 24.100):
        assert np.min(np.abs(lignes - attendu)) < 0.01, (
            f"aucune barre de mesure sur le temps fort {attendu}")
