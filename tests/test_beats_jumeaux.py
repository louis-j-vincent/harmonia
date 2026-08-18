"""tests/test_beats_jumeaux.py — deux marques pour un seul temps.

Louis, 2026-08-18, sur Another Day de Jamie Lidell : « les dernières sections
devraient toutes être un A et ils décalent, comment ça se fait ? ».

Le morceau tient 136,4 BPM du début à la fin, mais Beat This! pose quinze temps
de trop après 149,5 s, chacun à exactement 0,080 s du précédent — 0,18 fois le
temps médian. Onze des vingt-cinq mesures de la dernière section font alors 20
à 50 % de moins que les autres, dont quatre une DEMI-mesure, et la boucle de
quatre accords glisse d'une mesure toutes les huit.

Ce n'est pas l'erreur d'octave de `drop_inserted_beats` : un temps inséré tombe
à la MOITIÉ du temps, un jumeau à 0,18.
"""
from __future__ import annotations

import numpy as np

from harmonia_min.beats import (drop_duplicate_beats, drop_inserted_beats)

PER = 0.44          # 136,4 BPM, le tempo d'Another Day


def _grille(n=40, per=PER):
    return [round(i * per, 3) for i in range(n)]


def test_un_jumeau_est_retire():
    b = _grille()
    b.insert(20, round(b[19] + 0.08, 3))          # une marque en trop
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b) - 1
    d = np.diff(out)
    assert d.min() > 0.8 * PER, f"la grille reste irrégulière : {d.min():.3f}"


def test_on_garde_celui_qui_tombe_le_mieux():
    """Des deux jumeaux, on garde celui le plus proche de l'attendu — pas
    systématiquement le premier."""
    b = _grille()
    # le VRAI temps est a 20*PER ; le traceur le double 0,08 s AVANT
    b.insert(20, round(20 * PER - 0.08, 3))
    out, _ = drop_duplicate_beats(b, [])
    assert round(20 * PER, 3) in out, "on a gardé le mauvais des deux"


def test_un_demi_temps_nest_pas_un_jumeau():
    """L'erreur d'octave reste le domaine de drop_inserted_beats."""
    b = _grille()
    b.insert(20, round(b[19] + PER / 2, 3))
    out, _ = drop_duplicate_beats(b, [])
    assert len(out) == len(b), "drop_duplicate_beats a mangé un demi-temps"
    out2, _ = drop_inserted_beats(b, [])
    assert len(out2) == len(b) - 1, "drop_inserted_beats aurait dû le voir"


def test_une_grille_saine_nest_pas_touchee():
    b = _grille()
    out, db = drop_duplicate_beats(b, [b[0], b[4]])
    assert out == b and db == [b[0], b[4]]


def test_les_downbeats_suivent():
    b = _grille()
    jum = round(b[19] + 0.08, 3)
    b.insert(20, jum)
    out, db = drop_duplicate_beats(b, [b[0], jum])
    assert jum not in db or jum in out, "un downbeat pointe sur un temps retiré"


# ── la grille rigide (Louis, 2026-08-18) ────────────────────────────────────
#   « regarde la grille rigide des premières mesures, du milieu et des
#     dernières ; si elles ont toutes le même BPM, tu relies tout ensemble et
#     tu me fais une longue grille continue »

from harmonia_min.beats import grille_rigide           # noqa: E402


def test_un_morceau_metronomique_se_rigidifie():
    b = _grille(200)
    b.insert(150, round(b[149] + 0.08, 3))             # un doublon en plus
    r = grille_rigide(b, [b[0]])
    assert r is not None, "un morceau parfaitement régulier doit se rigidifier"
    g = np.asarray(r["beats"])
    assert np.std(np.diff(g)) < 1e-6, "la grille rendue n'est pas rigide"
    assert abs(r["periode"] - PER) < 1e-3


def test_les_doublons_ne_font_pas_refuser():
    """Les doublons se concentrent souvent dans UNE zone et y écrasent la
    période locale : sur Another Day la zone de fin mesurait 0,2933 s au lieu
    de 0,4442, soit 38 % d'écart, et le test refusait un morceau métronomique.
    On nettoie donc avant de mesurer."""
    b = _grille(200)
    # 15 doublons répartis dans le dernier tiers, comme sur Another Day : une
    # marque de trop 80 ms après un vrai temps, jamais deux d'affilée.
    for k in range(150, 195, 3):
        b.append(round(b[k] + 0.08, 3))
    b.sort()
    r = grille_rigide(b, [])
    assert r is not None, "les doublons d'une seule zone ne doivent pas faire refuser"
    assert abs(r["periode"] - PER) < 1e-3


def test_un_morceau_qui_derive_est_refuse():
    b = [0.0]
    per = PER
    for _ in range(200):                               # ralentit continument
        b.append(round(b[-1] + per, 4)); per *= 1.002
    assert grille_rigide(b, []) is None


def test_une_grille_qui_ne_colle_pas_est_refusee():
    """Même BPM aux trois zones, mais la phase a glissé au milieu."""
    b = _grille(80) + [round(x + PER/2, 3) for x in _grille(80, PER)[80:]]
    b = _grille(80)
    b += [round(80*PER + PER/2 + i*PER, 3) for i in range(80)]
    b += [round(160*PER + PER/2 + i*PER, 3) for i in range(80)]
    r = grille_rigide(b, [])
    assert r is None or r["colle_p95"] <= 0.20


# ── le tempo rigide (Louis, 2026-08-18) ──────────────────────────────────────
#   « souvent des intros sans tempo, donc il faut ajuster le tempo par rapport
#     au milieu du morceau... dans tous les cas trouve-moi un algo rigide pour
#     inférer le bpm »

from harmonia_min.beats import bpm_rigide                # noqa: E402


def test_le_tempo_dun_morceau_regulier():
    r = bpm_rigide(_grille(200))
    assert r is not None
    assert abs(r["bpm"] - 60 / PER) < 0.5
    assert r["inliers"] > 0.99


def test_une_intro_sans_tempo_ne_vote_pas():
    """Vingt battues n'importe où au début, puis le morceau. Le tempo doit
    sortir du corps du morceau, pas de l'intro."""
    intro = [round(0.31 * i + (0.07 if i % 3 else 0), 3) for i in range(20)]
    corps = [round(intro[-1] + 1.0 + i * PER, 3) for i in range(200)]
    r = bpm_rigide(intro + corps)
    assert r is not None
    assert abs(r["bpm"] - 60 / PER) < 1.0, f"l'intro a tire le tempo : {r['bpm']}"


def test_un_trou_au_milieu_ne_casse_pas_le_tempo():
    b = _grille(120) + [round(120 * PER + 8.0 + i * PER, 3) for i in range(120)]
    r = bpm_rigide(b)
    assert r is not None
    assert abs(r["bpm"] - 60 / PER) < 0.5


def test_la_phase_ne_derive_pas_sur_la_longueur():
    """Une erreur de 1 % sur la période fait quatre temps d'écart au bout de
    quatre cents battues : c'est ce que le verrouillage de phase corrige."""
    r = bpm_rigide(_grille(400))
    assert r is not None
    assert abs(r["periode"] - PER) < 1e-3, f"période {r['periode']}"
    assert r["ecart_median"] < 0.02


def test_deux_tempos_franchement_differents_sont_refuses():
    a = _grille(120)
    b = a + [round(a[-1] + (i + 1) * PER * 1.5, 3) for i in range(120)]
    r = bpm_rigide(b)
    assert r is None or r["inliers"] < 0.9
