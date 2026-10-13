"""Les régions de rythme (Louis, The APL Song, 2026-10-13).

« L'intro a un tempo différent du reste du morceau. Il faut une brique qui
détecte les différentes portions avec des rythmes différents. »

Mesuré sur APL : le tempo est le même (89 BPM) ; c'est la BARRE DE MESURE qui
saute d'un temps à 38,5 s (l'intro fait 8 mesures + 1 temps). Une phase
unique ne peut écrire juste ni l'intro ni le corps. Chaque portion d'au moins
une phrase dont l'harmonie ET les downbeats du traqueur s'accordent sur une
autre barre reçoit sa propre phase ; la mesure de jonction est courte.
"""
import json

import numpy as np
import pytest

from harmonia.bars import layout_bars
from harmonia.musx import FRAME_DT
from harmonia.regions_rythme import detecter_regions, lignes_de_mesure
from harmonia.rythme import verifier_changements

BPB = 4


def _triad(changes, dur):
    """Postérieures musx factices : un accord différent (jamais N.C.) à chaque
    instant de `changes`, one-hot sur 73 classes comme le vrai réseau."""
    n = int(dur / FRAME_DT) + 1
    tri = np.zeros((n, 73))
    bornes = [0.0] + list(changes) + [dur + 1]
    for k in range(len(bornes) - 1):
        a, z = int(bornes[k] / FRAME_DT), int(bornes[k + 1] / FRAME_DT)
        tri[a:z, 1 + (k * 5) % 24] = 1.0
    return tri


def _segments(beats, change_idx):
    """Les segments du re-décodage : bornes sur les temps, un accord par
    segment (étiquettes musx distinctes à chaque changement)."""
    names = ["C:maj", "G:maj", "A:min", "F:maj"]
    cuts = [0] + list(change_idx) + [len(beats) - 1]
    return [(float(beats[a]), float(beats[z]), names[k % 4])
            for k, (a, z) in enumerate(zip(cuts, cuts[1:])) if z > a]


def _morceau(intro_temps, p_intro, p_corps, n_corps_mesures=40):
    """Une intro de `intro_temps` temps à la période `p_intro`, puis un corps
    à `p_corps`. Un accord par mesure, sur le temps 1 de SA portion ; le
    traqueur marque les mêmes temps 1."""
    beats = [0.5 + i * p_intro for i in range(intro_temps)]
    t = beats[-1] + p_intro
    beats += [t + i * p_corps for i in range(n_corps_mesures * BPB + 1)]
    beats = np.round(beats, 4)
    intro_db = list(range(0, intro_temps, BPB))
    corps_db = list(range(intro_temps, len(beats), BPB))
    db_idx = intro_db + corps_db
    ch_idx = [k for k in db_idx if 0 < k < len(beats) - 1]
    tri = _triad([beats[k] for k in ch_idx], float(beats[-1]) + 1)
    return beats, [float(beats[k]) for k in db_idx], tri, ch_idx


# ── règle #1 : l'hypothèse porteuse ────────────────────────────────────────

def test_hypothese_porteuse_axes_des_temps():
    """musx : une image = 512/22050 s ; une battue est en secondes, comme les
    changements d'accords — sans quoi « le temps le plus proche » n'a pas de
    sens."""
    assert FRAME_DT == pytest.approx(512 / 22050)


def test_temps_reels_dans_la_duree_du_fichier_apl():
    """Règle #1 contre une référence externe : les battues en cache et les
    images musx d'APL tiennent dans la durée réelle du fichier (ffprobe)."""
    from harmonia import cache
    from harmonia.pipeline import duration_seconds
    from harmonia.settings import SETTINGS
    au = SETTINGS.audio_dir / "u3wUZw9S2PM.m4a"
    if not au.exists() or not cache.exists("beats", au) \
            or not cache.exists("musx_probs", au):
        pytest.skip("APL pas en cache sur cette machine")
    dur = duration_seconds(au)
    bj = cache.load_json("beats", au)
    tri = cache.load_npz("musx_probs", au)["triad"]
    assert 0 <= bj["beats"][0] and bj["beats"][-1] <= dur
    # une image musx de trop ou de moins, pas un facteur 2
    assert tri.shape[0] * FRAME_DT == pytest.approx(dur, abs=0.1)


# ── le cas APL, synthétique ────────────────────────────────────────────────

def test_intro_huit_mesures_plus_un_temps_donne_deux_regions():
    """APL : intro de 33 temps (8 mesures + 1), même tempo. Deux portions, la
    frontière sur le temps 33, et la 9e mesure de l'intro ne dure qu'un temps."""
    beats, db, tri, ch = _morceau(33, 0.674, 0.674)
    r = detecter_regions(beats, db, tri, FRAME_DT, BPB)["regions"]
    assert [(x["i0"], x["phase"]) for x in r] == [(0, 0), (33, 1)]
    lay = layout_bars(_segments(beats, ch), beats, db, tri, regions=r)
    longueurs = np.diff(lay.lines)
    assert list(longueurs[:8]) == [4] * 8 and longueurs[8] == 1
    assert set(longueurs[9:]) == {4}
    # chaque accord écrit sur le temps 1 de sa mesure
    assert all(c["beat"] == 0 for bar in lay.bars for c in bar)


def test_sans_regions_le_corps_est_decale_d_un_temps():
    """Le défaut d'avant, gardé comme témoin : une phase unique écrit tout le
    corps (ou toute l'intro) un temps à côté."""
    beats, db, tri, ch = _morceau(33, 0.674, 0.674)
    lay = layout_bars(_segments(beats, ch), beats, db, tri)
    hors_temps_1 = sum(1 for bar in lay.bars for c in bar
                       if c["beat"] != 0 and not c.get("carry"))
    assert hors_temps_1 >= 8


def test_intro_lente_plus_un_temps_puis_corps_rapide():
    """Le cas que Louis décrit : intro LENTE (8 mesures + 1 temps à 0,8 s),
    corps plus rapide (0,6 s). Les mesures suivent déjà le tempo (comptées en
    temps du traqueur) ; la brique ajoute la barre propre au corps."""
    beats, db, tri, ch = _morceau(33, 0.8, 0.6)
    r = detecter_regions(beats, db, tri, FRAME_DT, BPB)["regions"]
    assert [(x["i0"], x["phase"]) for x in r] == [(0, 0), (33, 1)]
    assert r[0]["bpm"] == pytest.approx(75, abs=0.5)
    assert r[1]["bpm"] == pytest.approx(100, abs=0.5)
    lay = layout_bars(_segments(beats, ch), beats, db, tri, regions=r)
    d = np.diff(lay.grid)
    assert np.allclose(d[:8], 3.2, atol=1e-3)       # intro : 4 × 0,8 s
    assert d[8] == pytest.approx(0.8, abs=1e-3)     # la mesure d'un temps
    assert np.allclose(d[9:-1], 2.4, atol=1e-3)     # corps : 4 × 0,6 s
    assert all(c["beat"] == 0 for bar in lay.bars for c in bar)


def test_intro_lente_de_huit_mesures_pile_n_a_pas_besoin_de_regions():
    """Changement de tempo SANS changement de barre : une seule phase suffit,
    et les mesures ont déjà la bonne durée dans les deux portions."""
    beats, db, tri, ch = _morceau(32, 0.8, 0.6)
    assert detecter_regions(beats, db, tri, FRAME_DT, BPB)["regions"] == []
    lay = layout_bars(_segments(beats, ch), beats, db, tri)
    d = np.diff(lay.grid)
    assert np.allclose(d[:8], 3.2, atol=1e-3) and np.allclose(d[8:-1], 2.4, atol=1e-3)


def test_la_marque_mesure_1_ne_rephase_que_sa_portion():
    beats, db, tri, ch = _morceau(33, 0.674, 0.674)
    r = detecter_regions(beats, db, tri, FRAME_DT, BPB)["regions"]
    # marque posée au temps 2 de l'intro : l'intro bouge, pas le corps
    lay = layout_bars(_segments(beats, ch), beats, db, tri,
                      bar1_time=float(beats[1]), regions=r)
    assert lay.lines[0] == 1
    assert [k for k in lay.lines if k >= 33][:3] == [33, 37, 41]


# ── ce qui doit rester refusé / inchangé ───────────────────────────────────

def test_morceau_a_une_seule_phase_inchange():
    beats, db, tri, ch = _morceau(32, 0.674, 0.674)
    assert detecter_regions(beats, db, tri, FRAME_DT, BPB)["regions"] == []
    a = layout_bars(_segments(beats, ch), beats, db, tri)
    b = layout_bars(_segments(beats, ch), beats, db, tri, regions=None)
    assert a == b


def test_le_traqueur_seul_ne_fait_pas_une_region():
    """Le traqueur saute d'un temps, l'harmonie non : pas de frontière."""
    beats, db, tri, ch = _morceau(32, 0.674, 0.674)
    db2 = [t for t in db if t < beats[32]] + [float(beats[k]) for k in range(33, len(beats), BPB)]
    assert detecter_regions(beats, db2, tri, FRAME_DT, BPB)["regions"] == []


def test_l_harmonie_seule_ne_fait_pas_une_region():
    """L'harmonie saute d'un temps, le traqueur garde sa barre : deux témoins
    exigés, pas de frontière."""
    beats, _db, tri, _ch = _morceau(33, 0.674, 0.674)
    db_global = [float(beats[k]) for k in range(0, len(beats), BPB)]
    out = detecter_regions(beats, db_global, tri, FRAME_DT, BPB)
    assert out["regions"] == []
    assert out["rejets"] and out["rejets"][0]["rejet"].startswith("le traqueur")


def test_triolets_de_that_s_life_toujours_recousus_et_sans_region():
    """Le hoquet de 007d51c : le garde-fou le recoud, et la brique n'y voit
    plus rien — elle passe APRÈS lui, sur les battues réparées."""
    P = 0.8
    vrais = [0.5 + i * P for i in range(32 * BPB + 1)]
    z0, z1 = 12 * BPB, 16 * BPB
    trq = vrais[:z0 + 1]
    for q in range(1, (z1 - z0) * 3 - 3):
        trq.append(vrais[z0] + q * (vrais[z1] - vrais[z0]) / ((z1 - z0) * 3 - 3))
    trq += vrais[z1:]
    dbs = [vrais[i] for i in range(0, z0 + 1, BPB)] + \
          [vrais[i] for i in range(z1, len(vrais), BPB)]
    tri = _triad([vrais[i] for i in range(BPB, len(vrais) - 1, BPB)], vrais[-1] + 1)
    nb, ndb, rap = verifier_changements(trq, dbs, tri, FRAME_DT, BPB)
    assert [x["verdict"][:7] for x in rap] == ["recousu"]
    assert detecter_regions(nb, ndb, tri, FRAME_DT, BPB)["regions"] == []


def test_une_portion_plus_courte_qu_une_phrase_est_ignoree():
    """Trois mesures décalées puis retour : moins d'une phrase, pas de région."""
    beats, db, tri, ch = _morceau(32, 0.674, 0.674)
    # décale d'un temps les changements des mesures 20 à 22 seulement
    ch2 = [k + 1 if 80 <= k < 92 else k for k in ch]
    tri2 = _triad([beats[k] for k in ch2], float(beats[-1]) + 1)
    db2 = [float(beats[k + 1]) if 80 <= k < 92 else float(beats[k])
           for k in range(0, len(beats), BPB)]
    assert detecter_regions(beats, db2, tri2, FRAME_DT, BPB)["regions"] == []


def test_lignes_de_mesure_une_portion_egale_l_arithmetique():
    r = [{"i0": 0, "i1": 50, "phase": 3}]
    assert lignes_de_mesure(r, 50, 4) == list(range(3, 50, 4))


# ── la vraie forme d'APL ───────────────────────────────────────────────────

def test_forme_d_apl_sur_ses_vrais_caches():
    """Sur les vrais caches d'APL (jamais de calcul à froid) : trois portions,
    intro sur une barre, corps décalé d'un temps à 38,5 s (la batterie entre),
    retour à 121,4 s."""
    from harmonia import cache
    from harmonia.settings import SETTINGS
    ch = SETTINGS.charts_dir / "min_u3wUZw9S2PM.json"
    au = SETTINGS.audio_dir / "u3wUZw9S2PM.m4a"
    if not ch.exists() or not au.exists() or not cache.exists("musx_probs", au) \
            or not cache.exists("beats", au):
        pytest.skip("APL pas en cache sur cette machine")
    bt = json.loads(ch.read_text())["beatTimes"]
    db = cache.load_json("beats", au)["downbeats"]
    tri = cache.load_npz("musx_probs", au)["triad"]
    r = detecter_regions(bt, db, tri, FRAME_DT, BPB)["regions"]
    assert [x["phase"] for x in r] == [0, 1, 0]
    assert r[1]["t0"] == pytest.approx(38.5, abs=0.1)
    assert r[2]["t0"] == pytest.approx(121.4, abs=0.1)
    assert all(x["bpm"] == pytest.approx(89.0, abs=0.2) for x in r)
