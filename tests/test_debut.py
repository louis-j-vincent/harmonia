"""tests/test_debut.py — la règle du vrai début, et ses garde-fous.

Le résultat mesuré (35/42 contre 28/42 pour le traqueur) vit dans le
docstring de `harmonia.debut` ; il demande l'audio et les postérieures en
cache, donc il ne tourne pas ici. Ce qui est épinglé ci-dessous, ce sont les
lois qui font que la règle ne peut PAS mentir : le plancher du silence, le
calage qui n'écrase pas un désaccord, et l'absence de repli muet.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.debut import (DUREE_TROU, PART_CALAGE, PAS_ENERGIE,
                            cale_sur_grille, dernier_trou,
                            premier_son, premiere_basse)
from harmonia.musx import FRAME_DT


def _basse(*plages, n=400):
    """Une tête de basse musx (n, 13) : `plages` = (trame0, trame1, masse)."""
    p = np.zeros((n, 13), dtype=np.float32)
    p[:, 0] = 1.0                                   # « pas de basse » partout
    for f0, f1, masse in plages:
        p[f0:f1, 0] = 1.0 - masse
        p[f0:f1, 1] = masse
    return p


# ── le plancher du silence ───────────────────────────────────────────────────

def test_le_silence_du_debut_ne_compte_pas_comme_un_son():
    """Hot N Cold : quatre secondes muettes, puis le morceau. Sans ce
    plancher, la tête de basse déclarait une basse à 0,00 s."""
    env = [0.0001] * 16 + [0.5] * 40                # 4 s de rien, puis du son
    assert premier_son(env) == pytest.approx(4.0, abs=PAS_ENERGIE)


def test_un_morceau_qui_demarre_tout_de_suite_commence_a_zero():
    assert premier_son([0.5] * 40) == 0.0


def test_une_enveloppe_vide_ne_fait_pas_tomber_le_calcul():
    assert premier_son([]) == 0.0


def test_la_recherche_de_basse_part_apres_le_plancher():
    """Une basse AVANT le premier son est un faux positif du modèle : on ne
    la voit pas, parce qu'on ne regarde pas avant."""
    p = _basse((0, 40, 0.9), (200, 300, 0.9))
    assert premiere_basse(p, apres=0.0) == pytest.approx(0.0, abs=FRAME_DT)
    tard = premiere_basse(p, apres=100 * FRAME_DT)
    assert tard == pytest.approx(200 * FRAME_DT, abs=FRAME_DT)


# ── la basse ─────────────────────────────────────────────────────────────────

def test_une_trame_isolee_ne_fait_pas_une_note():
    """Le modèle bruite. Une seule trame au-dessus du seuil n'est pas une
    entrée de basse — il en faut `TENUE_BASSE` d'affilée."""
    p = _basse((50, 52, 0.9), (150, 250, 0.9))      # 2 trames, puis une vraie
    assert premiere_basse(p) == pytest.approx(150 * FRAME_DT, abs=FRAME_DT)


def test_pas_de_basse_du_tout_rend_None_plutot_que_zero():
    """Pas de repli muet : un morceau sans basse déclarée doit le DIRE. Rendre
    0,0 ferait passer « je n'ai rien entendu » pour « ça commence au début »."""
    assert premiere_basse(_basse()) is None


# ── la grille tranche, mais n'écrase pas ─────────────────────────────────────

GRILLE = [0.0, 2.0, 4.0, 6.0, 8.0]                  # mesures de 2 s


def test_une_detection_proche_dune_ligne_est_posee_dessus():
    """« En cas de doute c'est lui qui tranche » — le traqueur est plus précis
    qu'un seuil franchi sur une postérieure."""
    t, sur = cale_sur_grille(4.17, GRILLE)
    assert (t, sur) == (4.0, True)


def test_une_detection_loin_de_toute_ligne_reste_brute_et_le_dit():
    """Le cas qu'il ne faut PAS arrondir : à plus d'un quart de mesure, l'un
    des deux se trompe, et l'effacer cacherait lequel."""
    t, sur = cale_sur_grille(5.0, GRILLE)           # pile entre deux lignes
    assert (t, sur) == (5.0, False)


@pytest.mark.parametrize("ecart,attendu", [
    (PART_CALAGE * 2 * 0.9, True),                  # juste dedans
    (PART_CALAGE * 2 * 1.1, False),                 # juste dehors
])
def test_la_frontiere_du_calage_est_un_quart_de_mesure(ecart, attendu):
    _t, sur = cale_sur_grille(4.0 + ecart, GRILLE)
    assert sur is attendu


def test_sans_grille_on_ne_cale_rien():
    assert cale_sur_grille(3.3, []) == (3.3, False)
    assert cale_sur_grille(None, GRILLE) == (None, False)


def test_une_grille_a_deux_lignes_a_quand_meme_une_mesure():
    """`median(diff)` sur deux lignes n'a qu'une valeur — le calcul ne doit pas
    partir en NaN et faire passer un calage pour impossible."""
    t, sur = cale_sur_grille(2.1, [0.0, 2.0])
    assert (t, sur) == (2.0, True)


# ── le trou : ce qui joue avant n'est pas le morceau ─────────────────────────

def test_un_trou_dharmonie_repousse_le_debut():
    """La règle de Louis : « un début de chanson puis plus rien derrière, puis
    toute l'identité musicale change ». Une intro de clip est de la VRAIE
    musique — la détecter n'était pas l'erreur ; la prendre pour le morceau,
    si."""
    p = _basse((0, 200, 0.9),          # l'intro du clip : de la vraie musique
               (330, 700, 0.9),        # puis le morceau
               n=800)                  # entre les deux, 3,0 s de rien
    trou = dernier_trou(p, apres=0.0)
    assert trou is not None
    # la FIN du trou, pas son début : c'est là que la musique reprend, et
    # c'est de là qu'on doit repartir chercher.
    assert trou == pytest.approx(330 * FRAME_DT, abs=2 * FRAME_DT)
    # sans le trou on tombe sur l'intro ; avec, sur le morceau
    assert premiere_basse(p) == pytest.approx(0.0, abs=FRAME_DT)
    assert premiere_basse(p, apres=trou) == pytest.approx(330 * FRAME_DT,
                                                          abs=2 * FRAME_DT)


def test_un_morceau_sans_trou_nest_pas_repousse():
    """Le cas majoritaire — 35 des 40 morceaux mesurés. Une règle qui gagne
    deux cas en cassant les autres ne vaut rien ; celle-ci n'en casse aucun."""
    assert dernier_trou(_basse((0, 700, 0.9), n=800)) is None


def test_une_respiration_courte_nest_pas_un_trou():
    """Louis : « sinon ça peut juste être une pause dans la musique ». Un
    silence d'une seconde est un break, pas une frontière de fichier."""
    court = int(1.0 / FRAME_DT)
    p = _basse((0, 200, 0.9), (200 + court, 700, 0.9), n=800)
    assert dernier_trou(p) is None


def test_cest_le_DERNIER_trou_qui_compte():
    """Un clip peut empiler deux préambules. Seul le dernier sépare l'intro du
    morceau — retenir le premier laisserait le second préambule dedans."""
    k = int(DUREE_TROU / FRAME_DT) + 4
    p = _basse((0, 100, 0.9),
               (100 + k, 200 + k, 0.9),
               (200 + 2 * k, 900, 0.9), n=1000)
    trou = dernier_trou(p)
    assert trou == pytest.approx((200 + 2 * k) * FRAME_DT, abs=2 * FRAME_DT)


def test_le_trou_ne_regarde_pas_avant_le_premier_son():
    """Le silence de tête d'un fichier n'est pas une frontière entre deux
    musiques : il n'y a rien avant lui."""
    p = _basse((300, 900, 0.9), n=1000)
    assert dernier_trou(p, apres=300 * FRAME_DT) is None


# ── les deux autres conditions du trou ───────────────────────────────────────
# Louis, 2026-09-17 : « c'est pas seulement la ligne de basse, c'est aussi pas
# de mélodie, et EN PLUS c'est une incohérence mélodique en pattern entre le
# début de la chanson et le trou. Il faut toutes ces règles-là. »

def _chroma(n, motif_aigu, energie=1.0):
    """Un bothchroma NNLS (n, 24) dont la moitié haute suit `motif_aigu`."""
    c = np.zeros((n, 24), dtype=np.float32)
    c[:, :12] = 0.05
    c[:, 12:] = np.asarray(motif_aigu, dtype=np.float32) * energie
    return c


def _temps(n, pas=0.05):
    return np.arange(n) * pas


NET = [1.0, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
PLAT = [1.0 / 12] * 12


def test_la_melodie_distingue_une_note_tenue_dun_bruit_etale():
    """Une mélodie concentre l'énergie sur peu de notes ; la parole l'étale.
    C'est la netteté, pas le volume : sur Sam Smith l'énergie aiguë vaut autant
    sur les 40 s de dialogue que sur le morceau."""
    from harmonia.debut import melodie
    n = 200
    net = float(melodie(_chroma(n, NET), _temps(n)).mean())
    plat = float(melodie(_chroma(n, PLAT), _temps(n)).mean())
    assert net > plat * 3


def test_un_passage_sans_basse_MAIS_avec_melodie_nest_pas_un_trou():
    """La 2e condition. Un pont où la basse se tait pendant que le chant
    continue n'est pas une frontière de fichier."""
    from harmonia.debut import dernier_trou
    n_ch, pas = 800, 0.05
    p = _basse((0, 200, 0.9), (330, 700, 0.9), n=800)
    assert dernier_trou(p) is not None                 # la basse seule dirait oui
    chroma = _chroma(n_ch, NET)                        # mais la mélodie ne cesse
    assert dernier_trou(p, chroma, _temps(n_ch, pas)) is None


def test_un_trou_sans_melodie_ET_avec_une_autre_matiere_compte():
    """Les trois conditions réunies : la basse se tait, la mélodie aussi, et
    ce qui joue avant ne ressemble pas à ce qui joue après."""
    from harmonia.debut import dernier_trou
    n_ch, pas = 800, 0.05
    p = _basse((0, 200, 0.9), (330, 700, 0.9), n=800)
    chroma = _chroma(n_ch, PLAT)
    avant = [0.0] * 12; avant[3] = 1.0                 # l'intro : une matière
    apres = [0.0] * 12; apres[9] = 1.0                 # le morceau : une autre
    chroma[:int(200 * FRAME_DT / pas), 12:] = avant
    chroma[int(330 * FRAME_DT / pas):, 12:] = apres
    assert dernier_trou(p, chroma, _temps(n_ch, pas)) is not None


def test_une_respiration_entre_deux_fois_la_MEME_matiere_nest_pas_un_trou():
    """La 3e condition, celle que Louis exige — « sinon ça peut juste être une
    pause dans la musique »."""
    from harmonia.debut import dernier_trou
    n_ch, pas = 800, 0.05
    p = _basse((0, 200, 0.9), (330, 700, 0.9), n=800)
    chroma = _chroma(n_ch, PLAT)
    meme = [0.0] * 12; meme[3] = 1.0
    chroma[:int(200 * FRAME_DT / pas), 12:] = meme
    chroma[int(330 * FRAME_DT / pas):, 12:] = meme     # la même, des deux côtés
    assert dernier_trou(p, chroma, _temps(n_ch, pas)) is None


def test_sans_chroma_on_retombe_sur_la_basse_seule_sans_mentir():
    """Repli dégradé assumé : deux des trois conditions manquent, et le module
    l'écrit dans son journal plutôt que de faire comme si de rien n'était."""
    from harmonia.debut import dernier_trou
    p = _basse((0, 200, 0.9), (330, 700, 0.9), n=800)
    assert dernier_trou(p, None, None) is not None
