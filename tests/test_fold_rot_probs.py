"""`folding._rot_probs` : transposer les postérieures APRÈS le modèle.

Louis, 2026-08-19, à l'oreille sur Bora Bora A (section à 3 passages dont un
un demi-ton plus haut) : la loi postérieure bat la loi CQT. La raison est
mécanique — transposer le spectre AVANT le modèle lui donne à entendre un son
qui n'a jamais existé, tourner les postérieures APRÈS est exact. Ces tests
pinent « exact ».
"""
import numpy as np

from harmonia_min import musx
from harmonia_min.folding import _rot_probs


def _flux(T=4):
    """Six flux au format musx : triade(73), basse(13), 7/9(4), 11/13(3)."""
    rng = np.random.default_rng(0)
    return [rng.random((T, k)).astype(np.float32) for k in (73, 13, 4, 4, 3, 3)]


def _col(root, famille=1):
    """La colonne du plan triade pour (fondamentale, famille 1-based)."""
    return 1 + (famille - 1) * 12 + root


def test_la_fondamentale_monte_du_bon_nombre_de_demi_tons():
    p = _flux()
    p[0][:] = 0.0
    p[0][:, _col(0)] = 1.0                      # C majeur partout
    monte = _rot_probs(p, 2)                    # +2 demi-tons → D
    assert monte[0][:, _col(2)].min() == 1.0
    assert monte[0][:, _col(0)].max() == 0.0


def test_chaque_famille_tourne_dans_son_propre_bloc():
    """Un mineur ne doit pas devenir majeur en montant d'un demi-ton."""
    p = _flux()
    p[0][:] = 0.0
    p[0][:, _col(11, famille=2)] = 1.0          # B mineur (famille 2)
    monte = _rot_probs(p, 1)                    # → C mineur, PAS C majeur
    assert monte[0][:, _col(0, famille=2)].min() == 1.0
    assert monte[0][:, _col(0, famille=1)].max() == 0.0


def test_la_colonne_N_ne_bouge_jamais():
    p = _flux()
    p[0][:, 0] = 0.7
    assert np.allclose(_rot_probs(p, 5)[0][:, 0], 0.7)


def test_la_basse_tourne_sans_toucher_sa_colonne_vide():
    p = _flux()
    p[1][:] = 0.0
    p[1][:, 0] = 0.3                            # « pas de basse »
    p[1][:, 1 + 7] = 1.0                        # basse G
    monte = _rot_probs(p, 3)                    # → B♭
    assert monte[1][:, 1 + 10].min() == 1.0
    assert np.allclose(monte[1][:, 0], 0.3)


def test_les_extensions_ne_tournent_pas():
    """7e, 9e, 11e, 13e sont relatives à la fondamentale : rien à tourner."""
    p = _flux()
    monte = _rot_probs(p, 4)
    for i in range(2, 6):
        assert np.array_equal(monte[i], p[i])


def test_monter_puis_redescendre_rend_l_original():
    p = _flux()
    aller = _rot_probs(p, 7)
    retour = _rot_probs(aller, -7)
    for a, b in zip(retour, p):
        assert np.allclose(a, b)


def test_zero_demi_ton_ne_copie_meme_pas():
    p = _flux()
    assert _rot_probs(p, 0) is p


def test_l_etiquette_lue_par_le_decodeur_suit_la_rotation():
    """Le test qui compte : `musx.label_confidence` — la fonction que le
    décodeur et le repli utilisent tous deux — doit voir l'accord monté."""
    p = _flux()
    p[0][:] = 0.0
    p[0][:, _col(5)] = 1.0                      # F majeur
    T = p[0].shape[0] * musx.FRAME_DT
    assert musx.label_confidence(p[0], 0.0, T, "F:maj") == 1.0
    monte = _rot_probs(p, 1)                    # → G♭ majeur
    assert musx.label_confidence(monte[0], 0.0, T, "Gb:maj") == 1.0
    assert musx.label_confidence(monte[0], 0.0, T, "F:maj") == 0.0
