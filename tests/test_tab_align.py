"""tests/test_tab_align.py — aligner un tab Ultimate Guitar sur l'audio.

Louis, 2026-09-18 : « détecte les raw priors donnés par musx, trouve la bonne
transposition, et matche les accords de guitar tabs DANS L'ORDRE en maximisant
la log-vraisemblance. Ça va te demander une notion de distance MUSICALE entre
accords, et un prior sur la grille. »

Les mesures réelles (transposition +3 retrouvée sur This Love, 79 % et 93 %
d'accord avec le top-1 de musx) vivent dans le docstring du module : elles
demandent l'audio, les postérieures en cache et le réseau. Ce qui est épinglé
ici, ce sont les lois — la lecture d'un accord, la distance, et le fait que
l'alignement ne revient jamais en arrière.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.integrations import tab_align as TA


# ── lire un accord de tab ───────────────────────────────────────────────────

@pytest.mark.parametrize("txt,root,q5,bass", [
    ("Am", 9, 1, None), ("Bb", 10, 0, None), ("F#m7b5", 6, 3, None),
    ("E7/G#", 4, 2, 8), ("Bdim7", 11, 4, None), ("Cmaj7", 0, 0, None),
    ("G13", 7, 2, None), ("Ebm9", 3, 1, None), ("A/C#", 9, 0, 1),
])
def test_les_accords_de_tab_se_lisent(txt, root, q5, bass):
    a = TA.lire_accord(txt)
    assert (a["root"], a["q5"], a["bass"]) == (root, q5, bass)


def test_maj7_n_est_pas_mineur():
    """Le piège du suffixe : « maj7 » contient un « m ». Tester les suffixes
    courts d'abord ferait de tout accord majeur septième un mineur."""
    assert TA.lire_accord("Cmaj7")["q5"] == 0
    assert TA.lire_accord("Cm7")["q5"] == 1


@pytest.mark.parametrize("txt", ["N.C.", "x2", "", "  ", "|", "Riff A"])
def test_ce_qui_n_est_pas_un_accord_est_ecarte(txt):
    """Les garder décalerait toute la séquence, donc tout l'alignement."""
    assert TA.lire_accord(txt) is None


def test_sus_et_add_gardent_la_fondamentale():
    """musx n'a que cinq familles : on perd la couleur, jamais l'accord."""
    for t in ("Dsus4", "Csus2", "Fadd9"):
        assert TA.lire_accord(t)["q5"] == 0


# ── la distance MUSICALE ────────────────────────────────────────────────────

def test_la_regle_de_Louis_Bb_est_plus_proche_de_Gm_que_de_F():
    """Sa règle en majuscules (2026-07-30) : « any SSM must be the chord-tone
    one where Bb is closer to Gm than to F »."""
    assert TA.proximite((10, 0), (7, 1)) > TA.proximite((10, 0), (5, 0))


def test_un_accord_est_identique_a_lui_meme_et_etranger_a_son_voisin():
    assert TA.proximite((0, 0), (0, 0)) == 1.0
    assert TA.proximite((0, 0), (1, 0)) == 0.0        # C et C# : aucune note


def test_la_couleur_coute_moins_que_la_fondamentale():
    """C7 contre C partage tout sauf la septième ; C contre G7 ne partage
    presque rien. Un tab qui écrit une couleur ne doit pas être rejeté."""
    assert TA.proximite((0, 2), (0, 0)) > 0.8
    assert TA.proximite((0, 0), (7, 2)) < 0.3


# ── la séquence et son prior ────────────────────────────────────────────────

BRUT = ("[Intro]\n[ch]Am[/ch] [ch]F[/ch]\n[Verse 1]\n"
        "[ch]Am[/ch]\n[ch]Am[/ch]\n[ch]F[/ch]\n[ch]C[/ch] [ch]N.C.[/ch]\n")


def test_la_suite_garde_l_ordre_et_la_section():
    seq = TA.sequence_du_tab(BRUT)
    assert [TA.nom_q5(a["root"], a["q5"]) for a in seq] == \
        ["Am", "F", "Am", "Am", "F", "C"]
    assert seq[0]["section"] == "Intro" and seq[2]["section"] == "Verse 1"


def test_les_repetitions_immediates_fondent_en_une_case():
    """Un tab écrit l'accord au-dessus de CHAQUE ligne qu'il couvre : trois
    `Am` de suite sont un Am tenu, pas trois changements."""
    seq = TA.compresser(TA.sequence_du_tab(BRUT))
    assert [TA.nom_q5(a["root"], a["q5"]) for a in seq] == ["Am", "F", "Am", "F", "C"]
    assert seq[2]["repetitions"] == 2


def test_le_prior_de_grille_dit_la_frequence_de_chaque_accord():
    pr = TA.prior_de_grille(TA.compresser(TA.sequence_du_tab(BRUT)))
    assert pr["part"][(9, 1)] == pytest.approx(3 / 6)     # Am, 3 apparitions sur 6
    assert pr["n_cases"] == 5


# ── l'alignement ────────────────────────────────────────────────────────────

def _priors(suite):
    """Une postérieure parfaite : toute la masse sur l'accord voulu."""
    P = np.full((len(suite), 60), 1e-6)
    for b, (root, q5) in enumerate(suite):
        P[b, root * 5 + q5] = 1.0
    return P


def test_l_alignement_retrouve_une_grille_parfaite():
    """Autant de cases que de mesures, une postérieure sans ambiguïté : chaque
    mesure doit tomber sur sa case."""
    grille = [(0, 0), (9, 1), (5, 0), (7, 0)]
    seq = [{"root": r, "q5": q, "repetitions": 1} for r, q in grille]
    assert TA.aligner(seq, _priors(grille)) == [0, 1, 2, 3]


def test_un_tab_TROP_COURT_et_un_audio_muet_font_du_surplace():
    """Le cas dégénéré, épinglé pour qu'il ne surprenne personne.

    Quatre cases pour douze mesures : le prior dit qu'une mesure consomme 0,33
    case, donc avancer coûte 1,2 nat. Si l'audio n'a AUCUN contraste, rien ne
    paie ce coût et le chemin reste sur la première case. Ce n'est pas un bug —
    c'est le maximum a posteriori du modèle — mais c'est la raison pour
    laquelle la page de contrôle affiche « cases utilisées ». Sur les deux
    morceaux du POC r vaut 1,44 et 1,05 et le chemin atteint la dernière case.
    """
    seq = [{"root": r, "q5": q, "repetitions": 1}
           for r, q in ((0, 0), (9, 1), (5, 0), (7, 0))]
    assert TA.aligner(seq, np.full((12, 60), 1.0 / 60)) == [0] * 12


def test_l_audio_paie_le_prior_quand_il_a_quelque_chose_a_dire():
    """Le pendant du précédent : même grille trop courte, mais une postérieure
    qui désigne les accords. La vraisemblance doit l'emporter sur le prior."""
    grille = [(0, 0), (9, 1), (5, 0), (7, 0)]
    seq = [{"root": r, "q5": q, "repetitions": 1} for r, q in grille]
    chemin = TA.aligner(seq, _priors([g for g in grille for _ in range(3)]))
    assert chemin == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_le_prior_punit_la_COURSE_autant_que_le_surplace():
    """Ce que `|k - r|` ne faisait pas : il classait un double saut MOINS cher
    qu'un surplace. Le Poisson pique en k ≈ r."""
    import math
    from harmonia.integrations.tab_align import RAIDEUR as R
    c = lambda k, r: R * (k * math.log(r) - r - math.lgamma(k + 1))
    for r in (1.05, 1.44):
        assert c(1, r) > c(0, r) and c(1, r) > c(2, r) > c(3, r)


def test_l_alignement_ne_revient_JAMAIS_en_arriere():
    """Un tab se lit dans l'ordre. Une case qui reculerait voudrait dire que
    le morceau rejoue le tab à l'envers."""
    seq = [{"root": r, "q5": 0, "repetitions": 1} for r in (0, 5, 7, 2, 9)]
    rng = np.random.default_rng(0)
    P = rng.random((40, 60))
    P /= P.sum(1, keepdims=True)
    chemin = TA.aligner(seq, P)
    assert all(b <= a for a, b in zip(chemin, chemin[1:])) or \
        all(a <= b for a, b in zip(chemin, chemin[1:])), "monotone"
    assert all(a <= b for a, b in zip(chemin, chemin[1:]))


def test_une_case_peut_tenir_plusieurs_mesures():
    """Un accord tenu quatre mesures ne doit pas forcer l'alignement à
    avancer."""
    seq = [{"root": 0, "q5": 0, "repetitions": 4},
           {"root": 7, "q5": 0, "repetitions": 1}]
    P = _priors([(0, 0)] * 4 + [(7, 0)])
    assert TA.aligner(seq, P) == [0, 0, 0, 0, 1]


def test_la_transposition_se_retrouve():
    """Le cas de This Love : le tab est écrit trois demi-tons plus bas que ce
    que le disque joue."""
    seq = [{"root": r, "q5": q, "repetitions": 1}
           for r, q in ((9, 1), (2, 1), (5, 0), (0, 0))]
    # l'audio joue la même grille transposée de +3
    P = _priors([((r + 3) % 12, q) for r, q in
                 ((9, 1), (2, 1), (5, 0), (0, 0))] * 4)
    dec, scores = TA.meilleure_transposition(seq, P)
    assert dec == 3
    assert scores[0][0] > scores[1][0]


def test_une_sequence_ou_un_audio_vide_ne_casse_pas():
    assert TA.aligner([], np.zeros((4, 60))) == []
    assert TA.aligner([{"root": 0, "q5": 0, "repetitions": 1}],
                      np.zeros((0, 60))) == []


# ── tout le tab posé sur tout le morceau ────────────────────────────────────
#
# Louis, 2026-09-18, en corrigeant le modèle : « le tab n'a aucune mesure ?? on
# est d'accord on matche bien toute la longueur des accords consécutifs à toute
# la longueur du chart, et ensuite la seule question qui permet de maximiser la
# log-proba totale c'est comment je pose mes accords du tab à l'intérieur, sans
# jamais en changer l'ordre (un accord toujours après un autre) ».
#
# Mesuré sur la même grille de temps, contre le top-1 de musx :
#     This Love  61,1 % (76/115 accords posés)  →  80,7 % (115/115)
#     Grenade    81,7 % (73/105)                →  96,7 % (105/105)

def _suite(*accords):
    return [{"root": r, "q5": q, "repetitions": n}
            for r, q, n in accords]


def test_les_DEUX_bouts_sont_ancres():
    """L'ancien alignement ancrait le début et laissait la fin flotter : rien
    n'obligeait le dernier accord du tab à tomber à la fin du morceau."""
    seq = _suite((0, 0, 1), (9, 1, 1), (5, 0, 1), (7, 0, 1))
    ch = TA.poser_tout(seq, np.full((40, 60), 1.0 / 60))
    assert ch[0] == 0, "le premier accord ouvre le morceau"
    assert ch[-1] == len(seq) - 1, "le dernier accord le ferme"


def test_AUCUN_accord_n_est_jete():
    """Le défaut que Louis a vu dans la ligne « cases utilisées » : 39 accords
    sur 115 disparaissaient, parce qu'une mesure ne peut en porter qu'un."""
    seq = _suite(*[(r % 12, 0, 1) for r in range(0, 24, 2)])
    ch = TA.poser_tout(seq, np.full((60, 60), 1.0 / 60))
    assert sorted(set(ch)) == list(range(len(seq)))


def test_l_ordre_n_est_JAMAIS_change():
    """« un accord toujours après un autre »."""
    seq = _suite((0, 0, 1), (5, 0, 3), (7, 0, 1), (2, 1, 2))
    rng = np.random.default_rng(3)
    P = rng.random((50, 60))
    P /= P.sum(1, keepdims=True)
    ch = TA.poser_tout(seq, P)
    assert all(a <= b for a, b in zip(ch, ch[1:]))


def test_un_tab_plus_long_que_le_morceau_le_DIT():
    """Infaisable : plus d'accords que de temps. On rend None plutôt que de
    rogner en silence — un repli silencieux est un bug qui se découvre trois
    semaines plus tard."""
    seq = _suite(*[(r % 12, 0, 1) for r in range(20)])
    assert TA.poser_tout(seq, np.full((10, 60), 1.0 / 60)) is None


def test_le_prior_de_duree_vient_de_CE_QUE_LE_TAB_ECRIT():
    """Un accord écrit au-dessus de trois lignes de paroles attend trois fois
    plus de temps qu'un accord écrit au-dessus d'une seule. Sans aucune aide
    de l'audio, le partage doit suivre ces poids."""
    seq = _suite((0, 0, 3), (7, 0, 1))
    ch = TA.poser_tout(seq, np.full((40, 60), 1.0 / 60))
    tenu = ch.count(0)
    assert 26 <= tenu <= 34, f"attendu ~30 temps sur 40, obtenu {tenu}"


def test_l_audio_l_emporte_sur_le_prior_de_duree():
    """Le prior propose, l'audio dispose : ici le tab annonce un premier
    accord trois fois plus long, mais l'audio dit le contraire."""
    seq = _suite((0, 0, 3), (7, 0, 1))
    P = _priors([(0, 0)] * 8 + [(7, 0)] * 32)
    ch = TA.poser_tout(seq, P)
    assert ch.count(0) == 8


def test_deux_accords_peuvent_partager_une_mesure():
    """C'est tout l'intérêt de travailler au TEMPS et non à la mesure : un
    vrai chart porte deux accords dans une mesure, et le tab en écrit plus
    que le morceau n'a de mesures."""
    seq = _suite((0, 0, 1), (5, 0, 1), (7, 0, 1), (2, 0, 1))
    P = _priors([(0, 0), (5, 0), (7, 0), (2, 0)])      # 4 temps = 1 mesure
    assert TA.poser_tout(seq, P) == [0, 1, 2, 3]


def test_l_origine_des_mesures_n_est_pas_toujours_le_temps_zero():
    """This Love a une levée d'un temps : ses premiers temps de mesure sont
    les indices 1, 5, 9… Compter la phase depuis zéro y donnait 5 % de
    changements « sur le temps fort » contre 63 % sur Grenade, et la
    conclusion — « ce morceau ne tombe pas sur la grille » — était fausse.
    """
    sans = TA._cout_depart(8, 4, 1.0, origine=0)
    avec = TA._cout_depart(8, 4, 1.0, origine=1)
    assert [i for i, c in enumerate(sans) if c == 0.0] == [0, 4, 8]
    assert [i for i, c in enumerate(avec) if c == 0.0] == [1, 5]


def test_le_milieu_de_mesure_coute_moins_qu_un_temps_faible():
    """Une demi-mesure est une place normale pour un changement d'accord ;
    le deuxième ou le quatrième temps, beaucoup moins."""
    c = TA._cout_depart(8, 4, 1.0)
    assert c[0] == 0.0 > c[2] > c[1] and c[1] == c[3]


def test_sans_chiffrage_le_prior_de_temps_fort_ne_fait_rien():
    """On ne devine pas une métrique qu'on n'a pas : `bpb=0` désactive."""
    assert not TA._cout_depart(8, 0, 1.0).any()


def test_le_prior_de_temps_fort_tire_un_changement_sur_la_barre():
    """Un accord dont l'audio hésite entre commencer au temps 3 ou au temps 4
    doit commencer au temps 4 — c'est le premier temps de la mesure."""
    seq = _suite((0, 0, 1), (7, 0, 1))
    P = _priors([(0, 0)] * 3 + [(0, 0)] + [(7, 0)] * 4)   # le 4e temps est ambigu
    P[3] = 0.5 * P[3] + 0.5 * _priors([(7, 0)])[0]
    assert TA.poser_tout(seq, P, bpb=4, temps_fort=2.0).index(1) == 4
