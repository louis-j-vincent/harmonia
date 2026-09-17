"""tests/test_traits_humains.py — un trait de Louis sort là où il l'a tracé.

Louis, 2026-09-17 : « relaxe les règles qu'on a mises sur les sections qui
doivent commencer sur un début de 4 barres ou autres quand on est en
annotation. Du moment qu'un humain annote une section, il n'y a pas à le
corriger, c'est LA vérité terrain, et c'est lui qui définit où commence la
chanson. »

Ce que faisait le code AVANT (mesuré sur ses 18 découpages annotés) : la
grille de jetons — des bi-mesures posées de deux en deux depuis la mesure 1 —
était calculée SANS lui, puis ses traits y étaient arrondis. **52 débuts de
section sur 191 reculaient d'une mesure, et 27 traits sur 191 étaient jetés
en silence** parce que leur jeton de départ était déjà pris par le trait
précédent. Sur Chain of Fools, un `intro` tracé mesure 1 ressortait
mesures 1-2, un `A` tracé 10-17 ressortait 9-18, et 5 traits sur 10
disparaissaient. Le symptôme connu — « un découpage parfait noté 0 %, décalé
d'un cran » — n'était pas une erreur de la machine : c'était la
quantification de ses propres traits.

La règle posée ici : **ses traits sont les bornes de la grille**. Les jetons
sont des bi-mesures À L'INTÉRIEUR de chaque région (un trait, ou un trou
entre deux traits), jamais à cheval sur une frontière qu'il a tracée.

Ce que ça ne fixe PAS : la mesure 1 mal détectée (entrée Sam Smith de
`docs/known_issues.md`) est un problème distinct et réel — elle décale la
grille de MESURES elle-même, donc l'audio sous les traits. Ici on garantit
seulement qu'un trait ressort aux mesures où il a été tracé.
"""
from __future__ import annotations

import pytest

from harmonia.soudure import jetons_sur_traits, traits_propres


# ── la grille se plie aux traits ─────────────────────────────────────────────

def test_chaque_frontiere_tracee_est_une_borne_de_jeton():
    """Le cas de Chain of Fools : des traits de 8 mesures à partir de la 10e.

    Sur la grille de deux-en-deux depuis 0, la mesure 9 (index) tombe AU
    MILIEU d'un jeton, et le trait reculait sur 8. Ici elle est une borne.
    """
    bornes = jetons_sur_traits(80, [(0, 0), (1, 8), (9, 16), (17, 24)])
    for arete in (0, 1, 9, 17, 25):
        assert arete in bornes, f"la mesure {arete} doit ouvrir un jeton"
    assert bornes[0] == 0 and bornes[-1] == 80


def test_un_trait_dune_seule_mesure_fait_son_propre_jeton():
    """L'intro d'une mesure de Chain of Fools : elle ne doit pas être avalée
    par la bi-mesure qui suit, ni allongée à deux mesures."""
    bornes = jetons_sur_traits(80, [(0, 0)])
    assert bornes[0] == 0 and bornes[1] == 1


def test_une_region_impaire_garde_sa_mesure_orpheline():
    """Un trait de 7 mesures : trois jetons, dont un de 3 — jamais 8 mesures,
    jamais 6 avec une mesure perdue."""
    bornes = jetons_sur_traits(40, [(2, 8)])
    dedans = [b for b in bornes if 2 <= b <= 9]
    assert dedans == [2, 4, 6, 9]          # 2+2+3 = 7 mesures, rien de perdu


def test_les_jetons_pavent_le_morceau_sans_trou_ni_recouvrement():
    for traits in ([], [(0, 0)], [(3, 10), (11, 18)], [(0, 3), (4, 4), (5, 39)]):
        bornes = jetons_sur_traits(40, traits)
        assert bornes == sorted(set(bornes)), "bornes en double ou désordonnées"
        assert bornes[0] == 0 and bornes[-1] == 40
        assert all(b < c for b, c in zip(bornes, bornes[1:]))


def test_sans_trait_la_grille_reste_des_bi_mesures():
    """Pas de trait, pas de changement : le morceau garde son grain."""
    assert jetons_sur_traits(10, []) == [0, 2, 4, 6, 8, 10]


# ── aucun trait n'est jeté en silence ────────────────────────────────────────

def test_deux_traits_voisins_survivent_tous_les_deux():
    """LE bug. Deux sections de 8 mesures collées : sous l'ancienne règle, la
    seconde tombait dans le jeton de la première et disparaissait."""
    traits = [{"label": "A", "mesure_debut": 2, "mesure_fin": 9},
              {"label": "B", "mesure_debut": 10, "mesure_fin": 17}]
    gardes, perdus = traits_propres(traits, 80)
    assert perdus == []
    assert [(b0, b1, lab) for b0, b1, lab in gardes] == [(1, 8, "A"), (9, 16, "B")]


def test_deux_traits_qui_se_chevauchent_vraiment_le_disent():
    """Un vrai recouvrement de mesures est le seul cas où un trait cède — et
    il est RENDU, pas avalé : l'appelant doit pouvoir le dire à l'écran."""
    traits = [{"label": "A", "mesure_debut": 1, "mesure_fin": 8},
              {"label": "B", "mesure_debut": 5, "mesure_fin": 12}]
    gardes, perdus = traits_propres(traits, 80)
    assert [g[2] for g in gardes] == ["A"]
    assert len(perdus) == 1 and perdus[0]["label"] == "B"


@pytest.mark.parametrize("trait", [
    {"label": "A", "mesure_debut": 0, "mesure_fin": 4},      # avant le morceau
    {"label": "A", "mesure_debut": 3, "mesure_fin": 2},      # à l'envers
    {"label": "A", "mesure_debut": "x", "mesure_fin": 4},    # illisible
    {"label": "A", "mesure_debut": 200, "mesure_fin": 210},  # après la fin
])
def test_un_trait_impossible_est_signale_pas_avale(trait):
    gardes, perdus = traits_propres([trait], 80)
    assert gardes == [] and len(perdus) == 1


def test_un_trait_qui_deborde_la_fin_est_rogne_pas_jete():
    """Le morceau s'arrête : on garde ce qu'il a tracé DEDANS plutôt que de
    perdre la section entière."""
    gardes, _ = traits_propres(
        [{"label": "outro", "mesure_debut": 78, "mesure_fin": 90}], 80)
    assert gardes == [(77, 79, "outro")]
