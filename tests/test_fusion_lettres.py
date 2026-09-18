"""Fusionner deux lettres : ce qu'on a DÉCODÉ, pas ce qu'on a entendu.

Les six morceaux analysés le 2026-09-18 (Fallin', Stand By Me, Sunny, Every
Breath You Take, Yesterday, Let It Be) et vérifiés à l'oreille par Louis
fixent le contrat. Les cas ci-dessous en sont la réduction : chacun reproduit
la forme d'un des six, en accords nus.

La fenêtre mesurée sur les six est (0.50, 0.75] pour la concordance ; le seuil
de production est 0.65, donc les tests d'ici doivent tenir à ±0.10 sans devenir
faux — sinon c'est que le réglage est plus fragile qu'annoncé.
"""
from __future__ import annotations

import pytest

from harmonia.folding import (MERGE_LETTERS_ACCORDS, MERGE_LETTERS_PREFIXE,
                              _sig_repli, concordance, famille_de,
                              merge_similar_letters, tete_commune)

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11


def acc(root, q=""):
    return {"root": root, "q": q, "nc": False}


def mesure(*accords):
    return [acc(*x) if isinstance(x, tuple) else acc(x) for x in accords]


MUETTE = [{"root": 0, "q": "", "nc": True}]


def monte(*blocs):
    """Construit (bars, sections) : chaque bloc est (label, [mesures])."""
    bars, sections, i = [], [], 0
    for label, mes in blocs:
        b0 = i
        bars.extend(mes)
        i += len(mes)
        vu = next((s for s in sections if s["label"] == label), None)
        if vu:
            vu["barRanges"].append([b0, i - 1])
        else:
            sections.append({"label": label, "barRanges": [[b0, i - 1]]})
    return bars, sections


def labels(sections):
    return [s["label"] for s in sections]


# ── la brique de comparaison ─────────────────────────────────────────────
def test_la_famille_confond_la_septieme_mais_pas_le_mode():
    assert famille_de("-") == famille_de("-7") == "m"
    assert famille_de("") == famille_de("^7") == famille_de("7") == "M"
    assert famille_de("-") != famille_de("")
    assert famille_de("o7") == famille_de("h7") == "o"


def test_une_mesure_muette_est_un_joker_mais_ne_prouve_rien():
    """Un joker s'accorde avec tout — sinon le couplet nu de Stand By Me ne
    ressemble plus à sa reprise. Mais une comparaison faite QUE de jokers ne
    prouve rien : sur le Syracuse d'iReal, une section de 2 mesures tombait
    en face de mesures tenues d'une section sans rapport et sortait à 1.00."""
    assert _sig_repli(MUETTE) is None
    X, Y = ((0, "M"),), ((7, "M"),)
    # une comparaison faite QUE de jokers ne conclut rien
    assert concordance([None, None], [X, Y]) == 0.0
    # une majorité de vrais accords conclut, les jokers ne pénalisent pas :
    # c'est exactement le couplet nu de Stand By Me contre sa reprise pleine
    plein = [X, Y, X, Y, X, Y]
    troue = [None, Y, None, Y, X, Y]
    assert concordance(troue, plein) == 1.0


def test_une_rotation_n_est_permise_que_dans_la_periode_de_la_boucle():
    """Autumn Leaves d'iReal : la section B est la section A décalée de 4
    mesures. Une boucle de 2 mesures a le droit de commencer sur l'une ou
    l'autre ; un passage de 8 mesures n'a pas le droit de commencer au
    milieu de lui-même."""
    from harmonia.folding import periode_interne
    boucle = [(("e",),), (("b",),), (("e",),), (("b",),)]
    assert periode_interne(boucle) == 2
    huit = [((i, "M"),) for i in range(8)]
    assert periode_interne(huit) == 8
    assert concordance(huit, huit[4:] + huit[:4]) < 0.5


def test_une_boucle_coupee_ailleurs_reste_la_meme_boucle():
    """C'est le cas de Fallin' : la boucle `E-|B-` y est découpée en 2, 4, 6,
    8 et 9 mesures selon la lettre. La rotation n'est permise que lorsque les
    blocs prouvent EUX-MÊMES qu'il s'agit d'une boucle — ici le bloc de 4
    répète sa cellule de 2."""
    cell = [(("em",),), (("bm",),)]
    quatre = cell * 2
    assert concordance(quatre, cell[1:] + cell[:1]) == 1.0   # décalée d'une mesure
    assert concordance(cell, cell * 3) == 1.0


def test_deux_blocs_de_deux_mesures_ne_se_tournent_pas_l_un_dans_l_autre():
    """Rien ne prouve que `E-|B-` et `B-|E-` sont la même boucle coupée
    ailleurs plutôt que deux cellules différentes. On refuse — under-fold,
    never over-fold."""
    cell = [(("em",),), (("bm",),)]
    assert concordance(cell, cell[1:] + cell[:1]) == 0.0


# ── les six, réduits ─────────────────────────────────────────────────────
def test_fallin_une_seule_boucle_devient_une_seule_lettre():
    """Huit lettres pour `E- | B-` répété, à cinq longueurs différentes, et
    avec la septième qui apparaît par endroits."""
    bars, sections = monte(
        ("A", [mesure((E, "-")), mesure((B, "-")), mesure((E, "-")), mesure((B, "-7"))]),
        ("B", [mesure((E, "-")), mesure((B, "-"))]),
        ("C", [mesure((E, "-")), mesure((B, "-7")), mesure((E, "-")),
               mesure((B, "-7")), mesure((E, "-")), mesure((B, "-"))]),
    )
    merge_similar_letters(sections, bars)
    assert set(labels(sections)) == {"A"}, labels(sections)


def test_stand_by_me_un_couplet_troue_de_nc_rejoint_sa_reprise():
    """« A » est « C » dont la moitié des mesures sont muettes (le couplet nu
    de basse). Sans le joker, la concordance tombait à 0.67 et elles ne se
    rencontraient pas."""
    plein = [mesure(A), mesure(A), mesure((F + 1, "-")), mesure((F + 1, "-")),
             mesure(D), mesure(E), mesure(A), mesure(A)]
    troue = [MUETTE, mesure(A), MUETTE, MUETTE,
             mesure(D), mesure(E), mesure(A), mesure(A)]
    bars, sections = monte(("A", troue), ("C", plein))
    merge_similar_letters(sections, bars)
    assert set(labels(sections)) == {"A"}, labels(sections)


def test_sunny_deux_queues_ne_sont_pas_deux_sections():
    """A et B partagent leur tête et divergent sur la queue : c'est une
    1re/2e fin. La concordance globale (3/8) ne suffit pas — c'est la règle
    de tête qui les réunit, et c'est ce qui permet ensuite à
    `_ireal_endings` de voir les queues."""
    tete = [mesure((E, "-")), mesure(G), mesure((C, "^7"))]
    a = tete + [mesure((F + 1, "o")), mesure((E, "-")), mesure(G),
                mesure((C, "^7")), mesure((B, "7"))]
    b = tete + [mesure((F, "7")), mesure((F + 1, "-7")), mesure((B, "7")),
                mesure((E, "-")), mesure(C)]
    bars, sections = monte(("A", a), ("B", b))
    assert concordance([_sig_repli(x) for x in a],
                       [_sig_repli(x) for x in b]) < MERGE_LETTERS_ACCORDS
    assert tete_commune([_sig_repli(x) for x in a],
                        [_sig_repli(x) for x in b]) >= MERGE_LETTERS_PREFIXE
    merge_similar_letters(sections, bars)
    assert set(labels(sections)) == {"A"}, labels(sections)


def test_let_it_be_couplet_et_refrain_restent_deux_sections():
    """LE CAS QUI A TUÉ LA VERSION PRÉCÉDENTE. Les mêmes quatre accords dans
    un ordre différent : un centroïde de chroma les donnait à 0.970 et les
    fusionnait. Ici ils ne partagent ni assez de suite, ni leur tête."""
    couplet = [mesure(C, G), mesure((A, "-"), F), mesure(C, G), mesure(F, C)]
    refrain = [mesure((A, "-"), G), mesure(F, C), mesure(C, G), mesure(F, C)]
    bars, sections = monte(("A", couplet), ("B", refrain))
    merge_similar_letters(sections, bars)
    assert sorted(set(labels(sections))) == ["A", "B"], labels(sections)


def test_yesterday_le_pont_reste_le_pont():
    """Le témoin : rien ne doit le casser."""
    a = [mesure(F), mesure((E, "-"), A), mesure((D, "-")), mesure(B - 1, C),
         mesure(F), mesure((D, "-"), G), mesure(B - 1, F)]
    pont = [mesure(A), mesure((D, "-"), C, B - 1), mesure((G, "-"), C), mesure(F)]
    bars, sections = monte(("A", a), ("B", pont), ("A", a))
    merge_similar_letters(sections, bars)
    assert sorted(set(labels(sections))) == ["A", "B"], labels(sections)


# ── les garde-fous ───────────────────────────────────────────────────────
def test_un_fragment_qui_est_vraiment_le_debut_rejoint_sa_lettre():
    """Un bloc de 2 mesures qui reprend EXACTEMENT les deux premières d'une
    lettre n'est pas une section : c'est un passage coupé. Il la rejoint, et
    `_ireal_cascade._merge_coupe` l'absorbe ensuite comme passage court —
    c'est le comportement voulu, pas un effet de bord."""
    couplet = [mesure(C, G), mesure((A, "-"), F), mesure(C, G), mesure(F, C)]
    fragment = [mesure(C, G), mesure((A, "-"), F)]
    bars, sections = monte(("A", couplet), ("Z", fragment))
    merge_similar_letters(sections, bars)
    assert set(labels(sections)) == {"A"}, labels(sections)


def test_un_fragment_qui_ne_partage_QUE_sa_premiere_mesure_reste_dehors():
    """La contre-épreuve : partager une mesure sur deux (0.50) ne suffit pas.
    C'est ce qui empêche n'importe quel élan d'absorber son voisin."""
    couplet = [mesure(C, G), mesure((A, "-"), F), mesure(C, G), mesure(F, C)]
    elan = [mesure(C, G), mesure((B, "-7"))]
    bars, sections = monte(("A", couplet), ("Z", elan))
    merge_similar_letters(sections, bars)
    assert sorted(set(labels(sections))) == ["A", "Z"], labels(sections)


def test_la_tete_ne_suffit_pas_sous_la_longueur_minimale():
    """La règle de TÊTE (passage à deux queues) exige des blocs assez longs :
    trois mesures de tête communes entre deux blocs de 3 ne prouvent rien."""
    from harmonia.folding import MERGE_LETTERS_MIN_BARS
    assert MERGE_LETTERS_MIN_BARS >= 4
    tete = [mesure(C), mesure(G), mesure((A, "-"))]
    bars, sections = monte(("A", tete), ("Z", tete[:] + []))
    # (identiques ⇒ la concordance les fusionne de toute façon ; on vérifie
    #  ici que la règle de tête ne se déclenche pas toute seule)
    assert tete_commune([_sig_repli(x) for x in tete],
                        [_sig_repli(x) for x in tete]) == 3


def test_intro_et_outro_ne_sont_jamais_fusionnees():
    """Ce sont des repères de structure, pas de la matière musicale."""
    a = [mesure(C), mesure(G), mesure((A, "-")), mesure(F)]
    bars, sections = monte(("intro", a), ("A", a), ("outro", a))
    merge_similar_letters(sections, bars)
    assert labels(sections) == ["intro", "A", "outro"]


def test_une_seule_lettre_ne_declenche_rien():
    a = [mesure(C), mesure(G), mesure((A, "-")), mesure(F)]
    bars, sections = monte(("A", a), ("A", a))
    merge_similar_letters(sections, bars)
    assert labels(sections) == ["A"]


@pytest.mark.parametrize("seuil", [0.55, 0.60, 0.65, 0.70, 0.75])
def test_le_reglage_tient_dans_toute_la_fenetre_mesuree(seuil):
    """La fenêtre mesurée sur les six est (0.50, 0.75]. Le contrat doit tenir
    partout dedans — sinon 0.65 est un point d'équilibre, pas un réglage."""
    couplet = [mesure(C, G), mesure((A, "-"), F), mesure(C, G), mesure(F, C)]
    refrain = [mesure((A, "-"), G), mesure(F, C), mesure(C, G), mesure(F, C)]
    boucle = [mesure((E, "-")), mesure((B, "-")), mesure((E, "-")), mesure((B, "-7"))]
    bars, sections = monte(("A", couplet), ("B", refrain))
    merge_similar_letters(sections, bars, seuil=seuil)
    assert sorted(set(labels(sections))) == ["A", "B"]
    bars, sections = monte(("A", boucle), ("B", boucle[1:] + boucle[:1]))
    merge_similar_letters(sections, bars, seuil=seuil)
    assert set(labels(sections)) == {"A"}
