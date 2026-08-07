"""scripts/section_metric.compare — la distance entre deux découpages.

Louis, 2026-08-07 : « vérifie que ta métrique de correction d'annotation est
bonne ». Ce fichier EST cette vérification. Chaque test est un cas construit
dont la bonne réponse est connue d'avance, et pas un chiffre relevé après coup.

Pourquoi il faut ce filet précisément ici : une métrique à quatre sens est
triviale à intervertir et la faute est silencieuse. C'est déjà arrivé le
2026-08-07 sur les entropies de `score_eval` — le sur-découpage affiché était le
sous-découpage, et une table de comparaison entière s'est lue à l'envers pendant
une soirée sans que rien n'ait l'air anormal. `test_dissymetrie` est là pour ça
et pour rien d'autre.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
from section_metric import compare                              # noqa: E402

N = 40
# la vérité de référence : une intro, deux A collés, un B, un A
TRUTH = "intro:0-7 A:8-15 A:16-23 B:24-31 A:32-39"


def S(spec):
    out = []
    for tok in spec.split():
        lab, rng = tok.split(":")
        a, b = rng.split("-")
        out.append({"b0": int(a), "b1": int(b), "label": lab})
    return out


def sc(spec, truth=TRUTH, **kw):
    return compare(S(spec), S(truth), N, **kw)


# ── les invariants qui doivent valoir exactement 1 ──────────────────────────

def test_identite():
    assert sc(TRUTH)["score"] == pytest.approx(1.0)


def test_renommage():
    """Son A peut être notre B : la métrique ne compare jamais les noms."""
    r = sc("I:0-7 X:8-15 X:16-23 Y:24-31 X:32-39")
    assert r["score"] == pytest.approx(1.0)
    assert r["mapping"]["X"] == "A" and r["mapping"]["Y"] == "B"


# ── ce que chaque moitié voit, et l'autre pas ───────────────────────────────

def test_fusion_de_reprises_adjacentes():
    """Ses deux A collés écrits comme un seul A de 16 mesures.

    C'est notre défaut le plus fréquent (`voice_sections` ne sait pas séparer
    deux occurrences adjacentes d'un même bloc) et il est INVISIBLE mesure par
    mesure : les lettres sont identiques partout. Seul l'appairage des segments
    l'attrape — c'est la raison d'être de la moitié `spans`.
    """
    r = sc("intro:0-7 A:8-23 B:24-31 A:32-39")
    assert r["letters"] == pytest.approx(1.0)
    assert r["spans"] < 0.95
    assert r["score"] < 0.95


def test_mauvais_nom_sur_une_reprise():
    """La même musique appelée A ici et C là : les lettres doivent le voir."""
    r = sc("intro:0-7 A:8-15 C:16-23 B:24-31 A:32-39")
    assert r["letters"] < 1.0
    assert r["spans"] == pytest.approx(1.0)      # les frontières, elles, tiennent


def test_dissymetrie():
    """LE test anti-inversion. Sur-découper et fusionner touchent des sens
    OPPOSÉS ; si un jour ils bougent ensemble, deux sens ont été échangés."""
    fus = sc("intro:0-7 A:8-23 B:24-31 A:32-39")          # on colle
    dec = sc("intro:0-7 A:8-11 A2:12-15 A:16-23 B:24-31 A:32-39")  # on coupe
    assert fus["span_p2r"] < fus["span_r2p"]
    assert dec["span_r2p"] < dec["span_p2r"]


# ── le sur-découpage constant, qu'il demande de tolérer ─────────────────────

T2 = "intro:0-7 A:8-15 B:16-23 A:24-31 B:32-39"


def test_surdecoupage_constant():
    """« Des fois on va merger un B et un C ensemble, et ça c'est à
    l'appréciation. » Couper CHAQUE B en deux de la même façon garde une
    correspondance constante : les lettres ne doivent rien perdre."""
    r = sc("intro:0-7 A:8-15 B:16-19 C:20-23 A:24-31 B:32-35 C:36-39", T2)
    assert r["letters"] == pytest.approx(1.0)
    assert r["score"] > 0.85


def test_surdecoupage_incoherent():
    """Le même découpage, mais sur UNE seule des deux reprises. C'est la faute
    que sa métrique doit distinguer de la précédente."""
    bon = sc("intro:0-7 A:8-15 B:16-19 C:20-23 A:24-31 B:32-35 C:36-39", T2)
    mal = sc("intro:0-7 A:8-15 B:16-19 C:20-23 A:24-31 B:32-39", T2)
    assert mal["letters"] < 1.0
    assert mal["score"] < bon["score"] - 0.05


# ── les queues, et ce qui les sépare d'un décalage ──────────────────────────

QUEUE = "intro:0-7 A:8-15 A:16-21 Q:22-23 B:24-31 A:32-39"
SHIFT = "intro:0-7 A:8-15 A:16-21 B:22-31 A:32-39"


def test_queue_excusee():
    """Les deux dernières mesures d'un A érigées en section à part : un
    désaccord de goût, pas une erreur. Coût réduit, et signalé comme tel."""
    r = sc(QUEUE)
    assert r["forgiven"] == 2
    assert r["score"] > 0.97


def test_decalage_plein_tarif():
    """Les deux MÊMES mesures, données au voisin : la frontière est fausse.
    Rien n'est excusé et ça coûte plus cher que la queue."""
    r = sc(SHIFT)
    assert r["forgiven"] == 0
    assert r["score"] < sc(QUEUE)["score"]


def test_queue_et_decalage_ne_sont_pas_confondus():
    """L'écart entre les deux est le cœur de la spécification de Louis. S'il
    tombe à zéro, la remise sur les queues excuse aussi les décalages et la
    métrique ne mesure plus l'alignement."""
    assert sc(QUEUE)["score"] - sc(SHIFT)["score"] > 0.02


def test_queue_trop_longue_nest_plus_une_queue():
    """Quatre mesures, ce n'est plus une queue : c'est une section."""
    r = sc("intro:0-7 A:8-15 A:16-19 Q:20-23 B:24-31 A:32-39")
    assert r["forgiven"] == 0


# ── le désalignement, qui doit coûter proportionnellement ───────────────────

def test_decalage_croissant():
    """« On perd des points quand il y a un désalignement » : le score doit
    décroître à mesure que la frontière s'éloigne, sans plateau."""
    prev = 1.0
    for k in (0, 1, 2, 3, 4):
        r = sc(f"intro:0-{7 + k} A:{8 + k}-{15 + k} A:{16 + k}-23 "
               f"B:24-31 A:32-39")
        assert r["score"] <= prev + 1e-9
        prev = r["score"]
    assert prev < 0.95


def test_intro_manquee():
    """Ne pas voir l'intro du tout : huit mesures placées ailleurs. C'est la
    faute vivante du moment, elle doit se voir franchement."""
    r = sc("A:0-15 A:16-23 B:24-31 A:32-39")
    assert r["score"] < 0.85


# ── les découpages dégénérés, qui doivent tomber ────────────────────────────

def test_degenere():
    """Le garde-fou. Un appairage par recouvrement a une faille connue : tout
    mettre sous une seule lettre rend le sens lui→nous parfait, et une moyenne
    des deux moitiés remonterait ce néant à 0,50. Le produit doit le laisser
    au sol, et les deux dégénérescences opposées avec lui."""
    tout_un = sc("X:0-39")
    une_par_mesure = " ".join(f"L{i}:{2 * i}-{2 * i + 1}" for i in range(20))
    hache = sc(une_par_mesure)
    for r in (tout_un, hache):
        assert r["score"] < 0.55
    assert tout_un["score"] < sc(TRUTH)["score"]
    # …et chacune s'effondre du côté attendu
    assert tout_un["span_p2r"] < 0.5 and tout_un["span_r2p"] > 0.9
    assert hache["span_r2p"] < 0.5 and hache["span_p2r"] > 0.9


def test_decalage_global_dune_mesure():
    """Tout le morceau poussé d'une mesure : petit mais réel, jamais gratuit."""
    r = sc("intro:0-8 A:9-16 A:17-24 B:25-32 A:33-39")
    assert 0.80 < r["score"] < 0.98


def test_bornes():
    """Aucun découpage ne peut sortir de [0, 1]."""
    for spec in (TRUTH, QUEUE, SHIFT, "X:0-39", "A:0-19 B:20-39"):
        r = sc(spec)
        assert 0.0 <= r["score"] <= 1.0
        for k in ("spans", "letters", "span_p2r", "span_r2p",
                  "letter_p2r", "letter_r2p", "pairwise"):
            assert 0.0 <= r[k] <= 1.0, (spec, k, r[k])


# ── les défauts de l'éditeur d'annotation ───────────────────────────────────
# Huit des dix-huit fichiers de Louis ne pavent pas proprement. La métrique doit
# les lire comme il les entend, pas comme le fichier les écrit.

def test_un_fragment_fantome_ne_compte_pas_pour_une_section():
    """Le cas réel : `A[33-34]` resté sous `A[33-44]` (Blue Lights). Sans
    réparation, ce fantôme compte comme une section qu'on n'aurait pas trouvée
    et fait chuter le score d'un morceau pourtant identique."""
    fantome = S("intro:0-7 A:8-9 A:8-15 B:16-23 A:24-31 B:32-39")
    propre = S(T2)
    assert compare(propre, fantome, N)["score"] == pytest.approx(1.0)


def test_apres_reparation_plus_aucun_chevauchement():
    """L'invariant, pour n'importe quelle entrée : les segments lus se suivent
    sans se marcher dessus. Un recouvrement PARTIEL entre deux lettres
    différentes n'est pas un fantôme — c'est un vrai désaccord de frontière — et
    on le tronque plutôt que de le supprimer, mais dans tous les cas ce qui sort
    est un découpage lisible."""
    from section_metric import segs
    for spec in ("intro:0-7 A:8-17 B:16-23 A:24-31 B:32-39",
                 "intro:0-7 A:8-9 A:8-15 B:16-23 A:24-31 B:32-39",
                 "A:0-39 B:4-8 C:4-8 D:0-39"):
        out = segs(S(spec), N)
        for a, b in zip(out, out[1:]):
            assert a[1] < b[0], (spec, a, b)


def test_un_trou_reste_un_trou():
    """Une mesure que Louis n'attribue à personne n'est pas une erreur à
    combler : c'est une information, et elle doit rester visible."""
    r = compare(S(TRUTH), S("intro:0-7 A:8-15 A:16-22 B:24-31 A:32-39"), N)
    assert r["score"] < 1.0
