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


# ── la régression du 2026-09-17 : une erreur d'UNITÉ ─────────────────────────

def _chart_bidon(n_mesures=60, duree=2.0):
    """Un chart minimal : une grille régulière et un accord par mesure."""
    grid = [round(i * duree, 3) for i in range(n_mesures + 1)]
    chords = [{"root": i % 4 * 3, "q": "", "bass": -1, "nc": False,
               "t0": grid[i], "t1": grid[i + 1]} for i in range(n_mesures)]
    return {"title": "bidon", "barGrid": grid, "nBars": n_mesures,
            "prompter": {"chords": chords}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": []}


def test_un_trait_dans_la_SECONDE_MOITIE_du_morceau_survit(tmp_path, monkeypatch):
    """LA régression du 2026-09-17, et le contrat de la route d'aujourd'hui.

    Le bug : `sections_inferer` passait à `traits_propres` le nombre de JETONS
    (des bi-mesures) au lieu du nombre de MESURES. Le nettoyage croyait donc le
    morceau deux fois plus court, et tout trait tracé après la moitié était
    jeté avec « commence après la fin du morceau ».

    Louis, sur Don't Want My Love : son trait « C » sur les mesures 37-48 d'un
    morceau de 56 mesures a disparu en silence, et la machine a rempli le trou.
    Une erreur d'unité, le premier motif du CLAUDE.md : elle produit des
    chiffres plausibles et faux.

    Le test vaut aussi pour la loi actée le même jour — brique + SongFormer :
    ses deux traits doivent ressortir AUX MESURES TRACÉES et SOUS LEUR NOM, et
    les trous porter ce que le modèle avait trouvé.
    """
    import json

    from harmonia.server import app as app_mod
    from harmonia.server.routes import sections as sec_mod

    from harmonia.sections import simulation as sim_mod

    (tmp_path / "min_bidon.json").write_text(json.dumps(_chart_bidon()),
                                             encoding="utf-8")
    (tmp_path / "absent.m4a").write_bytes(b"")     # le fichier doit EXISTER
    monkeypatch.setattr(sec_mod, "CHARTS_DIR", tmp_path)
    monkeypatch.setattr(sec_mod, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(sec_mod, "_AUTO_CACHE", {})
    # La recherche de reprises demande l'audio et le modèle : elle est mesurée
    # sur de vrais morceaux, pas ici. Ce test porte sur l'autre contrat — ses
    # traits survivent, aux bonnes mesures et sous leur nom — donc on la
    # neutralise en ne rendant que la brique elle-même.
    monkeypatch.setattr(sim_mod, "cherche_brique",
                        lambda chart, b0, b1, ad, **k: [(b0, b1, 1.0)])
    # SongFormer ne tourne pas dans un test : on lui substitue un découpage
    # fixe, et c'est LUI qu'on doit retrouver dans les trous.
    monkeypatch.setattr(sec_mod, "detect_sections",
                        lambda grid, audio, **k: [
                            {"b0": 0, "b1": 19, "label": "intro"},
                            {"b0": 20, "b1": 59, "label": "Z"}])
    client = app_mod.create_app().test_client()

    humain = [{"label": "A", "mesure_debut": 1, "mesure_fin": 8},
              {"label": "C", "mesure_debut": 41, "mesure_fin": 48}]
    out = client.post("/api/sections/inferer/min_bidon",
                      json={"humain": humain}).get_json()

    assert out.get("ecartes") == [], "aucun trait ne doit être jeté"
    siens = [s for s in out["sections"] if s["source"] == "humain"]
    assert {(s["label"], s["mesure_debut"], s["mesure_fin"]) for s in siens} == {
        ("A", 1, 8), ("C", 41, 48)}, "ses deux traits, aux mesures tracées"
    # les trous portent le découpage du modèle, pas des lettres inventées
    trous = {s["label"] for s in out["sections"] if s["source"] == "algo"}
    assert trous <= {"intro", "Z"}, f"les trous doivent venir du modèle : {trous}"
    # et le morceau est couvert de bout en bout, sans trou ni chevauchement
    plages = sorted((s["mesure_debut"], s["mesure_fin"])
                    for s in out["sections"])
    assert plages[0][0] == 1 and plages[-1][1] == 60
    for (_a, b), (c, _d) in zip(plages, plages[1:]):
        assert c == b + 1, f"trou ou chevauchement entre {b} et {c}"
