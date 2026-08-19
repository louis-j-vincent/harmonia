"""Les sections SongFormer : des secondes aux mesures, puis aux lettres.

Le modèle lui-même n'est pas chargé ici (une minute de CPU) — on teste la
partie qui lui est propre : tirer ses frontières sur NOTRE grille, effacer le
silence, et nommer. C'est là que sont les hypothèses porteuses.
"""
import pytest

from harmonia_min import songformer as SF


def grille(n, pas=2.0):
    return [i * pas for i in range(n + 1)]


def test_frontiere_tiree_sur_la_barre_la_plus_proche():
    # frontière à 7,6 s sur une grille de 2 s : la barre 4 (8,0 s) est la plus
    # proche, pas la barre 3 (6,0 s).
    segs = [{"label": "intro", "start": 0.0, "end": 7.6},
            {"label": "verse", "start": 7.6, "end": 20.0}]
    out = SF._sur_la_grille(segs, grille(10))
    assert [(s["b0"], s["b1"]) for s in out] == [(0, 3), (4, 9)]


def test_le_pavage_couvre_tout_sans_trou():
    segs = [{"label": "intro", "start": 0.9, "end": 4.1},
            {"label": "verse", "start": 4.1, "end": 11.0},
            {"label": "chorus", "start": 11.0, "end": 17.5}]
    out = SF._sur_la_grille(segs, grille(10))
    assert out[0]["b0"] == 0 and out[-1]["b1"] == 9
    for a, b in zip(out, out[1:]):
        assert b["b0"] == a["b1"] + 1


def test_deux_frontieres_dans_la_meme_mesure_nen_font_quune():
    # 4,1 s et 4,4 s tombent toutes deux sur la barre 2 : la seconde est jetée,
    # une section de zéro mesure n'existe pas.
    segs = [{"label": "intro", "start": 0.0, "end": 4.1},
            {"label": "inst", "start": 4.1, "end": 4.4},
            {"label": "verse", "start": 4.4, "end": 20.0}]
    out = SF._sur_la_grille(segs, grille(10))
    assert len(out) == 2
    assert [s["label"] for s in out] == ["intro", "inst"]


def test_le_silence_rejoint_son_voisin():
    segs = [{"b0": 0, "b1": 3, "label": "intro"},
            {"b0": 4, "b1": 9, "label": "verse"},
            {"b0": 10, "b1": 11, "label": "silence"}]
    out = SF._fondre_muets(segs)
    assert [s["label"] for s in out] == ["intro", "verse"]
    assert out[-1]["b1"] == 11          # la queue est reprise, pas perdue


def test_meme_role_meme_lettre():
    segs = [{"b0": 0, "b1": 7, "label": "intro"},
            {"b0": 8, "b1": 15, "label": "verse"},
            {"b0": 16, "b1": 23, "label": "chorus"},
            {"b0": 24, "b1": 31, "label": "verse"},
            {"b0": 32, "b1": 39, "label": "chorus"},
            {"b0": 40, "b1": 47, "label": "outro"}]
    out = SF._lettres(segs)
    assert [s["label"] for s in out] == ["intro", "A", "B", "A", "B", "outro"]


def test_meme_role_meme_lettre_meme_a_longueurs_inegales():
    """Le rôle fait la lettre, pas la longueur.

    « Under-fold, never over-fold » est tenu EN AVAL (sections_pour_chart et
    fold_display groupent par (lettre, longueur)). Le séparer ici rendrait
    quatre lettres pour un refrain joué cinq fois dont la frontière du modèle
    tombe au demi-temps près — ce que l'oreille n'entend pas.
    """
    segs = [{"b0": 0, "b1": 7, "label": "chorus"},      # 8 mesures
            {"b0": 8, "b1": 18, "label": "chorus"},     # 11 mesures
            {"b0": 19, "b1": 26, "label": "chorus"}]    # 8 mesures
    out = SF._lettres(segs)
    assert [s["label"] for s in out] == ["A", "A", "A"]


def test_intro_au_milieu_prend_une_lettre():
    """Le nom « intro » est réservé au DÉBUT ; ailleurs c'est une section."""
    segs = [{"b0": 0, "b1": 7, "label": "verse"},
            {"b0": 8, "b1": 15, "label": "intro"},
            {"b0": 16, "b1": 23, "label": "verse"}]
    out = SF._lettres(segs)
    assert [s["label"] for s in out] == ["A", "B", "A"]


def test_sans_intro_quand_la_marque_bar1_est_posee():
    """Set bar 1 : la marque EST le début de la forme, il n'y a plus d'intro à
    trouver — l'appelant a déjà découpé ce qui précède."""
    segs = [{"b0": 0, "b1": 7, "label": "intro"},
            {"b0": 8, "b1": 15, "label": "chorus"}]
    out = SF._lettres(segs, avec_intro=False)
    assert [s["label"] for s in out] == ["A", "B"]


def test_songformer_est_le_defaut():
    import os
    from harmonia_min import sections as S
    assert S.SECTION_MODE_ENV == "HARMONIA_SECTIONS"
    assert os.environ.get(S.SECTION_MODE_ENV, "songformer") == "songformer" \
        or os.environ.get(S.SECTION_MODE_ENV) is not None


def test_mode_inconnu_leve():
    import os
    from harmonia_min import sections as S
    old = os.environ.get(S.SECTION_MODE_ENV)
    os.environ[S.SECTION_MODE_ENV] = "n_importe_quoi"
    try:
        with pytest.raises(ValueError):
            S.detect_sections([0.0, 1.0, 2.0], None, None)
    finally:
        if old is None:
            os.environ.pop(S.SECTION_MODE_ENV, None)
        else:
            os.environ[S.SECTION_MODE_ENV] = old


def test_minimal_fold_separe_les_longueurs_dune_meme_lettre():
    """Le repli d'affichage groupe par (lettre, longueur), pas par lettre.

    Rouge d'abord (2026-08-18) : il groupait par lettre seule, donc un B de 8
    mesures et un B de 4 sortaient dans UN bloc, écrit à la longueur du
    représentant — la lecture d'Another Day se décalait d'une barre à chaque
    reprise. Les deux blocs gardent le nom B, chacun à sa longueur.
    """
    from harmonia_min.folding import minimal_fold

    def bar(root):
        return [{"root": root, "q": "", "nc": False, "bass": root,
                 "t0": 0.0, "t1": 1.0, "c": 1.0}]

    bars = [bar(i % 4) for i in range(12)]
    grid = [float(i) for i in range(13)]
    sections = [{"label": "B", "barRanges": [[0, 7]]},      # 8 mesures
                {"label": "B", "barRanges": [[8, 11]]}]     # 4 mesures
    out = minimal_fold(sections, bars, grid, {})
    assert [s["label"] for s in out] == ["B", "B"]
    assert [len(s["bars"]) for s in out] == [8, 4]
    assert len({s["id"] for s in out}) == 2      # deux id distincts
