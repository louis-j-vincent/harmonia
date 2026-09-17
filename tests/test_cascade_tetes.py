"""tests/test_cascade_tetes.py — les têtes d'extension arrivent intactes.

Louis, 2026-09-17 : « fais en sorte qu'on puisse avoir des extensions, et
ensuite j'ai envie que tu me fasses une démo d'un compas amélioré, où je
sélectionne d'abord l'accord en maj/min, ensuite dès qu'on le sélectionne on
select la 7ème, puis la 9ème, puis la 11ème, puis la 13ème si elle est
suggérée », puis « can you try integrating it into the app ».

POURQUOI CE CHAMP EXISTE. `sug` vit dans l'espace à 60 cases (12 racines ×
QUAL5) : c'est celui du DÉCODAGE, et il replie les septièmes (`maj7` -> `maj`,
`min7` -> `min`). Le compas en cascade a besoin de l'inverse — la triade nue
d'un côté, les degrés ajoutés de l'autre. `cascade_suggestions` lit donc les
têtes de musx là où elles sont encore séparées, sans rien décider.

CE QUE CES TESTS GARDENT, et c'est la partie fragile : le mapping des colonnes
de la tête de triade. Colonne 0 = « pas d'accord » ; colonne j>=1 -> racine
``(j-1) % 12``, famille ``(j-1) // 12`` dans l'ordre maj/min/sus4/sus2/dim/aug.
Une erreur d'un cran ici ferait proposer un accord faux avec l'air d'être sûr —
c'est exactement le « bug de calibration silencieux » que CLAUDE.md met en
tête de ses six motifs.

    .venv/bin/python -m pytest tests/test_cascade_tetes.py -v
"""
from __future__ import annotations

import numpy as np

from harmonia.musx import FRAME_DT
from harmonia.span_rescore import EXT_HEADS, TRIAD_FAMILIES, cascade_suggestions

N_TRI = 1 + 12 * len(TRIAD_FAMILIES)          # 73 : la colonne N plus 72 cases


def _probs(n_frames: int = 40, *, pic: int | None = None):
    """Six têtes plates, sauf la colonne `pic` de la triade si on la donne."""
    tri = np.full((n_frames, N_TRI), 1.0 / N_TRI)
    if pic is not None:
        tri[:] = 0.001
        tri[:, pic] = 1.0 - 0.001 * (N_TRI - 1)
    return [tri,
            np.full((n_frames, 13), 1 / 13),
            np.full((n_frames, 4), 0.25),
            np.full((n_frames, 4), 0.25),
            np.full((n_frames, 3), 1 / 3),
            np.full((n_frames, 3), 1 / 3)]


def _accord(t0=0.0, t1=1.0, **kw):
    return {"root": 0, "q": "", "t0": t0, "t1": t1, "nc": False, **kw}


def test_la_colonne_de_triade_se_lit_racine_et_famille():
    """Le mapping, cran par cran, sur les six familles et deux racines."""
    for fam in range(len(TRIAD_FAMILIES)):
        for root in (0, 7, 11):
            col = 1 + fam * 12 + root
            c = _accord()
            cascade_suggestions(_probs(pic=col), [c])
            top = c["casc"]["base"][0]
            assert (top["root"], top["type"]) == (root, fam), (
                f"colonne {col} devrait être {TRIAD_FAMILIES[fam]} sur {root}, "
                f"lue {TRIAD_FAMILIES[top['type']]} sur {top['root']}")


def test_la_colonne_zero_nest_jamais_un_candidat():
    """Colonne 0 = « pas d'accord du tout » : elle ne doit pas se déguiser en do."""
    probs = _probs(pic=0)
    c = _accord()
    cascade_suggestions(probs, [c])
    assert c["casc"]["base"], "un classement doit quand même sortir"
    # aucune probabilité ne doit valoir celle de la colonne N (0,927)
    assert max(b["c"] for b in c["casc"]["base"]) < 0.1


def test_les_quatre_tetes_dextension_arrivent_avec_leur_taille():
    c = _accord()
    cascade_suggestions(_probs(), [c])
    for nom, i, labels in EXT_HEADS:
        v = c["casc"][nom]
        assert len(v) == len(labels), (nom, len(v), len(labels))
        # arrondi à 4 décimales dans le chart — c'est de la place gagnée sur
        # un fichier que le téléphone télécharge, pas une perte d'information
        # (le compas dessine des orbes, pas la 5e décimale).
        assert abs(sum(v) - 1.0) < 1e-3, (nom, sum(v))


def test_la_mise_en_commun_est_bien_sur_lempan_de_laccord():
    """Deux accords, deux empans, deux réponses — pas la moyenne du morceau."""
    n = 40
    probs = _probs(n)
    tri = probs[0]
    do_maj, sol_min = 1, 1 + 12 + 7
    tri[:] = 0.001
    tri[:20, do_maj] = 0.9                # première moitié : do majeur
    tri[20:, sol_min] = 0.9               # seconde moitié : sol mineur
    mi = 20 * FRAME_DT
    a, b = _accord(0.0, mi), _accord(mi, 40 * FRAME_DT)
    assert cascade_suggestions(probs, [a, b]) == 2
    assert (a["casc"]["base"][0]["root"], a["casc"]["base"][0]["type"]) == (0, 0)
    assert (b["casc"]["base"][0]["root"], b["casc"]["base"][0]["type"]) == (7, 1)


def test_un_empan_plus_court_quune_trame_ne_rend_pas_le_vide():
    """La même règle d'intervalle vide que partout ailleurs : la trame la plus
    proche, jamais un zéro qui ferait disparaître l'accord de l'éditeur."""
    c = _accord(0.5 * FRAME_DT, 0.5 * FRAME_DT)
    assert cascade_suggestions(_probs(pic=1 + 7), [c]) == 1
    assert c["casc"]["base"][0]["root"] == 7


def test_les_nc_et_les_accords_illisibles_sont_sautes():
    nc = _accord(nc=True)
    casse = {"root": 0, "q": "", "nc": False}          # pas de t0/t1
    bon = _accord()
    assert cascade_suggestions(_probs(), [nc, casse, bon]) == 1
    assert "casc" not in nc and "casc" not in casse and "casc" in bon


def test_rien_a_annoter_ne_casse_pas():
    assert cascade_suggestions(_probs(), []) == 0
    assert cascade_suggestions(_probs(), [_accord(nc=True)]) == 0
