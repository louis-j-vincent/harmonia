"""tests/test_intro_en_tete.py — le trou AVANT la première section est l'intro.

Louis, 2026-09-30 : « quand il manque une section avant un A, c'est une
intro, pas un C ».

« Valider les sections » (`simulation.occurrences_de_tous` puis
`simulation.remplir_par_algo`) pose ses briques et leurs reprises, puis nomme
les zones libres avec l'algorithme des quatre mots — qui ne sait écrire que
des lettres. La zone libre de TÊTE (mesure 0 … début de la première section
posée − 1) devient désormais UNE section « intro ». Périmètre convenu avec
lui : la tête seulement. Les trous du milieu gardent leurs lettres ; aucune
règle d'outro.
"""
from __future__ import annotations

import json

import pytest

from harmonia.sections import simulation as sim_mod


def _chart(n=60, muettes=0, duree=2.0, sections=None):
    """Grille régulière, un accord par mesure ; les `muettes` premières
    mesures sont des N.C. (la tête silencieuse que la pipeline range dans une
    « intro » depuis e2d26cb)."""
    grid = [round(i * duree, 3) for i in range(n + 1)]
    chords = [{"root": i % 4 * 3, "q": "", "bass": -1, "nc": i < muettes,
               "t0": grid[i], "t1": grid[i + 1]} for i in range(n)]
    return {"title": "bidon", "barGrid": grid, "nBars": n,
            "prompter": {"chords": chords}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": sections or []}


def _occ(*plages):
    return [{"m0": a, "m1": b, "label": lab, "score": 1.0, "rot": 0,
             "trace": True} for a, b, lab in plages]


def _intros(blocs):
    return [b for b in blocs if b["label"] == "intro"]


def _couvre(blocs, n):
    pl = sorted((b["m0"], b["m1"]) for b in blocs)
    assert pl[0][0] == 0 and pl[-1][1] == n - 1
    for (_a, b), (c, _d) in zip(pl, pl[1:]):
        assert c == b + 1, f"trou ou chevauchement entre {b} et {c}"


def test_la_zone_de_tete_est_UNE_intro(tmp_path):
    chart = _chart(60)
    blocs = sim_mod.remplir_par_algo(chart, _occ((8, 15, "A"), (24, 31, "A")),
                                     tmp_path)
    intros = _intros(blocs)
    assert [(b["m0"], b["m1"]) for b in intros] == [(0, 7)], blocs
    assert intros[0]["source"] == "algo"
    _couvre(blocs, 60)


def test_les_lettres_neuves_commencent_juste_apres_celles_de_louis(tmp_path):
    """L'intro ne consomme pas de lettre : le premier bloc de l'algo après
    elle s'appelle B (Louis n'a que A), pas C."""
    chart = _chart(60)
    blocs = sim_mod.remplir_par_algo(chart, _occ((8, 15, "A"), (24, 31, "A")),
                                     tmp_path)
    lettres = [b["label"].rstrip("′") for b in blocs
               if b["source"] == "algo" and b["m0"] > 15]
    assert lettres, "il reste des trous au milieu, nommés par l'algo"
    assert lettres[0] == "B", lettres


def test_les_trous_du_milieu_gardent_leurs_lettres(tmp_path):
    chart = _chart(60)
    blocs = sim_mod.remplir_par_algo(chart, _occ((0, 7, "A"), (24, 31, "A")),
                                     tmp_path)
    assert not _intros(blocs), "pas de zone de tête : pas d'intro"
    assert any(b["source"] == "algo" for b in blocs)


def test_sans_aucune_section_posee_pas_dintro(tmp_path):
    """Pas de « première section posée » : rien ne dit où est la tête."""
    blocs = sim_mod.remplir_par_algo(_chart(40), [], tmp_path)
    assert not _intros(blocs)


def test_une_tete_muette_devient_intro_et_une_seule(tmp_path):
    """La tête N.C. que la pipeline appelait déjà « intro » (e2d26cb) ne
    produit pas deux intros : la sortie remplace les sections du chart."""
    from harmonia.pipeline import _one_section
    chart = _chart(40, muettes=4)
    bars = [[c] for c in chart["prompter"]["chords"]]
    chart["sections"] = _one_section(bars, chart["barGrid"], 40)
    assert [s["label"] for s in chart["sections"]] == ["intro", "A"]
    blocs = sim_mod.remplir_par_algo(chart, _occ((4, 11, "A"), (20, 27, "A")),
                                     tmp_path)
    assert [(b["m0"], b["m1"]) for b in _intros(blocs)] == [(0, 3)]
    _couvre(blocs, 40)


def test_louis_a_deja_trace_une_intro_on_nen_ajoute_pas(tmp_path):
    blocs = sim_mod.remplir_par_algo(
        _chart(40), _occ((2, 5, "intro"), (6, 13, "A"), (22, 29, "A")), tmp_path)
    assert len(_intros(blocs)) == 1 and _intros(blocs)[0]["m0"] == 2


def test_lintro_ne_se_replie_avec_rien_dans_le_chart(tmp_path):
    """Chemin refold / `sections_pour_chart` : l'intro reste UNE section,
    reps 1, et les lettres de Louis sont intactes."""
    from harmonia.soudure import sections_pour_chart
    chart = _chart(60, muettes=3)
    blocs = sim_mod.remplir_par_algo(chart, _occ((8, 15, "A"), (24, 31, "A")),
                                     tmp_path)
    secs = [{"label": b["label"], "mesure_debut": b["m0"] + 1,
             "mesure_fin": b["m1"] + 1} for b in blocs]
    neuves = sections_pour_chart(chart, secs)
    intros = [s for s in neuves if s["label"] == "intro"]
    assert len(intros) == 1
    assert intros[0]["barRanges"] == [[0, 7]] and intros[0]["reps"] == 1
    assert neuves[0]["label"] == "intro"
    a = [s for s in neuves if s["label"] == "A"]
    assert a and sorted(r for s in a for r in map(tuple, s["barRanges"])) == [
        (8, 15), (24, 31)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from harmonia.server import app as app_mod
    from harmonia.server.routes import sections as sec_mod
    (tmp_path / "min_bidon.json").write_text(json.dumps(_chart(60)),
                                             encoding="utf-8")
    (tmp_path / "absent.m4a").write_bytes(b"")
    monkeypatch.setattr(sec_mod, "CHARTS_DIR", tmp_path)
    monkeypatch.setattr(sec_mod, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(sim_mod, "cherche_brique",
                        lambda chart, b0, b1, ad, **k: (
                            [(b0, b1, 1.0, 0)],
                            {"label": k.get("label"), "m0": b0, "m1": b1,
                             "n_reprises": 0, "raison": None}))
    return app_mod.create_app().test_client()


def test_la_route_valider_rend_lintro_de_tete(client):
    humain = [{"label": "A", "mesure_debut": 9, "mesure_fin": 16},
              {"label": "B", "mesure_debut": 33, "mesure_fin": 40}]
    out = client.post("/api/sections/inferer/min_bidon",
                      json={"humain": humain}).get_json()
    tete = [s for s in out["sections"] if s["mesure_debut"] == 1]
    assert len(tete) == 1 and tete[0]["label"] == "intro", out["sections"]
    assert tete[0]["mesure_fin"] == 8
    # l'intro n'est pas un trait de Louis : elle est recalculée à chaque Valider
    assert tete[0]["source"] == "algo"
    assert sum(s["label"] == "intro" for s in out["sections"]) == 1
