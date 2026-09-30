"""Fins alternatives : l'équivalence sus-sur-basse et le curseur de fusion.

Louis, 2026-09-30, sur Claustrophobic : « t'as différencié un A où il y a un
Bb/C d'un A où il y a un C7sus4 (2 fins alternatives différentes) alors
qu'essentiellement c'est la même chose. Il faudrait un curseur sur lequel on
peut jouer qui force à quel point tu merges les sections ou tu les laisses
avoir des bouts alternatifs. »
"""
from __future__ import annotations

import pytest

from harmonia import folding
from harmonia.folding import (FUSION_DEFAUT, FUSION_NIVEAUX, _ireal_cascade,
                              _ireal_endings, accord_sonnant)

GRID = [float(i) for i in range(400)]
C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


def _ch(root, q="", bass=-1):
    return {"root": root, "q": q, "nc": False, "bass": bass}


def _blk(label, bars, ranges):
    L = len(bars)
    return {
        "id": "L" + label, "label": label, "tag": "", "reps": len(ranges),
        "spans": [[float(c0), float(c1 + 1)] for c0, c1 in ranges],
        "barRanges": [list(r) for r in ranges],
        "bars": [list(b) for b in bars],
        "barSpans": [[[float(c0 + i), float(c0 + i + 1)] for c0, _ in ranges]
                     for i in range(L)],
    }


def _song(passes):
    """{bar index: [chords]} + ranges, 8-bar passes laid every 8 bars."""
    bars, ranges = {}, []
    for k, p in enumerate(passes):
        c0 = 8 * k
        ranges.append((c0, c0 + len(p) - 1))
        for i, bar in enumerate(p):
            bars[c0 + i] = bar
    return bars, ranges


TRONC = [[_ch(Eb, "7")]] * 6


def _claustro():
    """Le A de Claustrophobic, ses six passes telles que décodées (2026-09-30)."""
    fins = [
        [[_ch(Eb, "7")], [_ch(C, "7sus4"), _ch(F, "^7")]],
        [[_ch(Eb, "7")], [_ch(Bb, "", C), _ch(F, "^7")]],
        [[_ch(Eb, "7")], [_ch(Bb, "", C), _ch(F, "^7")]],
        [[_ch(Bb, "-7")], [_ch(C, "7sus4"), _ch(F, "^7")]],
        [[_ch(Eb, "7")], [_ch(C, "9"), _ch(F, "7")]],
        [[_ch(Eb, "7")], [_ch(Bb, "7"), _ch(F, "^7")]],
    ]
    return _song([TRONC + f for f in fins])


# ── 1. l'équivalence : une triade un ton sous la basse est un sus sur la basse

@pytest.mark.parametrize("ch, attendu", [
    (_ch(Bb, "", C), (C, "7sus4")),      # Bb/C = C9sus4
    (_ch(F, "", G), (G, "7sus4")),       # F/G = G9sus4
    (_ch(Ab, "", Bb), (Bb, "7sus4")),    # Ab/Bb = Bb9sus4
    (_ch(G, "-7", C), (C, "7sus4")),     # G-7/C = C11
    (_ch(C, "9sus4"), (C, "7sus4")),     # la 9 est une couleur
    (_ch(C, "7sus4"), (C, "7sus4")),
    (_ch(C, "", E), (C, "")),            # renversement : rien ne change
    (_ch(Eb, "7", C), (Eb, "7")),        # tierce mineure sur la basse : pas un sus
    (_ch(D, "", C), (D, "")),            # pas de septième mineure : pas un sus
    (_ch(C, "7"), (C, "7")),
])
def test_accord_sonnant(ch, attendu):
    assert accord_sonnant(ch) == attendu


def test_bb_sur_c_et_c7sus4_ne_font_pas_deux_fins():
    """La queue ne diffère que par Bb/C contre C7sus4 : même accord sonné, pas
    de crochet 1./2."""
    bars, ranges = _song([
        TRONC + [[_ch(Eb, "7")], [_ch(C, "7sus4"), _ch(F, "^7")]],
        TRONC + [[_ch(Eb, "7")], [_ch(Bb, "", C), _ch(F, "^7")]],
    ])
    blk = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(blk, bars, GRID)
    assert "endings" not in blk


def test_f_sur_g_rejoint_un_passage_coupe_en_g7sus4():
    """Le raccord par famille (règle 3) lit aussi la fondamentale sonnée."""
    cell = [[_ch(C, "^7")], [_ch(A, "-7")], [_ch(D, "-7")], [_ch(G, "7sus4")]]
    host = _blk("A", cell + cell, [(0, 7), (16, 23)])
    court = _blk("A", cell[:3] + [[_ch(F, "", G)]], [(40, 43)])
    out = _ireal_cascade([host, court], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A"]


def test_une_vraie_fin_reste_une_fin():
    """This Love : `Bb Eb` contre `Ab G` — la fondamentale change pour de vrai."""
    bars, ranges = _song([
        TRONC + [[_ch(Bb)], [_ch(Eb)]],
        TRONC + [[_ch(Ab)], [_ch(G)]],
    ])
    blk = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(blk, bars, GRID)
    assert blk["endings"]["tail"] == 2


# ── 2. le curseur de fusion

def _fins(niveau):
    bars, ranges = _claustro()
    blk = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(blk, bars, GRID, fusion=niveau)
    return blk


def test_niveaux():
    assert FUSION_NIVEAUX == (0, 1, 2, 3, 4)
    assert FUSION_DEFAUT == 1


def test_defaut_egal_sans_parametre():
    bars, ranges = _claustro()
    a = _blk("A", [bars[i] for i in range(8)], ranges)
    b = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(a, bars, GRID)
    _ireal_endings(b, bars, GRID, fusion=FUSION_DEFAUT)
    assert a == b


def test_claustrophobic_au_defaut_bb_sur_c_rejoint_c7sus4():
    passes = [v["passes"] for v in _fins(FUSION_DEFAUT)["endings"]["variants"]]
    assert [0, 1, 2] in passes


def test_fusion_max_un_seul_bloc():
    """« Tout fusionner » : un bloc ×6, sans crochet, écrit avec la fin la
    plus jouée."""
    blk = _fins(max(FUSION_NIVEAUX))
    assert "endings" not in blk
    assert blk["reps"] == 6
    last = blk["bars"][-1]
    assert accord_sonnant(last[0]) == (C, "7sus4")
    # la carte de lecture reste : 8 rangées, chacune allumée à chaque passe
    assert [len(r) for r in blk["barSpans"]] == [6] * 8


def test_garder_les_fins_montre_meme_une_couleur():
    """Niveau 0 : C7 contre C-7 en dernière mesure fait une 2e fin (le défaut
    l'avale comme bruit de décodage)."""
    bars, ranges = _song([
        [[_ch(F)]] * 7 + [[_ch(C, "7")]],
        [[_ch(F)]] * 7 + [[_ch(C, "-7")]],
    ])
    d = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(d, bars, GRID)
    assert "endings" not in d
    k = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(k, bars, GRID, fusion=0)
    assert k["endings"]["tail"] == 1


def test_le_curseur_est_monotone():
    """Plus on pousse, moins il y a de fins distinctes — jamais plus."""
    n = []
    for niv in FUSION_NIVEAUX:
        blk = _fins(niv)
        n.append(len(blk.get("endings", {}).get("variants", [])) or 1)
    assert n == sorted(n, reverse=True), n
    assert n[0] > n[-1] == 1


def test_sections_par_fusion_ne_garde_que_ce_qui_change():
    """Les niveaux qui rendent la même chose que le défaut ne sont pas
    stockés ; deux niveaux identiques entre eux partagent une entrée."""
    x, d, y = ([{"label": k, "barRanges": [[0, 7]]}] for k in "xdy")
    rendus = {0: x, 1: d, 2: d, 3: y, 4: y}
    defaut, alt = folding.sections_par_fusion(lambda n: rendus[n])
    assert defaut == d
    assert alt["0"] == x and alt["3"] == y and alt["4"] == "3"
    assert alt["cle"] == folding.cle_sections(d)
    assert "1" not in alt and "2" not in alt


def test_appliquer_fusion_au_service():
    sec_d = [{"label": "A", "barRanges": [[0, 7]], "bars": [], "reps": 1}]
    sec_4 = [{"label": "A", "barRanges": [[0, 7]], "bars": ["m"], "reps": 1}]
    model = {"sections": sec_d,
             "sectionsFusion": {"cle": folding.cle_sections(sec_d),
                                "4": sec_4, "3": "4"}}
    m = folding.appliquer_fusion(dict(model), 3)
    assert m["sections"] == sec_4 and "sectionsFusion" not in m
    assert m["fusion"] == 3 and m["fusionDispo"] is True
    m = folding.appliquer_fusion(dict(model), None)
    assert m["sections"] == sec_d and m["fusion"] == FUSION_DEFAUT
    assert folding.appliquer_fusion({"sections": sec_d}, 4)["fusionDispo"] is False
    # des sections réécrites depuis (autre découpage) : les variantes sont
    # périmées, on sert le défaut
    stale = dict(model, sections=[{"label": "B", "barRanges": [[0, 3]]}])
    m = folding.appliquer_fusion(stale, 4)
    assert m["sections"][0]["label"] == "B"


# ── 3. le réglage par morceau : chart_meta.json, servi par /api/chart-model

@pytest.fixture()
def app_fusion(tmp_path, monkeypatch):
    import json as _json
    from harmonia import annotations as an
    from harmonia.server import jobs
    from harmonia.server.routes import library as lib
    charts = tmp_path / "charts"
    charts.mkdir()
    meta = tmp_path / "chart_meta.json"
    meta.write_text(_json.dumps({"min_x": {"artist": "PJ", "title": "Claus"}}))
    monkeypatch.setattr(lib, "CHARTS_DIR", charts)
    monkeypatch.setattr(lib, "META_PATH", meta)
    monkeypatch.setattr(jobs, "META_PATH", meta)
    monkeypatch.setattr(an, "ANNOT_DIR", tmp_path / "ann")
    sec_d = [{"id": "LA", "label": "A", "barRanges": [[0, 1], [2, 3]],
              "reps": 2, "bars": [[], []], "endings": {"tail": 1}}]
    sec_4 = [{"id": "LA", "label": "A", "barRanges": [[0, 1], [2, 3]],
              "reps": 2, "bars": [[], []]}]
    (charts / "min_x.json").write_text(_json.dumps({
        "file": "min_x", "key": {"tonic": 0, "mode": "major"},
        "barGrid": [0, 1, 2, 3, 4], "nBars": 4, "sections": sec_d,
        "sectionsFusion": {"cle": folding.cle_sections(sec_d), "4": sec_4}}))
    from harmonia.server import app as app_mod
    return app_mod.create_app().test_client(), meta


def test_route_curseur_persiste_et_sert(app_fusion):
    import json as _json
    client, meta = app_fusion
    m = client.get("/api/chart-model/min_x").get_json()
    assert m["fusion"] == FUSION_DEFAUT and m["fusionDispo"] is True
    assert "endings" in m["sections"][0] and "sectionsFusion" not in m
    r = client.post("/api/chart-meta/min_x", json={"fusion": 4})
    assert r.status_code == 200
    # la ligne garde artiste et titre
    assert _json.loads(meta.read_text())["min_x"] == {
        "artist": "PJ", "title": "Claus", "fusion": 4}
    m = client.get("/api/chart-model/min_x").get_json()
    assert m["fusion"] == 4 and "endings" not in m["sections"][0]
    # renommer l'artiste ne perd pas le cran
    client.post("/api/chart-meta/min_x", json={"artist": "Q", "title": "T"})
    assert _json.loads(meta.read_text())["min_x"]["fusion"] == 4
    # revenir au défaut retire la clé
    client.post("/api/chart-meta/min_x", json={"fusion": FUSION_DEFAUT})
    assert "fusion" not in _json.loads(meta.read_text())["min_x"]
    assert client.post("/api/chart-meta/min_x",
                       json={"fusion": 9}).status_code == 400


# ── l'équivalence n'enlève jamais une ressemblance qui existait

def test_bb_sur_c_se_lit_encore_bb():
    """Avant, la basse était ignorée : Bb/C ≡ Bb. Ça reste vrai — Bb/C gagne
    une lecture (C7sus4), il ne perd pas l'ancienne."""
    bars, ranges = _song([
        TRONC + [[_ch(Eb, "7")], [_ch(Bb)]],
        TRONC + [[_ch(Eb, "7")], [_ch(Bb, "", C)]],
    ])
    blk = _blk("A", [bars[i] for i in range(8)], ranges)
    _ireal_endings(blk, bars, GRID)
    assert "endings" not in blk


def test_g_sur_a_rejoint_encore_un_passage_en_g_maj7():
    """9zEl-FQLI4A (rapport d'or 2026-09-30) : `G/A` dans l'hôte, `G^7` dans
    le passage coupé. Remplacer la lecture écrite par la lecture sonnée
    cassait ce raccord ; les deux lectures le gardent."""
    cell = [[_ch(D, "^7")], [_ch(E, "-7")], [_ch(Gb, "-7")], [_ch(G, "", A)]]
    host = _blk("A", cell + cell, [(0, 7), (16, 23)])
    court = _blk("A", cell[:3] + [[_ch(G, "^7")]], [(40, 43)])
    out = _ireal_cascade([host, court], None, GRID, {"A": {"period": 4}})
    assert [b["label"] for b in out] == ["A"]
