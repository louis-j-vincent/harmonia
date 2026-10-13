"""Les changements de gamme sur le chart (Louis, 2026-10-13).

Règle : une ZONE de tonalité est montrée seulement si elle change la GAMME
(l'armure : la majeure relative, donc sol majeur → mi mineur n'en est pas
une) et si elle dure au moins une phrase (4 mesures de la grille du chart).
Un segment plus court est rendu à la zone qui le précède.
"""
import json

import pytest

from harmonia.key_zones import key_zones, PHRASE_BARS


def _seg(t0, t1, tonic, mode="major", x3=0.9):
    return {"i0": 0, "i1": 0, "t0": t0, "t1": t1, "tonic": tonic,
            "mode": mode, "x3": x3, "colours": []}


GRID = [float(i) for i in range(0, 101)]          # une mesure par seconde


def test_un_seul_segment_une_seule_zone():
    z = key_zones([_seg(0, 100, 7)], GRID)
    assert len(z) == 1 and z[0]["tonic"] == 7 and z[0]["mode"] == "major"


def test_modulation_d_une_phrase_est_gardee():
    z = key_zones([_seg(0, 60, 7), _seg(60, 100, 8)], GRID)
    assert [(q["tonic"], q["t0"]) for q in z] == [(7, 0), (8, 60)]
    assert z[1]["bars"] == 40


def test_segment_plus_court_qu_une_phrase_est_rendu_a_la_zone_d_avant():
    segs = [_seg(0, 50, 5), _seg(50, 50 + PHRASE_BARS - 1, 10, "minor"),
            _seg(50 + PHRASE_BARS - 1, 100, 5)]
    z = key_zones(segs, GRID)
    assert len(z) == 1 and z[0]["tonic"] == 5 and z[0]["t1"] == 100


def test_exactement_une_phrase_compte():
    segs = [_seg(0, 50, 5), _seg(50, 50 + PHRASE_BARS, 10, "minor"),
            _seg(50 + PHRASE_BARS, 100, 5)]
    z = key_zones(segs, GRID)
    assert [q["tonic"] for q in z] == [5, 10, 5]


def test_relatif_mineur_n_est_pas_un_changement_de_gamme():
    # sol majeur → mi mineur : même armure, aucune pastille
    z = key_zones([_seg(0, 50, 7), _seg(50, 100, 4, "minor")], GRID)
    assert len(z) == 1 and z[0]["tonic"] == 7


def test_premier_segment_court_est_rendu_a_la_zone_suivante():
    z = key_zones([_seg(0, 2, 2), _seg(2, 100, 9)], GRID)
    assert len(z) == 1 and z[0]["tonic"] == 9 and z[0]["t0"] == 0


def test_zones_couvrent_le_morceau_sans_trou():
    segs = [_seg(0, 40, 0), _seg(40, 41, 3), _seg(41, 70, 2), _seg(70, 100, 0)]
    z = key_zones(segs, GRID)
    assert [q["tonic"] for q in z] == [0, 2, 0]
    for a, b in zip(z, z[1:]):
        assert a["t1"] == b["t0"]
    assert z[1]["t0"] == 41


def test_sans_segments_ni_grille():
    assert key_zones([], GRID) == []
    assert key_zones(None, None) == []
    # sans grille de mesures : on compte sur les temps du chart
    z = key_zones([_seg(0, 10, 0), _seg(10, 30, 7)], None,
                  beat_times=[i * 0.5 for i in range(61)], bpb=4)
    assert [q["tonic"] for q in z] == [0, 7]


# ── servi par /api/chart-model, sans ré-analyse du morceau

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from harmonia import annotations as an
    from harmonia.server.routes import library as lib
    charts = tmp_path / "charts"
    charts.mkdir()
    meta = tmp_path / "chart_meta.json"
    meta.write_text("{}")
    monkeypatch.setattr(lib, "CHARTS_DIR", charts)
    monkeypatch.setattr(lib, "META_PATH", meta)
    monkeypatch.setattr(an, "ANNOT_DIR", tmp_path / "ann")
    (charts / "min_k.json").write_text(json.dumps({
        "file": "min_k", "key": {"tonic": 7, "mode": "major"},
        "keySegments": [_seg(0, 60, 7), _seg(60, 100, 8)],
        "barGrid": GRID, "nBars": 100,
        "sections": [{"id": "LA", "label": "A", "barRanges": [[0, 99]],
                      "reps": 1, "bars": [[] for _ in range(100)]}]}))
    from harmonia.server import app as app_mod
    return app_mod.create_app().test_client()


def test_route_sert_les_zones(client):
    m = client.get("/api/chart-model/min_k").get_json()
    assert [(z["tonic"], z["mode"], z["t0"]) for z in m["keyZones"]] == [
        (7, "major", 0), (8, "major", 60)]
