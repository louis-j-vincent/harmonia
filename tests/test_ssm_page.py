"""tests/test_ssm_page.py — la charge utile de la page SSM doit être cohérente.

Ce que ces tests protègent : la page place sa tête de lecture en convertissant
un temps en case, et une case en pixel. Si `t`, `n` et la matrice ne sont plus
d'accord entre eux, rien ne plante — la tête se pose simplement à côté, et ça
ne se voit qu'à l'oreille. C'est exactement le genre de décalage silencieux que
la règle #1 du projet demande de tester au niveau le plus bas.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from harmonia_min import ssm_page as SP


def test_demi_mesures_coupe_chaque_mesure_en_deux():
    assert SP._demi_mesures([0.0, 2.0, 4.0]) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_frontieres_disent_leur_source(tmp_path, monkeypatch):
    monkeypatch.setattr(SP, "SECTIONS_DIR", tmp_path)
    chart = {"sections": [{"label": "A", "barRanges": [[0, 7], [8, 15]]},
                          {"label": "B", "barRanges": [[16, 23]]}]}
    front, source = SP._frontieres(chart, "x")
    assert source == "le chart"
    assert [f["mesure"] for f in front] == [0, 8, 16]

    (tmp_path / "x.json").write_text(
        '{"sections": [{"label": "B", "b0": 9, "b1": 16},'
        ' {"label": "A", "b0": 1, "b1": 8}]}', encoding="utf-8")
    front, source = SP._frontieres(chart, "x")
    assert source == "tes annotations"          # sa main passe avant le chart
    assert [f["mesure"] for f in front] == [1, 9]


def test_charge_utile_coherente(monkeypatch, tmp_path):
    """`t`, `n` et la matrice doivent décrire la MÊME grille."""
    n_mes = 12
    grid = [float(i) for i in range(n_mes + 1)]
    chart = {"barGrid": grid, "nBars": n_mes, "title": "T",
             "audio_url": "/audio/x.m4a", "file": "min_x", "bar1": 1.0,
             "sections": [{"label": "A", "barRanges": [[0, 5]]},
                          {"label": "B", "barRanges": [[6, 11]]}]}
    (tmp_path / "x.m4a").write_bytes(b"pas vraiment un m4a")
    monkeypatch.setattr(SP, "SECTIONS_DIR", tmp_path / "vide")

    fake = np.zeros((400, 36), dtype=float)     # postérieurs musx bidon
    fake[:, 0] = 1.0
    import harmonia_min.musx as MX
    monkeypatch.setattr(MX, "frame_posteriors", lambda *a, **k: (fake,))

    d = SP.donnees(chart, audio_dir=tmp_path)
    assert d is not None
    assert d["n"] == n_mes * SP.PAR_MESURE      # une case par demi-mesure
    assert len(d["t"]) == d["n"] + 1            # n cases -> n+1 bornes
    assert len(base64.b64decode(d["m"])) == d["n"] ** 2
    assert d["t"][0] == grid[0] and d["t"][-1] == grid[-1]
    assert [f["mesure"] for f in d["frontieres"]] == [0, 6]
    assert d["mesure1"] == 1                    # sa marque, en index de mesure
    assert d["etirement"][0] <= d["etirement"][1]


def test_chart_trop_court_rend_none():
    assert SP.donnees({"barGrid": [0.0, 1.0], "audio_url": "/audio/x.m4a"}) is None


def test_page_est_autonome(monkeypatch, tmp_path):
    """Aucun appel réseau sortant : la page doit vivre hors ligne."""
    n_mes = 12
    chart = {"barGrid": [float(i) for i in range(n_mes + 1)], "nBars": n_mes,
             "audio_url": "/audio/x.m4a", "file": "min_x", "sections": []}
    (tmp_path / "x.m4a").write_bytes(b"x")
    monkeypatch.setattr(SP, "SECTIONS_DIR", tmp_path / "vide")
    fake = np.zeros((400, 36), dtype=float); fake[:, 0] = 1.0
    import harmonia_min.musx as MX
    monkeypatch.setattr(MX, "frame_posteriors", lambda *a, **k: (fake,))

    html = SP.page_html(chart, audio_dir=tmp_path, base_url="http://h:7772")
    assert html is not None
    assert "http://h:7772/audio/x.m4a" in html
    assert "requestAnimationFrame" not in html   # piège iPhone n°1
    assert 'addEventListener("timeupdate"' in html
    for interdit in ("<script src", "<link rel=\"stylesheet\"", "cdn."):
        assert interdit not in html


@pytest.mark.parametrize("mauvais", [None, "bof", float("nan")])
def test_marque_illisible_ne_casse_rien(monkeypatch, tmp_path, mauvais):
    n_mes = 12
    chart = {"barGrid": [float(i) for i in range(n_mes + 1)], "nBars": n_mes,
             "audio_url": "/audio/x.m4a", "bar1": mauvais, "sections": []}
    (tmp_path / "x.m4a").write_bytes(b"x")
    monkeypatch.setattr(SP, "SECTIONS_DIR", tmp_path / "vide")
    fake = np.zeros((400, 36), dtype=float); fake[:, 0] = 1.0
    import harmonia_min.musx as MX
    monkeypatch.setattr(MX, "frame_posteriors", lambda *a, **k: (fake,))
    d = SP.donnees(chart, audio_dir=tmp_path)
    assert d is not None and d["mesure1"] in (None, 0)
