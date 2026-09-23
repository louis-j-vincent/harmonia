"""La chaîne complète : une URL iReal → un chart que la bibliothèque sert.

Le test de `test_irealb_import.py` épingle la LECTURE de la grille ; celui-ci
épingle le geste, de bout en bout, à travers le serveur — parce que c'est là
que ça a cassé : l'écran de recherche appelait `importIrealChart`, une
fonction qui n'existait nulle part, et aucun résultat iReal ne pouvait
s'ouvrir.

Tout se passe dans un `tmp_path` : aucun test n'écrit dans la vraie
bibliothèque de Louis.
"""
from __future__ import annotations

import dataclasses
import json
import urllib.parse

import pytest
from pyRealParser.pyRealParser import Tune

from harmonia.settings import SETTINGS

AUTUMN = ("*A{T44C-7XyQ|F7XyQ|Bb^7XyQ|Eb^7XyQ|Ah7XyQ|D7b13XyQ|G-6XyQKcl  }"
          "*B[Ah7XyQ|D7b13XyQ|G-6XyQKcl LZC-7XyQ|F7XyQ|Bb^7XyQ|Eb^7XyQ]"
          "*C[Ah7XyQ|D7b13XyQ|G-7 Gb7LZF-7 E7LZAh7XyQ|D7b13XyQ|G-6XyQKcl  Z")


def _url(chords: str, *, title="Autumn Leaves", composer="Kosma Joseph",
         key="G-") -> str:
    scrambled = Tune._unscramble_chord_string(chords)
    raw = "=".join([title, composer, "Medium Swing", key,
                    Tune._chords_prefix + scrambled]) + "="
    return "irealb://" + urllib.parse.quote(raw, safe="=")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    reglages = dataclasses.replace(
        SETTINGS,
        charts_dir=tmp_path / "charts",
        human_dir=tmp_path / "human",
        annotations_dir=tmp_path / "human" / "annotations",
        chart_meta_path=tmp_path / "human" / "chart_meta.json",
    )
    from harmonia import annotations as an
    from harmonia.server import jobs
    from harmonia.server.routes import irealb, library as lib
    for mod in (jobs, irealb, lib):
        monkeypatch.setattr(mod, "CHARTS_DIR", reglages.charts_dir, raising=False)
        monkeypatch.setattr(mod, "META_PATH", reglages.chart_meta_path, raising=False)
    monkeypatch.setattr(lib, "SETTINGS", reglages)
    monkeypatch.setattr(an, "ANNOT_DIR", reglages.annotations_dir)
    reglages.charts_dir.mkdir(parents=True)
    reglages.human_dir.mkdir(parents=True, exist_ok=True)

    from harmonia.server import app as app_mod
    return app_mod.create_app().test_client()


def test_importer_puis_ouvrir_le_chart(client):
    r = client.post("/api/irealb-import",
                    json={"irealb_url": _url(AUTUMN), "composer": "Kosma Joseph"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["file"] == "ireal_autumn_leaves"
    assert d["nBars"] == 32 and d["sections"] == 3

    # Le chart s'ouvre par la MÊME route que tous les autres — un chart
    # importé n'est pas un objet à part dans l'app.
    m = client.get(f"/api/chart-model/{d['file']}").get_json()
    assert m["title"] == "Autumn Leaves"
    assert m["keyName"] == "G minor"
    assert [s["label"] for s in m["sections"]] == ["A", "B", "C"]
    assert m["sections"][0]["reps"] == 2
    assert m["audio_url"] == ""


def test_il_apparait_dans_la_bibliotheque_sans_audio(client):
    client.post("/api/irealb-import", json={"irealb_url": _url(AUTUMN)})
    ligne = next(c for c in client.get("/api/library").get_json()["charts"]
                 if c["file"] == "ireal_autumn_leaves")
    assert ligne["hasAudio"] is False      # la ligne dit « · no audio »
    assert ligne["bars"] == 32
    assert ligne["artist"] == "Kosma Joseph"   # le compositeur iReal
    assert ligne["key"] == {"tonic": 7, "mode": "minor"}


def test_deux_imports_du_meme_titre_ne_s_ecrasent_pas(client):
    """Écraser coûterait les annotations posées sur le premier : elles sont
    indexées par stem (`state/human/annotations/<stem>.json`)."""
    a = client.post("/api/irealb-import", json={"irealb_url": _url(AUTUMN)}).get_json()
    b = client.post("/api/irealb-import", json={"irealb_url": _url(AUTUMN)}).get_json()
    assert a["file"] == "ireal_autumn_leaves"
    assert b["file"] == "ireal_autumn_leaves_2"
    files = {c["file"] for c in client.get("/api/library").get_json()["charts"]}
    assert {a["file"], b["file"]} <= files


def test_une_url_illisible_repond_400_et_n_ecrit_rien(client, tmp_path):
    r = client.post("/api/irealb-import", json={"irealb_url": "irealb://n_importe_quoi"})
    assert r.status_code == 400 and "error" in r.get_json()
    assert client.get("/api/library").get_json()["charts"] == []


def test_sans_url_c_est_400(client):
    assert client.post("/api/irealb-import", json={}).status_code == 400


def test_l_aller_retour_avec_l_export_garde_les_accords(client):
    """`irealb_export.py` fait le chemin inverse. Un import puis un export
    doivent parler du même morceau — sinon les deux moitiés de la passerelle
    iReal ont divergé."""
    d = client.post("/api/irealb-import", json={"irealb_url": _url(AUTUMN)}).get_json()
    out = client.get(f"/api/irealb-export/{d['file']}").get_json()
    assert out["url"].startswith("irealb://")
    renvoye = Tune(urllib.parse.unquote(out["url"])[len("irealb://"):])
    assert renvoye.title == "Autumn Leaves"
    # l'export déroule les reprises : 32 mesures jouées, pas 24 écrites
    assert len(renvoye.measures_as_strings) == 32
    assert renvoye.measures_as_strings[0] == "C-7"
    # L'export écrit TOUT en dièses (`irealb_export.NOTE_SHARP`), donc le
    # B♭ du morceau revient en A♯ : même hauteur, mauvaise orthographe pour
    # un sol mineur. Ce test compare donc la HAUTEUR, et la graphie est
    # notée comme un défaut de l'exportateur (docs/known_issues.md), pas
    # comme un défaut de l'import.
    from harmonia.integrations.irealb_import import parse_root
    assert parse_root(renvoye.measures_as_strings[2][:2]) == parse_root("Bb")
    assert renvoye.measures_as_strings[2].endswith("^7")


# ── la recherche qui arrive au fur et à mesure (Louis, 2026-09-18) ──────────

def test_la_recherche_en_flux_rend_une_ligne_par_lot(client, monkeypatch):
    """NDJSON : une ligne JSON par lot, terminée par `{"done": true}`. Les
    standards (0,4 s) ne doivent pas attendre le forum (4 à 6 s)."""
    from harmonia.integrations import irealb_fetcher as f

    def faux(_q, max_results=8):
        yield "standards", [{"title": "Autumn Leaves", "irealb_url": "x"}]
        yield "forum", [{"title": "Autumn Leaves (RB5)", "irealb_url": "y"}]
    monkeypatch.setattr(f, "search_community_stream", faux)

    r = client.post("/api/irealb-search-stream", json={"title": "autumn"})
    assert r.status_code == 200
    assert r.mimetype == "application/x-ndjson"
    lignes = [json.loads(l) for l in r.get_data(as_text=True).splitlines() if l.strip()]
    assert [d.get("source") for d in lignes[:2]] == ["standards", "forum"]
    assert lignes[-1] == {"done": True}


def test_le_flux_sans_titre_est_400(client):
    assert client.post("/api/irealb-search-stream", json={}).status_code == 400


def test_un_forum_en_panne_ne_mange_pas_les_standards(monkeypatch):
    """Le forum est un SUPPLÉMENT : son échec ne doit jamais effacer ce que
    les standards ont déjà rendu."""
    from harmonia.integrations import irealb_fetcher as f
    monkeypatch.setattr(f, "_search_main_playlists",
                        lambda q, n: [{"title": "Blue Bossa", "irealb_url": "x"}])

    def casse(*_a, **_k):
        raise RuntimeError("forum down")
        yield
    monkeypatch.setattr(f, "_search_forum_stream", casse)
    lots = list(f.search_community_stream("blue bossa"))
    assert lots and lots[0][0] == "standards"
    assert [r["title"] for r in lots[0][1]] == ["Blue Bossa"]


def test_le_flux_ne_repete_jamais_le_meme_titre(monkeypatch):
    from harmonia.integrations import irealb_fetcher as f
    monkeypatch.setattr(f, "_search_main_playlists",
                        lambda q, n: [{"title": "Autumn Leaves", "irealb_url": "x"}])
    monkeypatch.setattr(f, "_search_forum_stream",
                        lambda q, n, **k: iter([[{"title": "autumn leaves", "irealb_url": "y"},
                                                 {"title": "Autumn Leaves 3", "irealb_url": "z"}]]))
    titres = [r["title"] for _s, lot in f.search_community_stream("autumn") for r in lot]
    assert titres == ["Autumn Leaves", "Autumn Leaves 3"]
