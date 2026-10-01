"""jobs.tempo_factor_for + the marks-file merge (analyze.py::_read_mark /
_write_mark) — the ÷2/×2 override's persistence layer.

Mirrors tests/test_marks_source_of_truth.py's pattern for `bar1_for`: same
sidecar file (`state/human/marks/<stem>.json`), one field per correction
kind. The merge behaviour is the one easy silent-data-loss bug here — setting
bar1 must not erase an existing tempo_factor, and vice versa, since both
`/api/bar1` and `/api/tempo` write the SAME file.
"""
from __future__ import annotations

import dataclasses
import json

from harmonia.server import jobs
from harmonia.server.routes import analyze as analyze_routes


def _avec_marks_dir(monkeypatch, marks_dir):
    monkeypatch.setattr(jobs, "SETTINGS",
                        dataclasses.replace(jobs.SETTINGS, marks_dir=marks_dir))
    monkeypatch.setattr(analyze_routes, "SETTINGS",
                        dataclasses.replace(analyze_routes.SETTINGS,
                                            marks_dir=marks_dir))


def test_tempo_factor_for_lit_le_fichier_de_marque(tmp_path, monkeypatch):
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "monstem.json").write_text(
        json.dumps({"tempo_factor": 0.5}), encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)

    assert jobs.tempo_factor_for("min_monstem") == 0.5


def test_tempo_factor_for_absent_rend_none(tmp_path, monkeypatch):
    _avec_marks_dir(monkeypatch, tmp_path / "vide")
    assert jobs.tempo_factor_for("min_jamais_marque") is None


def test_writing_tempo_factor_preserves_existing_bar1(tmp_path, monkeypatch):
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "monstem.json").write_text(
        json.dumps({"bar1": 5.59}), encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)

    analyze_routes._write_mark("monstem", tempo_factor=0.5)

    doc = json.loads((marks_dir / "monstem.json").read_text(encoding="utf-8"))
    assert doc == {"bar1": 5.59, "tempo_factor": 0.5}


def test_writing_bar1_preserves_existing_tempo_factor(tmp_path, monkeypatch):
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "monstem.json").write_text(
        json.dumps({"tempo_factor": 0.5}), encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)

    analyze_routes._write_mark("monstem", bar1=5.59)

    doc = json.loads((marks_dir / "monstem.json").read_text(encoding="utf-8"))
    assert doc == {"tempo_factor": 0.5, "bar1": 5.59}


# ── « Annuler » (2026-10-01, audit UX) ──────────────────────────────────────
# Poser la mesure 1 ou ÷2/×2 relance toute l'analyse. L'annulation ne fait
# rien de neuf côté serveur : ÷2 puis ×2 revient au départ (l'accumulation
# existante), et /api/bar1 rend la marque d'AVANT pour la reposer ou l'effacer.

def _client_avec_chart(monkeypatch, tmp_path):
    from harmonia.server import app as srv
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    _avec_marks_dir(monkeypatch, marks_dir)
    charts = tmp_path / "charts"
    charts.mkdir()
    (charts / "min_monstem.json").write_text(
        json.dumps({"audio_url": "audio/monstem.wav", "title": "T"}), encoding="utf-8")
    monkeypatch.setattr(analyze_routes, "CHARTS_DIR", charts)
    calls = []
    monkeypatch.setattr(analyze_routes, "start_job",
                        lambda stem, **kw: (calls.append((stem, kw)), "job1")[1])
    return srv.app.test_client(), marks_dir, calls


def test_bar1_rend_la_marque_precedente(tmp_path, monkeypatch):
    client, marks_dir, calls = _client_avec_chart(monkeypatch, tmp_path)
    (marks_dir / "monstem.json").write_text(json.dumps({"bar1": 5.59}), encoding="utf-8")
    r = client.post("/api/bar1/min_monstem", json={"t": 7.25})
    assert r.get_json() == {"job_id": "job1", "previous_bar1": 5.59}
    assert json.loads((marks_dir / "monstem.json").read_text())["bar1"] == 7.25
    assert calls[-1][1]["bar1_time"] == 7.25


def test_bar1_sans_marque_avant_rend_none(tmp_path, monkeypatch):
    client, marks_dir, _ = _client_avec_chart(monkeypatch, tmp_path)
    r = client.post("/api/bar1/min_monstem", json={"t": 3.0})
    assert r.get_json()["previous_bar1"] is None


def test_bar1_clear_retire_la_marque_et_garde_le_tempo(tmp_path, monkeypatch):
    client, marks_dir, calls = _client_avec_chart(monkeypatch, tmp_path)
    (marks_dir / "monstem.json").write_text(
        json.dumps({"bar1": 5.59, "tempo_factor": 0.5}), encoding="utf-8")
    r = client.post("/api/bar1/min_monstem", json={"clear": True})
    assert r.status_code == 200 and r.get_json()["previous_bar1"] == 5.59
    assert json.loads((marks_dir / "monstem.json").read_text()) == {"tempo_factor": 0.5}
    assert calls[-1][1].get("bar1_time") is None


def test_tempo_puis_son_inverse_revient_au_depart(tmp_path, monkeypatch):
    client, marks_dir, calls = _client_avec_chart(monkeypatch, tmp_path)
    client.post("/api/tempo/min_monstem", json={"factor": 0.5})
    assert json.loads((marks_dir / "monstem.json").read_text())["tempo_factor"] == 0.5
    client.post("/api/tempo/min_monstem", json={"factor": 2.0})
    assert "tempo_factor" not in json.loads((marks_dir / "monstem.json").read_text())
    assert calls[-1][1]["tempo_factor"] is None
