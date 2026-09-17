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
