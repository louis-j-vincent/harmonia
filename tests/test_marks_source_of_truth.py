"""tests/test_marks_source_of_truth.py — la marque « Set bar 1 » se lit dans
`state/human/marks/`, jamais dans le champ `bar1` d'un chart.

Avant le sprint 15, la marque de Louis ne vivait que dans le champ `bar1` du
chart régénérable — ce qui l'a fait disparaître le 2026-08-13 dès qu'un
rebake réécrivait le chart sans la repasser (voir `tests/test_bar1_persistence.py`,
qui pin le contrat de `analyze()`/du rebake ; celui-ci pin la couche
au-dessus : où `_run_job` va CHERCHER la marque avant même d'appeler
`analyze()`). Ce test épingle : `jobs.bar1_for(file_key)` lit le fichier de
marque, et un chart SANS `bar1` (donc qui ne peut plus la fournir) ne change
rien au résultat.

`SETTINGS` est un dataclass FROZEN : on ne peut pas faire
`monkeypatch.setattr(jobs.SETTINGS, "marks_dir", ...)` (lève
`FrozenInstanceError`). On remplace donc le NOM `jobs.SETTINGS` par un objet
équivalent (`dataclasses.replace`), ce que `bar1_for` relit à chaque appel
puisque c'est une globale du module.
"""
from __future__ import annotations

import dataclasses
import json

from harmonia.server import jobs


def _avec_marks_dir(monkeypatch, marks_dir):
    monkeypatch.setattr(jobs, "SETTINGS",
                        dataclasses.replace(jobs.SETTINGS, marks_dir=marks_dir))


def test_bar1_for_lit_le_fichier_de_marque(tmp_path, monkeypatch):
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "monstem.json").write_text(
        json.dumps({"bar1": 12.5}), encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)

    assert jobs.bar1_for("min_monstem") == 12.5


def test_bar1_for_ignore_le_champ_bar1_du_chart(tmp_path, monkeypatch):
    """Un chart SANS `bar1` (ou même un vieux chart qui n'a jamais porté ce
    champ) n'empêche pas la marque de ressortir : la seule source lue ici est
    le fichier de marque, jamais le chart."""
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "monstem.json").write_text(
        json.dumps({"bar1": 12.5}), encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    (charts_dir / "min_monstem.json").write_text(
        json.dumps({"file": "min_monstem", "nBars": 1}), encoding="utf-8")
    # `bar1_for` ne prend même pas `charts_dir` en paramètre — la preuve que
    # le champ `bar1` du chart n'entre pas en ligne de compte.
    assert jobs.bar1_for("min_monstem") == 12.5


def test_bar1_for_absent_rend_none(tmp_path, monkeypatch):
    _avec_marks_dir(monkeypatch, tmp_path / "vide")
    assert jobs.bar1_for("min_jamais_marque") is None


def test_bar1_for_fichier_illisible_rend_none(tmp_path, monkeypatch):
    marks_dir = tmp_path / "marks"
    marks_dir.mkdir()
    (marks_dir / "casse.json").write_text("{pas du json", encoding="utf-8")
    _avec_marks_dir(monkeypatch, marks_dir)
    assert jobs.bar1_for("min_casse") is None
