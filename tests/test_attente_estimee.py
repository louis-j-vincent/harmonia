"""Estimation de la durée d'attente (UX #42) : la médiane des durées mesurées
quand il y en a, des constantes sinon. Tout en tmp_path."""
import json

from harmonia.server import jobs


def _histo(tmp_path, monkeypatch, rows):
    p = tmp_path / "job_timings.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(jobs, "TIMINGS_PATH", p)
    return p


def test_repli_sans_historique(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "TIMINGS_PATH", tmp_path / "rien.jsonl")
    a = jobs.estimate_wait("youtube", 240)
    b = jobs.estimate_wait("youtube", 480)
    assert 0 < a < b < 300  # croît avec la durée du morceau


def test_mediane_proportionnelle_a_la_duree(tmp_path, monkeypatch):
    rows = [{"kind": "youtube", "dur_s": 200, "resolve_s": 10, "raw_s": 20},
            {"kind": "youtube", "dur_s": 300, "resolve_s": 14, "raw_s": 30},
            {"kind": "youtube", "dur_s": 100, "resolve_s": 12, "raw_s": 10},
            {"kind": "fichier", "dur_s": 100, "resolve_s": 1, "raw_s": 999}]
    _histo(tmp_path, monkeypatch, rows)
    # taux médian = 0,1 s par seconde d'audio ; résolution = celle observée
    assert jobs.estimate_wait("youtube", 400, resolve_s=15) == 15 + 0.1 * 400
    # sans durée : médiane des totaux (30, 44, 22 -> 30)
    assert jobs.estimate_wait("youtube") == 30


def test_ligne_ecrite_au_brut(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "TIMINGS_PATH", tmp_path / "d" / "t.jsonl")
    jobs._note_timing({"duration_s": 120.04, "resolve_s": 3.21, "head_s": 1.5},
                      "fichier", 7.77)
    (ligne,) = (tmp_path / "d" / "t.jsonl").read_text().splitlines()
    assert json.loads(ligne)["raw_s"] == 7.8 and json.loads(ligne)["kind"] == "fichier"
