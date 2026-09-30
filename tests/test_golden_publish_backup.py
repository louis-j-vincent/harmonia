import json

from tools import golden


def test_publish_copie_la_bibliotheque_dans_backup_avant_de_remplacer(tmp_path, monkeypatch):
    lib = tmp_path / "charts"
    lib.mkdir()
    (lib / "min_x.json").write_text(json.dumps({"v": "ancien"}), encoding="utf-8")
    out, bak = tmp_path / "out", tmp_path / "bak"
    bak.mkdir()

    monkeypatch.setattr(golden, "charger_moteur", lambda _e: {"charts": lib})

    def cuire(_eng, k, _charts, dest):
        (dest / f"{k}.json").write_text(json.dumps({"v": "neuf"}), encoding="utf-8")
        return {"status": "ok", "s": 0, "nBars": 1, "keyName": "C", "nSections": 1}

    monkeypatch.setattr(golden, "cuire", cuire)
    assert golden.main(["--engine", "harmonia", "--out", str(out),
                        "--publish", "--backup", str(bak)]) == 0
    assert json.loads((lib / "min_x.json").read_text())["v"] == "neuf"
    assert json.loads((bak / "min_x.json").read_text())["v"] == "ancien"
