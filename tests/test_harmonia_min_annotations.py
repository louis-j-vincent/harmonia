"""Red-first tests for harmonia_min annotation persistence (handoff P1).

The shell POSTs /api/annotations/<file> on every chord lock and swallows the
404 silently, so every confirmed chord is lost on reload. These tests pin the
contract the server must honour, using REAL charts from state/charts/ as
fixtures (handoff: prefer real charts over synthetic JSON).

The payload the shell actually sends (app_shell.html saveAnnotations) is
    {annotator:"", chords:[{bar,beat,root,bass:-1,q,t0,t1}], merges:[]}
and `bass` is HARDCODED to -1 there — hence test_overlay_keeps_slash_bass:
following the old app's "fix wins if not None" rule would silently destroy
every slash chord (This Love bar 0 is G/B) the moment a user confirms it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harmonia_min import annotations as an

CHARTS = Path(__file__).resolve().parent.parent / "harmonia_min" / "state" / "charts"
THIS_LOVE = CHARTS / "min_maroon_5_this_love.json"


@pytest.fixture(autouse=True)
def _tmp_annot_dir(tmp_path, monkeypatch):
    """Never touch the real sidecar directory from tests."""
    monkeypatch.setattr(an, "ANNOT_DIR", tmp_path / "annotations")


def _model() -> dict:
    return json.loads(THIS_LOVE.read_text(encoding="utf-8"))


def _chords(model: dict):
    for sec in model["sections"]:
        for bar in sec["bars"]:
            for ch in bar:
                yield ch


def _first_chord(model: dict) -> dict:
    return next(iter(_chords(model)))


# ── sidecar io ──────────────────────────────────────────────────────────────

def test_save_then_load_roundtrip():
    doc = {"annotator": "", "chords": [{"bar": 0, "beat": 0, "root": 5,
                                        "bass": -1, "q": "-7",
                                        "t0": 1.08, "t1": 3.6}],
           "merges": []}
    saved = an.save_annotation("min_maroon_5_this_love", doc)
    assert saved["schema"] == 1
    assert saved["chart"] == "min_maroon_5_this_love"
    assert saved["modified"]  # server-stamped
    back = an.load_annotation("min_maroon_5_this_love")
    assert back["chords"] == doc["chords"]


def test_load_missing_returns_empty_doc():
    empty = an.load_annotation("min_nonexistent")
    assert empty["chords"] == [] and empty["merges"] == []


def test_save_leaves_no_temp_file_behind():
    an.save_annotation("min_maroon_5_this_love", {"chords": [], "merges": []})
    leftovers = [p.name for p in an.ANNOT_DIR.iterdir() if p.suffix != ".json"]
    assert leftovers == []


def test_save_does_not_clobber_on_unserializable_payload():
    """A bad write must leave the previous good sidecar intact (atomicity)."""
    good = {"chords": [{"bar": 0, "beat": 0, "root": 5, "bass": -1, "q": "-7"}],
            "merges": []}
    an.save_annotation("min_maroon_5_this_love", good)
    with pytest.raises(Exception):
        an.save_annotation("min_maroon_5_this_love",
                           {"chords": [{"bar": object()}], "merges": []})
    assert an.load_annotation("min_maroon_5_this_love")["chords"] == good["chords"]


# ── overlay ─────────────────────────────────────────────────────────────────

def test_overlay_marks_chord_confirmed():
    model = _model()
    ch = _first_chord(model)
    fix = {"bar": ch["bar"], "beat": ch["beat"], "root": (ch["root"] + 5) % 12,
           "bass": -1, "q": "-7", "t0": ch["t0"], "t1": ch["t1"]}
    out = an.overlay(model, {"chords": [fix], "merges": []})
    got = _first_chord(out)
    assert got["confirmed"] is True
    assert got["c"] == 1.0
    assert got["root"] == fix["root"] and got["q"] == "-7"
    assert got["nc"] is False


def test_overlay_keeps_slash_bass_when_client_sends_minus_one():
    """This Love bar 0 is G/B: confirming it must NOT destroy the /B."""
    model = _model()
    ch = _first_chord(model)
    assert ch["bass"] == 11, "fixture drifted: expected a slash chord at bar 0"
    fix = {"bar": ch["bar"], "beat": ch["beat"], "root": ch["root"],
           "bass": -1, "q": ch["q"], "t0": ch["t0"], "t1": ch["t1"]}
    out = an.overlay(model, {"chords": [fix], "merges": []})
    assert _first_chord(out)["bass"] == 11


def test_overlay_uses_fix_times_for_a_shrunk_split_half():
    model = _model()
    ch = _first_chord(model)
    mid = (ch["t0"] + ch["t1"]) / 2
    fix = {"bar": ch["bar"], "beat": ch["beat"], "root": ch["root"],
           "bass": -1, "q": ch["q"], "t0": ch["t0"], "t1": mid}
    out = an.overlay(model, {"chords": [fix], "merges": []})
    assert _first_chord(out)["t1"] == pytest.approx(mid)


def test_overlay_synthesizes_the_second_half_of_a_split():
    model = _model()
    ch = _first_chord(model)
    mid = (ch["t0"] + ch["t1"]) / 2
    bpb = model.get("bpb", 4)
    new_beat = (ch.get("beat") or 0) + bpb / 2
    fixes = [
        {"bar": ch["bar"], "beat": ch["beat"], "root": ch["root"], "bass": -1,
         "q": ch["q"], "t0": ch["t0"], "t1": mid},
        {"bar": ch["bar"], "beat": new_beat, "root": 2, "bass": -1, "q": "-7",
         "t0": mid, "t1": ch["t1"]},
    ]
    out = an.overlay(model, {"chords": fixes, "merges": []})
    bar0 = out["sections"][0]["bars"][0]
    assert len(bar0) == 2, f"split half not synthesized: {bar0}"
    assert [c["beat"] for c in bar0] == sorted(c["beat"] for c in bar0)
    synth = bar0[1]
    assert (synth["root"], synth["q"]) == (2, "-7")
    assert synth["confirmed"] is True and synth["c"] == 1.0
    assert synth["t0"] == pytest.approx(mid)


def test_overlay_echoes_merges_so_the_shell_cannot_wipe_them():
    model = _model()
    assert "merges" not in model, "fixture drifted"
    merges = [{"id": 1, "spans": [[0, 3]]}]
    out = an.overlay(model, {"chords": [], "merges": merges})
    assert out["merges"] == merges


def test_overlay_applies_to_every_folded_repeat_of_a_bar():
    """Folded sections replay one bar identity; a lock must hit all copies."""
    model = None
    for p in sorted(CHARTS.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        seen: dict[tuple, int] = {}
        for ch in _chords(m):
            key = (ch.get("bar"), ch.get("beat"))
            seen[key] = seen.get(key, 0) + 1
        if any(v > 1 for v in seen.values()):
            model, dup = m, next(k for k, v in seen.items() if v > 1)
            break
    if model is None:
        pytest.skip("no chart with folded duplicate bar identities")
    fix = {"bar": dup[0], "beat": dup[1], "root": 4, "bass": -1, "q": "-",
           "t0": 0.0, "t1": 1.0}
    out = an.overlay(model, {"chords": [fix], "merges": []})
    hits = [c for c in _chords(out)
            if (c.get("bar"), c.get("beat")) == dup and c.get("confirmed")]
    assert len(hits) > 1, "only one folded copy was rehydrated"


def test_overlay_is_a_noop_without_annotations():
    model = _model()
    before = json.dumps(model, sort_keys=True)
    out = an.overlay(json.loads(before), {"chords": [], "merges": []})
    out.pop("merges", None)
    assert json.dumps(out, sort_keys=True) == before


# ── routes ──────────────────────────────────────────────────────────────────

def test_post_annotations_then_chart_model_rehydrates():
    from harmonia_min import server as srv

    client = srv.app.test_client()
    model = _model()
    ch = _first_chord(model)
    body = {"annotator": "",
            "chords": [{"bar": ch["bar"], "beat": ch["beat"], "root": 2,
                        "bass": -1, "q": "-7", "t0": ch["t0"], "t1": ch["t1"]}],
            "merges": []}
    r = client.post("/api/annotations/min_maroon_5_this_love", json=body)
    assert r.status_code == 200
    assert r.get_json() is not None, "shell calls r.json() on the response"

    r2 = client.get("/api/chart-model/min_maroon_5_this_love")
    assert r2.status_code == 200
    got = _first_chord(r2.get_json())
    assert got["confirmed"] is True and got["root"] == 2 and got["q"] == "-7"


def test_delete_chart_removes_its_sidecar():
    from harmonia_min import server as srv

    an.save_annotation("min_ghost", {"chords": [], "merges": []})
    assert an._path("min_ghost").exists()
    srv.app.test_client().delete("/api/chart/min_ghost")
    assert not an._path("min_ghost").exists()
