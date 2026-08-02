"""Red-first tests for the harmonia_min context-rescore endpoint (handoff P2).

Wire contract, taken from the shell (app_shell.html buildContextRescoreRequest
/ applyResp), not from a doc:

  request   {chords:[{t0,t1,root,q}], confirms:[{t0,t1,root,q}]}
            — the FULL displayed chart plus the locks; `q` is the iReal tail
  response  {key, tempo_bpm, n_changed, diff:[{index,start_s,end_s,
             old_label,new_label,old_confidence,new_confidence}], chords:[]}
            — labels are HARTE ("C:maj", "D:min7"), which parseLabel maps back

The load-bearing invariant: **confirms are hard evidence**. A confirmed span
must come back exactly as the user locked it, never rescored away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CHARTS = Path(__file__).resolve().parent.parent / "harmonia_min" / "state" / "charts"
THIS_LOVE = CHARTS / "min_maroon_5_this_love.json"


def _displayed_chords(model: dict) -> list[dict]:
    """The shell's buildContextRescoreRequest, in Python: flat, time-sorted,
    no-chord spans dropped."""
    out, seen = [], set()
    for sec in model.get("sections", []):
        for bar in sec.get("bars", []):
            for ch in bar:
                if ch.get("nc") or ch.get("t0") is None or ch.get("t1") is None:
                    continue
                key = (round(float(ch["t0"]), 3), round(float(ch["t1"]), 3))
                if key in seen or key[1] <= key[0]:
                    continue
                seen.add(key)
                out.append({"t0": ch["t0"], "t1": ch["t1"],
                            "root": ch["root"], "q": ch.get("q", "")})
    out.sort(key=lambda c: c["t0"])
    return out


@pytest.fixture(scope="module")
def chords() -> list[dict]:
    return _displayed_chords(json.loads(THIS_LOVE.read_text(encoding="utf-8")))


@pytest.fixture()
def client():
    from harmonia_min import server as srv
    return srv.app.test_client()


def test_zero_confirms_changes_nothing(client, chords):
    """Differential design: no locks -> baseline == locked run, by construction."""
    r = client.post("/api/context_rescore/min_maroon_5_this_love",
                    json={"chords": chords, "confirms": []})
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_changed"] == 0
    assert body["diff"] == []


def test_confirmed_span_comes_back_bit_identical(client, chords):
    """Hard evidence: the locked span must never be rescored away."""
    j = len(chords) // 2
    lock = {**chords[j], "root": (chords[j]["root"] + 5) % 12, "q": "-7"}
    r = client.post("/api/context_rescore/min_maroon_5_this_love",
                    json={"chords": chords, "confirms": [lock]})
    assert r.status_code == 200
    body = r.get_json()
    hit = [d for d in body["diff"]
           if abs(d["start_s"] - lock["t0"]) < 1e-6]
    if hit:  # if reported at all, it must be reported AS LOCKED
        assert hit[0]["new_label"] == "F:min7" or hit[0]["new_label"].endswith(":min7")
    # and the returned chord list must carry the lock verbatim
    same = [c for c in body["chords"] if abs(c["t0"] - lock["t0"]) < 1e-6]
    assert same and same[0]["root"] == lock["root"] and same[0]["q"] == lock["q"]


def test_response_shape_is_what_the_shell_parses(client, chords):
    j = len(chords) // 2
    lock = {**chords[j], "root": (chords[j]["root"] + 7) % 12}
    r = client.post("/api/context_rescore/min_maroon_5_this_love",
                    json={"chords": chords, "confirms": [lock]})
    body = r.get_json()
    for k in ("key", "tempo_bpm", "n_changed", "diff", "chords"):
        assert k in body, f"missing {k}"
    for d in body["diff"]:
        assert set(d) >= {"index", "start_s", "end_s", "old_label", "new_label",
                          "old_confidence", "new_confidence"}
        # Harte on the wire: "<Note>:<quality>"
        assert ":" in d["new_label"] and ":" in d["old_label"]
        assert d["end_s"] > d["start_s"]


def test_reinfer_alias_routes_to_the_same_handler(client, chords):
    r = client.post("/api/reinfer/min_maroon_5_this_love",
                    json={"chords": chords, "confirms": []})
    assert r.status_code == 200
    assert r.get_json()["n_changed"] == 0


def test_unknown_chart_is_a_clean_404(client):
    r = client.post("/api/context_rescore/min_does_not_exist",
                    json={"chords": [], "confirms": []})
    assert r.status_code == 404
    assert r.get_json()["error"]


def test_empty_chords_does_not_explode(client):
    r = client.post("/api/context_rescore/min_maroon_5_this_love",
                    json={"chords": [], "confirms": []})
    assert r.status_code == 200
    assert r.get_json()["n_changed"] == 0
