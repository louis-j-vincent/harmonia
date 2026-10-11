"""tests/test_pales_meme_moteur.py — les suggestions pâles = le calcul de Valider.

Louis, 2026-10-11, sur Evening Star : « les sections proposées en pâle
c'était du n'importe quoi, comment ça se fait que l'annotation pâle ne
correspond pas à l'annotation finalement décidée ? ». Les pâles passaient par
`section_tool.find_repeats`, qui cherche une CELLULE (la plus petite taille
cohérente : 2 mesures sur Evening Star) ; Valider passe par
`simulation.cherche_brique`, qui cherche la brique entière. Ce test épingle que
`/api/section-repeats` répond désormais avec `cherche_brique`, brique exclue,
et sans les mesures déjà prises.
"""
from __future__ import annotations

import json

import pytest

from harmonia.sections import simulation as sim_mod


def _chart(n=40):
    grid = [round(i * 2.0, 3) for i in range(n + 1)]
    chords = [{"root": i % 4 * 3, "q": "", "bass": -1, "nc": False,
               "t0": grid[i], "t1": grid[i + 1]} for i in range(n)]
    return {"title": "bidon", "barGrid": grid, "nBars": n,
            "prompter": {"chords": chords}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": []}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from harmonia.server import app as app_mod
    from harmonia.server.routes import sections as sec_mod
    (tmp_path / "min_bidon.json").write_text(json.dumps(_chart()), encoding="utf-8")
    (tmp_path / "absent.m4a").write_bytes(b"")
    monkeypatch.setattr(sec_mod, "CHARTS_DIR", tmp_path)
    monkeypatch.setattr(sec_mod, "AUDIO_DIR", tmp_path)
    appels = []

    def faux(chart, b0, b1, ad, label="", trace=None):
        appels.append((b0, b1))
        # la brique elle-même, une reprise libre, une reprise sur des mesures
        # prises, une transposée
        return ([(b0, b1, 1.0, 0), (20, 27, 0.93, 0), (30, 37, 0.91, 0),
                 (10, 17, 0.95, 3)],
                {"label": label, "m0": b0, "m1": b1, "n_reprises": 3, "raison": None})
    monkeypatch.setattr(sim_mod, "cherche_brique", faux)
    c = app_mod.create_app().test_client()
    c.appels = appels
    return c


def test_les_pales_viennent_de_cherche_brique(client):
    out = client.post("/api/section-repeats/min_bidon",
                      json={"b0": 2, "b1": 9, "claimed": [32, 33]}).get_json()
    assert client.appels == [(2, 9)], "les pâles doivent appeler le moteur de Valider"
    assert out["cell_bars"] == 8
    occ = [(o["b0"], o["b1"], o["rot"], o["source"]) for o in out["occurrences"]]
    # la brique n'est pas proposée ; 30–37 touche une mesure prise ; la
    # transposée reste (le client l'écarte, règle « même tonalité seulement »)
    assert (20, 27, 0, "found") in occ
    assert (10, 17, 3, "found-transposed") in occ
    assert all(not (a <= 2 <= b) for a, b, _r, _s in occ)
    assert all(not (a <= 32 <= b) for a, b, _r, _s in occ)
