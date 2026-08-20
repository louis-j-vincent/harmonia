"""La propagation d'un accord sur ses voisins est retirée (Louis, 2026-08-20).

« quand on annote un nouvel accord, ça se propage sur les accords suivants,
mais cette fonction est deprecated, enlève-la ».

Ce qui compte ici : qu'elle ne revienne pas par la fenêtre. La route existe
encore mais REFUSE — un 404 serait avalé en silence par le shell, et le chemin
« merge » passait par la même adresse en n'y recevant déjà qu'une réponse sans
effet.
"""
import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client():
    from harmonia_min.server import app
    app.config["TESTING"] = True
    return app.test_client()


def test_le_module_de_propagation_n_existe_plus():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("harmonia_min.context_rescore")


@pytest.mark.parametrize("route", ["/api/context_rescore/x", "/api/reinfer/x"])
def test_la_route_refuse_et_le_dit(client, route):
    r = client.post(route, json={"chords": [], "confirms": []})
    assert r.status_code == 410            # ni 200 muet, ni 404 avalé
    assert "retirée" in r.get_json()["error"]


def test_le_shell_n_appelle_plus_la_propagation():
    src = (REPO / "harmonia_min" / "app_shell.html").read_text(encoding="utf-8")
    for mort in ("runReinfer", "buildContextRescoreRequest",
                 "renderReinferAction("):
        assert mort not in src, f"{mort} est encore appelé dans le shell"


def test_les_candidats_du_modele_survivent_a_la_suppression():
    """`ireal_q_to_q5` vivait dans le module retiré et sert aux candidats."""
    from harmonia_min.span_rescore import ireal_q_to_q5
    assert (ireal_q_to_q5("-7"), ireal_q_to_q5("^7"), ireal_q_to_q5("h7"),
            ireal_q_to_q5("o"), ireal_q_to_q5("7")) == (1, 0, 3, 4, 2)
