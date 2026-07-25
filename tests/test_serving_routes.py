"""Behavioural pins for the extracted serving routes + the create_app() factory
(Phase 6c serving-route refactor).

These lock the two invariants the whole extraction was gated on:
  * ``create_app()`` builds a fresh, independent Flask app whose url_map carries
    the same routes as the module-level instance, with BARE, un-namespaced
    endpoint names (the ``name=""`` blueprint + the add_url_rule collector).
  * a few moved routes still answer their validation/404 paths as before.

Plus a regression pin for the Phase 6c ``/api/reinfer-from-beats`` numpy bugfix
(bare ``np`` with no import -> the audio-present path used to 500).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harmonia_server  # noqa: E402  (needs scripts/ on sys.path first)


@pytest.fixture
def client():
    # Build via the factory (not harmonia_server.app) so the test also exercises
    # create_app() producing a usable, isolated app.
    app = harmonia_server.create_app()
    app.testing = True
    return app.test_client()


def test_create_app_is_a_factory():
    a = harmonia_server.create_app()
    b = harmonia_server.create_app()
    assert a is not b  # fresh instance each call — no import-time singleton
    # reproduces the full route table of the module-level default instance
    assert len(list(a.url_map.iter_rules())) == len(
        list(harmonia_server.app.url_map.iter_rules())
    )


def test_endpoints_are_bare_and_unnamespaced():
    # name="" on the api blueprint + add_url_rule (endpoint defaults to the view
    # __name__) => every endpoint is bare; no "api." / "pages." prefixes. This is
    # exactly what keeps app.url_map byte-identical to the pre-refactor wiring and
    # keeps url_for("serve_audio")/url_for("index") resolving.
    eps = {r.endpoint for r in harmonia_server.create_app().url_map.iter_rules()}
    assert "serve_audio" in eps      # api blueprint (first batch)
    assert "api_analyze" in eps      # api blueprint (final cluster)
    assert "api_waveform_peaks" in eps
    assert "index" in eps            # page route via the collector
    assert all("." not in e for e in eps)


def test_moved_route_validation_paths(client):
    assert client.post("/api/analyze", json={}).status_code == 400
    assert client.post("/api/irealb-align", json={}).status_code == 400
    assert client.post("/api/reinfer/__nope__.html", json={}).status_code == 400
    assert client.get("/api/waveform-peaks/__nope__").status_code == 404
    assert client.get("/api/beat-grid-audio/__nope__").status_code == 404
    r = client.get("/api/section-labels/__nope__.html")
    assert r.status_code == 200 and r.get_json() == {"labels": {}}


def test_reinfer_from_beats_happy_path_no_500(client):
    """Regression pin for the Phase 6c numpy fix: the route used a bare ``np``
    with no numpy import, so its audio-present path raised NameError -> caught
    -> HTTP 500 on every real call. With a real cached-audio slug + empty beats
    it must now reach ``np.array`` and return a clean 400 (n_locked < 1), never
    a 500. (Empty beats short-circuit before any librosa/inference work, so this
    stays a fast unit-style check.)"""
    audio = sorted((REPO / "docs" / "audio").glob("*.m4a"))
    if not audio:
        pytest.skip("no cached audio available to exercise the audio-present path")
    slug = audio[0].stem
    r = client.post(f"/api/reinfer-from-beats/{slug}", json={"corrected_beat_times": []})
    body = r.get_data(as_text=True)
    assert r.status_code == 400, body            # NOT 500 (the old NameError)
    assert "n_locked_beats must be >= 1" in body
