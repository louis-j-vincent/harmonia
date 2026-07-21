"""SOTA downbeat-phase anchor (2026-07-21) — the pure-numeric pieces that
don't need real audio or the beat_this model. beat_this_downbeats/
sota_downbeat_phase's real-audio behaviour is validated in
scratchpad/downbeat_triangulation.py against 8 real library songs (not a
unit test — needs real audio + the model checkpoint download).
"""
from __future__ import annotations

import numpy as np

from harmonia.models.downbeat_anchor import _regularity


class TestRegularity:
    def test_perfectly_regular_scores_one(self):
        times = np.arange(0, 20, 1.82)
        assert _regularity(times) == 1.0

    def test_one_real_gap_barely_dents_the_score(self):
        # a spoken-bridge-style single dropout shouldn't tank an otherwise
        # rock-solid track (this is exactly what a coefficient-of-variation
        # metric gets backwards — see the module's own docstring).
        times = list(np.arange(0, 40, 1.82)) + [200.0]  # one huge gap at the end
        conf = _regularity(np.asarray(times))
        assert conf > 0.9

    def test_irregular_spacing_scores_low(self):
        rng = np.random.default_rng(0)
        times = np.cumsum(rng.uniform(1.5, 4.2, size=24))
        assert _regularity(times) < 0.5

    def test_too_few_points_is_zero_confidence(self):
        assert _regularity(np.array([0.0, 1.0, 2.0])) == 0.0


class TestSotaDownbeatPhaseAbstains:
    def test_abstains_when_beat_this_unavailable(self, monkeypatch):
        import harmonia.models.downbeat_anchor as mod

        def _boom(_path):
            raise RuntimeError("no model")
        monkeypatch.setattr(mod, "beat_this_downbeats", _boom)
        assert mod.sota_downbeat_phase("x.wav", bar_period=2.0) is None

    def test_abstains_on_low_confidence(self, monkeypatch):
        import harmonia.models.downbeat_anchor as mod
        monkeypatch.setattr(mod, "beat_this_downbeats",
                            lambda _path: (np.arange(10, dtype=float), 0.3))
        assert mod.sota_downbeat_phase("x.wav", bar_period=2.0) is None

    def test_returns_phi_and_sentinel_ratio_on_confidence(self, monkeypatch):
        import harmonia.models.downbeat_anchor as mod
        # downbeats at 1.0, 3.0, 5.0, ... with bar_period=2.0 -> phase 1.0 -> phi=1 of 2
        db = np.arange(1.0, 21.0, 2.0)
        monkeypatch.setattr(mod, "beat_this_downbeats", lambda _path: (db, 0.98))
        result = mod.sota_downbeat_phase("x.wav", bar_period=2.0, beats_per_bar=2)
        assert result is not None
        phi, ratio = result
        assert phi == 1
        assert ratio == 999.0
