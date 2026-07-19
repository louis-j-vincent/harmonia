"""Confidence calibration on the live nnls24 path (2026-07-19 audit).

The deployed path showed a raw root-mass share as `confidence` — uncalibrated
for the displayed chord (joint ECE 0.145 on RWC) and never touched by the #26
machinery. These tests pin the shipped isotonic map and its loader.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

import harmonia.models.chord_pipeline_v1 as cp

MAP_PATH = Path(cp.__file__).parent / "nnls24_conf_calibration.npz"


@pytest.fixture(autouse=True)
def _reset_map_cache():
    cp._nnls24_conf_map = False
    yield
    cp._nnls24_conf_map = False


def test_map_file_ships_and_is_monotone():
    d = np.load(MAP_PATH)
    x, y = d["x"], d["y"]
    assert len(x) == len(y) >= 50
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(y) >= -1e-9), "isotonic map must be non-decreasing"
    assert 0.0 <= y.min() and y.max() <= 1.0
    # provenance fields the audit requires
    assert str(d["score_kind"]) == "nnls_root_softmax_max_oracle_block"
    assert float(d["oof_ece_joint"]) < 0.05  # the #26 gate, on the live path


def test_map_deflates_overconfident_scores():
    # The audit's headline: shown 0.97 vs joint accuracy 0.844 in the top bin.
    # The calibrated value at a 0.97 raw score must sit clearly below it.
    cp._nnls24_conf_map = False
    m = cp._get_nnls24_conf_map()
    assert m is not None
    cal = float(np.interp(0.97, m[0], m[1]))
    assert cal < 0.93
    assert cal > 0.5


def test_kill_switch_disables_map(monkeypatch):
    monkeypatch.setenv("HARMONIA_NNLS24_CALIB", "off")
    cp._nnls24_conf_map = False
    assert cp._get_nnls24_conf_map() is None


def test_loader_is_cached():
    m1 = cp._get_nnls24_conf_map()
    m2 = cp._get_nnls24_conf_map()
    assert m1 is m2
