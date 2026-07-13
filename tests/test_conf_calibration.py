"""Tests for the display-layer confidence calibration (audit step 1b).

Covers the two new pieces in chord_pipeline_v1: the span root-posterior helper
(`_span_root_conf`, the root half of the fused confidence) and the isotonic
calibration singleton (`_get_conf_calibrator`, inert when the artifact is
absent). Decision-safety is by construction — both run at output-assembly
time, after every label and gate — so these tests pin the value contract only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import chord_pipeline_v1 as P  # noqa: E402


def _reset_cal_singleton():
    # Mission 4 replaced the single (_CONF_CAL, _CONF_CAL_LOADED) pair with a
    # per-domain cache dict — clear whichever exists so suite order can't leak
    # a previously-loaded map into these tests.
    if hasattr(P, "_CONF_CAL_CACHE"):
        P._CONF_CAL_CACHE.clear()
    if hasattr(P, "_CONF_CAL_LOADED"):
        P._CONF_CAL = None
        P._CONF_CAL_LOADED = False


# ── _span_root_conf ───────────────────────────────────────────────────────────

def test_span_root_conf_reads_the_labels_root_row():
    bt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    proba = np.zeros((4, 12))
    proba[:, 3] = 0.7          # D# / Eb row
    proba[:, 0] = 0.2
    rc = P._span_root_conf(proba, bt, 1.0, 3.0, "D#:maj7")
    assert rc == pytest.approx(0.7)
    rc = P._span_root_conf(proba, bt, 1.0, 3.0, "C:maj")
    assert rc == pytest.approx(0.2)


def test_span_root_conf_none_cases():
    bt = np.array([0.0, 1.0, 2.0])
    proba = np.ones((2, 12)) / 12
    assert P._span_root_conf(None, bt, 0.0, 1.0, "C:maj") is None      # no root model
    assert P._span_root_conf(proba, bt, 0.0, 1.0, "Cmaj") is None      # no colon
    assert P._span_root_conf(proba, bt, 0.0, 1.0, "H:maj") is None     # unknown root
    # span past the last beat clamps instead of crashing
    assert P._span_root_conf(proba, bt, 0.5, 99.0, "C:maj") is not None


# ── _get_conf_calibrator ──────────────────────────────────────────────────────

def test_calibrator_absent_returns_none(tmp_path, monkeypatch):
    _reset_cal_singleton()
    monkeypatch.setattr(P, "CONF_CALIBRATION_PATH", tmp_path / "missing.npz")
    assert P._get_conf_calibrator() is None
    _reset_cal_singleton()


def test_calibrator_interpolates_saved_breakpoints(tmp_path, monkeypatch):
    _reset_cal_singleton()
    path = tmp_path / "confidence_calibration.npz"
    np.savez(path, x=np.array([0.0, 0.5, 1.0]), y=np.array([0.1, 0.4, 0.9]))
    monkeypatch.setattr(P, "CONF_CALIBRATION_PATH", path)
    cal = P._get_conf_calibrator()
    assert cal is not None
    assert cal(0.0) == pytest.approx(0.1)
    assert cal(0.25) == pytest.approx(0.25)   # midway 0.1→0.4
    assert cal(1.0) == pytest.approx(0.9)
    assert cal(2.0) == pytest.approx(0.9)     # clipped beyond range
    _reset_cal_singleton()
