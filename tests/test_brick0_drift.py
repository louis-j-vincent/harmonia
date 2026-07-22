"""Unit tests for the Brick 0 aligner-v5 TEMPO-DRIFT detector + piecewise-constant-
per-section tempo model (scripts/brick0_propose.py).

Audio-free: they exercise the classifier/grid math on synthetic offset ramps that
mirror the real batch-1 measurements, so the GUARDRAIL (fire only on a monotone,
materially-large, well-fit ramp; leave constant/rubato songs untouched) is locked in
without a Beat This! / librosa run.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import brick0_propose as bp  # noqa: E402


def _rows(ts, deltas, agr=0.35, n_beats=64):
    """(t_center, delta, agr_best, n_beats) tuples in the offset_ramp format."""
    return [(float(t), float(d), float(agr), n_beats) for t, d in zip(ts, deltas)]


# ── the DRIFT case: a clean monotone ramp (Blue Bossa: +0.45 -> -1.25 over 22) ──

def test_monotone_ramp_classified_as_drift():
    bp_grid = 60.0 / 170.75                      # ~0.3515 s (Blue Bossa)
    ts = np.linspace(22.0, 494.0, 22)
    deltas = np.linspace(0.45, -1.25, 22)
    deltas += 0.05 * np.sin(np.arange(22))       # mild jitter, still monotone-ish
    d = bp.detect_drift(_rows(ts, deltas), bp_grid)
    assert d["classification"] == "drift"
    assert d["apply"] is True
    assert abs(d["pearson"]) >= bp._DRIFT_R_MIN
    assert d["span_beats"] >= bp._DRIFT_SPAN_BEATS
    # coeffs must encode a real negative slope (accelerating band)
    assert d["slope_s_per_s"] < 0


# ── the GUARDRAIL cases: constant + rubato songs must NOT get drift ─────────────

def test_flat_tiny_ramp_not_applied():
    """Blue Bossa backing: monotone (Pearson ~1) but span 0.2s = half a beat.
    The MAGNITUDE gate must veto it (else the guardrail is too weak)."""
    bp_grid = 60.0 / 150.0                        # 0.4 s
    ts = np.linspace(17.5, 299.0, 12)
    deltas = np.linspace(-0.10, 0.10, 12)
    d = bp.detect_drift(_rows(ts, deltas, agr=0.53), bp_grid)
    assert d["apply"] is False
    assert d["classification"] == "flat"
    assert d["span_beats"] < bp._DRIFT_SPAN_BEATS


def test_erratic_ramp_not_applied():
    """Close-To-You-style section jitter: swings both ways, low monotonicity, even
    with a large span -> 'erratic', never chased (that would be the banned free warp)."""
    bp_grid = 0.5
    ts = np.linspace(10.0, 60.0, 6)
    deltas = [0.0, 1.5, -0.2, 1.6, -0.1, 1.4]
    d = bp.detect_drift(_rows(ts, deltas), bp_grid)
    assert d["apply"] is False
    assert d["classification"] in ("erratic", "flat")
    assert abs(d["pearson"]) < bp._DRIFT_R_MIN


def test_insufficient_units_not_applied():
    d = bp.detect_drift(_rows([10.0, 20.0], [0.0, -1.0]), 0.35)
    assert d["apply"] is False
    assert d["classification"] == "insufficient"


# ── the grid warp: smooth, monotone, and it FLATTENS a synthetic drift ─────────

def test_drift_grid_is_monotone_and_matches_offset_model():
    beats = np.arange(0.0, 200.0, 0.3515)
    coeffs = [0.0, -0.004, 0.5]                   # linear: delta(t) = -0.004 t + 0.5
    w = bp.drift_grid(beats, coeffs)
    assert np.all(np.diff(w) > 0)                 # strictly increasing (usable grid)
    expect = beats + (coeffs[1] * beats + coeffs[2])
    assert np.allclose(w, expect, atol=1e-9)


def test_drift_grid_flattens_a_known_ramp():
    """If chords truly sit at grid + delta_true(t), warping the grid by the fitted
    delta model must drive the residual offset (delta_true - warp) toward ~0."""
    beats = np.arange(0.0, 500.0, 0.3515)
    coeffs = [0.0, -0.004, 0.5]                   # the drift the band actually has
    true_pos = beats + bp._drift_offset(coeffs, beats)     # where the audio really is
    w = bp.drift_grid(beats, coeffs)
    resid = true_pos - w                          # residual after correction
    assert np.max(np.abs(resid)) < 1e-6           # fully flattened


# ── per-section BPM series: smooth, monotone, no jumps (Louis's guardrail) ─────

def test_section_bpms_smooth_and_monotone():
    # A QUADRATIC offset (constant acceleration) => the per-section BPM genuinely
    # RAMPS (a pure-linear offset would give a single re-tempo = equal BPMs). These
    # coeffs mirror Blue Bossa's flat-then-drop curve (~170 -> ~173 BPM).
    beat_period = 0.3515
    beats = np.arange(0.0, 520.0, beat_period)
    coeffs = [-1.5e-5, 0.003, 0.4]                # c2<0 => band accelerates
    w = bp.drift_grid(beats, coeffs)
    assert np.all(np.diff(w) > 0)                 # grid still usable
    pls = [bp.Placement(chorus=c, sec_idx=0, label="A", start_beat=c * 64,
                        n_beats=64, agreement=0.35, occ=c) for c in range(22)]
    series = np.array([s["bpm"] for s in bp.section_bpms(pls, w, beat_period)])
    assert not np.any(np.isnan(series))
    steps = np.diff(series)
    # monotone (band accelerates) and — the GUARDRAIL — NO jumps: each section can
    # only TRACK the single global ramp, so successive BPM steps are tiny + smooth.
    assert np.all(steps > 0)                      # strictly increasing (real ramp)
    assert series.max() - series.min() > 1.0      # a genuine drift, not degenerate
    assert np.max(np.abs(steps)) < 0.5            # cannot jump — only track


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
