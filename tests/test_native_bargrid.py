"""Phase 2 sub-steps 1+2 — native per-bar-downbeat bar grid (2026-07-21).

Red-first: exercises the NEW native-bargrid resolver (beat_grid.native_bar_grid
/ best_supported_phase / native_bargrid_enabled) and the explicit-bnds path
threaded through _flux_anchored_bar_root. All of these fail before the feature
is implemented. Also pins the load-bearing invariant that the kill-switch
defaults OFF and that the OFF grid is byte-identical to the prior np.arange grid
(CLAUDE.md #1/#4/#6 — the live default must not move).
"""
import numpy as np

from harmonia.models.beat_grid import (
    best_supported_phase,
    native_bar_grid,
    native_bargrid_enabled,
    NATIVE_BARGRID_ENV,
)


# ── kill-switch (default OFF) ────────────────────────────────────────────────
def test_kill_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(NATIVE_BARGRID_ENV, raising=False)
    assert native_bargrid_enabled() is False


def test_kill_switch_on_values(monkeypatch):
    for v in ("1", "on", "ON", "true", "yes"):
        monkeypatch.setenv(NATIVE_BARGRID_ENV, v)
        assert native_bargrid_enabled() is True
    for v in ("0", "off", "", "no"):
        monkeypatch.setenv(NATIVE_BARGRID_ENV, v)
        assert native_bargrid_enabled() is False


# ── best-supported single phase (fallback, sub-step 2) ───────────────────────
def test_best_supported_phase_recovers_offset_phase():
    # 120 bpm real beat grid; true downbeats land on beat-phase 2 of every bar.
    bts = np.arange(0.0, 60.0, 0.5)
    downbeats = bts[2::4]
    assert best_supported_phase(downbeats, bts, beats_per_bar=4) == 2


def test_best_supported_phase_beats_circular_mean_style_wraparound():
    # A near-perfect but phase-1 native track — best_supported must pick 1, not 0.
    bts = np.arange(0.0, 40.0, 0.5)
    downbeats = bts[1::4] + 0.01  # tiny jitter, still within tol of phase 1
    assert best_supported_phase(downbeats, bts, beats_per_bar=4) == 1


# ── native_bar_grid: primary / fallback / absent ─────────────────────────────
def test_native_bar_grid_primary_uses_native_downbeats():
    bts = np.arange(0.0, 60.0, 0.5)
    downbeats = bts[2::4]            # regular, on phase 2
    bnds, anchor, mode = native_bar_grid(
        downbeats, conf=1.0, bts=bts, period=0.5, flux_phi=0)
    assert mode == "native"
    assert anchor == 2              # bar1 anchor = phase of first native downbeat
    np.testing.assert_allclose(bnds, downbeats)   # the bars ARE the native downbeats


def test_native_bar_grid_fallback_best_phase_when_irregular():
    bts = np.arange(0.0, 60.0, 0.5)
    downbeats = bts[2::4]
    bnds, anchor, mode = native_bar_grid(
        downbeats, conf=0.40, bts=bts, period=0.5, flux_phi=0)  # low regularity
    assert mode == "phase"
    assert anchor == 2
    np.testing.assert_allclose(bnds, bts[2::4])   # uniform subsample at best phase


def test_native_bar_grid_absent_keeps_flux_chain():
    bts = np.arange(0.0, 60.0, 0.5)
    downbeats = bts[2:6]            # only 4 downbeats (<5) -> native unusable
    bnds, anchor, mode = native_bar_grid(
        downbeats, conf=1.0, bts=bts, period=0.5, flux_phi=3)
    assert mode == "flux"
    assert bnds is None            # caller keeps its existing circular-mean grid
    assert anchor == 3             # flux_phi passed straight through


# ── explicit-bnds path through _flux_anchored_bar_root ───────────────────────
class _ZeroHeads:
    def root_proba(self, feat):
        return np.zeros((len(feat), 12), dtype=float)


def test_flux_anchored_bar_root_honors_explicit_bnds():
    from harmonia.models.chord_pipeline_v1 import _flux_anchored_bar_root

    fps = 50
    dur = 20.0
    times = np.arange(0.0, dur, 1.0 / fps)
    arr = np.zeros((len(times), 24), dtype=float)
    bar_period = 2.0                       # 4 beats * 0.5s
    # variable-width native downbeats (NOT a uniform np.arange grid)
    bnds = np.array([0.5, 2.4, 4.5, 6.3, 8.5, 10.6, 12.4])
    bar_root, bar_times = _flux_anchored_bar_root(
        arr, times, _ZeroHeads(), phi=0, bar_period=bar_period, bnds=bnds)
    eps = 0.25 * (bar_period / 4)
    starts = [t0 - eps for (t0, _t1) in bar_times]
    # each emitted bar start comes from a native boundary, and a trailing bar
    # is appended to reach the end of the audio (variable widths preserved).
    assert len(bar_times) >= len(bnds) - 1
    np.testing.assert_allclose(starts[: len(bnds) - 1], bnds[:-1], atol=1e-6)
    widths = [round((bar_times[i + 1][0] - bar_times[i][0]), 3)
              for i in range(len(bnds) - 2)]
    assert len(set(widths)) > 1            # genuinely variable-width (not uniform)


def test_flux_anchored_bar_root_default_is_uniform_arange():
    # bnds=None (the OFF default) reproduces the exact prior uniform grid.
    from harmonia.models.chord_pipeline_v1 import _flux_anchored_bar_root

    fps = 50
    times = np.arange(0.0, 20.0, 1.0 / fps)
    arr = np.zeros((len(times), 24), dtype=float)
    bar_period = 2.0
    phi = 1
    _, bar_times = _flux_anchored_bar_root(
        arr, times, _ZeroHeads(), phi=phi, bar_period=bar_period)
    beat = bar_period / 4
    eps = 0.25 * beat
    expected = np.arange(phi * beat, float(times[-1]) + bar_period, bar_period)
    starts = [t0 - eps for (t0, _t1) in bar_times]
    np.testing.assert_allclose(starts, expected[:-1], atol=1e-6)
