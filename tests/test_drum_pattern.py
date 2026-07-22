"""Tests for harmonia.align.drum_pattern (Stage-1 drum beat/tempo tracker).

Audio-free wherever possible: pure-numpy synthetic onset envelopes exercise the
octave anchoring, phase tracking, strong-beat pair, local tempo and reliability.
A single fast end-to-end smoke test builds a synthetic drum WAVEFORM (kicks +
snares) and runs the whole percussive-onset -> track path (no external file, no
Beat This!). The real-corpus lock numbers live in the scratch validation harness,
not here (they need the audio + the Beat This! backend).
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.align import drum_pattern as dp

FPS = dp.SR / dp.HOP


# ── helpers ──────────────────────────────────────────────────────────────────

def _click_env(period_s: float, dur_s: float = 40.0, amp: float = 1.0,
               t0: float = 0.0, n: int | None = None) -> np.ndarray:
    """A synthetic broadband onset envelope: unit impulses every ``period_s``."""
    if n is None:
        n = int(dur_s * FPS)
    env = np.zeros(n)
    k = 0
    while True:
        t = t0 + k * period_s
        f = int(round(t * FPS))
        if f >= n:
            break
        env[f] = amp
        k += 1
    return env


# ── octave anchoring (requirement #1, Stage-0 calibration caveat) ────────────

def test_estimate_period_stays_in_octave():
    """The refined period must land in the prior's octave, never 2x/0.5x."""
    period = 0.5
    env = _click_env(period, dur_s=30.0)
    est = dp.estimate_period_in_octave(env, prior_period=period)
    assert abs(est - period) < 0.01


def test_octave_anchor_rejects_subdivision_slip():
    """Env with strong energy on both beats AND their midpoints (subdivision):
    a naive autocorr would slip to the half-period octave; the anchored estimate
    stays at the beat period."""
    period = 0.5
    beats = _click_env(period, dur_s=30.0, amp=1.0)
    subs = _click_env(period / 2, dur_s=30.0, amp=0.9)   # extra energy at P/2
    env = np.maximum(beats, subs)
    # naive full-range peak (octave_tol huge) can pick the subdivision...
    est_naive = dp.estimate_period_in_octave(env, prior_period=0.35, octave_tol=1.0)
    # ...but anchored to the beat octave it stays put.
    est_locked = dp.estimate_period_in_octave(env, prior_period=period,
                                              octave_tol=0.18)
    assert abs(est_locked - period) < 0.02
    assert est_naive <= period + 0.02          # naive is <= beat period (slipped)


@pytest.mark.parametrize("period", [0.30, 0.5, 0.85])
def test_track_period_and_octave_lock(period):
    pytest.importorskip("librosa")
    env = _click_env(period, dur_s=40.0)
    tr = dp.track_drum_beats(env, prior_period=period)
    assert tr.octave_locked is True
    assert abs(tr.period - period) < 0.02
    # tracked spacing is at the beat octave (not doubled/halved)
    med = float(np.median(np.diff(tr.beat_times)))
    assert 0.8 * period < med < 1.25 * period


def test_track_does_not_slip_when_prior_is_correct_but_env_has_subdivisions():
    pytest.importorskip("librosa")
    period = 0.5
    env = np.maximum(_click_env(period, 40.0, 1.0), _click_env(period / 2, 40.0, 0.8))
    tr = dp.track_drum_beats(env, prior_period=period)
    med = float(np.median(np.diff(tr.beat_times)))
    assert 0.8 * period < med < 1.25 * period       # locked to the beat, not P/2


# ── beat likelihood = the DBN observation term ───────────────────────────────

def test_beat_likelihood_peaks_on_grid():
    pytest.importorskip("librosa")
    period = 0.5
    tr = dp.track_drum_beats(_click_env(period, 40.0), prior_period=period)
    # sample the pulse on a tracked beat vs a quarter-beat off it
    b = tr.beat_times[len(tr.beat_times) // 2]
    on = tr.beat_likelihood_at(b)
    off = tr.beat_likelihood_at(b + 0.25 * period)
    assert 0.0 <= off < on <= 1.0
    # per-beat likelihood array is populated and in range
    assert tr.beat_likelihood.min() >= 0.0 and tr.beat_likelihood.max() <= 1.0 + 1e-9
    assert np.median(tr.beat_likelihood) > 0.3


def test_beat_likelihood_at_vectorised():
    pytest.importorskip("librosa")
    tr = dp.track_drum_beats(_click_env(0.5, 30.0), prior_period=0.5)
    ts = np.array([5.0, 10.0, 15.0])
    out = tr.beat_likelihood_at(ts)
    assert out.shape == ts.shape
    assert np.all((out >= 0.0) & (out <= 1.0))


# ── local tempo tracking ─────────────────────────────────────────────────────

def test_local_tempo_reports_bpm():
    pytest.importorskip("librosa")
    period = 0.5      # 120 BPM
    tr = dp.track_drum_beats(_click_env(period, 40.0), prior_period=period)
    bpm = tr.local_tempo(20.0)
    assert abs(bpm - 120.0) < 6.0
    assert abs(tr.local_period(20.0) - period) < 0.03


def test_local_tempo_clamped_to_octave():
    """local period is octave-clamped so a single dropped beat can't halve it."""
    pytest.importorskip("librosa")
    period = 0.5
    tr = dp.track_drum_beats(_click_env(period, 40.0), prior_period=period)
    lp = tr.local_period(np.linspace(2, 38, 50))
    lo, hi = period / (1 + dp._OCTAVE_TOL), period * (1 + dp._OCTAVE_TOL)
    assert np.all((lp >= lo - 1e-6) & (lp <= hi + 1e-6))


# ── strong-beat PAIR (NOT a downbeat) ────────────────────────────────────────

def test_strong_beat_pair_kick_snare():
    """Kick on 1&3, snare on 2&4 -> a clean, confident half-tempo pair grid whose
    two grids interleave. Must NOT claim a downbeat (only the pair)."""
    pytest.importorskip("librosa")
    period = 0.5
    n = int(40 * FPS)
    broad = _click_env(period, n=n)
    low = np.zeros(n)
    high = np.zeros(n)
    for k in range(int(40 / period)):
        f = int(round(k * period * FPS))
        if f < n:
            (low if k % 2 == 0 else high)[f] = 1.0
    tr = dp.track_drum_beats(broad, prior_period=period, low=low, high=high)
    assert tr.strong_beat_confidence > 0.4
    assert len(tr.strong_beat_mask) == len(tr.beat_times)
    # strong (1&3) and backbeat (2&4) partition the beats, roughly half each
    ns, nb = len(tr.strong_beat_times), len(tr.backbeat_times)
    assert ns + nb == len(tr.beat_times)
    assert abs(ns - nb) <= 2
    # the two grids interleave (adjacent strong beats are ~2 beats apart)
    if ns > 2:
        gaps = np.diff(tr.strong_beat_times)
        assert abs(np.median(gaps) - 2 * period) < 0.15 * period


def test_strong_beat_phase_at_alternates():
    pytest.importorskip("librosa")
    period = 0.5
    n = int(40 * FPS)
    broad = _click_env(period, n=n)
    low = np.zeros(n); high = np.zeros(n)
    for k in range(int(40 / period)):
        f = int(round(k * period * FPS))
        if f < n:
            (low if k % 2 == 0 else high)[f] = 1.0
    tr = dp.track_drum_beats(broad, prior_period=period, low=low, high=high)
    # consecutive tracked beats should mostly flip strong/backbeat phase
    phases = tr.strong_beat_phase_at(tr.beat_times[2:-2])
    flips = np.mean(phases[1:] != phases[:-1])
    assert flips > 0.8


# ── reliability = the DBN weight driver (Stage-0 scoping catch) ──────────────

def test_reliability_higher_where_drums_present():
    """Drums only in the second half -> reliability rises there (down-weights the
    drum-less first half, like Let It Be's piano intro)."""
    period = 0.5
    n = int(60 * FPS)
    env = np.zeros(n)
    half = n // 2
    env[half:] = _click_env(period, n=n)[half:]     # clicks only in 2nd half
    rt, rel = dp.drum_reliability(env, period)
    first = rel[rt < 25.0].mean()
    second = rel[rt > 35.0].mean()
    assert second > first
    assert second > 0.3


# ── anchor window selection (requirement #2) ─────────────────────────────────

def test_anchor_window_picks_steady_region():
    """Steady clicks only in a mid window -> the anchor lands inside it."""
    period = 0.5
    n = int(60 * FPS)
    env = np.zeros(n)
    lo_t, hi_t = 25.0, 40.0
    lo_f, hi_f = int(lo_t * FPS), int(hi_t * FPS)
    clicks = _click_env(period, n=n)
    env[lo_f:hi_f] = clicks[lo_f:hi_f]
    aw = dp.select_drum_anchor(env, period)
    assert lo_t - 4 <= aw.t0 <= hi_t
    assert aw.clarity > 0.2


# ── end-to-end smoke: synthetic drum WAVEFORM through the whole path ─────────

def test_track_from_audio_synthetic_waveform():
    """Full path percussive_onset_envelopes -> track on a synthetic kick/snare
    waveform (no file, no Beat This!). Confirms the extractor + tracker wire up
    and the tempo is recovered in-octave."""
    pytest.importorskip("librosa")
    sr = dp.SR
    period = 0.5
    dur = 24.0
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    # kick (60 Hz burst) on 1&3, snare (noise burst) on 2&4
    rng = np.random.default_rng(0)
    for k in range(int(dur / period)):
        onset = k * period
        i0 = int(onset * sr)
        seg = slice(i0, min(i0 + int(0.05 * sr), len(y)))
        env = np.exp(-np.linspace(0, 6, seg.stop - seg.start))
        if k % 2 == 0:
            tt = np.arange(seg.stop - seg.start) / sr
            y[seg] += 0.9 * np.sin(2 * np.pi * 60 * tt) * env
        else:
            y[seg] += 0.5 * rng.standard_normal(seg.stop - seg.start) * env
    tr = dp.track_from_audio(y, prior_bpm=120.0, run_beat_this=False)
    assert tr.octave_locked is True
    assert abs(tr.tempo_bpm - 120.0) < 8.0
    assert len(tr.beat_times) > 30
    med = float(np.median(np.diff(tr.beat_times)))
    assert 0.8 * period < med < 1.25 * period
