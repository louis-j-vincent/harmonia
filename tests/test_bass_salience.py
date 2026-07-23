"""Tests for harmonia.align.bass_salience (stream #4 — the bass-salience instrument).

Audio-free wherever possible: synthetic ``BassChroma`` objects exercise the
duration-integrated pc read, the energy x concentration reliability, and the
fifth/third harmonic guard. A single librosa smoke test drives the whole
CQT -> flat-fold path on a synthetic low sine and checks the pc is recovered.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.align import bass_salience as bs

FPS = bs.SR / bs.HOP


def _bass_chroma(chroma, low=None, energy=None, dur_s=None):
    """Build a synthetic BassChroma from a (F, 12) frame array."""
    chroma = np.asarray(chroma, float)
    F = chroma.shape[0]
    if low is None:
        low = chroma.copy()
    if energy is None:
        energy = chroma.sum(axis=1)
    times = np.arange(F) / FPS
    return bs.BassChroma(chroma=chroma, low_chroma=np.asarray(low, float),
                         times=times, energy=np.asarray(energy, float),
                         energy_ref=max(float(np.percentile(energy, 95)), 1e-9))


def _peak(pc, F=40, amp=1.0, floor=0.02):
    """(F, 12) chroma peaked on one pitch class (plus a small flat floor)."""
    m = np.full((F, 12), floor)
    m[:, pc] = amp
    return m


# ── duration-integrated pc read (never the instant) ─────────────────────────

def test_pc_over_span_reads_argmax():
    bass = _bass_chroma(_peak(7))               # G dominates
    pc, rel = bs.bass_pc_over_span(bass, 0.1, 0.4)
    assert pc == 7
    assert 0.0 < rel <= 1.0


def test_pc_over_span_integrates_over_span_not_onset():
    """A broadband ATTACK on the first frame (masking the pitch) must NOT win —
    the sustained pitch over the span does. Mirrors the load-bearing recipe fix."""
    m = _peak(5, F=40)                          # sustained F over the whole span
    m[0, :] = 5.0                               # loud broadband attack on frame 0
    bass = _bass_chroma(m)
    pc, _ = bs.bass_pc_over_span(bass, 0.0, 0.4)
    assert pc == 5                              # sustain wins, not the attack frame


# ── reliability = energy x concentration (the auto-downweight) ───────────────

def test_reliability_high_for_peaky_low_for_flat():
    peaky = _bass_chroma(_peak(0, floor=0.02))
    flat = _bass_chroma(np.ones((40, 12)))     # walking-bass analogue: all pcs equal
    r_peaky = bs.bass_reliability(peaky, 0.1, 0.4)
    r_flat = bs.bass_reliability(flat, 0.1, 0.4)
    assert r_peaky > 0.5
    assert r_flat < 0.05                        # flat chroma self-zeroes
    assert r_peaky > r_flat


def test_reliability_scales_with_energy():
    loud = _bass_chroma(_peak(2, amp=1.0))
    quiet_ch = _peak(2, amp=1.0)
    # a low-energy span (relative to the song's 95th-pct energy) reads lower rel
    bass = _bass_chroma(np.vstack([_peak(2, F=38, amp=1.0), _peak(2, F=2, amp=1.0)]))
    # scale one span down by making its frames tiny
    bass.chroma[:5] *= 0.05
    bass.energy[:5] *= 0.05
    r_quiet = bs.bass_reliability(bass, 0.0, 5 / FPS)
    r_loud = bs.bass_reliability(bass, 20 / FPS, 30 / FPS)
    assert r_loud > r_quiet


# ── fifth / third harmonic guard ─────────────────────────────────────────────

def test_fifth_guard_recovers_low_fundamental():
    """Argmax is G (7) but the LOW-octave fundamental energy sits on C (0 == 7-7):
    the guard returns the true root C."""
    folded = _peak(7, F=40, amp=1.0, floor=0.02)
    folded[:, 0] = 0.7                          # C is a strong second peak
    low = np.full((40, 12), 0.02)
    low[:, 0] = 1.0                             # ...and dominates the lowest octave
    low[:, 7] = 0.1
    bass = _bass_chroma(folded, low=low)
    pc, _ = bs.bass_pc_over_span(bass, 0.1, 0.4, guard=True)
    assert pc == 0
    # with the guard OFF the raw argmax (the fifth) is returned
    pc_raw, _ = bs.bass_pc_over_span(bass, 0.1, 0.4, guard=False)
    assert pc_raw == 7


def test_fifth_guard_does_not_fire_on_clean_root():
    """When the argmax IS the low fundamental, the guard leaves it alone."""
    folded = _peak(0, F=40, amp=1.0, floor=0.02)
    low = _peak(0, F=40, amp=1.0, floor=0.02)  # fundamental strongest on the root
    bass = _bass_chroma(folded, low=low)
    pc, _ = bs.bass_pc_over_span(bass, 0.1, 0.4, guard=True)
    assert pc == 0


# ── series convenience ───────────────────────────────────────────────────────

def test_bass_pc_series_matches_spans():
    # two spans, roots C then G
    m = np.vstack([_peak(0, F=20), _peak(7, F=20)])
    bass = _bass_chroma(m)
    bounds = np.array([0.0, 20 / FPS, 40 / FPS])
    pcs, rels = bs.bass_pc_series(bass, bounds)
    assert pcs.tolist() == [0, 7]
    assert np.all(rels > 0.4)


# ── empty / degenerate spans ─────────────────────────────────────────────────

def test_silent_span_zero_reliability():
    bass = _bass_chroma(np.zeros((40, 12)) + 1e-9)
    _, rel = bs.bass_pc_over_span(bass, 0.1, 0.4)
    assert rel == 0.0


# ── librosa smoke: whole CQT -> flat-fold path on a synthetic bass note ──────

def test_bass_chroma_recovers_synthetic_note():
    pytest.importorskip("librosa")
    import librosa
    sr = bs.SR
    dur = 2.0
    f_c2 = librosa.note_to_hz("C2")            # a bass C -> pitch class 0
    t = np.arange(int(dur * sr)) / sr
    y = 0.5 * np.sin(2 * np.pi * f_c2 * t).astype(np.float32)
    bass = bs.bass_chroma(y)
    assert bass.chroma.shape[1] == 12
    pc, rel = bs.bass_pc_over_span(bass, 0.3, 1.7)
    assert pc == 0                             # C recovered
    assert rel > 0.3
