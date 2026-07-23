"""Tests for harmonia.align.downbeat (the global-phase downbeat resolver).

Audio-free: synthetic per-beat evidence arrays drive ``resolve_phase`` directly
to check the global-phase argmax, the reliability weighting (bass auto-downweight
vs harmonic-rhythm carry), the drum strong-pair narrowing, the precision-first
flag, and the mid-song phase-flip detector. Two extractor tests use synthetic
chroma frames / BassChroma. A tiny fake ``DrumBeatTrack`` exercises the
``resolve_downbeat`` wiring with no audio.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.align import downbeat as db
from harmonia.align.bass_salience import BassChroma


def _beats(K, period=0.5, t0=0.0):
    return t0 + np.arange(K) * period


def _on_phase(K, phase, meter=4, amp=1.0):
    """Per-beat evidence with all mass on one metrical phase."""
    ev = np.zeros(K)
    ev[np.arange(K) % meter == phase] = amp
    return ev


# ── global-phase argmax ──────────────────────────────────────────────────────

def test_resolves_phase_zero():
    K = 64
    res = db.resolve_phase(_beats(K), meter=4, harm=_on_phase(K, 0))
    assert res.phase_offset == 0
    assert res.confidence > 0.5
    assert not res.flagged
    # downbeats are every 4th beat starting at 0
    assert np.allclose(res.downbeat_times, _beats(K)[::4])


def test_resolves_nonzero_phase():
    K = 64
    res = db.resolve_phase(_beats(K), meter=4, harm=_on_phase(K, 2))
    assert res.phase_offset == 2


def test_per_term_contributions_recorded():
    K = 48
    res = db.resolve_phase(_beats(K), meter=4, harm=_on_phase(K, 1))
    c = res.per_term_contributions["harm"]
    assert c["favored_phase"] == 1
    assert c["concentration"] > 0.9
    assert abs(sum(c["phase_fracs"]) - 1.0) < 1e-6


# ── reliability weighting: bass auto-downweights, harmonic rhythm carries ─────

def test_bass_downweighted_when_stream_unreliable():
    """Walking-bass analogue: bass evidence points at the WRONG phase but its
    stream reliability (bass_weight) is ~0, so harmonic rhythm (right phase)
    wins — the Autumn behaviour."""
    K = 64
    harm = _on_phase(K, 0)                       # harmony says phase 0 (correct)
    bass = _on_phase(K, 1)                       # bass (mis)points at phase 1
    res = db.resolve_phase(_beats(K), meter=4, harm=harm, bass=bass,
                           bass_weight=0.02)      # near-zero bass reliability
    assert res.phase_offset == 0
    assert res.per_term_contributions["bass"]["weight"] < 0.05


def test_bass_reinforces_when_reliable():
    """Pop analogue: reliable bass agrees with harmony on phase 0 -> both terms
    carry weight and the winner is unambiguous."""
    K = 64
    res = db.resolve_phase(_beats(K), meter=4, harm=_on_phase(K, 0),
                           bass=_on_phase(K, 0), bass_weight=0.9)
    assert res.phase_offset == 0
    assert res.per_term_contributions["bass"]["weight"] > 0.5
    assert res.confidence > 0.6


# ── drum strong-pair narrowing (which of 1&3 is beat 1, resolved by harmony) ──

def test_drum_pair_narrows_candidates():
    K = 64
    # strong grid on EVEN indices (parity 0) -> candidates {0, 2}
    strong = (np.arange(K) % 2 == 0)
    harm = _on_phase(K, 2)                       # harmony picks phase 2 of the pair
    res = db.resolve_phase(_beats(K), meter=4, harm=harm, strong_mask=strong,
                           strong_conf=0.5)
    assert set(res.candidate_phases) == {0, 2}
    assert res.phase_offset == 2                 # harmony breaks the 1-vs-3 tie


def test_drum_pair_not_narrowed_when_unconfident():
    K = 64
    strong = (np.arange(K) % 2 == 0)
    res = db.resolve_phase(_beats(K), meter=4, harm=_on_phase(K, 1),
                           strong_mask=strong, strong_conf=0.05)
    assert set(res.candidate_phases) == {0, 1, 2, 3}
    assert res.phase_offset == 1                 # odd phase still reachable


# ── precision-first flagging ─────────────────────────────────────────────────

def test_flags_when_no_evidence():
    K = 40
    res = db.resolve_phase(_beats(K), meter=4, harm=np.zeros(K))
    assert res.flagged
    assert res.confidence == 0.0


def test_flags_weak_margin():
    K = 64
    rng = np.random.default_rng(0)
    harm = 0.25 + 0.001 * rng.standard_normal(K)   # essentially uniform
    res = db.resolve_phase(_beats(K), meter=4, harm=np.abs(harm))
    assert res.confidence < db._FLAG_CONF
    assert res.flagged


def test_detects_mid_song_phase_flip():
    """A dropped beat flips the phase: harmony concentrates on phase 0 in the
    first half, phase 2 in the second -> FLAG."""
    K = 64
    harm = np.zeros(K)
    idx = np.arange(K)
    harm[(idx < K // 2) & (idx % 4 == 0)] = 1.0
    harm[(idx >= K // 2) & (idx % 4 == 2)] = 1.0
    res = db.resolve_phase(_beats(K), meter=4, harm=harm)
    assert res.flagged
    assert any("FLIP" in n for n in res.notes)


# ── evidence extractors ──────────────────────────────────────────────────────

def test_beat_chroma_flux_peaks_on_changes():
    # chroma frames: chord C for 2 beats, then F for 2 beats (change at beat 2)
    period = 0.5
    fps = 43.0
    beats = _beats(4, period)
    F = int(4 * period * fps) + 5
    ftimes = np.arange(F) / fps
    frames = np.zeros((F, 12))
    half = np.searchsorted(ftimes, 2 * period)
    frames[:half, 0] = 1.0                       # C
    frames[half:, 5] = 1.0                       # F
    flux = db.beat_chroma_flux(frames, ftimes, beats)
    assert flux[0] == 0.0                         # no predecessor
    assert flux[2] > 0.5                          # the change beat
    assert flux[1] < 0.1 and flux[3] < 0.1        # held beats


def test_bass_beat_evidence_marks_root_changes():
    # bass root C for beats 0-1, G for beats 2-3 -> change at beat 2
    period = 0.5
    fps = 22050 / 512
    beats = _beats(4, period)
    F = int(4 * period * fps) + 5
    times = np.arange(F) / fps
    chroma = np.full((F, 12), 0.02)
    half = np.searchsorted(times, 2 * period)
    chroma[:half, 0] = 1.0
    chroma[half:, 7] = 1.0
    bass = BassChroma(chroma=chroma, low_chroma=chroma.copy(), times=times,
                      energy=chroma.sum(axis=1),
                      energy_ref=float(np.percentile(chroma.sum(axis=1), 95)))
    ev, rel = db.bass_beat_evidence(bass, beats)
    assert ev[2] > 0.4                            # root change on beat 2, reliable
    assert ev[1] == 0.0 and ev[3] == 0.0


# ── resolve_downbeat wiring (fake DrumBeatTrack, no audio) ────────────────────

class _FakeTrack:
    def __init__(self, beat_times, strong_mask, strong_conf, octave_locked=True):
        self.beat_times = np.asarray(beat_times, float)
        self.strong_beat_mask = np.asarray(strong_mask, bool)
        self.strong_beat_confidence = float(strong_conf)
        self.octave_locked = octave_locked


def test_resolve_downbeat_end_to_end_fake():
    K = 64
    period = 0.5
    beats = _beats(K, period)
    # synthetic chroma: chord change every 4 beats on phase 0
    fps = 43.0
    F = int(K * period * fps) + 5
    ftimes = np.arange(F) / fps
    frames = np.full((F, 12), 0.02)
    roots = [0, 5, 7, 9]
    for bar, k0 in enumerate(range(0, K, 4)):
        t = k0 * period
        lo = np.searchsorted(ftimes, t)
        hi = np.searchsorted(ftimes, t + 4 * period)
        frames[lo:hi, roots[bar % 4]] = 1.0
    track = _FakeTrack(beats, np.arange(K) % 2 == 0, strong_conf=0.4)
    res = db.resolve_downbeat(track, chart_alignment=None,
                              chroma=(frames, ftimes), bass=None, meter=4)
    assert res.phase_offset == 0
    assert not res.flagged
    assert set(res.candidate_phases) == {0, 2}


def test_resolve_downbeat_flags_unlocked_grid():
    K = 32
    beats = _beats(K)
    fps = 43.0
    F = int(K * 0.5 * fps) + 5
    ftimes = np.arange(F) / fps
    frames = np.full((F, 12), 0.02)
    frames[:, 0] = 1.0
    track = _FakeTrack(beats, np.arange(K) % 2 == 0, 0.4, octave_locked=False)
    res = db.resolve_downbeat(track, chroma=(frames, ftimes), meter=4)
    assert res.flagged
    assert any("octave-locked" in n for n in res.notes)


def test_constant_lattice_recovers_clean_grid():
    beats = _beats(64, 0.5, t0=0.3)
    lat, P = db.constant_lattice(beats)
    assert abs(P - 0.5) < 0.01
    assert np.max(np.abs(np.diff(lat) - P)) < 1e-6      # perfectly even
    # every tracked beat sits on a lattice point
    d = np.abs(beats[:, None] - lat[None, :]).min(axis=1)
    assert np.max(d) < 0.05


def test_constant_lattice_cleans_inserted_beats():
    """An inserted (extra) beat must NOT make the lattice jitter — the constant
    grid stays evenly spaced, which is the whole point (a jittery grid slips the
    global phase)."""
    beats = list(_beats(40, 0.5))
    beats.insert(20, beats[20] + 0.25)              # a spurious extra beat
    lat, P = db.constant_lattice(np.array(beats), period=0.5)
    assert abs(P - 0.5) < 0.02
    assert np.max(np.abs(np.diff(lat) - P)) < 1e-6   # still perfectly even


def test_harmonic_change_evidence_is_sparse():
    period = 0.5
    K = 32
    beats = _beats(K, period)
    fps = 43.0
    F = int(K * period * fps) + 5
    ftimes = np.arange(F) / fps
    frames = np.full((F, 12), 0.02)
    for bar, k0 in enumerate(range(0, K, 4)):        # change every 4 beats
        t = k0 * period
        lo = np.searchsorted(ftimes, t); hi = np.searchsorted(ftimes, t + 4 * period)
        frames[lo:hi, [0, 5, 7, 9][bar % 4]] = 1.0
    ev = db.harmonic_change_evidence(frames, ftimes, beats)
    nz = np.flatnonzero(ev)
    assert len(nz) < K // 2                           # sparse (peak-picked)
    assert np.all(nz % 4 == 0)                        # all changes on phase 0


def test_chart_hits_from_array():
    K = 32
    beats = _beats(K)
    bar1_times = beats[::4]                       # chart bar-1 = every 4th beat
    hits = db._chart_hits(bar1_times, beats)
    assert hits is not None
    assert hits[::4].sum() == len(bar1_times)
    assert hits[1::4].sum() == 0
