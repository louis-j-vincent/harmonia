"""Tests for the harmonic-rhythm downbeat-phase brick.

Pins the load-bearing assumptions (CLAUDE.md #1): the cosine change metric, the
ramp-shape scorer's phase recovery, its CORRECT abstention (low confidence) on
the half-bar alias that sinks the naive argmax, and the conservative arbiter's
override policy including the 421 audio-veto guardrail.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.harmonic_downbeat import (
    HarmonicDownbeatResult,
    arbitrate_downbeat_phase,
    beat_change_strength,
    harmonic_downbeat_enabled,
    harmonic_downbeat_phase,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _evidence_from_bar_changes(
    per_pos_delta_deg: list[float], n_bars: int, phi_star: int = 0,
) -> np.ndarray:
    """Build (n_beats, 2) unit-vector evidence whose consecutive-beat cosine
    change realizes a chosen within-bar profile.

    ``per_pos_delta_deg[k]`` = angle jump (degrees) applied at within-bar
    position k; change strength there is ``1 - cos(delta)``. The whole pattern
    is rotated so the ``per_pos_delta_deg`` profile lands with its index 0 at
    beat-index ``phi_star`` (the intended downbeat phase)."""
    bar_len = len(per_pos_delta_deg)
    n = n_bars * bar_len
    theta = 0.0
    angles = []
    for i in range(n):
        pos = (i - phi_star) % bar_len
        theta += np.deg2rad(per_pos_delta_deg[pos])
        angles.append(theta)
    a = np.asarray(angles)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


# --------------------------------------------------------------------------- #
# beat_change_strength
# --------------------------------------------------------------------------- #
def test_change_zero_when_held():
    ev = np.tile([0.0, 1.0, 0.0], (6, 1))
    c = beat_change_strength(ev)
    assert c[0] == 0.0
    assert np.allclose(c, 0.0)


def test_change_one_when_orthogonal():
    ev = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    c = beat_change_strength(ev)
    assert c[0] == 0.0
    assert np.allclose(c[1:], 1.0)  # cosine distance of orthogonal = 1


def test_change_invariant_to_energy():
    # same direction, different magnitude -> NOT a change (cosine, not L2)
    ev = np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 0.2]])
    c = beat_change_strength(ev)
    assert np.allclose(c, 0.0, atol=1e-9)


def test_change_zero_row_is_not_a_change():
    ev = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])  # silence beat in middle
    c = beat_change_strength(ev)
    assert np.allclose(c, 0.0)  # both edges of the zero row are 0


# --------------------------------------------------------------------------- #
# harmonic_downbeat_phase — phase recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phi_star", [0, 1, 2, 3])
@pytest.mark.parametrize("method", ["ramp", "auto"])
def test_recovers_descending_ramp_phase(phi_star, method):
    # a clean descending ramp of change (big jump on the downbeat, less after).
    # both the transparent ramp scorer and the shipped (auto->learned) scorer
    # must recover the phase; only the ramp scorer's confidence SCALE is pinned.
    ev = _evidence_from_bar_changes([90.0, 60.0, 30.0, 0.0], n_bars=8,
                                    phi_star=phi_star)
    r = harmonic_downbeat_phase(None, ev, bar_len=4, method=method)
    assert r.valid
    assert r.best_phase == phi_star
    assert len(r.per_phase_score) == 4
    assert np.argmax(r.per_phase_score) == phi_star
    if method == "ramp":
        assert r.confidence > 0.5           # strong, unambiguous margin


def test_half_bar_alias_yields_low_confidence():
    # equal change on beats 0 and 2, none on 1 and 3 -> phase 0 and 2 tie.
    ev = _evidence_from_bar_changes([90.0, 0.0, 90.0, 0.0], n_bars=8, phi_star=0)
    r = harmonic_downbeat_phase(None, ev, bar_len=4, method="ramp")
    assert r.valid
    # the naive argmax of change_mass cannot separate 0 vs 2; the ramp scorer
    # must therefore report LOW confidence (correct abstention), which the
    # arbiter turns into "keep the audio phase".
    assert r.confidence < 0.1
    assert set(np.argsort(r.per_phase_score)[-2:]) == {0, 2}


def test_generalizes_to_three_four():
    # non-4/4 always uses the ramp scorer (learned model is 4/4-only)
    ev = _evidence_from_bar_changes([90.0, 45.0, 0.0], n_bars=10, phi_star=1)
    r = harmonic_downbeat_phase(None, ev, bar_len=3)
    assert r.valid and r.best_phase == 1
    assert len(r.per_phase_score) == 3


def test_deterministic_tie_keeps_lowest_phase():
    # uniform change everywhere -> all phases equal -> lowest index wins, conf 0
    ev = _evidence_from_bar_changes([90.0, 90.0, 90.0, 90.0], n_bars=8)
    r = harmonic_downbeat_phase(None, ev, bar_len=4, method="ramp")
    assert r.best_phase == 0
    assert r.confidence < 1e-6


def test_ml_requires_coeffs_when_forced():
    import harmonia.models.harmonic_downbeat as H
    ev = _evidence_from_bar_changes([90.0, 60.0, 30.0, 0.0], n_bars=8)
    if H.learned_available():
        r = harmonic_downbeat_phase(None, ev, bar_len=4, method="ml")
        assert r.valid and abs(sum(r.per_phase_score) - 1.0) < 1e-6  # a distribution
    else:
        with pytest.raises(ValueError):
            harmonic_downbeat_phase(None, ev, bar_len=4, method="ml")


def test_bad_method_raises():
    ev = _evidence_from_bar_changes([90.0, 60.0, 30.0, 0.0], n_bars=8)
    with pytest.raises(ValueError):
        harmonic_downbeat_phase(None, ev, bar_len=4, method="bogus")


# --------------------------------------------------------------------------- #
# harmonic_downbeat_phase — invalid / guard paths
# --------------------------------------------------------------------------- #
def test_too_few_beats_is_invalid():
    ev = np.eye(4, 3)[:5]  # only 5 beats < 2*bar_len
    r = harmonic_downbeat_phase(None, ev, bar_len=4)
    assert not r.valid and r.best_phase == 0 and r.confidence == 0.0


def test_all_zero_evidence_is_invalid():
    ev = np.zeros((16, 5))
    r = harmonic_downbeat_phase(None, ev, bar_len=4)
    assert not r.valid


def test_rejects_1d_evidence():
    with pytest.raises(ValueError):
        harmonic_downbeat_phase(None, np.zeros(16), bar_len=4)


def test_rejects_beat_time_length_mismatch():
    ev = np.eye(8, 4)
    with pytest.raises(ValueError):
        harmonic_downbeat_phase(np.arange(7.0), ev, bar_len=4)


def test_rejects_bad_bar_len():
    with pytest.raises(ValueError):
        harmonic_downbeat_phase(None, np.eye(8, 4), bar_len=1)


def test_accepts_matching_beat_times():
    ev = _evidence_from_bar_changes([90.0, 60.0, 30.0, 0.0], n_bars=6, phi_star=2)
    bt = np.arange(ev.shape[0], dtype=float) * 0.5
    r = harmonic_downbeat_phase(bt, ev, bar_len=4)
    assert r.valid and r.best_phase == 2


# --------------------------------------------------------------------------- #
# arbitrate_downbeat_phase — conservative override policy
# --------------------------------------------------------------------------- #
def _res(best_phase, confidence, valid=True):
    return HarmonicDownbeatResult(
        best_phase=best_phase, confidence=confidence,
        per_phase_score=[0, 0, 0, 0], change_mass=[0, 0, 0, 0],
        n_beats=32, valid=valid)


def test_agreement_keeps_audio_no_disagreement():
    final, src, dis = arbitrate_downbeat_phase(1, 0.4, _res(1, 1.0))
    assert final == 1 and src == "audio" and dis is False


def test_strong_disagreement_overrides():
    # adjacent (non-half-bar) flip 0->1 uses the normal min_harmonic_conf gate
    final, src, dis = arbitrate_downbeat_phase(
        0, 0.4, _res(1, 0.5), min_harmonic_conf=0.20, halfbar_min_conf=0.85)
    assert final == 1 and src == "harmonic" and dis is True


def test_weak_disagreement_keeps_audio_but_flags():
    final, src, dis = arbitrate_downbeat_phase(
        0, 0.4, _res(1, 0.1), min_harmonic_conf=0.20)
    assert final == 0 and src == "audio" and dis is True  # flag still raised


def test_halfbar_flip_needs_higher_confidence():
    # 0->2 is a half-bar flip (the aliasing failure): passes the normal gate but
    # NOT the higher half-bar gate -> keep audio.
    final, src, dis = arbitrate_downbeat_phase(
        0, 0.4, _res(2, 0.5), min_harmonic_conf=0.20, halfbar_min_conf=0.85)
    assert final == 0 and src == "audio" and dis is True


def test_halfbar_flip_overrides_when_very_confident():
    final, src, dis = arbitrate_downbeat_phase(
        0, 0.4, _res(2, 0.9), min_harmonic_conf=0.20, halfbar_min_conf=0.85)
    assert final == 2 and src == "harmonic" and dis is True


def test_audio_veto_protects_confident_audio_from_override():
    # the 421 guardrail: audio confidently right -> harmony can't overturn it
    final, src, dis = arbitrate_downbeat_phase(
        0, 0.97, _res(1, 1.5), min_harmonic_conf=0.20, audio_veto_conf=0.90)
    assert final == 0 and src == "audio" and dis is True


def test_arbiter_defaults_are_calibrated():
    import harmonia.models.harmonic_downbeat as H
    assert H.DEFAULT_MIN_CONF == 0.20
    assert H.DEFAULT_AUDIO_VETO_CONF == 0.70
    assert H.DEFAULT_HALFBAR_MIN_CONF == 0.85
    assert H.learned_available() is True  # coefficients are embedded


def test_invalid_harmonic_result_keeps_audio():
    final, src, dis = arbitrate_downbeat_phase(3, 0.1, _res(0, 2.0, valid=False))
    assert final == 3 and src == "audio" and dis is False


# --------------------------------------------------------------------------- #
# kill-switch
# --------------------------------------------------------------------------- #
def test_kill_switch_default_off(monkeypatch):
    monkeypatch.delenv("HARMONIA_HARMONIC_DOWNBEAT", raising=False)
    assert harmonic_downbeat_enabled() is False


def test_kill_switch_on(monkeypatch):
    monkeypatch.setenv("HARMONIA_HARMONIC_DOWNBEAT", "on")
    assert harmonic_downbeat_enabled() is True
