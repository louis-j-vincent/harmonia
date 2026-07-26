"""Tests for the default-OFF seventh_upgrade brick (2026-07-26)."""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.seventh_upgrade import (ENV_FLAG, _parse, _relabel_min7,
                                             seventh_evidence,
                                             upgrade_minor_sevenths)

# nnls_features stores bothchroma A-first and rolls by +9 to get a C-first frame
# (rolled[i] == a_frame[(i-9)%12]).  To place energy at a C-relative pitch class
# we therefore write it at a_frame[(pc-9)%12], i.e. np.roll(c_frame, +3).
_ROLL_BACK = 3


def _frames(pcs_energy: dict[int, float], n_frames: int = 20) -> np.ndarray:
    """A fake bothchroma matrix (A-indexed) whose treble half holds `pcs_energy`
    (keys are C-relative pitch classes)."""
    tre_c = np.full(12, 0.02)
    for pc, e in pcs_energy.items():
        tre_c[pc % 12] = e
    tre_a = np.roll(tre_c, _ROLL_BACK)       # undo the +9 roll applied downstream
    row = np.concatenate([np.full(12, 0.01), tre_a])
    return np.tile(row, (n_frames, 1)).astype(np.float32)


def _times(n_frames: int = 20, step: float = 0.04644) -> np.ndarray:
    return np.arange(n_frames) * step


def _chord(label: str, t0: float = 0.0, t1: float = 0.8) -> dict:
    return {"label": label, "start_s": t0, "end_s": t1, "confidence": 0.5}


# ── label plumbing ────────────────────────────────────────────────────────
def test_parse_handles_colon_slash_and_colonless():
    assert _parse("A:min") == (9, "min", "")
    assert _parse("A:min/C") == (9, "min", "C")
    assert _parse("Bb:maj7") == (10, "maj7", "")
    assert _parse("Cmin") == (0, "min", "")
    assert _parse("N")[0] is None


def test_relabel_preserves_the_slash_bass_and_ignores_non_min():
    assert _relabel_min7("A:min") == "A:min7"
    assert _relabel_min7("A:min/C") == "A:min7/C"
    assert _relabel_min7("A:min7") == "A:min7"      # already a 7th
    assert _relabel_min7("A:maj") == "A:maj"
    assert _relabel_min7("A:minmaj7") == "A:minmaj7"  # never touch minmaj7


# ── OFF is an exact no-op ─────────────────────────────────────────────────
def test_off_by_default_is_an_exact_no_op(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    arr, times = _frames({9: 0.5, 0: 0.4, 4: 0.4, 7: 0.5}), _times()
    chords = [_chord("A:min")]
    out, n = upgrade_minor_sevenths(chords, arr, times)
    assert n == 0
    assert out == chords
    assert out[0] is chords[0]          # same dict objects, nothing mutated


def test_env_flag_turns_it_on(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "on")
    arr = _frames({9: 0.5, 0: 0.4, 4: 0.5, 7: 0.5})   # A C E + G(=b7 of A)
    out, n = upgrade_minor_sevenths([_chord("A:min")], arr, _times(), theta=0.5)
    assert n == 1 and out[0]["label"] == "A:min7"


# ── the rule itself ───────────────────────────────────────────────────────
def test_upgrades_only_when_the_seventh_is_present():
    strong = _frames({9: 0.5, 0: 0.4, 4: 0.5, 7: 0.5})    # b7 of A = G (pc 7)
    weak = _frames({9: 0.5, 0: 0.4, 4: 0.5})              # no G at all
    hi = seventh_evidence(strong, _times(), 0.0, 0.8, 9, is_minor=True)
    lo = seventh_evidence(weak, _times(), 0.0, 0.8, 9, is_minor=True)
    assert hi > lo * 3

    out_hi, n_hi = upgrade_minor_sevenths([_chord("A:min")], strong, _times(),
                                          theta=1.5, enabled=True)
    out_lo, n_lo = upgrade_minor_sevenths([_chord("A:min")], weak, _times(),
                                          theta=1.5, enabled=True)
    # single-chord songs normalise against themselves -> F/median == 1 < theta
    assert n_hi == 0 and n_lo == 0
    # with a second, 7th-less chord in the song the strong one clears the bar
    song = [_chord("A:min", 0.0, 0.4), _chord("A:min", 0.4, 0.8)]
    mixed = np.concatenate([strong[:10], weak[:10]])
    out, n = upgrade_minor_sevenths(song, mixed, _times(), theta=1.2, enabled=True)
    assert [c["label"] for c in out] == ["A:min7", "A:min"]
    assert n == 1


def test_major_chords_are_never_touched():
    """The maj-side upgrade measured -0.50pp and is deliberately not implemented."""
    arr = _frames({0: 0.5, 4: 0.5, 7: 0.5, 10: 0.6})   # C E G Bb -> a dom7
    song = [_chord("C:maj", 0.0, 0.4), _chord("C:maj", 0.4, 0.8)]
    out, n = upgrade_minor_sevenths(song, arr, _times(), theta=0.1, enabled=True)
    assert n == 0 and [c["label"] for c in out] == ["C:maj", "C:maj"]


def test_decision_is_scale_invariant():
    """Per-song relative normalisation => a global gain change must not matter."""
    a = np.concatenate([_frames({9: .5, 0: .4, 4: .5, 7: .5})[:10],
                        _frames({9: .5, 0: .4, 4: .5})[:10]])
    song = [_chord("A:min", 0.0, 0.4), _chord("A:min", 0.4, 0.8)]
    o1, n1 = upgrade_minor_sevenths(song, a, _times(), theta=1.2, enabled=True)
    o2, n2 = upgrade_minor_sevenths(song, a * 7.5, _times(), theta=1.2, enabled=True)
    assert n1 == n2 == 1
    assert [c["label"] for c in o1] == [c["label"] for c in o2]


def test_empty_and_degenerate_inputs_are_safe():
    arr, times = _frames({0: 0.5}), _times()
    assert upgrade_minor_sevenths([], arr, times, enabled=True) == ([], 0)
    out, n = upgrade_minor_sevenths([_chord("N", 0.0, 0.4)], arr, times, enabled=True)
    assert n == 0 and out[0]["label"] == "N"
    # a span with no frames at all
    out, n = upgrade_minor_sevenths([_chord("A:min", 99.0, 99.5)], arr, times,
                                    enabled=True)
    assert n == 0


@pytest.mark.parametrize("flag", ["off", "0", "", "no"])
def test_falsey_env_values_stay_off(monkeypatch, flag):
    monkeypatch.setenv(ENV_FLAG, flag)
    arr = _frames({9: 0.5, 0: 0.4, 4: 0.5, 7: 0.9})
    out, n = upgrade_minor_sevenths([_chord("A:min")], arr, _times())
    assert n == 0 and out[0]["label"] == "A:min"
