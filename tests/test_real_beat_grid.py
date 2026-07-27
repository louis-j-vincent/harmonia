"""Unit tests for the REAL-BEAT GRID brick (harmonia/models/beat_grid.py).

The load-bearing property is the one CLAUDE.md's kill-switch convention demands:
with ``HARMONIA_REAL_BEAT_GRID`` unset (or ``off``) both entry points are EXACT
no-ops — ``apply_real_beat_grid`` returns the very same array object and
``snap_chord_times_to_beats`` does not touch a single field.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import beat_grid as bg


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(bg.REAL_BEAT_GRID_ENV, raising=False)


def _lattice(period=0.5, dur=20.0, phase=0.1):
    bt = np.arange(phase, dur + period, period)
    return np.unique(np.concatenate([[0.0], bt, [dur]]))


def _real(period=0.5, dur=20.0, phase=0.1, jitter=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(phase, dur, period)
    return t + rng.normal(0, jitter, len(t))


def _chords(bt, labels=("C:maj", "A:min", "F:maj", "G:7")):
    """Contiguous chords whose boundaries sit exactly on the lattice."""
    idx = np.linspace(0, len(bt) - 1, len(labels) + 1).astype(int)
    out, segs = [], []
    for i, lab in enumerate(labels):
        t0, t1 = float(bt[idx[i]]), float(bt[idx[i + 1]])
        out.append({"label": lab, "start_s": round(t0, 3), "end_s": round(t1, 3),
                    "duration_beats": 4, "confidence": 0.5})
        segs.append({"start_s": round(t0, 3), "end_s": round(t1, 3),
                     "key": "C major", "n_beats": 4})
    return out, segs


# ── mode parsing ───────────────────────────────────────────────────────────

def test_default_mode_is_off():
    assert bg.real_beat_grid_mode() == "off"


@pytest.mark.parametrize("val,want", [
    ("off", "off"), ("0", "off"), ("", "off"), ("snap", "snap"),
    ("GRID", "grid"), (" Snap ", "snap"), ("banana", "off"),
])
def test_mode_parsing(monkeypatch, val, want):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, val)
    assert bg.real_beat_grid_mode() == want


# ── OFF is an EXACT no-op ──────────────────────────────────────────────────

def test_grid_off_returns_the_same_object():
    bt = _lattice()
    out, info = bg.apply_real_beat_grid(bt, _real(), 20.0, 0.5)
    assert out is bt                      # identity, not merely equal
    assert info["applied"] is False


def test_snap_off_does_not_mutate(monkeypatch):
    bt = _lattice()
    chords, segs = _chords(bt)
    before = [dict(c) for c in chords]
    info = bg.snap_chord_times_to_beats(chords, segs, _real(), 20.0, 0.5)
    assert info["applied"] is False
    assert chords == before


def test_snap_mode_still_no_op_without_beats(monkeypatch):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "snap")
    bt = _lattice()
    chords, segs = _chords(bt)
    before = [dict(c) for c in chords]
    assert bg.snap_chord_times_to_beats(chords, segs, None, 20.0, 0.5)["applied"] is False
    assert chords == before


# ── guards ─────────────────────────────────────────────────────────────────

def test_guard_passes_on_a_dense_grid():
    ok, stats = bg.real_grid_guard(_real(), 20.0, 0.5)
    assert ok, stats
    assert abs(stats["gapfrac"]) < 0.1


def test_guard_refuses_a_gappy_grid():
    """Blue Bossa's failure mode: 25% of the beats never detected."""
    t = _real()
    keep = np.ones(len(t), dtype=bool)
    keep[::4] = False                     # drop a quarter of the beats
    ok, stats = bg.real_grid_guard(t[keep], 20.0, 0.5)
    assert not ok
    assert "coverage" in stats["reason"]


def test_guard_refuses_a_tempo_octave():
    """CLAUDE.md song-002 trap: the tracker locked the 2x-fast octave."""
    ok, stats = bg.real_grid_guard(_real(period=0.25, dur=20.0), 20.0, 0.5)
    assert not ok
    assert "octave" in stats["reason"]


def test_grid_mode_refused_grid_returns_the_lattice(monkeypatch):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "grid")
    bt = _lattice()
    t = _real()
    keep = np.ones(len(t), dtype=bool)
    keep[::4] = False
    out, info = bg.apply_real_beat_grid(bt, t[keep], 20.0, 0.5)
    assert out is bt
    assert info["applied"] is False


# ── ON behaviour ───────────────────────────────────────────────────────────

def test_grid_mode_uses_detected_beats(monkeypatch):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "grid")
    bt = _lattice()
    t = _real()
    out, info = bg.apply_real_beat_grid(bt, t, 20.0, 0.5)
    assert info["applied"] is True
    assert out[0] == 0.0 and out[-1] == 20.0     # endpoint convention preserved
    assert np.all(np.diff(out) > 0)              # strictly increasing
    # every interior cell edge is a detected beat
    interior = out[1:-1]
    assert np.max(np.abs(interior[:, None] - t[None, :]).min(1)) < 1e-9


def test_snap_moves_boundaries_onto_beats_and_stays_monotone(monkeypatch):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "snap")
    bt = _lattice()
    t = _real()
    chords, segs = _chords(bt)
    first0, last0 = chords[0]["start_s"], chords[-1]["end_s"]
    info = bg.snap_chord_times_to_beats(chords, segs, t, 20.0, 0.5)
    assert info["applied"] is True
    starts = [c["start_s"] for c in chords]
    ends = [c["end_s"] for c in chords]
    assert starts == sorted(starts) and all(e > s for s, e in zip(starts, ends))
    for a, b in zip(chords[:-1], chords[1:]):
        assert a["end_s"] == b["start_s"]        # contiguity
    # endpoints pinned -> the scored span is identical before and after
    assert chords[0]["start_s"] == pytest.approx(first0)
    assert chords[-1]["end_s"] == pytest.approx(last0)
    assert segs[0]["start_s"] == chords[0]["start_s"]


def test_snap_never_moves_more_than_half_a_beat(monkeypatch):
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "snap")
    bt = _lattice()
    t = _real()
    chords, segs = _chords(bt)
    orig = [c["start_s"] for c in chords] + [chords[-1]["end_s"]]
    bg.snap_chord_times_to_beats(chords, segs, t, 20.0, 0.5)
    now = [c["start_s"] for c in chords] + [chords[-1]["end_s"]]
    assert max(abs(a - b) for a, b in zip(orig, now)) <= 0.25 + 1e-6


def test_grid_mode_also_snaps_boundaries_second_lattice_fix(monkeypatch):
    """SECOND-LATTICE FIX (2026-07-27): the snap must ALSO run in ``grid`` mode.

    The Occam post-pass re-grids the chart onto its own uniform bar lattice, so
    without this the ``grid`` decode was silently undone wherever Occam fired
    (stand_by_me came out bit-identical to ``off``).  Here the chords sit on the
    synthetic lattice (== Occam's re-grid); after the snap every interior
    boundary must land on a DETECTED beat, and ``applied`` must be True (it was
    False — a no-op — before the fix)."""
    monkeypatch.setenv(bg.REAL_BEAT_GRID_ENV, "grid")
    bt = _lattice()
    t = _real()
    chords, segs = _chords(bt)
    info = bg.snap_chord_times_to_beats(chords, segs, t, 20.0, 0.5)
    assert info["applied"] is True                     # ran in grid mode
    for c in chords[1:]:                               # interior starts (b_1..b_n-1)
        assert min(abs(c["start_s"] - float(x)) for x in t) < 1e-3
    # contiguity + monotonicity preserved
    for a, b in zip(chords[:-1], chords[1:]):
        assert a["end_s"] == b["start_s"]
