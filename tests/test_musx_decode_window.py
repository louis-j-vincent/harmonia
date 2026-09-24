"""tests/test_musx_decode_window.py — decode_window reads the raw triad-plane
posterior for a time window directly, no chart, no template (harmonia/musx.py,
added 2026-09-24 for the San Francisco Street repliement session: a repeated
section-letter's other occurrences are never independently checked in
production, see docs/known_issues.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.musx import FRAME_DT, decode_window


def _triad(n_frame=100):
    return np.zeros((n_frame, 73))


def test_decode_window_picks_the_dominant_root_and_family():
    triad = _triad()
    # root D (pc=2), maj family (1) -> col 1 + 0*12 + 2 = 3
    triad[:, 3] = 0.9
    triad[:, 0] = 0.1
    label, conf = decode_window(triad, 0.0, 50 * FRAME_DT)
    assert label == "D:maj"
    assert conf == pytest.approx(0.9)


def test_decode_window_reads_min_family_not_maj():
    triad = _triad()
    # root E (pc=4), min family (2) -> col 1 + 1*12 + 4 = 17
    triad[:, 17] = 0.8
    triad[:, 0] = 0.2
    label, conf = decode_window(triad, 0.0, 50 * FRAME_DT)
    assert label == "E:min"
    assert conf == pytest.approx(0.8)


def test_decode_window_no_chord_column_reads_n():
    triad = _triad()
    triad[:, 0] = 0.7
    triad[:, 3] = 0.3
    label, conf = decode_window(triad, 0.0, 50 * FRAME_DT)
    assert label == "N"
    assert conf == pytest.approx(0.7)


def test_decode_window_uses_only_its_own_frames_not_a_neighbours():
    """The whole point: a window must never borrow a sibling occurrence's
    evidence. Two adjacent, differently-decoded windows stay independent."""
    triad = _triad(n_frame=200)
    triad[:100, 3] = 0.9    # first half: D:maj
    triad[:100, 0] = 0.1
    triad[100:, 17] = 0.85  # second half: E:min
    triad[100:, 0] = 0.15

    lab1, _ = decode_window(triad, 0.0, 100 * FRAME_DT)
    lab2, _ = decode_window(triad, 100 * FRAME_DT, 200 * FRAME_DT)
    assert lab1 == "D:maj"
    assert lab2 == "E:min"


def test_decode_window_empty_range_returns_neutral():
    triad = _triad()
    label, conf = decode_window(triad, 5.0, 5.0)
    assert label == "N"
    assert conf == 0.0

