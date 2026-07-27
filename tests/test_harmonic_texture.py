"""Contract tests for the STATED/IMPLIED detector + the implied-harmony policy.

Two load-bearing properties, both cheap and both the kind that has silently
broken before in this repo:

  * the PRE-REGISTERED threshold constant is exactly 0.45 and sits strictly
    between the two ear-anchor clusters (CLAUDE.md #1: unit-test the constant
    against its external reference, here Louis's anchors);
  * the policy is an exact no-op with an empty span list, so wiring it changes
    nothing until a detector actually fires.

Audio-free: the span/mask/policy layer is pure, so it is tested on synthetic
curves.  The one audio-dependent claim (that the promoted module reproduces the
overlay spans bit-identically) is a research check, not a unit test.
"""
import numpy as np
import pytest

from harmonia.models import harmonic_texture as ht
from harmonia.models.no_chord_policy import (
    hold_short_implied, hold_through_implied, implied_policy_mode,
    implied_segments, stated_chord_duration, state_implied_mask,
)


# ── the pre-registered constant ──────────────────────────────────────────────
def test_threshold_is_pinned_and_in_the_anchor_gap():
    """0.45 was fixed on the ear anchors BEFORE any score was computed."""
    assert ht.IMPLIED_REL_THR == 0.45
    implied_anchor_max = 0.33     # blue_bossa contrabass solo, the loudest implied
    stated_anchor_min = 0.67      # bein_green, the thinnest stated
    assert implied_anchor_max < ht.IMPLIED_REL_THR < stated_anchor_min


def test_band_edges_sit_above_the_bass_register():
    """MIDI 55 = G3: an upright bass tops out around here, so energy above it
    is comping/melody rather than the bass line being counted twice."""
    assert ht.BAND_LO_MIDI < ht.BASS_TOP_MIDI < ht.BAND_HI_MIDI
    assert ht.BASS_TOP_MIDI == 55


# ── span extraction ─────────────────────────────────────────────────────────
def _curve(rel, dt=0.1):
    rel = np.asarray(rel, float)
    t = np.arange(len(rel)) * dt
    return ht.TextureCurve(times=t, rel=rel, raw=rel, median=1.0)


def test_implied_spans_finds_a_long_dip():
    rel = np.ones(200)
    rel[50:150] = 0.2                      # 10 s dip at dt=0.1
    spans = ht.implied_spans(_curve(rel))
    assert len(spans) == 1
    a, b = spans[0]
    assert a == pytest.approx(5.0, abs=0.15)
    assert b == pytest.approx(15.0, abs=0.15)


def test_short_dips_are_dropped_not_merged():
    """A 1 s gap between two comped chords is a gap in the comping, not an
    implied passage."""
    rel = np.ones(200)
    rel[50:60] = 0.2                       # 1 s only, below MIN_SPAN_S
    assert ht.implied_spans(_curve(rel)) == []


def test_threshold_is_honoured():
    rel = np.full(200, 0.5)                # between 0.45 and 0.60
    assert ht.implied_spans(_curve(rel)) == []
    assert len(ht.implied_spans(_curve(rel), threshold=0.6)) == 1


def test_implied_mask_marks_only_inside_spans():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = ht.implied_mask(t, [[1.0, 3.0]])
    assert list(m) == [False, True, True, False, False]


def test_env_override_is_diagnostic_only(monkeypatch):
    assert ht.spans_from_env() == ht.IMPLIED_REL_THR
    monkeypatch.setenv("HARMONIA_IMPLIED_THR", "0.6")
    assert ht.spans_from_env() == 0.6
    monkeypatch.setenv("HARMONIA_IMPLIED_THR", "not-a-number")
    assert ht.spans_from_env() == ht.IMPLIED_REL_THR


# ── the policy layer ────────────────────────────────────────────────────────
def test_policy_default_is_off():
    assert implied_policy_mode() == "off"


def test_state_implied_mask_is_exact_noop_without_spans():
    mask = np.array([True, False, True])
    bounds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    out = state_implied_mask(mask, bounds, [])
    assert np.array_equal(out, mask)


def test_state_implied_mask_clears_only_inside_spans():
    mask = np.array([True, True, True])
    bounds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    out = state_implied_mask(mask, bounds, [[0.9, 2.1]])
    assert list(out) == [True, False, True]      # only the middle segment


def test_implied_segments_uses_the_midpoint_not_overlap():
    """Span edges are soft to ~1 s, so an overlap test would drag whole
    neighbouring chords in."""
    bounds = [(0.0, 10.0), (10.0, 11.0)]
    # the span clips only the tail of segment 0 -> its midpoint (5 s) is outside,
    # while segment 1's midpoint (10.5 s) is inside
    assert list(implied_segments(bounds, [[9.5, 10.8]])) == [False, True]


def test_hold_is_exact_noop_without_spans():
    chords = [{"start_s": 0.0, "end_s": 1.0, "label": "C:maj"},
              {"start_s": 1.0, "end_s": 2.0, "label": "N"}]
    assert hold_through_implied(chords, []) == chords


def test_hold_replaces_n_and_marks_it_inferred():
    chords = [{"start_s": 0.0, "end_s": 1.0, "label": "C:maj"},
              {"start_s": 1.0, "end_s": 2.0, "label": "N"},
              {"start_s": 2.0, "end_s": 3.0, "label": "N"}]
    out = hold_through_implied(chords, [[1.0, 2.0]])
    assert out[1]["label"] == "C:maj"
    assert out[1]["inferred"] is True
    assert out[2]["label"] == "N"          # outside the span: untouched
    assert "inferred" not in out[2]
    assert chords[1]["label"] == "N"       # input not mutated


def test_hold_leaves_a_leading_n_alone():
    """No predecessor to hold — inventing one would be worse than the hole."""
    chords = [{"start_s": 0.0, "end_s": 1.0, "label": "N"}]
    out = hold_through_implied(chords, [[0.0, 1.0]])
    assert out[0]["label"] == "N"
    assert "inferred" not in out[0]


def test_stated_chord_duration_ignores_n_and_implied_cells():
    chords = [{"start_s": 0.0, "end_s": 2.0, "label": "C:maj"},
              {"start_s": 2.0, "end_s": 4.0, "label": "F:maj"},
              {"start_s": 4.0, "end_s": 30.0, "label": "N"},
              {"start_s": 30.0, "end_s": 40.0, "label": "G:maj"}]
    # the 26 s N cell must not count; the 10 s cell inside the span must not either
    assert stated_chord_duration(chords, [[30.0, 40.0]]) == pytest.approx(2.0)


def test_hold_short_refuses_spans_longer_than_a_chord():
    """The measured outcome on all 7 frozen songs: no span qualifies."""
    chords = [{"start_s": 0.0, "end_s": 2.0, "label": "C:maj"},
              {"start_s": 2.0, "end_s": 4.0, "label": "F:maj"},
              {"start_s": 4.0, "end_s": 14.0, "label": "N"}]
    out = hold_short_implied(chords, [[4.0, 14.0]])       # 10 s span, 2 s chords
    assert out[2]["label"] == "N"
    assert not any("inferred" in c for c in out)


def test_hold_short_allows_a_span_shorter_than_a_chord():
    chords = [{"start_s": 0.0, "end_s": 4.0, "label": "C:maj"},
              {"start_s": 4.0, "end_s": 6.0, "label": "N"},
              {"start_s": 6.0, "end_s": 10.0, "label": "F:maj"}]
    out = hold_short_implied(chords, [[4.0, 6.0]])        # 2 s span, 4 s chords
    assert out[1]["label"] == "C:maj"
    assert out[1]["inferred"] is True


def test_hold_short_is_noop_without_spans():
    chords = [{"start_s": 0.0, "end_s": 1.0, "label": "C:maj"},
              {"start_s": 1.0, "end_s": 2.0, "label": "N"}]
    assert hold_short_implied(chords, []) == chords


def test_hold_does_not_touch_stated_chords():
    chords = [{"start_s": 0.0, "end_s": 1.0, "label": "C:maj"},
              {"start_s": 1.0, "end_s": 2.0, "label": "F:maj"}]
    out = hold_through_implied(chords, [[0.0, 2.0]])
    assert [c["label"] for c in out] == ["C:maj", "F:maj"]
    assert not any("inferred" in c for c in out)
