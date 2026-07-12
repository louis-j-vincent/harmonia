"""Unit tests for the issue-#20 diatonic quality prior (chord_pipeline_v1)."""
from harmonia.models.chord_pipeline_v1 import apply_diatonic_prior as A

# C major throughout: tonic=0, mode="major", reliable key (key_conf=0.9),
# uncertain acoustics (conf=0.4) unless noted.


def test_flips_nondiatonic_triad_to_diatonic():
    # ii (D) mislabelled maj -> should snap to min
    assert A(2, "maj", 0.4, 0, "major", 0.9) == "min"


def test_preserves_seventh_extension():
    # ii mislabelled maj7 -> min7 (keeps the seventh, only fixes third)
    assert A(2, "maj7", 0.4, 0, "major", 0.9) == "min7"


def test_keeps_already_diatonic():
    assert A(0, "maj", 0.4, 0, "major", 0.9) == "maj"      # I
    assert A(7, "7", 0.4, 0, "major", 0.9) == "7"          # V dom7


def test_confident_acoustics_bypass_prior():
    # conf >= threshold_chromatic -> trust acoustic, no override
    assert A(2, "maj", 0.9, 0, "major", 0.9) == "maj"


def test_unreliable_key_bypasses_prior():
    # key_conf < key_conf_min -> no override
    assert A(2, "maj", 0.4, 0, "major", 0.1) == "maj"


def test_chromatic_root_is_passthrough():
    # deg 1 (Db) is not in the major diatonic table -> keep acoustic call
    assert A(1, "maj", 0.4, 0, "major", 0.9) == "maj"


def test_sus_and_aug_untouched():
    assert A(2, "sus4", 0.4, 0, "major", 0.9) == "sus4"


def test_boost_tunes_flip_threshold():
    # weak boost + fairly-high conf (still < thresh) should not flip
    assert A(2, "maj", 0.6, 0, "major", 0.9, diatonic_boost=1.0) == "maj"
    # strong boost flips the same case
    assert A(2, "maj", 0.6, 0, "major", 0.9, diatonic_boost=4.0) == "min"


def test_minor_key_degrees():
    # A minor (tonic=9). ii° (B, deg 2) mislabelled maj -> dim
    assert A(11, "maj", 0.4, 9, "minor", 0.9) == "dim"
    # bIII (C, deg 3) mislabelled min -> maj
    assert A(0, "min", 0.4, 9, "minor", 0.9) == "maj"
