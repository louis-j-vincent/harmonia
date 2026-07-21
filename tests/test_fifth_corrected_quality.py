"""_fifth_corrected_quality (2026-07-21) — the min7<->hdim7 / min<->dim
direct-audio tie-break. Confirmed empirically on real "Autumn Leaves" audio
first (see the function's own docstring + docs/known_issues.md) before
writing this: several `min7` calls had 2-3x more diminished-5th chroma
energy than perfect-5th — stronger evidence than the one bar the model
itself correctly called `hdim7`.
"""
from __future__ import annotations

import numpy as np

from harmonia.models.chord_pipeline_v1 import _fifth_corrected_quality


def _treble(root: int, p5_energy: float, b5_energy: float) -> np.ndarray:
    t = np.full(12, 0.05, dtype=np.float32)   # low baseline everywhere else
    t[root] = 1.0
    t[(root + 7) % 12] = p5_energy
    t[(root + 6) % 12] = b5_energy
    return t


class TestFifthCorrectedQuality:
    def test_corrects_min7_to_hdim7_when_b5_dominates(self):
        treble = _treble(root=2, p5_energy=0.2, b5_energy=1.5)   # like the real G:min7 case
        assert _fifth_corrected_quality("min7", 2, treble) == "hdim7"

    def test_corrects_hdim7_to_min7_when_p5_dominates(self):
        treble = _treble(root=2, p5_energy=1.5, b5_energy=0.2)
        assert _fifth_corrected_quality("hdim7", 2, treble) == "min7"

    def test_leaves_min7_alone_when_p5_dominates_already_correct(self):
        treble = _treble(root=5, p5_energy=1.6, b5_energy=0.3)
        assert _fifth_corrected_quality("min7", 5, treble) == "min7"

    def test_abstains_on_ambiguous_margin(self):
        # b5 slightly louder but not by a decisive margin -> leave alone
        treble = _treble(root=0, p5_energy=0.9, b5_energy=1.0)
        assert _fifth_corrected_quality("min7", 0, treble) == "min7"

    def test_abstains_when_both_bins_are_noise_floor(self):
        treble = np.full(12, 0.02, dtype=np.float32)
        assert _fifth_corrected_quality("min7", 3, treble) == "min7"

    def test_min_dim_triad_pair_also_corrected(self):
        treble = _treble(root=7, p5_energy=0.15, b5_energy=1.2)
        assert _fifth_corrected_quality("min", 7, treble) == "dim"

    def test_unrelated_qualities_pass_through_untouched(self):
        treble = _treble(root=0, p5_energy=0.1, b5_energy=2.0)
        assert _fifth_corrected_quality("maj7", 0, treble) == "maj7"
        assert _fifth_corrected_quality("dom7", 0, treble) == "dom7"
        assert _fifth_corrected_quality("sus4", 0, treble) == "sus4"
