"""Tests for the hierarchical chord tree (harmonia.theory.chord_tree)."""

import numpy as np
import pytest

from harmonia.theory.chord_tree import (
    Family,
    HierarchicalReporter,
    base_seventh_of,
    family_label,
    family_of,
    label_at_depth,
)
from harmonia.theory.chord_vocabulary import ChordQuality, build_index


class TestFamilyMapping:
    @pytest.mark.parametrize("quality,family", [
        (ChordQuality.MAJOR, Family.MAJOR),
        (ChordQuality.MAJ7, Family.MAJOR),
        (ChordQuality.DOM7, Family.MAJOR),
        (ChordQuality.DOM13B9, Family.MAJOR),
        (ChordQuality.MINOR, Family.MINOR),
        (ChordQuality.MIN7, Family.MINOR),
        (ChordQuality.MIN_MAJ7, Family.MINOR),
        (ChordQuality.HALF_DIM7, Family.DIMINISHED),
        (ChordQuality.DIM7, Family.DIMINISHED),
        (ChordQuality.AUG7, Family.AUGMENTED),
        (ChordQuality.SUS4, Family.SUSPENDED),
        (ChordQuality.DOM7SUS4, Family.SUSPENDED),
        (ChordQuality.NO_CHORD, Family.NO_CHORD),
    ])
    def test_family(self, quality, family):
        assert family_of(quality) == family

    def test_every_quality_has_a_family(self):
        """No vocabulary quality may be missing from the tree — else it would
        crash at report time."""
        for q in ChordQuality:
            assert family_of(q) is not None
            assert base_seventh_of(q) is not None


class TestBaseSeventh:
    @pytest.mark.parametrize("quality,base", [
        (ChordQuality.MAJ9, ChordQuality.MAJ7),
        (ChordQuality.MAJ13, ChordQuality.MAJ7),
        (ChordQuality.DOM9, ChordQuality.DOM7),
        (ChordQuality.DOM7B9, ChordQuality.DOM7),
        (ChordQuality.DOM13, ChordQuality.DOM7),
        (ChordQuality.MIN9, ChordQuality.MIN7),
        (ChordQuality.MIN11, ChordQuality.MIN7),
        (ChordQuality.MAJOR, ChordQuality.MAJOR),   # already a triad
    ])
    def test_strips_extensions(self, quality, base):
        assert base_seventh_of(quality) == base


class TestLabels:
    def test_family_labels(self):
        assert family_label(0, ChordQuality.MAJ7) == "C"
        assert family_label(9, ChordQuality.MIN7) == "Am"
        assert family_label(2, ChordQuality.DOM7) == "D"
        assert family_label(11, ChordQuality.DIM7) == "Bdim"
        assert family_label(-1, ChordQuality.NO_CHORD) == "N"

    def test_label_at_depth(self):
        # C13b9 → family "C", seventh "C7", exact "C13b9"
        assert label_at_depth(0, ChordQuality.DOM13B9, 1) == "C"
        assert label_at_depth(0, ChordQuality.DOM13B9, 2) == "C7"
        assert label_at_depth(0, ChordQuality.DOM13B9, 3) == "C13b9"


class TestHierarchicalReporter:
    def setup_method(self):
        self.idx_to_chord, self.chord_to_idx = build_index(max_phase=1)
        self.reporter = HierarchicalReporter(self.idx_to_chord, confidence=0.5)

    def _idx(self, root, quality):
        return self.chord_to_idx[(root, quality)]

    def test_confident_evidence_reports_exact(self):
        """All mass on the exact chord → report depth 3 (exact label)."""
        idx = self._idx(0, ChordQuality.DOM7)
        post = np.full(len(self.idx_to_chord), 1e-6)
        post[idx] = 1.0
        r = self.reporter.report(idx, post)
        assert r.depth == 3
        assert r.label == "C7"
        assert r.family == "C"

    def test_diffuse_evidence_falls_back_to_family(self):
        """Mass spread across the major family but not concentrated on the exact
        seventh → report the family only."""
        post = np.full(len(self.idx_to_chord), 1e-6)
        # spread mass over C major, Cmaj7, C7, C6-less-vocab: several major-family roots=0
        for q in (ChordQuality.MAJOR, ChordQuality.MAJ7, ChordQuality.DOM7):
            post[self._idx(0, q)] = 0.33
        idx = self._idx(0, ChordQuality.DOM7)
        r = self.reporter.report(idx, post)
        assert r.family == "C"
        assert r.depth <= 2      # not confident enough for the exact chord
        assert r.label in ("C", "C7")

    def test_no_chord(self):
        idx = self._idx(-1, ChordQuality.NO_CHORD)
        post = np.full(len(self.idx_to_chord), 1e-6)
        post[idx] = 1.0
        r = self.reporter.report(idx, post)
        assert r.label == "N"
        assert r.family == "N"

    def test_confidence_one_pins_to_family(self):
        reporter = HierarchicalReporter(self.idx_to_chord, confidence=1.0)
        idx = self._idx(0, ChordQuality.DOM7)
        post = np.full(len(self.idx_to_chord), 1e-6)
        post[idx] = 0.9  # high but a hair of mass elsewhere → ratio < 1.0
        post[self._idx(0, ChordQuality.MAJ7)] = 0.1
        r = reporter.report(idx, post)
        assert r.depth == 1
        assert r.label == "C"

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            HierarchicalReporter(self.idx_to_chord, confidence=0.0)
        with pytest.raises(ValueError):
            HierarchicalReporter(self.idx_to_chord, confidence=1.5)
