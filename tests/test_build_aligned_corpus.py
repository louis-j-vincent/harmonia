"""scripts/build_aligned_corpus.py — the pure quality-family mapper.

Verified against real iReal raw tokens (2026-07-21, via
harmonia.irealb_fetcher._parse_ireal_chord_token on Autumn Leaves/Billie
Jean's actual chart tokens) before trusting this: e.g. "Ah7" ->
(9, "hdim7"), "D7b13" -> (2, "dom13"), "G-6" -> (7, "min"). This module maps
those "sev" strings down to the 7-way family the trained heads use.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_aligned_corpus import _family, QUALITIES


class TestFamily:
    def test_maj_variants(self):
        for sev in ("maj", "maj7", "maj9", "maj11", "maj13", "maj9#11"):
            assert _family(sev) == "maj"

    def test_min_variants(self):
        for sev in ("min", "min7", "min9", "min11", "min13", "minmaj7"):
            assert _family(sev) == "min"

    def test_dom_variants_including_bare_extensions(self):
        for sev in ("7", "9", "dom11", "dom13", "7b9", "7#9"):
            assert _family(sev) == "dom"

    def test_hdim(self):
        assert _family("hdim7") == "hdim"

    def test_dim(self):
        for sev in ("dim", "dim7"):
            assert _family(sev) == "dim"

    def test_aug(self):
        for sev in ("aug", "aug7", "augmaj7"):
            assert _family(sev) == "aug"

    def test_sus(self):
        for sev in ("sus2", "sus4", "7sus4", "9sus4"):
            assert _family(sev) == "sus"

    def test_every_family_is_a_valid_quality_index(self):
        for sev in ("maj", "min7", "9", "hdim7", "dim", "aug", "sus4"):
            assert _family(sev) in QUALITIES
