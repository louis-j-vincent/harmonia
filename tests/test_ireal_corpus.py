"""Unit tests for the iReal → MMA conversion layer (harmonia.data.ireal_corpus)."""

import pytest

from harmonia.data.ireal_corpus import (
    IREAL_TO_MMA,
    chord_root_pc,
    split_chords,
    style_to_groove,
    to_mma_chord,
)


class TestSplitChords:
    def test_single_chord(self):
        assert split_chords("D-7") == ["D-7"]

    def test_two_chords_concatenated(self):
        assert split_chords("Eh7A7b9") == ["Eh7", "A7b9"]

    def test_two_chords_with_qualities(self):
        assert split_chords("G-7C7") == ["G-7", "C7"]

    def test_slash_bass_not_split(self):
        assert split_chords("Ab/Gb") == ["Ab/Gb"]

    def test_slash_bass_then_chord(self):
        assert split_chords("C/EG7") == ["C/E", "G7"]

    def test_flat_root(self):
        assert split_chords("Bb7Eb^7") == ["Bb7", "Eb^7"]

    def test_glued_no_chord(self):
        assert split_chords("A7n") == ["A7", "n"]

    def test_invisible_root_tokens(self):
        assert split_chords("EW/GW/A") == ["E", "W/G", "W/A"]


class TestToMMAChord:
    @pytest.mark.parametrize(
        "ireal,expected",
        [
            ("F^7", "FM7"),
            ("E-7", "Em7"),
            ("Eh7", "Em7b5"),
            ("A7b9", "A7b9"),
            ("C", "C"),
            ("D-", "Dm"),
            ("Bo7", "Bdim7"),
            ("G7sus", "G7sus4"),
            ("C7alt", "C7alt"),
            ("F#-7", "F#m7"),
            ("Bb^9", "BbM9"),
            ("A-^7", "AmM7"),
            ("C69", "C69"),
            ("Ab/Gb", "Ab/Gb"),
            ("n", "z"),
            ("W", "z"),
            ("W/G", "G5"),
            ("W/DN2", "D5"),
            ("Ebo^7", "Ebdim7(addM7)"),
            ("D^7U", "DM7"),
            ("D7at", "D7alt"),
            ("A7N2", "A7"),
        ],
    )
    def test_mapping(self, ireal, expected):
        assert to_mma_chord(ireal) == expected

    def test_unknown_quality_returns_none(self):
        assert to_mma_chord("Cweird") is None

    def test_all_mapped_qualities_are_valid_mma(self):
        """Every target quality in the map must exist in MMA's chord table.

        Skipped when MMA isn't installed (data/tools) — run
        scripts/fetch_accompaniment_deps.sh first.
        """
        import sys
        from pathlib import Path

        mma_dir = Path(__file__).parent.parent / "data" / "tools" / "mma-bin-25.05.3"
        if not mma_dir.exists():
            pytest.skip("MMA not installed")
        sys.path.insert(0, str(mma_dir))
        from MMA import chordtable

        valid = set(chordtable.chordlist.keys())
        for ireal_q, mma_q in IREAL_TO_MMA.items():
            if mma_q:  # '' = plain major, always valid
                assert mma_q in valid, f"{ireal_q!r} maps to invalid MMA quality {mma_q!r}"


class TestChordRootPC:
    @pytest.mark.parametrize(
        "chord,pc", [("C", 0), ("C#m7", 1), ("Bb7", 10), ("F#m7b5", 6), ("z", None)]
    )
    def test_roots(self, chord, pc):
        assert chord_root_pc(chord) == pc


class TestStyleToGroove:
    def test_medium_swing(self):
        assert style_to_groove("Medium Swing", (4, 4)) == ("Swing", 140)

    def test_bossa(self):
        groove, _ = style_to_groove("Bossa Nova", (4, 4))
        assert groove == "BossaNova"

    def test_waltz_time_overrides_style(self):
        groove, _ = style_to_groove("Medium Swing", (3, 4))
        assert groove == "JazzWaltz"

    def test_unknown_style_falls_back_to_swing(self):
        assert style_to_groove("Zydeco Stomp", (4, 4))[0] == "Swing"


class TestSectionizedMeasures:
    def test_sections_survive_flattening(self):
        """Integration: parse a real mini iReal chart and check labels + repeat expansion."""
        pytest.importorskip("pyRealParser")
        from harmonia.data.ireal_corpus import sectionized_measures

        class FakeTune:
            chord_string = "*A{T44C |A-7 |D-7 |G7 }*B[F^7 |E-7 |D-7 |C6 ]"

        measures = sectionized_measures(FakeTune())
        labels = [lab for lab, _ in measures]
        chords = [m for _, m in measures]
        # A section repeats: 8 bars of A, then 4 bars of B
        assert labels == ["A"] * 8 + ["B"] * 4
        assert chords[0] == "C"
        assert chords[4] == "C"  # repeat expanded
        assert chords[8] == "F^7"
