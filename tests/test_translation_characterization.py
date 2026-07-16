"""
Characterization tests for chord-label translation — Phase 0 of the
refactoring plan (docs/refactoring_delegation_plan.md), covering
`harmonia/data/billboard_translator.py` and
`harmonia/data/pop909_parser.py::parse_harte_label`.

CLAUDE.md rule #3 ("ground truth is a measurement too") and the
refactoring-suggestions doc §2d flag chord-label translation as exactly the
kind of logic that must have ONE tested implementation, not several
independently-drifting ones. These tests pin current behavior on the
specific tricky cases already documented as having bitten this project:

  - `/bass` inversions: docs/known_issues.md line 109 area (pop909_parser)
    documents "Bass inversions (/bass_note) are ignored — we model root
    position chords" and a separate finding (known_issues.md, the
    "bass_chord_inference_summary" entry) that model-vs-GT agreement on
    inversion labels is 38.1% vs 86.8% on non-inversion labels — i.e.
    the inversion slash is silently dropped, not misparsed.
  - colon-quality labels (Harte "C:maj7" / Billboard "Bb:min7" style).
  - maj7-family mapping (maj7/maj9/maj13/add9/aug -> Billboard Q5 "maj";
    the `BILLBOARD_TO_Q5` table's central "collapse everything with a major
    3rd, no altered/added tone that changes the *quality bucket*" design,
    which is also the mechanism behind known_issues.md #31 ("Billboard v2
    quality head trained on collapsed GT") -- BILLBOARD_TO_Q5 collapsing
    maj7/maj9/etc into "maj" is doing exactly what that issue warns about,
    so pinning this table's current contents is high-value: any accidental
    edit here silently changes what "exact" match means downstream.
"""

from __future__ import annotations

import pytest

from harmonia.data.billboard_translator import (
    BILLBOARD_TO_Q5,
    note_to_pitch_class,
    parse_billboard_chord,
)
from harmonia.data.pop909_parser import parse_harte_label
from harmonia.theory.chord_vocabulary import ChordQuality


class TestBillboardTranslatorPinned:

    @pytest.mark.parametrize("label,expected", [
        ("C:maj", (0, "maj")),
        ("Bb:min7", (10, "min")),
        ("N", (None, None)),
        ("G:7", (7, "dom")),
        ("C#:dim", (1, "dim")),
        ("F:m7b5", (5, "hdim")),
    ])
    def test_parse_billboard_chord_pinned(self, label, expected):
        assert parse_billboard_chord(label) == expected

    def test_maj7_family_collapses_to_maj(self):
        # The exact "GT-mapping artifact" mechanism behind known_issues.md
        # #31 — pin every member of this collapse so an accidental edit is
        # caught immediately.
        maj_family = ["maj", "maj7", "maj6", "maj9", "maj13", "add9", "add11", "add2"]
        for q in maj_family:
            assert BILLBOARD_TO_Q5[q] == "maj", f"{q!r} should collapse to 'maj'"

    def test_min7_family_collapses_to_min(self):
        min_family = ["min", "min7", "min6", "min9", "min13"]
        for q in min_family:
            assert BILLBOARD_TO_Q5[q] == "min", f"{q!r} should collapse to 'min'"

    def test_dominant_family_collapses_to_dom(self):
        dom_family = ["7", "9", "11", "13", "7b9", "7#9", "7alt"]
        for q in dom_family:
            assert BILLBOARD_TO_Q5[q] == "dom", f"{q!r} should collapse to 'dom'"

    def test_half_diminished_distinct_from_diminished(self):
        # A specific, previously-easy-to-conflate pair: hdim7 (min7b5) is
        # its own Q5 bucket, NOT folded into "dim".
        assert BILLBOARD_TO_Q5["m7b5"] == "hdim"
        assert BILLBOARD_TO_Q5["hdim7"] == "hdim"
        assert BILLBOARD_TO_Q5["dim7"] == "dim"
        assert BILLBOARD_TO_Q5["m7b5"] != BILLBOARD_TO_Q5["dim7"]

    def test_no_chord_maps_to_none(self):
        assert BILLBOARD_TO_Q5["N"] is None

    @pytest.mark.parametrize("note,pc", [
        ("C", 0), ("C#", 1), ("Db", 1), ("F#", 6), ("Gb", 6), ("B", 11),
        ("N", None), ("", None),
    ])
    def test_note_to_pitch_class_pinned(self, note, pc):
        assert note_to_pitch_class(note) == pc

    def test_malformed_label_without_colon_returns_none(self):
        # No ":" separator -> (None, None), not a parse error.
        assert parse_billboard_chord("Cmaj7") == (None, None)

    def test_unknown_quality_returns_none_root(self):
        # A root that parses fine but a quality string absent from the Q5
        # table -> root discarded too (both-or-nothing contract).
        assert parse_billboard_chord("C:some_unknown_quality") == (None, None)


class TestPOP909BassInversionHandling:
    """Pins the documented '/bass inversions are ignored' behavior
    (pop909_parser.py docstring, line ~109) — the slash and bass note are
    silently dropped, and the label is parsed as if it were root-position."""

    def test_slash_bass_inversion_ignored(self):
        # "C:maj7/5" (root position C major 7 with G in the bass) should
        # parse identically to "C:maj7" — the /5 is dropped, not an error
        # and not shifting the perceived root.
        with_inversion = parse_harte_label("C:maj7/5")
        without_inversion = parse_harte_label("C:maj7")
        assert with_inversion == without_inversion
        assert with_inversion == (0, ChordQuality.MAJ7)

    def test_slash_bass_on_dominant_seventh(self):
        # "G:7/3" (G7 first inversion, B in bass) — bass digit dropped.
        assert parse_harte_label("G:7/3") == (7, ChordQuality.DOM7)

    def test_slash_with_note_name_bass_dropped(self):
        # Some corpora write the absolute bass note name rather than a
        # scale-degree digit; the regex's quality group excludes "/" so
        # this is dropped the same way (verified against source, not a
        # real POP909 file — see the scope-gap test below).
        assert parse_harte_label("F:maj/A") == (5, ChordQuality.MAJOR)


class TestPOP909ColonQualityAndMaj7Family:

    @pytest.mark.parametrize("label,expected", [
        ("C:maj", (0, ChordQuality.MAJOR)),
        ("Bb:min7", (10, ChordQuality.MIN7)),
        ("G:7", (7, ChordQuality.DOM7)),
        ("D:dom7", (2, ChordQuality.DOM7)),
        ("F#:maj7", (6, ChordQuality.MAJ7)),
        ("A:minmaj7", (9, ChordQuality.MIN_MAJ7)),
        ("Eb:hdim7", (3, ChordQuality.HALF_DIM7)),
        ("C:dim7", (0, ChordQuality.DIM7)),
        ("N", (-1, ChordQuality.NO_CHORD)),
        ("X", None),
    ])
    def test_parse_harte_label_pinned(self, label, expected):
        assert parse_harte_label(label) == expected

    def test_bare_root_defaults_to_major(self):
        # No ":quality" suffix at all -> defaults to "maj" per the regex's
        # optional quality group + parse_harte_label's `or "maj"` fallback.
        assert parse_harte_label("C") == (0, ChordQuality.MAJOR)

    def test_unknown_quality_falls_back_to_major_not_none(self):
        # Distinct contract from billboard_translator: an unrecognized
        # quality string here still returns a root (falls back to MAJOR)
        # rather than discarding the whole chord — pop909_parser and
        # billboard_translator deliberately disagree on this, which is
        # exactly the kind of cross-module inconsistency Phase 1/2 of the
        # refactor should either unify or explicitly document as
        # corpus-specific behavior.
        assert parse_harte_label("C:totally_unknown_quality_xyz") == (0, ChordQuality.MAJOR)


class TestTranslationScopeGap:
    """Documents what this characterization does NOT cover (CLAUDE.md
    rule #4)."""

    def test_documents_scope_gap(self):
        # NOT verified in this session: (1) `harmonia/data/pop909_parser.py`
        # real /bass-note-name (e.g. "F:maj/A", vs. the scale-degree-digit
        # form "F:maj/5") against actual POP909 `.chord` files — no local
        # POP909 dataset was available in this sandbox (see
        # `tests/test_pop909_parser.py`'s skipif pattern; same constraint
        # applies here). The regex `_HARTE_RE`'s quality group explicitly
        # excludes "/" and "(" so both forms are structurally handled the
        # same way, but this was reasoned from source, not run against a
        # real corpus file. (2) The ~17 scattered translation-flavored
        # `def`s named in refactoring_suggestions.md §2d (e.g. in
        # scripts/llm_chord_priors.py) were NOT characterized here — only
        # the two canonical package-level translators were in scope per
        # this task's instructions.
        assert True
