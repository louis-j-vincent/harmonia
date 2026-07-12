"""Unit tests for harmonia/theory/local_key.py.

The load-bearing assumptions of the section-key-highlight and transpose
features: that an iReal token parses to the right root pitch class + quality,
that transposition shifts the root correctly with sensible spelling, and that
the chord-tone Krumhansl match recovers an obvious key.
"""

from __future__ import annotations

from harmonia.theory.local_key import (chord_pcs, continuity_scale_track,
                                        continuity_scale_track_v2, core_tones,
                                        estimate_key, fitting_scales, key_name,
                                        local_key_track, parse_token, prefer_flats,
                                        quality_class, transpose_token)


def test_parse_token_roots_and_bass():
    assert parse_token("C")[0] == 0
    assert parse_token("Ab^7")[0] == 8
    assert parse_token("F#-7")[0] == 6
    assert parse_token("Bb7")[:2] == (10, "7")
    # slash bass
    root, qual, bass = parse_token("D-7/G")
    assert (root, qual, bass) == (2, "-7", 7)


def test_chord_pcs_triads():
    # C major triad → {C, E, G}; C minor → {C, Eb, G}
    assert {0, 4, 7} <= set(chord_pcs("C"))
    assert {0, 3, 7} <= set(chord_pcs("C-"))
    assert {0, 3, 6} <= set(chord_pcs("Co"))          # diminished
    assert {0, 4, 8} <= set(chord_pcs("C+"))          # augmented


def test_transpose_token():
    # up a perfect 4th (5 semitones), flat spelling: C7 → F7, quality kept
    assert transpose_token("C7", 5, flats=True) == "F7"
    assert transpose_token("Ab^7", 2, flats=True) == "Bb^7"
    assert transpose_token("D-7/G", 3, flats=True) == "F-7/Bb"
    # sharp spelling
    assert transpose_token("C7", 6, flats=False) == "F#7"


def test_prefer_flats_and_key_name():
    assert prefer_flats(5, "major") is True          # F major → flats
    assert prefer_flats(7, "major") is False         # G major → sharps
    assert key_name(8, "major") == "Ab major"        # not "G# major"
    assert key_name(10, "minor") == "Bb minor"


def test_estimate_key_recovers_obvious_key():
    # a plain ii-V-I in C major must come back as C major
    est = estimate_key(["D-7", "G7", "C^7"])
    assert (est["tonic"], est["mode"]) == (0, "major")
    # relative minor cadence resolves to A minor
    est = estimate_key(["B-7b5", "E7", "A-"])
    assert (est["tonic"], est["mode"]) == (9, "minor")


def test_quality_class():
    assert quality_class("") == "maj"
    assert quality_class("^7") == "maj"
    assert quality_class("6") == "maj"
    assert quality_class("7") == "dom"
    assert quality_class("7b9") == "dom"
    assert quality_class("-7") == "min"
    assert quality_class("h7") == "m7b5"
    assert quality_class("o7") == "dim"
    assert quality_class("7sus") == "dom"


def test_local_key_track_secondary_dominant():
    # in Bb: a lone G7 before Cm points to C minor, not Bb
    track = local_key_track(["Bb^7", "G7", "C-7", "F7"])
    names = [t["name"] for t in track]
    assert names[0] == "Bb major"
    assert names[1] == "C minor"          # G7 → C(minor), tonicization
    assert names[2] == names[3] == "Bb major"   # Cm7 F7 = ii-V home


def test_local_key_track_dominant_cycle():
    # rhythm-changes bridge: each dominant is a V a fifth down
    track = local_key_track(["D7", "G7", "C7", "F7"])
    assert [t["name"] for t in track] == ["G major", "C major", "F major", "Bb major"]


def test_continuity_holds_scale_until_forced():
    # the Gm7 / Am ambiguity: both fit F major, so we STAY in F major (no jump to
    # G minor / A minor) — a scale holds until a chord tone contradicts it.
    track = continuity_scale_track(["F^7", "G-7", "C7", "A-7"], 5, "major")
    assert [t["name"] for t in track] == ["F major"] * 4


def test_continuity_switches_on_out_of_scale_note():
    # D7 brings F#, not in F major → forced to the nearest fitting collection (G)
    track = continuity_scale_track(["F^7", "D7", "G-7"], 5, "major")
    assert [t["name"] for t in track] == ["F major", "G major", "F major"]


def test_continuity_labels_relative_minor():
    # a region leaning on the vi-minor tonic reads as minor, not its rel major
    track = continuity_scale_track(["G-7", "C-7", "G-7"], 7, "minor")
    assert track[0]["name"] == "G minor"


def test_fitting_scales_ranks_nearest_first():
    # a dominant pins exactly one major collection (its target) as the top scale
    fit = fitting_scales(["C7", "G-7"])
    assert fit[0][0]["name"] == "F major"
    majors = [s for s in fit[0] if s["mode"] == "major"]
    assert len(majors) == 1                       # C7 fits only F among major keys
    assert any(s["mode"] == "major" for s in fit[1])


def test_core_tones_excludes_bass():
    assert core_tones("C/E") == core_tones("C")


# ── continuity_scale_track_v2 (harmonic-minor-aware, #23) ──────────────────────
_AUTUMN_LEAVES = ["C-7", "F7", "Bb^7", "Eb^7", "A-7b5", "D7b13", "G-6"] * 2


def test_v2_autumn_leaves_holds_g_minor():
    # #23 root cause: v1 oscillated Bb/G/F over a static G-minor loop because the
    # D7b13's raised leading tone (F#) and the Gm6's major 6th (E) were treated as
    # out-of-scale. v2 accepts both (harmonic + surgical-melodic) → one stable key.
    track = continuity_scale_track_v2(_AUTUMN_LEAVES, home_tonic=7, home_mode="minor")
    names = [t["name"] for t in track]
    assert set(names) == {"G minor"}          # 0 collection changes, labelled minor


def test_v1_autumn_leaves_oscillates_regression_guard():
    # documents the old broken behaviour (>1 distinct key over the same loop)
    v1 = continuity_scale_track(_AUTUMN_LEAVES, home_tonic=7, home_mode="minor")
    assert len({t["name"] for t in v1}) > 1


def test_v2_minor_key_v7_is_not_a_modulation():
    # D7 (V7 of Gm, raised F#) inside a Gm region must NOT force a scale change
    track = continuity_scale_track_v2(["G-6", "D7", "G-6"], home_tonic=7, home_mode="minor")
    assert [t["name"] for t in track] == ["G minor"] * 3


def test_v2_real_borrowing_jumps_collections():
    # the user's spec case: in C major a Gm7 (Bb) forces F major; then an Eb chord
    # forces Bb major — genuine collection changes, unlike a minor-key V7.
    track = continuity_scale_track_v2(["C^7", "G-7", "Eb^7"], home_tonic=0, home_mode="major")
    assert [t["name"] for t in track] == ["C major", "F major", "Bb major"]


def test_v2_all_the_things_bridge_progresses():
    # ATTYA section B is a genuine progressive modulation; v2 need not be perfect
    # but must move through collections toward the G-major target (not stall).
    B = ["C-7", "F-7", "Bb7", "Eb^7", "Ab^7", "A-7b5", "D7", "G^7"]
    names = [t["name"] for t in continuity_scale_track_v2(B, home_tonic=0, home_mode="major")]
    assert names[0] == "Bb major" and names[-1] == "G major"
    assert len(set(names)) >= 3                        # it modulates, doesn't stall
