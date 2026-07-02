"""
Unit tests for the core theory layer.

These tests run without any audio files or ML models — they only exercise
the music theory logic, which must be correct before anything else can work.
"""

import numpy as np
import pytest

from harmonia.theory.chord_vocabulary import (
    ChordQuality,
    build_index,
    chord_label,
    get_template,
    get_vocabulary,
    n_chords,
    pitch_class,
)
from harmonia.theory.jazz_priors import (
    PROGRESSIONS,
    STYLE_PRIORS,
    RelativeChord,
    infer_style_posteriors,
)
from harmonia.theory.key_profiles import (
    KEY_PROFILES,
    N_KEYS,
    activations_to_chroma,
    detect_modulations,
    infer_key,
)


# ── Chord vocabulary ──────────────────────────────────────────────────────────

class TestChordVocabulary:

    def test_phase1_count(self):
        # 15 qualities × 12 roots + 1 no-chord = 181
        assert n_chords(1) == 181

    def test_phase2_larger_than_phase1(self):
        assert n_chords(2) > n_chords(1)

    def test_build_index_bijective(self):
        idx_to_chord, chord_to_idx = build_index(max_phase=1)
        # Round-trip
        for i, c in enumerate(idx_to_chord):
            assert chord_to_idx[c] == i

    def test_no_chord_is_last(self):
        idx_to_chord, _ = build_index(max_phase=1)
        root, quality = idx_to_chord[-1]
        assert quality == ChordQuality.NO_CHORD

    def test_chord_label(self):
        assert chord_label(0, ChordQuality.MAJ7) == "Cmaj7"
        assert chord_label(7, ChordQuality.DOM7) == "G7"
        assert chord_label(10, ChordQuality.MIN7) == "A#min7"
        assert chord_label(-1, ChordQuality.NO_CHORD) == "N"

    def test_template_intervals(self):
        # C major triad must contain intervals 0, 4, 7
        t = get_template(ChordQuality.MAJOR)
        assert frozenset({0, 4, 7}) == t.intervals

    def test_template_weights_in_range(self):
        for q, t in [(q, get_template(q)) for q in get_vocabulary(max_phase=4)]:
            if q == ChordQuality.NO_CHORD:
                continue
            for interval, w in t.weights.items():
                assert 0 < w <= 1.0, f"{q}: weight {w} out of range for interval {interval}"

    def test_dom7_has_tritone(self):
        # Dominant 7th must have the tritone (3rd + b7th = interval 4 and 10)
        t = get_template(ChordQuality.DOM7)
        assert 4 in t.intervals   # major 3rd
        assert 10 in t.intervals  # minor 7th

    def test_dim7_symmetry(self):
        # Fully diminished 7th has equal intervals (all minor 3rds = 3 semitones)
        t = get_template(ChordQuality.DIM7)
        intervals = sorted(t.intervals)
        diffs = [intervals[i+1] - intervals[i] for i in range(len(intervals)-1)]
        assert all(d == 3 for d in diffs)

    def test_weight_vector_length(self):
        t = get_template(ChordQuality.MAJ7)
        vec = t.to_weight_vector()
        assert len(vec) == 12


# ── Jazz priors ───────────────────────────────────────────────────────────────

class TestJazzPriors:

    def test_all_progressions_have_chords(self):
        for name, prog in PROGRESSIONS.items():
            assert len(prog.chords) >= 2, f"{name} has fewer than 2 chords"

    def test_ii_v_i_major_relative(self):
        prog = PROGRESSIONS["ii_V_I_major"]
        intervals = [rc.interval for rc in prog.chords]
        # iim7 = interval 2, V7 = interval 7, Imaj7 = interval 0
        assert intervals == [2, 7, 0]

    def test_ii_v_i_instantiation_in_c(self):
        prog = PROGRESSIONS["ii_V_I_major"]
        chords = prog.instantiate(tonic=0)  # C major
        roots = [c[0] for c in chords]
        qualities = [c[1] for c in chords]
        assert roots == [2, 7, 0]          # D, G, C
        assert qualities == [ChordQuality.MIN7, ChordQuality.DOM7, ChordQuality.MAJ7]

    def test_ii_v_i_instantiation_in_g(self):
        prog = PROGRESSIONS["ii_V_I_major"]
        chords = prog.instantiate(tonic=7)  # G major
        roots = [c[0] for c in chords]
        assert roots == [9, 2, 7]           # A, D, G

    def test_ii_v_i_instantiation_in_bb(self):
        prog = PROGRESSIONS["ii_V_I_major"]
        chords = prog.instantiate(tonic=10)  # Bb major
        roots = [c[0] for c in chords]
        assert roots == [0, 5, 10]           # C, F, Bb

    def test_tritone_sub_interval(self):
        prog = PROGRESSIONS["tritone_sub_major"]
        intervals = [rc.interval for rc in prog.chords]
        # bII = interval 1 (tritone sub)
        assert 1 in intervals

    def test_tritone_sub_is_tritone_from_v(self):
        # bII is exactly a tritone (6 semitones) above/below V (interval 7)
        # bII = interval 1 — difference from V (7) = 6 semitones mod 12
        assert (7 - 1) % 12 == 6

    def test_all_styles_have_tempo_range(self):
        for name, style in STYLE_PRIORS.items():
            lo, hi = style.tempo_range
            assert lo < hi, f"{name}: tempo range invalid"
            assert style.tempo_mode >= lo and style.tempo_mode <= hi

    def test_style_posteriors_sum_to_one(self):
        posteriors = infer_style_posteriors(130.0)
        assert abs(sum(posteriors.values()) - 1.0) < 1e-9

    def test_bebop_wins_at_high_tempo(self):
        posteriors = infer_style_posteriors(260.0)
        best = max(posteriors, key=posteriors.get)
        assert best == "bebop"

    def test_ballad_wins_at_low_tempo(self):
        posteriors = infer_style_posteriors(50.0)
        best = max(posteriors, key=posteriors.get)
        assert best == "jazz_ballad"

    def test_blues_has_12bar_structure(self):
        prog = PROGRESSIONS["blues_I_IV_V"]
        assert len(prog.chords) == 12


# ── Key profiles ──────────────────────────────────────────────────────────────

class TestKeyProfiles:

    def test_profile_matrix_shape(self):
        assert KEY_PROFILES.shape == (N_KEYS, 12)

    def test_profiles_sum_to_one(self):
        # Each row should sum to 1 (normalised distributions)
        row_sums = KEY_PROFILES.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_c_major_tonic_highest_weight(self):
        # In C major profile (row 0), C (index 0) should have highest weight
        assert KEY_PROFILES[0].argmax() == 0

    def test_g_major_tonic_highest_weight(self):
        # In G major profile (row 7), G (index 7) should have highest weight
        assert KEY_PROFILES[7].argmax() == 7

    def test_infer_key_c_major(self):
        # Strong C major chroma: C, E, G, B prominent
        chroma = np.zeros(12)
        chroma[[0, 4, 7, 11]] = [1.0, 0.8, 0.6, 0.7]  # C, E, G, B
        chroma[[2, 5, 9]] = [0.4, 0.5, 0.3]            # D, F, A (also in C major)
        result = infer_key(chroma)
        assert result.tonic == 0
        assert result.mode == "major"

    def test_infer_key_a_minor(self):
        # Strong A minor chroma: A, C, E prominent, with D, G, F
        chroma = np.zeros(12)
        chroma[[9, 0, 4]] = [1.0, 0.8, 0.7]   # A, C, E (Am triad)
        chroma[[2, 7, 5]] = [0.5, 0.5, 0.4]   # D, G, F (diatonic to A minor)
        result = infer_key(chroma)
        assert result.tonic == 9  # A
        assert result.mode == "minor"

    def test_infer_key_returns_top3(self):
        chroma = np.ones(12) / 12  # flat chroma
        result = infer_key(chroma)
        top3 = result.top_k(3)
        assert len(top3) == 3
        # Probabilities should sum close to something reasonable
        assert all(p > 0 for _, p in top3)

    def test_activations_to_chroma_shape(self):
        fake_activations = np.random.rand(100, 88).astype(np.float32)
        chroma = activations_to_chroma(fake_activations)
        assert chroma.shape == (12,)

    def test_activations_to_chroma_scales_with_frame_count_not_amplitude(self):
        # infer_key needs the aggregate magnitude to reflect real evidence
        # (how many frames of signal, not how loud they are) to calibrate
        # its posterior — see TestKeyInferenceCalibration. Each frame is
        # L1-normalised individually before summing, so:
        #   - uniformly scaling amplitude must NOT change the result (each
        #     frame's shape is invariant to its own overall scale)
        #   - more frames of the same content MUST increase the total
        #     (that's the real "more evidence" signal)
        rng = np.random.default_rng(0)
        fake_activations = rng.random((100, 88)).astype(np.float32)
        chroma = activations_to_chroma(fake_activations)
        scaled_chroma = activations_to_chroma(fake_activations * 10.0)
        np.testing.assert_allclose(scaled_chroma, chroma, rtol=1e-4)

        doubled_activations = np.concatenate([fake_activations, fake_activations], axis=0)
        doubled_chroma = activations_to_chroma(doubled_activations)
        np.testing.assert_allclose(doubled_chroma, chroma * 2.0, rtol=1e-4)

    def test_detect_modulations_none(self):
        # Same key throughout = no modulations
        from harmonia.theory.key_profiles import KeyPosterior
        kp = KeyPosterior(
            log_probs=np.zeros(N_KEYS),
            tonic=0, mode="major", key_name="C major", confidence=0.9,
        )
        mods = detect_modulations([kp, kp, kp])
        assert mods == []

    def test_detect_modulations_one(self):
        from harmonia.theory.key_profiles import KeyPosterior
        c_major = KeyPosterior(np.zeros(N_KEYS), 0, "major", "C major", 0.9)
        g_major = KeyPosterior(np.zeros(N_KEYS), 7, "major", "G major", 0.85)
        mods = detect_modulations([c_major, c_major, g_major, g_major])
        assert mods == [2]


# ── Key inference calibration (docs/known_issues.md #0) ────────────────────
#
# infer_key() used to treat a bounded correlation score (dot product of two
# L1-normalised distributions, mathematically confined to a narrow range) as
# a log-likelihood. exp() of a value confined to ~[0.06, 0.16] can never
# produce more than ~10% relative concentration across 24 keys, no matter
# how unambiguous the input — every segment of every song came out at
# ~0.043 confidence (~1/24 = uniform), regardless of how clean the chroma
# was. These tests pin down calibrated behaviour: confidence must be able
# to concentrate sharply given strong evidence, and must scale with the
# amount of evidence (more observed energy => more concentrated posterior),
# which is the whole point of treating this as a proper multinomial
# likelihood over raw (unnormalised) chroma counts rather than a bounded
# correlation over pre-normalised distributions.
class TestKeyInferenceCalibration:

    def test_confidence_concentrates_given_strong_unambiguous_evidence(self):
        # Raw chroma proportional to the C-major KS profile itself, scaled up
        # to a realistic total activation count for a real segment. This is
        # about as unambiguous as chroma evidence gets for "C major" — the
        # posterior should concentrate sharply on it, not sit near 1/24.
        chroma_raw = KEY_PROFILES[0] * 500.0  # tonic=0 => "C major" row
        result = infer_key(chroma_raw)
        assert result.key_name == "C major"
        assert result.confidence > 0.5, (
            f"confidence {result.confidence:.4f} still near-uniform "
            f"(1/24={1/24:.4f}) despite unambiguous, high-magnitude evidence"
        )

    def test_confidence_increases_with_evidence_magnitude(self):
        # Same normalised shape (same relative pitch-class distribution),
        # different total magnitude. More evidence, same shape, must yield a
        # *more* concentrated posterior -- this is the secondary bug: the
        # old code's Dirichlet-style sharpening term was neutralised because
        # the caller pre-normalised chroma to sum to 1 before infer_key ever
        # saw it, so "total" was ~1.0 regardless of real evidence.
        shape = KEY_PROFILES[7]  # "G major" row, used as an arbitrary shape
        weak = infer_key(shape * 2.0)
        strong = infer_key(shape * 200.0)
        assert strong.confidence > weak.confidence

    def test_confidence_uniform_when_no_evidence(self):
        # All-zero chroma = no evidence at all. With a uniform prior, the
        # posterior must reduce exactly to the prior: 1/24 everywhere.
        result = infer_key(np.zeros(12))
        assert result.confidence == pytest.approx(1.0 / N_KEYS, abs=1e-9)

    def test_song001_segment_no_longer_near_uniform(self):
        # Real chroma shape from docs/handoff_2026-07-02_key_inference.md §4
        # (song 001, 35-beat segment, 38.6s-62.1s) -- the exact case that
        # exposed the bug: all 16 segments of song 001 resolved to "F#
        # major" with bit-for-bit identical confidence, 0.043, regardless of
        # segment length/content. The shape below is that segment's
        # L1-normalised chroma; scaled to a plausible raw activation total
        # for a 35-beat segment, it should no longer be a coin flip.
        chroma_norm = np.array([
            0.0079, 0.2111, 0.0197, 0.1004, 0.0146, 0.0540,
            0.2588, 0.0094, 0.0769, 0.0185, 0.1215, 0.1071,
        ])
        chroma_raw = chroma_norm * 500.0
        result = infer_key(chroma_raw)
        assert result.key_name == "F# major"
        assert result.confidence > 0.3, (
            f"confidence {result.confidence:.4f} -- the old bug capped this "
            f"around 0.043 (~1/24) for every segment regardless of content"
        )
