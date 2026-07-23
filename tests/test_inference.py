"""Pure-core unit tests for the chord-inference brick (``harmonia.align.inference``).

Numpy-only + audio-free: every test drives the decoder through synthetic chroma /
bass arrays (the ``decode_from_features`` entry) or the pure helpers, so the suite
runs with no audio, no Beat This!, and no network. This mirrors the fusion suite's
convention (validate the STATE-SPACE core; the audio front-end is exercised only by
the offline validation harness).
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.align import inference as inf


# ── synthetic chroma helpers ─────────────────────────────────────────────────

def _chord_chroma(root: int, ivs, noise=0.0, rng=None):
    """A clean 12-chroma for a chord (unit mass on the chord tones), + noise."""
    v = np.zeros(12)
    for iv in ivs:
        v[(root + iv) % 12] = 1.0
    v = v / v.sum()
    if noise and rng is not None:
        v = v + noise * rng.random(12)
        v = v / v.sum()
    return v


# ═════════════════════════════════════════════════════════════════════════════
# Vocabulary + templates
# ═════════════════════════════════════════════════════════════════════════════

def test_vocab_shape_and_roots():
    v = inf.build_vocab()
    C = 12 * len(inf.DECODE_QUALITIES)
    assert v.templates.shape == (C, 12)
    assert len(v.roots) == C and len(v.qualities) == C and len(v.families) == C
    # every template is centre-normed (zero mean, unit norm)
    assert np.allclose(v.templates.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(v.templates, axis=1), 1.0, atol=1e-9)


def test_vocab_root_transposition():
    """A C-maj and a G-maj template are the same shape rolled by 7 semitones."""
    v = inf.build_vocab(["maj"])
    cmaj = v.templates[np.where((v.roots == 0))[0][0]]
    gmaj = v.templates[np.where((v.roots == 7))[0][0]]
    assert np.allclose(np.roll(cmaj, 7), gmaj, atol=1e-9)


# ═════════════════════════════════════════════════════════════════════════════
# Emission — the right chord scores highest on its own chroma
# ═════════════════════════════════════════════════════════════════════════════

def test_emission_picks_correct_chord():
    v = inf.build_vocab()
    # a clean C major triad chroma
    cn = inf._centre_norm(_chord_chroma(0, [0, 4, 7])[None, :])
    E = inf.emission_matrix(cn, v)
    best = int(np.argmax(E[0]))
    assert v.roots[best] == 0 and v.qualities[best] == "maj"


def test_emission_min_vs_maj_third():
    """A minor triad chroma must prefer min over maj (the 3rd is the only diff)."""
    v = inf.build_vocab()
    cn = inf._centre_norm(_chord_chroma(9, [0, 3, 7])[None, :])   # A minor
    E = inf.emission_matrix(cn, v)
    best = int(np.argmax(E[0]))
    assert v.roots[best] == 9 and inf._family_of(v.qualities[best]) == "min"


def test_bass_stream_breaks_root_tie():
    """With an ambiguous chroma, a reliable bass on G should pull the root to G."""
    v = inf.build_vocab(["maj", "min"])
    # flat-ish chroma (near tie); bass firmly on G (pc 7)
    cn = inf._centre_norm(np.ones((1, 12)) + 0.001 * np.eye(12)[0][None, :])
    apc = np.array([7])
    rel = np.array([1.0])
    E = inf.emission_matrix(cn, v, apc, rel, w_bass=1.0)
    best = int(np.argmax(E[0]))
    assert v.roots[best] == 7


# ═════════════════════════════════════════════════════════════════════════════
# Key estimation (Krumhansl-Schmuckler)
# ═════════════════════════════════════════════════════════════════════════════

def test_estimate_key_c_major():
    # a C-major scale mass -> C:maj
    chroma = np.zeros(12)
    for pc in inf._MAJOR_SCALE:
        chroma[pc] = 1.0
    chroma[0] += 2.0  # emphasise the tonic/dominant so KS locks C not A-min
    chroma[7] += 1.0
    k = inf.estimate_key(chroma)
    assert k.tonic_pc == 0 and k.mode == "maj"


def test_estimate_key_a_minor():
    chroma = np.zeros(12)
    for pc in (9, 11, 0, 2, 4, 5, 7):   # A natural minor
        chroma[pc] = 1.0
    chroma[9] += 2.5
    chroma[4] += 1.0
    k = inf.estimate_key(chroma)
    assert k.tonic_pc == 9 and k.mode == "min"


# ═════════════════════════════════════════════════════════════════════════════
# Transition prior — self-transition free, diatonic + fifth bonuses
# ═════════════════════════════════════════════════════════════════════════════

def test_transition_self_is_free_and_max():
    v = inf.build_vocab()
    key = inf.KeyEstimate(0, "maj", 0.5)
    T = inf.build_transition(v, key)
    assert np.allclose(np.diag(T), 0.0)
    # every off-diagonal transition costs less than the free self-transition
    off = T.copy()
    np.fill_diagonal(off, -np.inf)
    assert off.max() < 0.0


def test_transition_prefers_diatonic_target():
    """In C major, a change into G:maj (diatonic V) beats a change into Ab:maj."""
    v = inf.build_vocab(["maj"])
    key = inf.KeyEstimate(0, "maj", 0.5)
    T = inf.build_transition(v, key)
    c = int(np.where(v.roots == 0)[0][0])
    g = int(np.where(v.roots == 7)[0][0])
    ab = int(np.where(v.roots == 8)[0][0])
    assert T[c, g] > T[c, ab]


def test_fifth_bonus_prefers_fourth_over_tritone():
    assert inf._fifth_bonus(0, 5) > inf._fifth_bonus(0, 6)
    assert inf._fifth_bonus(0, 7) == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Viterbi — the DBN prior smooths flicker the argmax leaves in
# ═════════════════════════════════════════════════════════════════════════════

def test_viterbi_smooths_single_beat_flicker():
    """A held chord with ONE beat that WEAKLY prefers a neighbour (margin below the
    change cost): argmax flickers, the DBN persistence prior holds the chord.

    Emission is built by hand so the flicker margin is controlled (a small +eps at
    one beat) — the point is that a sub-change-cost preference must NOT trigger a
    change, exactly the flicker the no-prior baseline leaves in."""
    v = inf.build_vocab()
    key = inf.KeyEstimate(0, "maj", 0.5)
    C = len(v.qualities)
    c0 = int(np.where((v.roots == 0) & (np.array(v.qualities) == "maj"))[0][0])
    c1 = int(np.where((v.roots == 2) & (np.array(v.qualities) == "min"))[0][0])
    E = np.zeros((8, C))
    E[:, c0] = 5.0                       # every beat strongly prefers C:maj
    E[4, c0] = 0.0                       # beat 4: the C evidence drops out ...
    E[4, c1] = 0.1                       # ... and a neighbour wins by a tiny margin
    path = inf.viterbi_decode(E, inf.build_transition(v, key))
    assert len(np.unique(path)) == 1 and int(path[0]) == c0   # DBN holds C:maj
    amax = inf.argmax_decode(E)
    assert amax[4] == c1 and len(np.unique(amax)) > 1          # baseline flickers


def test_viterbi_follows_a_real_change():
    """C for 4 beats then G for 4 beats: the DBN must still emit both chords."""
    v = inf.build_vocab()
    beats = np.arange(8) * 0.5
    rows = [_chord_chroma(0, [0, 4, 7])] * 4 + [_chord_chroma(7, [0, 4, 7])] * 4
    cn = inf._centre_norm(np.array(rows))
    E = inf.emission_matrix(cn, v)
    key = inf.KeyEstimate(0, "maj", 0.5)
    path = inf.viterbi_decode(E, inf.build_transition(v, key))
    roots = [int(v.roots[c]) for c in path]
    assert roots[:4] == [0, 0, 0, 0]
    assert roots[4:] == [7, 7, 7, 7]


# ═════════════════════════════════════════════════════════════════════════════
# Coalescing + sounding bass + the end-to-end pure decode
# ═════════════════════════════════════════════════════════════════════════════

def test_coalesce_spans_and_times():
    v = inf.build_vocab(["maj"])
    beats = np.array([0.0, 0.5, 1.0, 1.5])
    path = np.array([0, 0, 7 * 1, 7 * 1])  # two roots (C then whatever idx maps)
    # build a plausible emission (only used for the confidence field)
    cn = inf._centre_norm(np.array([_chord_chroma(0, [0, 4, 7])] * 2
                                   + [_chord_chroma(1, [0, 4, 7])] * 2))
    E = inf.emission_matrix(cn, v)
    # map path to real chord indices for roots 0 and 1
    c0 = int(np.where(v.roots == 0)[0][0])
    c1 = int(np.where(v.roots == 1)[0][0])
    path = np.array([c0, c0, c1, c1])
    spans = inf.coalesce_path(path, beats, v, E)
    assert len(spans) == 2
    assert spans[0]["t0"] == 0.0 and spans[0]["root_pc"] == 0
    assert spans[1]["root_pc"] == 1
    # contiguous, non-overlapping
    assert spans[0]["t1"] == spans[1]["t0"]


def test_coalesce_uses_sounding_bass():
    """A slash bass: chord root C but a reliable sounding bass on E -> bass_pc=E."""
    v = inf.build_vocab(["maj"])
    beats = np.array([0.0, 0.5])
    c0 = int(np.where(v.roots == 0)[0][0])
    path = np.array([c0, c0])
    cn = inf._centre_norm(np.array([_chord_chroma(0, [0, 4, 7])] * 2))
    E = inf.emission_matrix(cn, v)
    apc = np.array([4, 4])       # sounding bass on E
    rel = np.array([1.0, 1.0])
    spans = inf.coalesce_path(path, beats, v, E, audio_bass_pc=apc, bass_rel=rel)
    assert spans[0]["root_pc"] == 0 and spans[0]["bass_pc"] == 4
    assert spans[0]["label"].endswith("/E")


def test_decode_from_features_end_to_end():
    """Full pure decode: C-C-G-G chroma -> two spans C then G, with confidence."""
    v_q = inf.DECODE_QUALITIES
    dec = inf.FusionChordDecoder(method="dbn")
    beats = np.arange(8) * 0.5
    rows = [_chord_chroma(0, [0, 4, 7])] * 4 + [_chord_chroma(7, [0, 4, 7])] * 4
    cn = inf._centre_norm(np.array(rows))
    mean_chroma = np.array(rows).mean(axis=0)
    out = dec.decode_from_features(beats, cn, mean_chroma)
    assert isinstance(out, inf.ChordInference)
    roots = [c["root_pc"] for c in out.chords]
    assert roots == [0, 7]
    assert 0.0 <= out.whole_song_confidence <= 1.0
    assert out.chords[0]["quality"] == "maj"


def test_tonic_baseline_is_constant():
    dec = inf.FusionChordDecoder(method="tonic")
    beats = np.arange(6) * 0.5
    # A-minor-ish chroma
    rows = [_chord_chroma(9, [0, 3, 7])] * 6
    cn = inf._centre_norm(np.array(rows))
    out = dec.decode_from_features(beats, cn, np.array(rows).mean(axis=0))
    assert len(out.chords) == 1  # a single held tonic
    assert out.chords[0]["root_pc"] == out.key.tonic_pc


def test_empty_input_is_safe():
    dec = inf.FusionChordDecoder()
    out = dec.decode_from_features(np.zeros(0), np.zeros((0, 12)), np.ones(12) / 12)
    assert out.chords == [] and out.whole_song_confidence == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
