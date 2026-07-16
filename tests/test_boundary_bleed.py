"""Boundary-bleed regression test (docs/known_issues.md "boundary bleed").

Constructs a synthetic 2-chord audio timeline where chord A (C major) occupies
[0.0, 1.0) and chord B (F major) occupies [1.0, 2.0). Beats fall on a grid whose
last A-beat STRADDLES the 1.0 s boundary. The old whole-beat-snap pooling
(seg_feature over b0:b1) pulls F-major energy into chord A's feature; the
frame-clipped pooling (seg_feature_clipped) does not.
"""
import numpy as np
import pytest

from harmonia.data.yt_chord_corpus import (
    seg_feature_abs, seg_feature_abs_clipped,
)

FR = 86.1328125  # basic-pitch frame rate


def _midi(pc, octave=4):
    return 12 * (octave + 1) + pc - 21  # -> 88-key index (A0=0)


def _synth():
    dur = 2.0
    n = int(dur * FR)
    ft = np.arange(n) / FR
    onset_f = np.zeros((n, 88), np.float32)
    note_f = np.zeros((n, 88), np.float32)
    # chord A = C major triad in [0,1); chord B = F major triad in [1,2)
    A = [_midi(0), _midi(4), _midi(7)]      # C E G
    B = [_midi(5), _midi(9), _midi(0)]      # F A C
    for i, t in enumerate(ft):
        keys = A if t < 1.0 else B
        for k in keys:
            onset_f[i, k] = 1.0
            note_f[i, k] = 1.0
    return ft, onset_f, note_f


def _beat_pool(ft, onset_f, note_f, beat_times):
    nb = len(beat_times) - 1
    ob = np.zeros((nb, 88), np.float32); nb_ = np.zeros((nb, 88), np.float32)
    for b in range(nb):
        m = (ft >= beat_times[b]) & (ft < beat_times[b + 1])
        if m.any():
            ob[b] = onset_f[m].sum(0); nb_[b] = note_f[m].sum(0)
    return ob, nb_


def test_beatgrid_bleeds_and_clipped_does_not():
    ft, onset_f, note_f = _synth()
    # beats at 0,0.5,0.9,1.3,... -> the beat [0.9,1.3) straddles the 1.0 boundary
    beat_times = np.array([0.0, 0.5, 0.9, 1.3, 1.7, 2.0])
    ob, nb = _beat_pool(ft, onset_f, note_f, beat_times)

    t0, t1 = 0.0, 1.0
    # OLD whole-beat snap: b0=beat containing t0, b1=beat containing t1 (inclusive)
    b0 = int(np.searchsorted(beat_times, t0, side="right")) - 1
    b1 = int(np.searchsorted(beat_times, t1, side="right"))
    grid_feat = seg_feature_abs(ob, nb, max(b0, 0), b1)
    clip_feat = seg_feature_abs_clipped(ft, onset_f, note_f, t0, t1)

    # chroma of the onset block (first 12 dims). F=pc5, A=pc9 are B-only tones.
    grid_chroma = grid_feat[:12]
    clip_chroma = clip_feat[:12]
    # The straddling beat [0.9,1.3) includes ~0.3s of chord B, so grid pooling
    # leaks F (pc5) and A (pc9) energy into chord A.
    assert grid_chroma[5] > 0.05, "expected F-major bleed in beat-grid feature"
    assert grid_chroma[9] > 0.05, "expected A bleed in beat-grid feature"
    # Clipped pooling sees ONLY [0,1) = pure C major -> no F/A onset energy.
    assert clip_chroma[5] < 1e-6, f"clipped feature leaked F: {clip_chroma[5]}"
    assert clip_chroma[9] < 1e-6, f"clipped feature leaked A: {clip_chroma[9]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
