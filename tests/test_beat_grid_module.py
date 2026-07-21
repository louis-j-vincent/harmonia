"""harmonia/models/beat_grid.py — direct tests against the module's own public
API (2026-07-21 modularity refactor). test_beat_grid.py already covers
bestfit_beat_period through its chord_pipeline_v1 re-export; this file checks
the functions that moved alongside it import and behave correctly from their
new home, independent of chord_pipeline_v1.
"""

import numpy as np

from harmonia.models.beat_grid import (
    attach_musx_onset_hints,
    bestfit_beat_period,
    chroma_flux,
    flux_downbeat_phase,
    structure_anchor_phase,
)


def test_bestfit_beat_period_importable_from_beat_grid():
    beats = 0.1 + (60.0 / 120.0) * np.arange(50)
    fit = bestfit_beat_period(beats, 60.0 / 120.0 * 1.01)
    assert abs(fit - 60.0 / 120.0) / (60.0 / 120.0) < 1e-6


def test_chroma_flux_peaks_at_chord_changes():
    # 4 chords x 20 frames each, treble chroma delta-function per chord —
    # flux should peak exactly on the change frames.
    n_per, n_chords = 20, 4
    arr = np.zeros((n_per * n_chords, 24))
    times = np.arange(n_per * n_chords) / 10.0
    for k in range(n_chords):
        arr[k * n_per:(k + 1) * n_per, 12 + k] = 1.0
    d, fps = chroma_flux(arr, times)
    assert fps == 10.0
    change_frames = [n_per * k for k in range(1, n_chords)]
    non_change = [i for i in range(len(d)) if i not in change_frames and i != 0]
    assert min(d[f] for f in change_frames) > max(d[i] for i in non_change)


def test_flux_downbeat_phase_recovers_known_phase():
    # A 2-beat bar (period=2 frames-of-fps=1 -> bar_period=2s), chord changes
    # every bar starting at phase 1 (not 0) — flux_downbeat_phase should find phi=1.
    fps = 4  # 4 frames/sec
    bar_period = 2.0
    beats_per_bar = 2
    n_bars = 10
    arr = np.zeros((n_bars * int(bar_period * fps), 24))
    times = np.arange(arr.shape[0]) / fps
    frame_per_beat = int(bar_period * fps) // beats_per_bar
    # Put a chord-change spike at beat-phase 1 of every bar (offset by 1 beat
    # from the bar start), to make bar 1 the correct downbeat phase.
    for b in range(n_bars):
        idx = b * beats_per_bar * frame_per_beat + 1 * frame_per_beat
        if idx < arr.shape[0]:
            arr[idx, 12 + (b % 12)] = 5.0
    phi, ratio = flux_downbeat_phase(arr, times, bar_period, beats_per_bar)
    assert phi == 1
    assert ratio > 1.0


def test_structure_anchor_phase_prefers_peaked_loop():
    # 2 phases x 16 bars x 4 beats; phase 0 alternates two crisp one-hot
    # chords per bar (peaked + a lag-2 loop), phase-shifted content at other
    # phases is smeared (mixed root mass) -> phase 0 should win.
    beats_per_bar, max_bars = 4, 16
    nb = max_bars * beats_per_bar
    beat_proba = np.full((nb, 12), 1.0 / 12, dtype=float)
    for b in range(max_bars):
        root = b % 2
        beat_proba[b * beats_per_bar:(b + 1) * beats_per_bar] = 0.0
        beat_proba[b * beats_per_bar:(b + 1) * beats_per_bar, root] = 1.0
    phi, scores = structure_anchor_phase(beat_proba, beats_per_bar, max_bars)
    assert phi == 0
    assert len(scores) == beats_per_bar


def test_attach_musx_onset_hints_snaps_within_tolerance_only():
    period = 0.5
    chords = [
        {"label": "C:maj", "start_s": 1.42},
        {"label": "N", "start_s": 3.0},
        {"label": "G:maj", "start_s": 4.0},
    ]
    mx_labels = [(0.0, 1.18, "C:maj"), (1.18, 4.9, "G:maj"), (4.9, 6.0, "N")]
    n = attach_musx_onset_hints(chords, mx_labels, period)
    assert n == 1
    assert chords[0]["onset_s"] == 1.18
    assert "onset_s" not in chords[1]     # N.C. cell skipped
    # G:maj's start (4.0) is farther than `period` from the nearest change
    # time (1.18) -> no hint attached.
    assert "onset_s" not in chords[2]
