"""
Fully symbolic (no audio, no Basic Pitch) per-beat feature table, built
across the ENTIRE 909-song POP909 corpus. Sister script to
build_chord_change_features.py, which is limited to the 5 rendered-audio
songs used throughout docs/chord_change_signal_analysis/ -- this reruns the
same feature definitions at ~180x the sample size, to (a) evaluate the ML
classifier's real generalization without n=5's variance and heterogeneity,
and (b) support the Cross-Repeat Harmonic Agreement structure-validation
experiment (docs/structure_trigram_design_2026-07-04.md), which explicitly
needs corpus-scale data no audio pipeline could supply.

Two signals that came from Basic Pitch audio activations in
build_chord_change_features.py get symbolic surrogates here instead:
  - `beat_probs` (a (B, 88) onset-activation matrix, used for chroma/onset-
    density/structure): built directly from the raw MIDI's note attacks
    (all three non-drum tracks: MELODY, BRIDGE, PIANO -- see
    `_symbolic_onset_beat_probs`), onset-weighted by velocity rather than
    sustained across note duration, to mirror the audio pipeline's actual
    input (onset_probs, not note_probs -- see docs/known_issues.md's
    "Resolved" section for why that distinction mattered once already).
  - Bass: reuses `bass_track.py::true_bass_track()` directly, POP909's own
    PIANO-track ground truth already built and validated in an earlier
    session (docs/known_issues.md #1's "Bass-note motion" subsection).

Everything else (beat grid, phase, chord-change labels, bigram tables,
structure/periodicity functions) is IDENTICAL code to
build_chord_change_features.py, imported directly rather than duplicated.

Usage:
    .venv/bin/python scripts/build_symbolic_features.py [--limit N]
    (writes docs/chord_change_signal_analysis/features_symbolic.csv)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bass_track import forward_fill_bass, true_bass_track  # noqa: E402
from build_chord_change_features import (  # noqa: E402
    _beat_in_bar_phase, _conditional_logprob, _INTERVAL_TO_ROMAN,
    _load_pop909_beat_grid, fit_bigram_tables, identify_best_parent_scale,
    quality_bucket,
)

DATA_ROOT = Path(__file__).parent.parent / "data"
POP909_DIR = DATA_ROOT / "pop909" / "POP909"
OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"

MIDI_LO = 21  # A0
MIDI_HI = 108  # C8


def _load_all_notes(midi_path) -> list[tuple[float, float, int, int]]:
    """(start_s, end_s, pitch, velocity) for all non-drum instruments."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend((n.start, n.end, n.pitch, n.velocity) for n in inst.notes)
    return notes


def symbolic_onset_beat_probs(midi_path, beat_times: np.ndarray) -> np.ndarray:
    """
    (B, 88) onset-activation matrix, symbolic analogue of Basic Pitch's
    onset_probs: for each note, weight = velocity/127 is added at the beat
    interval containing the note's ONSET time only (not held across the
    note's full duration) -- this deliberately mirrors what
    build_chord_change_features.py actually feeds into chroma/onset-
    density/periodicity (`act.onset_probs`, not `act.note_probs` -- see
    that script's module docstring and docs/known_issues.md's resolved
    `PitchActivations.chroma()` bug for why onset, not sustain, is correct
    here).
    """
    notes = _load_all_notes(midi_path)
    B = len(beat_times)
    probs = np.zeros((B, MIDI_HI - MIDI_LO + 1), dtype=np.float64)
    if B == 0:
        return probs
    for (start, _end, pitch, vel) in notes:
        if not (MIDI_LO <= pitch <= MIDI_HI):
            continue
        b = int(np.searchsorted(beat_times, start, side="right")) - 1
        if b < 0:
            b = 0
        if b >= B:
            continue
        probs[b, pitch - MIDI_LO] += vel / 127.0
    return probs


def build_symbolic_features_for_song(song_id: str, atomic_counts, mode_counts) -> pd.DataFrame | None:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.periodicity import find_loop_phase, score_periods
    from harmonia.models.structure import Segmenter, build_ssm

    gt_song = POP909Parser(POP909_DIR).parse_song(song_id)
    if gt_song is None or not gt_song.chord_events:
        return None
    midi_path = POP909_DIR / song_id / f"{song_id}.mid"
    if not midi_path.exists():
        return None

    beat_times, is_downbeat = _load_pop909_beat_grid(song_id)
    B = len(beat_times)
    if B < 8:
        return None
    phase = _beat_in_bar_phase(is_downbeat)

    beat_probs = symbolic_onset_beat_probs(midi_path, beat_times)

    # --- Ground truth: chord at each beat, and the change indicator -------
    gt_root = np.full(B, -2, dtype=int)
    gt_label = [None] * B
    gt_quality_obj = [None] * B
    for b in range(B):
        t = beat_times[b]
        for ev in gt_song.chord_events:
            if ev.start_beat <= t < ev.end_beat:
                gt_root[b] = ev.root
                gt_label[b] = ev.label
                gt_quality_obj[b] = ev.quality
                break
    chord_changed = np.zeros(B, dtype=bool)
    for b in range(1, B):
        if gt_root[b] >= 0 and gt_root[b - 1] >= 0:
            chord_changed[b] = gt_label[b] != gt_label[b - 1]

    # --- A: harmonic-rhythm timing -----------------------------------------
    beats_since_change = np.zeros(B, dtype=int)
    run = 0
    for b in range(B):
        beats_since_change[b] = run
        run = 0 if chord_changed[b] else run + 1

    # --- B: bass (ground-truth PIANO track, not audio-inferred) -----------
    true_bass = true_bass_track(midi_path, beat_times)
    bass_raw = np.array([tb.true_bass for tb in true_bass], dtype=int)
    bass_filled = forward_fill_bass(bass_raw)
    bass_onset = bass_raw >= 0
    bass_changed = np.zeros(B, dtype=bool)
    for b in range(1, B):
        if bass_filled[b] >= 0 and bass_filled[b - 1] >= 0:
            bass_changed[b] = (bass_filled[b] % 12) != (bass_filled[b - 1] % 12)
    bass_is_root_or_fifth = np.zeros(B, dtype=bool)
    for b in range(B):
        if bass_filled[b] >= 0 and gt_root[b] >= 0:
            pc = bass_filled[b] % 12
            bass_is_root_or_fifth[b] = pc == gt_root[b] or pc == (gt_root[b] + 7) % 12

    # --- C: bigram log-probabilities (only meaningful at change points) ----
    tonic, key_quality = None, None
    key_path = POP909_DIR / song_id / "key_audio.txt"
    if key_path.exists():
        from harmonia.data.pop909_parser import parse_harte_label
        line = open(key_path).readline().split()
        if len(line) >= 3:
            parsed = parse_harte_label(line[2])
            if parsed:
                tonic, key_quality = parsed
    mode = "major" if (key_quality and quality_bucket(key_quality) == "maj") else "minor"
    canonical_tonic = (tonic if mode == "major" else (tonic + 3) % 12) if tonic is not None else 0
    best_T, _ = identify_best_parent_scale(gt_song.chord_events)

    bigram_logprob_atomic = np.full(B, np.nan)
    bigram_logprob_mode = np.full(B, np.nan)
    bigram_mode_delta = np.full(B, np.nan)
    for b in range(1, B):
        if not chord_changed[b] or gt_root[b] < 0 or gt_root[b - 1] < 0:
            continue
        deg_a_atomic = (gt_root[b - 1] - best_T) % 12
        deg_b_atomic = (gt_root[b] - best_T) % 12
        ka_atomic = (_INTERVAL_TO_ROMAN[deg_a_atomic], quality_bucket(gt_quality_obj[b - 1]))
        kb_atomic = (_INTERVAL_TO_ROMAN[deg_b_atomic], quality_bucket(gt_quality_obj[b]))
        bigram_logprob_atomic[b] = _conditional_logprob(atomic_counts, ka_atomic, kb_atomic)

        deg_a_mode = (gt_root[b - 1] - canonical_tonic) % 12
        deg_b_mode = (gt_root[b] - canonical_tonic) % 12
        ka_mode = (_INTERVAL_TO_ROMAN[deg_a_mode], quality_bucket(gt_quality_obj[b - 1]))
        kb_mode = (_INTERVAL_TO_ROMAN[deg_b_mode], quality_bucket(gt_quality_obj[b]))
        bigram_logprob_mode[b] = _conditional_logprob(mode_counts[mode], ka_mode, kb_mode)
        bigram_mode_delta[b] = bigram_logprob_mode[b] - bigram_logprob_atomic[b]

    # --- D: raw note/chroma (from the symbolic onset matrix) --------------
    onset_density = beat_probs.sum(axis=1)
    chroma = np.zeros((B, 12))
    for k in range(beat_probs.shape[1]):
        chroma[:, (MIDI_LO + k) % 12] += beat_probs[:, k]
    chroma_cosine_dist = np.zeros(B)
    for b in range(1, B):
        na, nb = np.linalg.norm(chroma[b - 1]), np.linalg.norm(chroma[b])
        if na > 0 and nb > 0:
            chroma_cosine_dist[b] = 1.0 - float(np.dot(chroma[b - 1], chroma[b]) / (na * nb))

    # --- E: structure --------------------------------------------------------
    segments = Segmenter().segment(beat_probs, beat_times)
    boundaries = sorted({s.start_beat for s in segments} | {segments[-1].end_beat} if segments else {0, B})
    dist_to_boundary = np.zeros(B, dtype=int)
    for b in range(B):
        dist_to_boundary[b] = min(abs(b - bd) for bd in boundaries)

    periods = score_periods(beat_probs, beats_per_bar=4, top_k=1)
    period = list(periods.keys())[0] if periods else 0
    loop_phase = find_loop_phase(period, is_downbeat)
    position_in_loop = ((np.arange(B) - loop_phase) % period) if period > 0 else np.zeros(B, dtype=int)

    return pd.DataFrame({
        "song_id": song_id, "beat_idx": np.arange(B), "time_s": beat_times,
        "chord_changed": chord_changed, "gt_root": gt_root, "gt_label": gt_label,
        "A_beat_phase": phase, "A_beats_since_change": beats_since_change,
        "B_bass_changed": bass_changed, "B_bass_onset": bass_onset,
        "B_bass_is_root_or_fifth": bass_is_root_or_fifth,
        "C_bigram_logprob_atomic": bigram_logprob_atomic,
        "C_bigram_logprob_mode": bigram_logprob_mode,
        "C_bigram_mode_delta": bigram_mode_delta,
        "D_onset_density": onset_density, "D_chroma_cosine_dist": chroma_cosine_dist,
        "E_dist_to_segment_boundary": dist_to_boundary, "E_position_in_loop": position_in_loop,
        "E_detected_period": period, "E_loop_phase": loop_phase,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N songs (debugging)")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    print("Fitting corpus-wide bigram tables (909 songs, symbolic)...")
    atomic_counts, mode_counts = fit_bigram_tables()

    song_ids = sorted(p.name for p in POP909_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    if args.limit:
        song_ids = song_ids[:args.limit]

    frames = []
    t0 = time.time()
    n_skipped = 0
    for i, song_id in enumerate(song_ids):
        try:
            df = build_symbolic_features_for_song(song_id, atomic_counts, mode_counts)
        except Exception as e:  # noqa: BLE001 -- corpus run, log and continue
            print(f"  [{song_id}] SKIPPED ({type(e).__name__}: {e})")
            n_skipped += 1
            continue
        if df is None or len(df) < 8:
            n_skipped += 1
            continue
        frames.append(df)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(song_ids)} songs, {elapsed:.0f}s elapsed, "
                  f"{elapsed / (i + 1) * (len(song_ids) - i - 1):.0f}s remaining (est.)")

    result = pd.concat(frames, ignore_index=True)
    print(f"\nSkipped {n_skipped}/{len(song_ids)} songs (missing files / too short / parse errors)")
    print(f"Total: {len(result)} rows across {result['song_id'].nunique()} songs, "
          f"{time.time() - t0:.0f}s")
    out_path = OUT_DIR / "features_symbolic.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
