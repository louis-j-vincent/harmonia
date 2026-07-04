"""
Shared per-beat feature table for the chord-change-signal investigation
(2026-07-03/04): one row per beat, per song, with ~10 candidate predictive
metrics spanning 5 categories, plus the real chord-change label -- built
once here so every pairwise/joint analysis downstream works off the same,
independently-validated numbers instead of each recomputing its own
(possibly subtly different) version.

Categories (see docs/chord_change_signal_analysis/README.md for the full
rationale):
  A. Harmonic-rhythm timing   -- beat_phase, beats_since_change
  B. Bass patterns            -- bass_pc_changed, bass_onset, bass_is_root_or_fifth
  C. Key/bigram patterns      -- bigram_logprob_atomic, bigram_logprob_mode,
                                  bigram_mode_delta
  D. Raw note/chroma patterns -- onset_density, chroma_cosine_dist
  E. Song structure           -- dist_to_segment_boundary, position_in_loop

Ground truth (chord_changed, gt_root, gt_quality) and the beat/downbeat
grid both come from POP909's own annotation files (beat_midi.txt,
chord_midi.txt), not our audio beat tracker -- consistent with
scripts/plot_chord_change_correlates.py, for the same reason: this
analysis is ABOUT timing, so the least noisy available reference is used
for the parts ground truth can supply. B/D metrics still come from real
audio (Basic Pitch activations on POP909's beat grid).

Usage:
    .venv/bin/python scripts/build_chord_change_features.py
    (writes docs/chord_change_signal_analysis/features.csv)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bass_track import forward_fill_bass, infer_bass_track_learned  # noqa: E402
from scale_taxonomy import (  # noqa: E402
    MAJOR_FAMILY, DIATONIC_MAJOR_FAMILY, precise_triad_quality,
)

DATA_ROOT = Path(__file__).parent.parent / "data"
POP909_DIR = DATA_ROOT / "pop909" / "POP909"
OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"
SONGS = ["001", "002", "003", "004", "005"]

_INTERVAL_TO_ROMAN = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III",
    5: "IV", 6: "bV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
}


def quality_bucket(quality) -> str:
    from harmonia.theory.chord_vocabulary import get_template
    t = get_template(quality)
    if 3 in t.intervals:
        return "min"
    if 4 in t.intervals:
        return "maj"
    return "other"


# ---------------------------------------------------------------------------
# One-time corpus-wide bigram probability fit (909 songs, symbolic only)
# ---------------------------------------------------------------------------

def identify_best_parent_scale(chord_events) -> tuple[int, float]:
    real_events = [ev for ev in chord_events if ev.root >= 0]
    if not real_events:
        return 0, 0.0
    best_T, best_frac = 0, -1.0
    for T in range(12):
        matched = sum(
            1 for ev in real_events
            if DIATONIC_MAJOR_FAMILY.get((ev.root - T) % 12) == precise_triad_quality(ev.quality)
        )
        frac = matched / len(real_events)
        if frac > best_frac:
            best_T, best_frac = T, frac
    return best_T, best_frac


def fit_bigram_tables():
    """
    Returns (atomic_counts, mode_counts) -- both {(key_a, key_b): count},
    key = (roman_numeral, quality_bucket). atomic_counts is pooled by each
    song's own best-fit parent scale (mode-agnostic); mode_counts is split
    {"major": Counter, "minor": Counter}, canonicalised to the relative-
    major tonic per song (see plot_structure_proposal_illustrations.py's
    illustrate_ngrams_canonical/illustrate_atomic_bigrams for the original,
    separately-validated versions of this logic).
    """
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    atomic_counts = Counter()
    mode_counts = {"major": Counter(), "minor": Counter()}

    for song in songs:
        if not song.chord_events:
            continue
        real_events = [ev for ev in song.chord_events if ev.root >= 0]

        # atomic: pooled by song's own best-fit parent scale
        best_T, _ = identify_best_parent_scale(song.chord_events)
        collection_pcs = {(best_T + iv) % 12 for iv in MAJOR_FAMILY}
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            if a.root not in collection_pcs or b.root not in collection_pcs:
                continue
            deg_a, deg_b = (a.root - best_T) % 12, (b.root - best_T) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            atomic_counts[(key_a, key_b)] += 1

        # mode-specific: canonicalised to relative major, split by annotated mode
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists():
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = "major" if quality_bucket(key_quality) == "maj" else "minor"
        canonical_tonic = tonic if mode == "major" else (tonic + 3) % 12
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            deg_a = (a.root - canonical_tonic) % 12
            deg_b = (b.root - canonical_tonic) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            mode_counts[mode][(key_a, key_b)] += 1

    return atomic_counts, mode_counts


def _conditional_logprob(counts: Counter, key_a, key_b, smoothing: float = 1.0) -> float:
    """log P(key_b | key_a), Laplace-smoothed over all keys seen after key_a."""
    outcomes = {kb: c for (ka, kb), c in counts.items() if ka == key_a}
    total = sum(outcomes.values())
    n_outcomes = len(outcomes) or 1
    num = outcomes.get(key_b, 0) + smoothing
    denom = total + smoothing * n_outcomes
    return float(np.log(num / denom)) if denom > 0 else float(np.log(1e-9))


# ---------------------------------------------------------------------------
# Per-song beat-level feature extraction
# ---------------------------------------------------------------------------

def _load_pop909_beat_grid(song_id: str):
    """Thin wrapper over the canonical parser (kept because
    build_symbolic_features.py imports it by name). POP909Parser now reads
    the downbeat column itself — see docs/known_issues.md #7."""
    from harmonia.data.pop909_parser import POP909Parser
    song = POP909Parser(POP909_DIR).parse_song(song_id)
    return song.beat_times, song.is_downbeat


def _beat_in_bar_phase(is_downbeat: np.ndarray) -> np.ndarray:
    phase = np.zeros(len(is_downbeat), dtype=int)
    counter, started = 0, False
    for b in range(len(is_downbeat)):
        if is_downbeat[b]:
            counter, started = 0, True
        phase[b] = counter if started else -1
        counter += 1
    return phase


class _PseudoBeatGrid:
    def __init__(self, beat_times):
        self.beat_times = beat_times

    @property
    def beat_duration_s(self):
        return float(np.median(np.diff(self.beat_times))) if len(self.beat_times) > 1 else 0.5

    def quantise_frames(self, frame_times, note_probs):
        from harmonia.models.rhythm import BeatGrid
        return BeatGrid.quantise_frames(self, frame_times, note_probs)


def build_features_for_song(song_id: str, atomic_counts, mode_counts) -> pd.DataFrame:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.periodicity import find_loop_phase, score_periods
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.structure import Segmenter, build_ssm

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    gt_song = POP909Parser(POP909_DIR).parse_song(song_id)

    beat_times, is_downbeat = _load_pop909_beat_grid(song_id)
    grid = _PseudoBeatGrid(beat_times)
    B = len(beat_times)
    phase = _beat_in_bar_phase(is_downbeat)

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    act = extractor.extract(wav)
    beat_probs = grid.quantise_frames(act.frame_times, act.onset_probs)

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

    # --- B: bass ------------------------------------------------------------
    bass_raw = infer_bass_track_learned(beat_probs)
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

    # --- D: raw note/chroma --------------------------------------------------
    onset_density = beat_probs.sum(axis=1)
    midi_start = 21
    chroma = np.zeros((B, 12))
    for k in range(88):
        chroma[:, (midi_start + k) % 12] += beat_probs[:, k]
    chroma_cosine_dist = np.zeros(B)
    for b in range(1, B):
        na, nb = np.linalg.norm(chroma[b - 1]), np.linalg.norm(chroma[b])
        if na > 0 and nb > 0:
            chroma_cosine_dist[b] = 1.0 - float(np.dot(chroma[b - 1], chroma[b]) / (na * nb))

    # --- E: structure ----------------------------------------------------------
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
    import logging
    logging.basicConfig(level=logging.WARNING)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Fitting corpus-wide bigram tables (909 songs, symbolic)...")
    atomic_counts, mode_counts = fit_bigram_tables()
    print(f"  atomic bigram contexts: {len(atomic_counts)}, "
          f"major contexts: {len(mode_counts['major'])}, minor contexts: {len(mode_counts['minor'])}")

    frames = []
    for song_id in SONGS:
        print(f"Building features for song {song_id}...")
        df = build_features_for_song(song_id, atomic_counts, mode_counts)
        frames.append(df)
        print(f"  {len(df)} beats, {df['chord_changed'].sum()} chord changes")

    full = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / "features.csv"
    full.to_csv(out, index=False)
    print(f"\nSaved: {out} ({len(full)} rows)")


if __name__ == "__main__":
    main()
