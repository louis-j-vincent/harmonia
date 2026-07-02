"""
Middle-point experiment (2026-07-02 sprint, 1hr time-boxed): given ORACLE
chord-change boundaries (GT chord_events' real start/end times), can we
reconstruct the right chord label (root + quality) at each segment, using:
  - ground-truth key/scale (key_audio.txt)
  - a bass-register-weighted pitch-class score, with a root>fifth
    preference (both learned/motivated by this session's earlier findings
    -- see docs/known_issues.md issue #1's bass-pattern subsection)
  - the segment's full chroma, template-matched for quality

This decouples "what chord is it" from "when does it change" -- if this
works well, chord *timing* (not labeling) becomes the remaining problem,
learnable jointly with bass motion (see docs/known_issues.md #1).

Usage:
    .venv/bin/python scripts/experiment_bass_chord_inference.py --songs 001 002 003 004 005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bass_track import compress_to_runs  # noqa: E402 (unused for now, kept for later)

DATA_ROOT = Path(__file__).parent.parent / "data"
MIDI_START = 21
PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------------------
# Bass-register weighting (learned from scripts/learn_bass_distribution.py:
# true bass register is MIDI 37-61, C#2-C#4)
# ---------------------------------------------------------------------------

def bass_octave_weight(midi: np.ndarray, center: float = 46.0, sigma: float = 9.0) -> np.ndarray:
    return np.exp(-0.5 * ((midi - center) / sigma) ** 2)


_MIDI_RANGE = np.arange(MIDI_START, MIDI_START + 88)
_BASS_WEIGHT = bass_octave_weight(_MIDI_RANGE)
_PC_OF_KEY = _MIDI_RANGE % 12


def segment_chroma_and_bass(seg_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    seg_probs: (n_beats, 88) onset activation for one oracle chord segment.
    Each beat is L1-normalised individually before summing (one unit of
    evidence per beat -- docs/known_issues.md #0's calibration principle),
    both for the plain chroma (quality matching) and the bass-weighted
    chroma (root/bass evidence).

    Returns: (chroma_12, bass_pc_12)
    """
    n = seg_probs.shape[0]
    chroma = np.zeros(12)
    bass_pc = np.zeros(12)
    for b in range(n):
        row = seg_probs[b]
        row_sum = row.sum()
        if row_sum <= 0:
            continue
        row_norm = row / row_sum
        for k in range(88):
            chroma[_PC_OF_KEY[k]] += row_norm[k]

        weighted = row * _BASS_WEIGHT
        wsum = weighted.sum()
        if wsum > 0:
            weighted_norm = weighted / wsum
            for k in range(88):
                bass_pc[_PC_OF_KEY[k]] += weighted_norm[k]
    return chroma, bass_pc


# ---------------------------------------------------------------------------
# Chord scoring
# ---------------------------------------------------------------------------

def quality_bucket(quality) -> str:
    from harmonia.theory.chord_vocabulary import get_template
    t = get_template(quality)
    if 3 in t.intervals:
        return "min"
    if 4 in t.intervals:
        return "maj"
    return "other"


def score_chord(
    chroma: np.ndarray, bass_pc: np.ndarray, diatonic_pcs: set[int],
    fifth_weight: float, w_bass: float, w_key: float, w_chroma: float,
    diatonic_boost: float, vocabulary,
) -> tuple[int, object, dict]:
    from harmonia.theory.chord_vocabulary import get_template

    eps = 1e-9
    best = None
    for root in range(12):
        root_bass_score = bass_pc[root] + fifth_weight * bass_pc[(root + 7) % 12]
        key_bonus = diatonic_boost if root in diatonic_pcs else 1.0
        root_score = w_bass * np.log(root_bass_score + eps) + w_key * np.log(key_bonus)

        for q in vocabulary:
            template = get_template(q)
            vec = np.array(template.to_weight_vector())
            rotated = np.roll(vec, root)  # index i -> pitch class (root+i)%12
            denom = (np.linalg.norm(chroma) * np.linalg.norm(rotated)) or 1.0
            cos_sim = float(np.dot(chroma, rotated) / denom)
            total = root_score + w_chroma * np.log(cos_sim + eps)
            if best is None or total > best[0]:
                best = (total, root, q)
    return best[1], best[2], {}


def _load_slash_flags(song_id: str) -> list[bool]:
    """
    Whether each chord_midi.txt line encodes a bass-note inversion (e.g.
    "F#:maj7/5"). POP909Parser discards this ("Bass inversions are ignored
    -- we model root position chords"), but it matters here: a GT label's
    "root" is the *functional* root, not necessarily the *sounding bass
    note* under an inversion -- and this model deliberately follows the
    sounding bass note. Line order matches gt_song.chord_events 1:1 for
    real (non-N, parseable) events; see run_song's zip usage.
    """
    path = DATA_ROOT / "pop909" / "POP909" / song_id / "chord_midi.txt"
    flags = []
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 3:
            flags.append("/" in parts[2])
    return flags


def run_song(
    song_id: str, params: dict, verbose: bool = False, exclude_slash: bool = False,
) -> dict | None:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.theory.chord_vocabulary import get_vocabulary

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        return None
    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    if gt_song is None or not gt_song.chord_events or not gt_song.key_events:
        return None

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs_onset = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs_onset.shape[0]

    gt_key = gt_song.key_events[0]
    diatonic_pcs = _diatonic_pcs(gt_key.tonic, gt_key.mode)
    vocabulary = get_vocabulary(max_phase=1)
    slash_flags = _load_slash_flags(song_id) if exclude_slash else []

    n_root_correct = n_bucket_correct = n_total = 0
    for i, ev in enumerate(gt_song.chord_events):
        if ev.root < 0:
            continue
        if exclude_slash and i < len(slash_flags) and slash_flags[i]:
            continue
        b_start = int(np.searchsorted(bg.beat_times, ev.start_beat, side="left"))
        b_end = int(np.searchsorted(bg.beat_times, ev.end_beat, side="left"))
        b_start, b_end = min(b_start, B), min(max(b_end, b_start + 1), B)
        if b_start >= B or b_end <= b_start:
            continue
        seg_probs = beat_probs_onset[b_start:b_end]
        if seg_probs.sum() <= 0:
            continue

        chroma, bass_pc = segment_chroma_and_bass(seg_probs)
        pred_root, pred_q, _ = score_chord(
            chroma, bass_pc, diatonic_pcs, vocabulary=vocabulary, **params,
        )
        gt_bucket, pred_bucket = quality_bucket(ev.quality), quality_bucket(pred_q)

        n_total += 1
        if pred_root == ev.root:
            n_root_correct += 1
        if pred_root == ev.root and pred_bucket == gt_bucket:
            n_bucket_correct += 1
        if verbose:
            print(f"    [{ev.start_beat:6.1f}s-{ev.end_beat:6.1f}s] GT={ev.label:<8s} "
                  f"pred={PC_NAMES[pred_root]}{pred_q.value:<8s} "
                  f"{'OK' if pred_root==ev.root else 'MISS'}")

    if n_total == 0:
        return None
    return {
        "song_id": song_id, "n_total": n_total,
        "root_acc": n_root_correct / n_total,
        "majmin_acc": n_bucket_correct / n_total,
    }


def _diatonic_pcs(tonic: int, mode: str) -> set[int]:
    major_degrees = [0, 2, 4, 5, 7, 9, 11]
    minor_degrees = [0, 2, 3, 5, 7, 8, 10]
    degrees = major_degrees if mode == "major" else minor_degrees
    return {(tonic + d) % 12 for d in degrees}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--fifth-weight", type=float, default=0.8)  # tuned, see docs/known_issues.md #1
    parser.add_argument("--w-bass", type=float, default=1.5)        # tuned, see docs/known_issues.md #1
    parser.add_argument("--w-key", type=float, default=1.0)
    parser.add_argument("--w-chroma", type=float, default=1.0)
    parser.add_argument("--diatonic-boost", type=float, default=3.0)
    parser.add_argument("--exclude-slash", action="store_true",
                         help="Skip GT events with a bass-note inversion (e.g. \"/5\") -- "
                              "see docs/known_issues.md #1's slash-chord finding")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    params = dict(
        fifth_weight=args.fifth_weight, w_bass=args.w_bass, w_key=args.w_key,
        w_chroma=args.w_chroma, diatonic_boost=args.diatonic_boost,
    )
    print(f"params: {params}  exclude_slash={args.exclude_slash}")

    results = []
    for song_id in args.songs:
        r = run_song(song_id, params, verbose=args.verbose, exclude_slash=args.exclude_slash)
        if r is None:
            print(f"  {song_id}: skipped (no data)")
            continue
        results.append(r)
        print(f"  {song_id}: n={r['n_total']:3d}  root_acc={r['root_acc']:.1%}  majmin_acc={r['majmin_acc']:.1%}")

    if results:
        mean_root = np.mean([r["root_acc"] for r in results])
        mean_majmin = np.mean([r["majmin_acc"] for r in results])
        print(f"\nMEAN: root_acc={mean_root:.1%}  majmin_acc={mean_majmin:.1%}")


if __name__ == "__main__":
    main()
