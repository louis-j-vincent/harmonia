"""
Scale-fitting validation harness — the Layer-2 check from the 2026-07-02
key-inference handoff (docs/handoff_2026-07-02_key_inference.md §4).

Before trusting "key" as a combined concept anywhere downstream, this
validates the simpler underlying capability in isolation: given a
segment's (now-fixed, raw) chroma, does infer_key() reliably identify the
tonic + mode, with genuinely calibrated confidence, across real segments
and real songs? Synthetic unambiguous cases are covered by
tests/test_theory.py::TestKeyInferenceCalibration and
tests/test_structure.py::TestSyntheticUnambiguousKey — this script covers
the next two rungs: one real segment, whole-song consistency, and
generalisation across all 5 available POP909 songs against key_audio.txt
ground truth (not used anywhere in this project before).

Usage:
    .venv/bin/python scripts/validate_key_inference.py --songs 001 002 003 004 005
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"

logger = logging.getLogger(__name__)


def run_song(song_id: str, verbose: bool = False) -> dict | None:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.structure import Segmenter
    from harmonia.theory.key_profiles import infer_key

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        print(f"  [{song_id}] WAV not found: {wav}")
        return None

    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    gt_song = POP909Parser(pop909_dir).parse_song(song_id)
    if gt_song is None or not gt_song.key_events:
        print(f"  [{song_id}] No key_audio.txt ground truth")
        return None

    pitch_extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    activations = pitch_extractor.extract(wav)
    rhythm = RhythmAnalyser(prefer_madmom=False)
    beat_grid = rhythm.analyse(wav)
    beat_probs = beat_grid.quantise_frames(activations.frame_times, activations.onset_probs)

    segments = Segmenter().segment(beat_probs, beat_grid.beat_times)

    global_chroma = activations.chroma()
    global_key = infer_key(global_chroma)

    seg_results = []
    for seg in segments:
        kp = infer_key(seg.chroma)
        mid_t = (seg.start_time_s + seg.end_time_s) / 2.0
        gt = gt_song.key_at_time(mid_t)
        match = gt is not None and gt.tonic == kp.tonic and gt.mode == kp.mode
        seg_results.append({
            "start_s": seg.start_time_s, "end_s": seg.end_time_s,
            "n_beats": seg.n_beats, "key_name": kp.key_name,
            "confidence": kp.confidence, "gt_label": gt.label if gt else None,
            "match": match,
        })
        if verbose:
            gt_str = gt.label if gt else "?"
            print(f"    [{seg.start_time_s:6.1f}s-{seg.end_time_s:6.1f}s] "
                  f"{seg.n_beats:3d} beats  {kp.key_name:<10s} "
                  f"conf={kp.confidence:.3f}  GT={gt_str:<8s} "
                  f"{'OK' if match else 'MISS'}")

    total_dur = sum(s["end_s"] - s["start_s"] for s in seg_results)
    matched_dur = sum(s["end_s"] - s["start_s"] for s in seg_results if s["match"])
    duration_weighted_acc = matched_dur / total_dur if total_dur > 0 else 0.0

    confidences = [s["confidence"] for s in seg_results]
    n_unique_confidences = len(set(round(c, 6) for c in confidences))

    gt_global = gt_song.key_events[0]  # all 5 songs: single span, no modulation
    global_match = gt_global.tonic == global_key.tonic and gt_global.mode == global_key.mode

    return {
        "song_id": song_id,
        "n_segments": len(segments),
        "gt_key": gt_global.label,
        "global_key": global_key.key_name,
        "global_confidence": global_key.confidence,
        "global_match": global_match,
        "duration_weighted_acc": duration_weighted_acc,
        "n_unique_confidences": n_unique_confidences,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "segments": seg_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    results = []
    for song_id in args.songs:
        print(f"\n{'='*70}\nSong {song_id}\n{'='*70}")
        r = run_song(song_id, verbose=args.verbose)
        if r is None:
            continue
        results.append(r)
        print(f"  GT key: {r['gt_key']}   Global inferred: {r['global_key']} "
              f"(conf={r['global_confidence']:.3f})  "
              f"{'MATCH' if r['global_match'] else 'MISMATCH'}")
        print(f"  {r['n_segments']} segments, duration-weighted accuracy: "
              f"{r['duration_weighted_acc']:.1%}")
        print(f"  Confidence: min={r['min_confidence']:.3f} "
              f"mean={r['mean_confidence']:.3f} max={r['max_confidence']:.3f}  "
              f"({r['n_unique_confidences']} distinct values across "
              f"{r['n_segments']} segments)")

    print(f"\n{'='*70}\nSummary ({len(results)} songs)\n{'='*70}")
    if results:
        global_acc = sum(r["global_match"] for r in results) / len(results)
        seg_acc = sum(r["duration_weighted_acc"] for r in results) / len(results)
        print(f"  Global-key accuracy: {global_acc:.1%} ({sum(r['global_match'] for r in results)}/{len(results)})")
        print(f"  Mean duration-weighted per-segment accuracy: {seg_acc:.1%}")
        print(f"  {'song':<6}{'GT':<10}{'global pred':<14}{'conf':<8}{'seg acc':<10}{'distinct confs'}")
        for r in results:
            print(f"  {r['song_id']:<6}{r['gt_key']:<10}{r['global_key']:<14}"
                  f"{r['global_confidence']:<8.3f}{r['duration_weighted_acc']:<10.1%}"
                  f"{r['n_unique_confidences']}/{r['n_segments']}")


if __name__ == "__main__":
    main()
