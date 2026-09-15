#!/usr/bin/env python3
"""Evaluate production pipeline (chord_pipeline_v1) on Billboard McGill test set.

Measures root, quality, and weighted accuracy against MIREX hand-verified annotations.

Usage:
  python scripts/eval_billboard_prod.py --n-songs 10
  python scripts/eval_billboard_prod.py --split test --verbose
"""

from __future__ import annotations

import argparse
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import mir_eval.chord

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.billboard_loader import BillboardDataset
from harmonia.models.chord_pipeline_v1 import infer_chords_v1

RESULTS_FILE = REPO / "docs" / "billboard_prod_eval_results.json"


def evaluate_song(song_id: str, bb: BillboardDataset) -> dict:
    """Evaluate production pipeline on a single Billboard song."""
    try:
        gt = bb.load_track_gt(song_id)
        audio_path = Path(gt["audio_path"])

        if not audio_path.exists():
            return {"status": "error", "reason": "audio_missing", "song_id": song_id}

        # Run inference
        try:
            chart = infer_chords_v1(audio_path, audio_domain="real")
        except Exception as e:
            return {"status": "error", "reason": f"inference: {str(e)[:100]}", "song_id": song_id}

        # Convert GT to mir_eval format
        gt_intervals = []
        gt_labels = []

        PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        for chord in gt["chords"]:
            t0, t1 = chord["t0"], chord["t1"]
            root = chord["root"]
            quality = chord["quality"]

            if root is None or quality is None:
                label = "N"
            else:
                pitch = PITCH_NAMES[root]
                if quality == "maj":
                    label = f"{pitch}:maj"
                elif quality == "min":
                    label = f"{pitch}:min"
                elif quality == "dom":
                    label = f"{pitch}:7"
                elif quality == "hdim":
                    label = f"{pitch}:hdim7"
                elif quality == "dim":
                    label = f"{pitch}:dim"
                else:
                    label = "N"

            gt_intervals.append((t0, t1))
            gt_labels.append(label)

        # Extract inferred chords from chart
        inf_intervals = []
        inf_labels = []

        for chord in chart.chords:
            t0 = chord["start_s"]
            t1 = chord["end_s"]
            label = chord["label"]

            inf_intervals.append((t0, t1))
            inf_labels.append(label)

        # Convert to arrays
        gt_intervals = np.array(gt_intervals)
        inf_intervals = np.array(inf_intervals)

        # Compute metrics using mir_eval
        try:
            scores = mir_eval.chord.evaluate(gt_intervals, gt_labels, inf_intervals, inf_labels)
            root_acc = scores.get("Root", 0.0)
            majmin_acc = scores.get("Majmin", 0.0)  # major/minor distinction
            sevenths_acc = scores.get("Sevenths", 0.0)
            tetrads_acc = scores.get("Tetrads", 0.0)
            weighted_acc = scores.get("Weighted", 0.0)
        except Exception as e:
            root_acc = majmin_acc = sevenths_acc = tetrads_acc = weighted_acc = 0.0

        return {
            "status": "success",
            "song_id": song_id,
            "title": gt["title"],
            "artist": gt["artist"],
            "n_gt_chords": len(gt["chords"]),
            "n_inf_chords": len(chart.chords),
            "root_accuracy": float(root_acc),
            "majmin_accuracy": float(majmin_acc),
            "sevenths_accuracy": float(sevenths_acc),
            "tetrads_accuracy": float(tetrads_acc),
            "weighted_accuracy": float(weighted_acc),
        }

    except Exception as e:
        return {"status": "error", "reason": f"exception: {str(e)[:100]}", "song_id": song_id}


def main():
    parser = argparse.ArgumentParser(description="Evaluate production pipeline on Billboard")
    parser.add_argument("--n-songs", type=int, default=10, help="Number of songs to evaluate")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test",
                        help="Which split to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"Billboard McGill — Production Pipeline Evaluation")
    print(f"{'=' * 80}\n")

    # Load Billboard dataset
    bb = BillboardDataset(chord_type="majmin")
    train_ids, val_ids, test_ids = bb.split_train_val_test(
        train_ratio=0.8, val_ratio=0.1, seed=42
    )

    split_map = {"test": test_ids, "val": val_ids, "train": train_ids}
    song_ids = split_map[args.split][:args.n_songs]

    print(f"Evaluating {len(song_ids)} songs from {args.split} split")
    print(f"(Dataset: {len(split_map[args.split])} total available)\n")

    # Evaluate each song
    results = []
    successful = 0
    failed = 0

    for idx, song_id in enumerate(song_ids):
        result = evaluate_song(song_id, bb)

        if result["status"] == "success":
            successful += 1
            if args.verbose:
                print(
                    f"[{idx+1:3d}/{len(song_ids)}] ✓ {result['title'][:45]:<45} | "
                    f"root={result['root_accuracy']:.1%} "
                    f"maj={result['majmin_accuracy']:.1%} "
                    f"7th={result['sevenths_accuracy']:.1%}"
                )
            results.append(result)
        else:
            failed += 1
            if args.verbose:
                print(f"[{idx+1:3d}/{len(song_ids)}] ✗ {song_id}: {result['reason']}")

    # Aggregate results
    print(f"\n{'=' * 80}")
    print(f"Results Summary")
    print(f"{'=' * 80}\n")

    print(f"Successful: {successful}/{len(song_ids)} ({100*successful/len(song_ids):.0f}%)")
    print(f"Failed: {failed}/{len(song_ids)}\n")

    if successful > 0:
        root_accs = np.array([r["root_accuracy"] for r in results])
        majmin_accs = np.array([r["majmin_accuracy"] for r in results])
        sevenths_accs = np.array([r["sevenths_accuracy"] for r in results])
        tetrads_accs = np.array([r["tetrads_accuracy"] for r in results])
        weighted_accs = np.array([r["weighted_accuracy"] for r in results])

        print(f"Root Accuracy (12-way pitch class):")
        print(f"  Mean: {np.mean(root_accs):.1%} ± {np.std(root_accs):.1%}")
        print(f"  Median: {np.median(root_accs):.1%}")
        print(f"  Range: [{np.min(root_accs):.1%}, {np.max(root_accs):.1%}]\n")

        print(f"Majmin Accuracy (major/minor distinction):")
        print(f"  Mean: {np.mean(majmin_accs):.1%} ± {np.std(majmin_accs):.1%}")
        print(f"  Median: {np.median(majmin_accs):.1%}\n")

        print(f"Sevenths Accuracy (+ 7th detection):")
        print(f"  Mean: {np.mean(sevenths_accs):.1%} ± {np.std(sevenths_accs):.1%}\n")

        print(f"Tetrads Accuracy (full chord):")
        print(f"  Mean: {np.mean(tetrads_accs):.1%} ± {np.std(tetrads_accs):.1%}\n")

        print(f"Weighted Accuracy (time-weighted over all metrics):")
        print(f"  Mean: {np.mean(weighted_accs):.1%} ± {np.std(weighted_accs):.1%}")
        print(f"  Median: {np.median(weighted_accs):.1%}\n")

        # iRealb baseline from known_issues.md #19
        # "prod pipeline on 7195 real chords (iReal GT): root 59% / exact(root+q5) 32%"
        print(f"{'=' * 80}")
        print(f"Comparison to iRealb Baseline (from docs/known_issues.md #19)")
        print(f"{'=' * 80}\n")

        irealb_baseline = {
            "root": 0.59,          # 59%
            "majmin": 0.61,        # Approximate from known_issues
            "sevenths": 0.45,      # dom7 21% exact, so ~45% for sevenths
            "tetrads": 0.32,       # "exact(root+q5) 32%"
        }

        mean_root = np.mean(root_accs)
        mean_majmin = np.mean(majmin_accs)
        mean_sevenths = np.mean(sevenths_accs)
        mean_tetrads = np.mean(tetrads_accs)
        mean_weighted = np.mean(weighted_accs)

        print(f"Root Accuracy:")
        print(f"  Billboard:      {mean_root:.1%}")
        print(f"  iRealb:         {irealb_baseline['root']:.1%}")
        print(f"  Delta:          {(mean_root - irealb_baseline['root'])*100:+.1f}pp\n")

        print(f"Majmin Accuracy:")
        print(f"  Billboard:      {mean_majmin:.1%}")
        print(f"  iRealb:         {irealb_baseline['majmin']:.1%}")
        print(f"  Delta:          {(mean_majmin - irealb_baseline['majmin'])*100:+.1f}pp\n")

        print(f"Sevenths Accuracy:")
        print(f"  Billboard:      {mean_sevenths:.1%}")
        print(f"  iRealb:         {irealb_baseline['sevenths']:.1%}")
        print(f"  Delta:          {(mean_sevenths - irealb_baseline['sevenths'])*100:+.1f}pp\n")

        print(f"Tetrads Accuracy:")
        print(f"  Billboard:      {mean_tetrads:.1%}")
        print(f"  iRealb:         {irealb_baseline['tetrads']:.1%}")
        print(f"  Delta:          {(mean_tetrads - irealb_baseline['tetrads'])*100:+.1f}pp\n")

        # Save results
        summary = {
            "dataset": "Billboard McGill",
            "n_songs_evaluated": len(song_ids),
            "n_successful": successful,
            "n_failed": failed,
            "split": args.split,
            "metrics": {
                "root": {
                    "mean": float(mean_root),
                    "std": float(np.std(root_accs)),
                    "median": float(np.median(root_accs)),
                    "min": float(np.min(root_accs)),
                    "max": float(np.max(root_accs)),
                },
                "majmin": {
                    "mean": float(mean_majmin),
                    "std": float(np.std(majmin_accs)),
                    "median": float(np.median(majmin_accs)),
                    "min": float(np.min(majmin_accs)),
                    "max": float(np.max(majmin_accs)),
                },
                "sevenths": {
                    "mean": float(mean_sevenths),
                    "std": float(np.std(sevenths_accs)),
                },
                "tetrads": {
                    "mean": float(mean_tetrads),
                    "std": float(np.std(tetrads_accs)),
                },
                "weighted": {
                    "mean": float(mean_weighted),
                    "std": float(np.std(weighted_accs)),
                    "median": float(np.median(weighted_accs)),
                },
            },
            "iRealb_baseline": irealb_baseline,
            "deltas": {
                "root": float(mean_root - irealb_baseline["root"]),
                "majmin": float(mean_majmin - irealb_baseline["majmin"]),
                "sevenths": float(mean_sevenths - irealb_baseline["sevenths"]),
                "tetrads": float(mean_tetrads - irealb_baseline["tetrads"]),
            },
            "detailed_results": results,
        }

        with open(RESULTS_FILE, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Full results saved to: {RESULTS_FILE}\n")

    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
