#!/usr/bin/env python3
"""Billboard McGill Quick Start

Usage:
  python scripts/billboard_quickstart.py

Demonstrates:
- Loading a single track
- Iterating through chord events
- Exporting to JSONL
- Creating train/val/test splits
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harmonia.data.billboard_loader import BillboardDataset
from harmonia.data.billboard_translator import parse_billboard_chord


def main():
    print("=" * 70)
    print("Billboard McGill Quick Start")
    print("=" * 70 + "\n")

    # Initialize dataset
    print("1. Loading Billboard dataset...")
    try:
        bb = BillboardDataset(chord_type="majmin")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo download Billboard, run:")
        print("  python -c \"import mirdata; ds = mirdata.initialize('billboard'); ds.download()\"")
        return 1

    print(f"✓ Loaded {len(bb)} tracks\n")

    # Load a single track
    print("2. Loading a single track...")
    track_ids = bb.track_ids()
    track_id = track_ids[0]
    gt = bb.load_track_gt(track_id)

    print(f"Track: {gt['title']} by {gt['artist']}")
    print(f"Chords: {len(gt['chords'])} events")
    print(f"Audio: {gt['audio_path']}\n")

    # Show first 5 chords
    print("3. First 5 chords:")
    for chord in gt["chords"][:5]:
        root_str = str(chord["root"]) if chord["root"] is not None else "N"
        print(
            f"  {chord['t0']:6.2f}s-{chord['t1']:6.2f}s: "
            f"{chord['label']:12} → root={root_str:>2} quality={chord['quality']}"
        )
    print()

    # Demonstrate translator
    print("4. Direct chord translation:")
    examples = ["C:maj", "F#:min7", "Bb:7", "C:dim", "N"]
    for label in examples:
        root, quality = parse_billboard_chord(label)
        print(f"  {label:12} → root={str(root):>2} quality={quality}")
    print()

    # Create splits
    print("5. Creating train/val/test split...")
    train_ids, val_ids, test_ids = bb.split_train_val_test(
        train_ratio=0.8, val_ratio=0.1, seed=42
    )
    print(f"  Train: {len(train_ids):3} tracks")
    print(f"  Val:   {len(val_ids):3} tracks")
    print(f"  Test:  {len(test_ids):3} tracks\n")

    # Show export capability
    print("6. Export capability:")
    print("  To export all tracks to JSONL:")
    print("    bb = BillboardDataset()")
    print("    bb.export_to_jsonl(Path('billboard_gt.jsonl'))")
    print()

    print("=" * 70)
    print("✓ Quick start complete!")
    print("=" * 70)
    print("\nFor more examples, see:")
    print("  - docs/billboard_integration_summary.md")
    print("  - docs/billboard_translation_validation.md")
    print("  - docs/plots/billboard_translation_demo.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
