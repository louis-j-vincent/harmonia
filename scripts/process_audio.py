"""
harmonia — audio inference script

Usage:
    python scripts/process_audio.py path/to/audio.wav
    python scripts/process_audio.py path/to/audio.wav --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonia: transcribe audio to chord chart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Input audio file (.wav/.mp3/.flac)")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache"),
        help="Directory for caching Basic Pitch activations (default: data/cache)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path. Defaults to <audio_stem>_chords.json",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3, 4],
        help="Chord vocabulary phase (1=triads+7ths, 2=+9ths, ...)",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    print(f"Harmonia — processing {args.audio.name}")
    print(f"  Vocab phase : {args.phase}")
    print(f"  Cache dir   : {args.cache_dir}")

    # Stage 1: pitch extraction
    print("\n[1/3] Extracting pitch activations (Basic Pitch)...")
    from harmonia.models.stage1_pitch import PitchExtractor
    extractor = PitchExtractor(cache_dir=args.cache_dir)
    activations = extractor.extract(args.audio)
    print(f"  Frames: {activations.n_frames}  Duration: {activations.duration_s:.1f}s")

    # Stage 2: key inference
    print("\n[2/3] Inferring key...")
    from harmonia.theory.key_profiles import infer_key
    chroma = activations.chroma()
    key = infer_key(chroma)
    print(f"  Key: {key.key_name}  (confidence: {key.confidence:.2f})")
    print(f"  Top candidates: {key.top_k(3)}")

    # Stage 3: chord inference (HMM — coming in v0.2)
    print("\n[3/3] Chord inference...")
    print("  ⚠  Full Bayesian chord HMM not yet implemented.")
    print("     Returning key inference result only.")

    # Output
    result = {
        "file": str(args.audio),
        "duration_s": activations.duration_s,
        "key": key.key_name,
        "key_confidence": round(key.confidence, 4),
        "key_candidates": key.top_k(3),
        "chords": [],   # populated in v0.2
    }

    out_path = args.out or args.audio.with_name(args.audio.stem + "_chords.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
