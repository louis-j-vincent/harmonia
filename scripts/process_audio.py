"""
harmonia — full inference CLI

Usage:
    python scripts/process_audio.py song.wav
    python scripts/process_audio.py song.wav --phase 2 --out chart.json
    python scripts/process_audio.py song.wav --no-madmom   # force librosa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonia: transcribe audio to chord chart",
    )
    parser.add_argument("audio", type=Path, help="Input audio (.wav/.mp3/.flac)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON (default: <stem>_chords.json)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Chord vocabulary phase")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                        help="Cache dir for Basic Pitch activations")
    parser.add_argument("--no-madmom", action="store_true",
                        help="Use librosa beat tracker instead of madmom")
    parser.add_argument("--min-segment-beats", type=int, default=8,
                        help="Minimum beats per structural segment")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if not args.audio.exists():
        print(f"Error: {args.audio} not found", file=sys.stderr)
        sys.exit(1)

    from harmonia.pipeline import HarmoniaPipeline

    pipeline = HarmoniaPipeline(
        max_phase=args.phase,
        cache_dir=args.cache_dir,
        prefer_madmom=not args.no_madmom,
        min_segment_beats=args.min_segment_beats,
    )

    chart = pipeline.run(args.audio)
    chart.print()

    out = args.out or args.audio.with_name(args.audio.stem + "_chords.json")
    chart.save_json(out)


if __name__ == "__main__":
    main()
