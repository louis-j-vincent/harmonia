"""
harmonia — full inference CLI

Usage:
    python scripts/process_audio.py song.wav
    python scripts/process_audio.py song.wav --out chart.json
    python scripts/process_audio.py song.wav --beat-backend librosa

Runs the live pipeline, `chord_pipeline_v1.infer_chords_v1`. Ported to it
2026-07-30 when the Gen-1 `HarmoniaPipeline` was deleted; the old
`--phase` / `--no-madmom` / `--min-segment-beats` flags were Gen-1-only knobs
(vocabulary phase, madmom preference, Segmenter granularity) with no
equivalent here, so they are gone rather than silently ignored.
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
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                        help="Cache dir for feature activations")
    parser.add_argument("--beat-backend", default="beatthis",
                        choices=["beatthis", "librosa", "madmom"],
                        help="Beat tracker (default beatthis)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if not args.audio.exists():
        print(f"Error: {args.audio} not found", file=sys.stderr)
        sys.exit(1)

    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    chart = infer_chords_v1(
        args.audio,
        cache_dir=args.cache_dir,
        beat_backend=args.beat_backend,
    )
    chart.print()

    out = args.out or args.audio.with_name(args.audio.stem + "_chords.json")
    chart.save_json(out)


if __name__ == "__main__":
    main()
