"""
harmonia — analyze a YouTube video's chords

Usage:
    python scripts/analyze_youtube.py https://youtu.be/XYZ
    python scripts/analyze_youtube.py https://youtu.be/XYZ --out chart.json
    python scripts/analyze_youtube.py https://youtu.be/XYZ --phase 2 --keep-audio

Requires yt-dlp:
    pip install yt-dlp

The audio is downloaded to a temp file (or --audio-dir), the live pipeline
(`chord_pipeline_v1.infer_chords_v1`) is run, and the JSON chart is saved.
The temp file is removed afterwards unless --keep-audio is given.

Ported to `infer_chords_v1` 2026-07-30 when the Gen-1 `HarmoniaPipeline` was
deleted; `--phase` / `--no-madmom` / `--min-segment-beats` were Gen-1-only
knobs with no equivalent and are gone rather than silently ignored. For a
rendered interactive chart rather than raw JSON, use
`scripts/render_youtube_chart.py`, which runs the same pipeline.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_ytdlp() -> None:
    if shutil.which("yt-dlp") is None:
        print(
            "Error: yt-dlp not found.  Install it with:  pip install yt-dlp",
            file=sys.stderr,
        )
        sys.exit(1)


def _download_audio(url: str, dest_dir: Path, verbose: bool) -> Path:
    """Download best audio from *url* into *dest_dir* as an opus/m4a/mp3 file.

    Returns the path of the downloaded file.
    """
    template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "best",   # keeps native codec (opus/m4a); avoids re-encode
        "--audio-quality", "0",     # best quality
        "-o", template,
        "--print", "after_move:filepath",  # emit final path to stdout
        "--no-playlist",
        url,
    ]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]

    log = logging.getLogger(__name__)
    log.info("Downloading audio from %s", url)
    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)

    if result.returncode != 0:
        stderr = result.stderr or ""
        print(f"Error: yt-dlp failed (exit {result.returncode})\n{stderr}", file=sys.stderr)
        sys.exit(1)

    # --print after_move:filepath prints one line per track
    path_line = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    if path_line and Path(path_line).exists():
        return Path(path_line)

    # Fallback: find the newest file in dest_dir
    files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    audio_exts = {".opus", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".wav"}
    for f in files:
        if f.suffix in audio_exts:
            return f

    print("Error: could not locate downloaded audio file.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonia: analyze chords in a YouTube video",
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON (default: <video_id>_chords.json in cwd)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                        help="Cache dir for feature activations")
    parser.add_argument("--beat-backend", default="beatthis",
                        choices=["beatthis", "librosa", "madmom"],
                        help="Beat tracker (default beatthis)")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Directory to write downloaded audio (default: temp dir)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Keep downloaded audio file after analysis")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    _check_ytdlp()

    use_temp = args.audio_dir is None
    audio_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_")) if use_temp else args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path = _download_audio(args.url, audio_dir, args.verbose)
        logging.getLogger(__name__).info("Audio saved to %s", audio_path)

        from harmonia.models.chord_pipeline_v1 import infer_chords_v1

        chart = infer_chords_v1(
            audio_path,
            cache_dir=args.cache_dir,
            beat_backend=args.beat_backend,
        )
        chart.print()

        out = args.out or Path(audio_path.stem + "_chords.json")
        chart.save_json(out)
        logging.getLogger(__name__).info("Chart saved to %s", out)

    finally:
        if use_temp and not args.keep_audio:
            shutil.rmtree(audio_dir, ignore_errors=True)
        elif args.keep_audio:
            logging.getLogger(__name__).info("Audio kept at %s", audio_dir)


if __name__ == "__main__":
    main()
