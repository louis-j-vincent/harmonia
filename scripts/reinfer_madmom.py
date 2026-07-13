"""Re-infer the docs/audio corpus with the madmom beat backend and render charts.

Runs infer_chords_v1(beat_backend="madmom") on every docs/audio/*.m4a, writes an
interactive HTML chart to docs/plots/reinferred_madmom_<slug>.html, and records
the tempo each backend reports (librosa vs madmom) for the report.

Usage:
    .venv/bin/python scripts/reinfer_madmom.py [--only SONGSTEM ...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_WAV_CACHE = Path(tempfile.gettempdir()) / "harmonia_reinfer_wav"


def to_wav(path: Path) -> Path:
    """infer_chords_v1 loads via soundfile, which cannot decode m4a. Transcode
    to a cached 44.1k mono wav with ffmpeg (madmom reads m4a fine, but the
    Gen-2 pipeline's sf.read does not)."""
    if path.suffix.lower() == ".wav":
        return path
    _WAV_CACHE.mkdir(parents=True, exist_ok=True)
    out = _WAV_CACHE / (path.stem + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", "44100", str(out)], check=True)
    return out

# Force canonical repo ahead of the editable finder → stale ~/harmonia clone.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

AUDIO_DIR = Path("docs/audio")
OUT_DIR = Path("docs/plots")
CACHE_DIR = Path("data/cache")
SUMMARY = Path("docs/reinfer_madmom_summary.json")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "chart"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="song stems to run")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    from harmonia.output.chart_interactive import render_interactive
    from scripts.render_youtube_chart import chart_to_interactive_inputs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(AUDIO_DIR.glob("*.m4a"))
    if args.only:
        paths = [p for p in paths if p.stem in set(args.only)]
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    for path in paths:
        name = path.stem
        print(f"\n=== {name} ===", flush=True)
        wav = to_wav(path)
        t0 = time.time()
        chart = infer_chords_v1(wav, beat_backend="madmom", cache_dir=CACHE_DIR)
        dt = time.time() - t0

        title = name.replace("_", " ").title()
        chart_obj, chord_dicts = chart_to_interactive_inputs(
            chart, title, f"inferred (madmom) from {path.name}")
        out_html = OUT_DIR / f"reinferred_madmom_{slug(name)}.html"
        render_interactive(chart_obj, chord_dicts, out_html, bars_per_row=4,
                           sections=chart.sections)

        row = {
            "song": name,
            "tempo_bpm": round(float(chart.tempo_bpm), 1),
            "n_chords": len(chart.chords),
            "key": chart.global_key,
            "html": str(out_html),
            "infer_s": round(dt, 1),
        }
        rows.append(row)
        print(f"  tempo={row['tempo_bpm']} BPM  {row['n_chords']} chords  "
              f"key={row['key']}  → {out_html.name}  ({dt:.0f}s)", flush=True)

    SUMMARY.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {SUMMARY}  ({len(rows)} songs)")


if __name__ == "__main__":
    main()
