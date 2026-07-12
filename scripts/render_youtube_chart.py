"""
render_youtube_chart.py — download a YouTube video and render an interactive chord chart.

Usage:
    .venv/bin/python scripts/render_youtube_chart.py https://youtu.be/XYZ
    .venv/bin/python scripts/render_youtube_chart.py https://youtu.be/XYZ --title "My Song"
    .venv/bin/python scripts/render_youtube_chart.py --audio song.wav  # skip download

Requires yt-dlp:
    pip install yt-dlp

Writes an HTML file (same interactive viewer as the existing inferred charts) and
opens it in the default browser.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Harmonia label → iReal token mapping ─────────────────────────────────────
#
# ChordQuality.value strings (from chord_vocabulary.py) mapped to iReal tokens.
# iReal uses: "" = major triad, "-" = minor, "^7" = maj7, "-7" = min7, "7" = dom7,
# "o" = dim, "o7" = dim7, "h7" = half-dim, "+" = aug, "sus" = sus4, etc.

_QUALITY_TO_IREAL: dict[str, str] = {
    # Triads
    "maj":      "",
    "min":      "-",
    "dim":      "o",
    "aug":      "+",
    # Suspended
    "sus2":     "sus2",
    "sus4":     "sus",
    # Seventh chords
    "maj7":     "^7",
    "min7":     "-7",
    "7":        "7",
    "mMaj7":    "-^7",
    "ø7":       "h7",
    "°7":       "o7",
    "augMaj7":  "+^7",
    "aug7":     "+7",
    "7sus4":    "7sus",
    # sev_h spellings from chord_pipeline_v1.py (_SEV_TO_Q5) — this table was
    # originally written against chord_vocabulary.ChordQuality.value strings
    # (an older pipeline's vocabulary); chord_pipeline_v1's sev_h uses
    # different names for the same three qualities. Missing entries here
    # silently fell through .get(quality, quality) and printed the raw sev_h
    # string as the ireal token (e.g. "hdim7" instead of "h7").
    "hdim7":    "h7",
    "dim7":     "o7",
    "minmaj7":  "-^7",
    # Phase 2 — dominant altered / 9ths
    "9":        "9",
    "min9":     "-9",
    "maj9":     "^9",
    "7b9":      "7b9",
    "7#9":      "7#9",
    "9sus4":    "9sus",
    "maj9#11":  "^9#11",
    # Phase 3 — 11ths
    "min11":    "-11",
    "dom11":    "11",
    "maj11":    "^11",
    # Phase 4 — 13ths
    "dom13":    "13",
    "min13":    "-13",
    "maj13":    "^13",
    # No chord
    "N":        "N.C.",
}

# Collapse a full quality to seventh level (family + 7th, no extensions)
_QUALITY_TO_SEVENTH: dict[str, str] = {
    "9":        "7",
    "min9":     "min7",
    "maj9":     "maj7",
    "7b9":      "7",
    "7#9":      "7",
    "9sus4":    "7sus4",
    "maj9#11":  "maj7",
    "min11":    "min7",
    "dom11":    "7",
    "maj11":    "maj7",
    "dom13":    "7",
    "min13":    "min7",
    "maj13":    "maj7",
}

# Collapse any quality to family (triad only)
_QUALITY_TO_FAMILY: dict[str, str] = {
    "min7":     "min",
    "dom7":     "maj",  # dominant = major family
    "7":        "maj",
    "maj7":     "maj",
    "mMaj7":    "min",
    "ø7":       "min",
    "°7":       "dim",
    "augMaj7":  "aug",
    "aug7":     "aug",
    "7sus4":    "sus4",
    "sus2":     "sus2",
    "sus4":     "sus4",
    "hdim7":    "min",
    "dim7":     "dim",
    "minmaj7":  "min",
    "9":        "maj",
    "min9":     "min",
    "maj9":     "maj",
    "7b9":      "maj",
    "7#9":      "maj",
    "9sus4":    "sus4",
    "maj9#11":  "maj",
    "min11":    "min",
    "dom11":    "maj",
    "maj11":    "maj",
    "dom13":    "maj",
    "min13":    "min",
    "maj13":    "maj",
}

# Valid root names in Harmonia (SEMITONE_NAMES order)
_ROOTS = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

# q5 family name (see chord_pipeline_v1._top_chord_suggestions) -> iReal token,
# at family/seventh granularity (no extensions — that's the depth the q5
# posterior actually distinguishes).
_Q5_TO_IREAL: dict[str, str] = {
    "maj": "", "min": "-", "dom": "7", "hdim": "h7", "dim": "o7",
}


def _split_label(label: str) -> tuple[str, str]:
    """Split "D:maj7" -> ("D", "maj7"), or "F#min7" -> ("F#", "min7") for
    callers still passing the older concatenated format. Returns ("", label)
    on failure.

    chord_pipeline_v1.infer_chords_v1 always emits the colon form
    (``f"{NOTE[root]}:{sev_h}"``) — this was previously unhandled here, so
    every quality token coming out of a *fresh* analysis carried a leading
    ":" (e.g. ireal token ":maj7" instead of "^7"), silently corrupting the
    chart display for any newly-analyzed song while every already-rendered
    chart (generated before this bug, or via a different pipeline) looked
    fine. Caught 2026-07-12 while adding chord-suggestion data, which flows
    through this same function.
    """
    if ":" in label:
        root, _, quality = label.partition(":")
        if root in _ROOTS:
            return root, quality
    if len(label) >= 2 and label[:2] in _ROOTS:
        return label[:2], label[2:]
    if label[:1] in _ROOTS:
        return label[:1], label[1:]
    return "", label


def _quality_ireal(quality: str, level: str) -> str:
    """Convert a Harmonia quality string to an iReal token at the given level.

    level: "exact" | "seventh" | "family"
    """
    if level == "seventh":
        quality = _QUALITY_TO_SEVENTH.get(quality, quality)
    elif level == "family":
        quality = _QUALITY_TO_FAMILY.get(quality, quality)
        quality = _QUALITY_TO_SEVENTH.get(quality, quality)  # two-step collapse
    return _QUALITY_TO_IREAL.get(quality, quality)


def label_to_ireal(label: str, level: str = "exact") -> str:
    """Convert a Harmonia chord label to an iReal token at the given depth."""
    root, quality = _split_label(label)
    if not root:
        return label
    return root + _quality_ireal(quality, level)


# ── Audio download ────────────────────────────────────────────────────────────

def _check_ytdlp() -> None:
    if shutil.which("yt-dlp") is None:
        print("Error: yt-dlp not found.  Install with:  pip install yt-dlp", file=sys.stderr)
        sys.exit(1)


def _download_audio(url: str, dest_dir: Path, verbose: bool) -> tuple[Path, str]:
    """Download audio from a YouTube URL. Returns (audio_path, video_title)."""
    template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "best",
        "--audio-quality", "0",
        "-o", template,
        "--print", "after_move:filepath",
        "--print", "title",
        "--no-playlist",
        url,
    ]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]

    log = logging.getLogger(__name__)
    log.info("Downloading audio from %s", url)
    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)

    if result.returncode != 0:
        print(f"Error: yt-dlp failed (exit {result.returncode})\n{result.stderr or ''}",
              file=sys.stderr)
        sys.exit(1)

    lines = (result.stdout or "").strip().splitlines()
    title = ""
    audio_path: Path | None = None

    # yt-dlp prints filepath then title (one --print per line)
    for line in lines:
        p = Path(line)
        if p.exists():
            audio_path = p
        elif not title:
            title = line

    if audio_path is None:
        # Fallback: newest audio file in dest_dir
        audio_exts = {".opus", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".wav"}
        files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            if f.suffix in audio_exts:
                audio_path = f
                break

    if audio_path is None:
        print("Error: could not locate downloaded audio file.", file=sys.stderr)
        sys.exit(1)

    return audio_path, title


# ── ChordChart → render_interactive adapter ───────────────────────────────────

def _parse_time_signature(ts_str: str) -> tuple[int, int]:
    """Parse "4/4" → (4, 4). Falls back to (4, 4)."""
    try:
        num, den = ts_str.split("/")
        return int(num), int(den)
    except Exception:
        return 4, 4


def chart_to_interactive_inputs(pipeline_chart, title: str, source_desc: str):
    """Convert a ChordChart (from HarmoniaPipeline) to inputs for render_interactive.

    Returns (chart_obj, chord_dicts) where chord_dicts have the {bar, beat, levels}
    format that render_interactive expects.
    """
    from harmonia.output.chart_render import Chart

    bpb, _ = _parse_time_signature(pipeline_chart.time_signature)
    beat_dur_s = 60.0 / max(pipeline_chart.tempo_bpm, 1.0)

    chord_dicts = []
    for ch in pipeline_chart.chords:
        abs_beat = ch["start_s"] / beat_dur_s
        bar = int(abs_beat) // bpb
        beat = int(abs_beat) % bpb
        conf = float(ch["confidence"])
        label = ch["label"]

        # Synthesise the three-level breakdown from the single pipeline prediction.
        # All levels share the same confidence — the UI's Auto mode will just show
        # the exact level when confidence >= threshold (default 0.6).
        ireal_exact   = label_to_ireal(label, "exact")
        ireal_seventh = label_to_ireal(label, "seventh")
        ireal_family  = label_to_ireal(label, "family")

        suggestions = [
            {"root": s["root"], "ireal": _Q5_TO_IREAL.get(s["q5"], ""), "conf": s["prob"]}
            for s in ch.get("suggestions", [])
        ]

        chord_dicts.append({
            "bar": bar,
            "beat": beat,
            "start_s": ch["start_s"],
            "end_s": ch["end_s"],
            "levels": {
                "family":  {"ireal": ireal_family,  "conf": conf},
                "seventh": {"ireal": ireal_seventh, "conf": conf},
                "exact":   {"ireal": ireal_exact,   "conf": conf},
            },
            "suggestions": suggestions,
        })

    n_bars = max((c["bar"] for c in chord_dicts), default=0) + 1

    # section_per_bar: bars with the same consecutive value form one section block.
    # We label every bar with its segment key so the interactive chart groups them
    # correctly (the renderer shows a section-start marker only when the label changes).
    section_per_bar = [""] * n_bars
    for seg in pipeline_chart.segments:
        start_bar = int(seg["start_s"] / beat_dur_s) // bpb
        end_bar   = min(int(seg["end_s"]   / beat_dur_s) // bpb + 1, n_bars)
        key_tag   = seg.get("key", "")
        for b in range(start_bar, end_bar):
            section_per_bar[b] = key_tag

    chart_obj = Chart(
        title=title,
        composer="",
        key=pipeline_chart.global_key,
        style=source_desc,
        tempo=int(pipeline_chart.tempo_bpm),
        time_signature=(bpb, 4),
        n_bars=n_bars,
        section_per_bar=section_per_bar,
    )
    return chart_obj, chord_dicts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download a YouTube video and render an interactive Harmonia chord chart.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("url", nargs="?", default=None, help="YouTube URL")
    src.add_argument("--audio", type=Path, default=None,
                     help="Analyse a local audio file instead of downloading")

    ap.add_argument("--title", default=None,
                    help="Chart title (default: video title or filename)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML path (default: <title>_chart.html in cwd)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                    help="Cache dir for Basic Pitch activations")
    ap.add_argument("--theta", type=float, default=0.08,
                    help="Cosine novelty threshold for chord-change detection (default 0.08)")
    ap.add_argument("--cols", type=int, default=4,
                    help="Bars per row in the chart")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Keep downloaded audio after analysis")
    ap.add_argument("--no-open", action="store_true",
                    help="Don't open the HTML in the browser")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    # ── Acquire audio ─────────────────────────────────────────────────────────
    tmp_dir: Path | None = None
    video_title = ""

    if args.audio:
        if not args.audio.exists():
            sys.exit(f"Error: {args.audio} not found")
        audio_path = args.audio
        video_title = args.audio.stem
        source_desc = f"inferred from {args.audio.name}"
    else:
        _check_ytdlp()
        tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
        try:
            audio_path, video_title = _download_audio(args.url, tmp_dir, args.verbose)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        source_desc = f"inferred from YouTube · {args.url}"
        logging.getLogger(__name__).info("Audio: %s  title: %s", audio_path, video_title)

    title = args.title or video_title or audio_path.stem

    # ── Run pipeline (Gen-2 v1) ───────────────────────────────────────────────
    try:
        from harmonia.models.chord_pipeline_v1 import infer_chords_v1

        logging.getLogger(__name__).info("Running v1 pipeline on %s…", audio_path.name)
        pipeline_chart = infer_chords_v1(
            audio_path,
            seventh_gate=0.0,
            cache_dir=args.cache_dir,
        )
        pipeline_chart.print()

    finally:
        if tmp_dir and not args.keep_audio:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Convert and render ────────────────────────────────────────────────────
    from harmonia.output.chart_interactive import render_interactive

    chart_obj, chord_dicts = chart_to_interactive_inputs(pipeline_chart, title, source_desc)

    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "chart"
    out = args.out or Path(f"{slug}_chart.html")
    render_interactive(chart_obj, chord_dicts, out, bars_per_row=args.cols,
                       sections=pipeline_chart.sections)
    print(f"→ {out.resolve()}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
