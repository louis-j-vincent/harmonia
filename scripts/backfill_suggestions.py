"""Backfill chord-suggestion data onto already-rendered charts.

The suggestion feature (chord_pipeline_v1._top_chord_suggestions) only
populates P.chords[i].sug for *freshly analyzed* songs — it's baked at
render time, and the already-rendered docs/plots/inferred_*.html files
predate the feature. Re-running full inference needs the source audio;
most already-analyzed songs' downloads were cleaned up after analysis
(--keep-audio wasn't the default), but a few have a cached copy in
docs/audio/ (kept for in-app playback). For those, this re-runs the
(now-fixed, see known_issues.md #24) pipeline against the cached audio and
PitchExtractor cache (fast — no re-download, no re-extraction if cached),
then merges just the "sug" field onto the existing baked chords by nearest
start time. Everything else already displayed (labels, manual corrections,
motifs) is untouched.

Songs with no cached local audio are left alone — their Suggestions tab
correctly shows "no data, re-run analysis" until re-analyzed via the app.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
AUDIO_DIR = REPO / "docs" / "audio"
PLOTS_DIR = REPO / "docs" / "plots"
CACHE_DIR = REPO / "data" / "cache"

from render_youtube_chart import _Q5_TO_IREAL  # noqa: E402
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402


def backfill_one(audio_path: Path, chart_path: Path) -> int:
    print(f"  running pipeline on {audio_path.name} (cached activations if available)...")
    # infer_chords_v1 reads via soundfile, which can't open .m4a directly —
    # transcode to a throwaway wav first (harmless, cheap, ffmpeg already a
    # hard dependency elsewhere in this pipeline).
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "44100", str(wav_path)],
            check=True, capture_output=True,
        )
        chart = infer_chords_v1(wav_path, cache_dir=CACHE_DIR)

    pipe_chords = [
        {
            "start_s": c["start_s"],
            "sug": [
                {"root": s["root"], "ireal": _Q5_TO_IREAL.get(s["q5"], ""), "conf": s["prob"]}
                for s in c.get("suggestions", [])
            ],
        }
        for c in chart.chords
    ]
    if not pipe_chords:
        return 0

    text = chart_path.read_text(encoding="utf-8")
    m = re.search(r"const P = (\{.*?\});\n", text)
    if not m:
        return 0
    P = json.loads(m.group(1))

    n = 0
    pi = 0
    for c in P.get("chords", []):
        t0 = c.get("t0")
        if t0 is None:
            continue
        # advance pi to the pipeline segment whose start is nearest t0
        while pi + 1 < len(pipe_chords) and abs(pipe_chords[pi + 1]["start_s"] - t0) <= abs(pipe_chords[pi]["start_s"] - t0):
            pi += 1
        sug = pipe_chords[pi]["sug"]
        if sug:
            c["sug"] = [
                {"root": s["root"], "q": s["ireal"], "c": round(s["conf"], 4)}
                for s in sug
            ]
            n += 1

    if n:
        new_json = json.dumps(P)
        text = text[: m.start(1)] + new_json + text[m.end(1) :]
        chart_path.write_text(text, encoding="utf-8")
    return n


def main() -> None:
    total = 0
    for audio_path in sorted(AUDIO_DIR.glob("*.m4a")):
        chart_path = PLOTS_DIR / f"inferred_{audio_path.stem}.html"
        if not chart_path.exists():
            print(f"skip {audio_path.name}: no matching {chart_path.name}")
            continue
        n = backfill_one(audio_path, chart_path)
        print(f"{'backfilled' if n else 'no suggestions produced for'} {n} chords in {chart_path.name}")
        total += n
    print(f"\n{total} chords backfilled total")


if __name__ == "__main__":
    main()
