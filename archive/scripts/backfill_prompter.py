"""Backfill model["prompter"] into existing harmonia_min chart JSONs.

New analyses get the key from pipeline.analyze(); this stamps the same
pre-section, pre-fold chord list onto charts baked before 2026-08-05,
WITHOUT re-running the full pipeline (so nothing else in the chart can
change — no clobbering charts other sessions validated).

It re-decodes from the chart's OWN stored grid:
  * beat times   = model["beatTimes"]   (what the chart's playhead uses)
  * downbeats    = model["barGrid"]     (the bar boundaries ARE the downbeats)
  * beats/bar    = model["bpb"]
and the musx frame posteriors come from their disk cache (cache-hit for
every library song), so a song takes seconds, not minutes.

Usage: .venv/bin/python scripts/backfill_prompter.py [--force]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx  # noqa: E402
from harmonia_min.pipeline import prompter_chords  # noqa: E402

CHARTS_DIR = REPO / "harmonia_min" / "state" / "charts"
AUDIO_DIR = REPO / "docs" / "audio"


def backfill(path: Path, force: bool = False) -> str:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("prompter") and not force:
        return "skip (already has prompter)"
    audio_url = model.get("audio_url") or ""
    audio = AUDIO_DIR / Path(audio_url).name if audio_url else None
    if audio is None or not audio.exists():
        return f"skip (no audio: {audio_url!r})"
    beat_times = model.get("beatTimes") or []
    bar_grid = model.get("barGrid") or []
    if len(beat_times) < 8 or len(bar_grid) < 2:
        return "skip (no stored beat/bar grid)"
    probs = musx.frame_posteriors(audio)
    triad = probs[0]
    segments, _latency = musx.redecode(
        beat_times, probs, downbeat_times=bar_grid,
        beats_per_bar=int(model.get("bpb") or 4))
    # same short-leading-N rule as pipeline.analyze()
    if segments and segments[0][2] == "N" and \
            (segments[0][1] - segments[0][0]) < 2.0 * float(
                np.median(np.diff(np.asarray(beat_times)))):
        segments = segments[1:]
    model["prompter"] = {"chords": prompter_chords(segments, triad)}
    path.write_text(json.dumps(model), encoding="utf-8")
    return f"ok ({len(segments)} chords)"


def main() -> None:
    force = "--force" in sys.argv
    files = sorted(CHARTS_DIR.glob("*.json"))
    if not files:
        print(f"no charts in {CHARTS_DIR}")
        return
    for p in files:
        try:
            print(f"{p.stem:60s} {backfill(p, force)}")
        except Exception as exc:  # noqa: BLE001 — keep going, report at end
            print(f"{p.stem:60s} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
