"""Backfill musx-ranked ``sug`` onto every chart in harmonia_min/state/charts/.

The annotation editor's candidate list is musx's own top-3 as of 2026-08-07
(span_rescore.musx_suggestions) — but only charts analysed AFTER that carry
it. This script gives the already-analysed library the same data, from the
musx_probs cache alone: a chart whose cache entry is missing is SKIPPED
loudly, never re-run (same doctrine as span_rescore.compute_acoustic_logp).

Frame mapping uses the latency=0 convention shared by label_confidence and
compute_acoustic_logp; ``meta.musx_latency_ms`` is deliberately not applied.

Usage:  .venv/bin/python scripts/backfill_musx_sug.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.musx import frame_posteriors          # noqa: E402
from harmonia_min.span_rescore import (                 # noqa: E402
    musx_cache_path, musx_suggestions)

CHARTS_DIR = REPO / "harmonia_min" / "state" / "charts"
AUDIO_DIR = REPO / "docs" / "audio"


def backfill(dry_run: bool = False) -> int:
    n_done = n_skip = 0
    for p in sorted(CHARTS_DIR.glob("*.json")):
        model = json.loads(p.read_text(encoding="utf-8"))
        audio_url = model.get("audio_url") or ""
        if not audio_url:
            print(f"SKIP  {p.name}: no audio_url")
            n_skip += 1
            continue
        audio = AUDIO_DIR / Path(audio_url).name
        cache = musx_cache_path(audio)
        if not cache.exists():
            print(f"SKIP  {p.name}: no musx_probs cache ({cache.name})")
            n_skip += 1
            continue
        probs = frame_posteriors(audio)          # cache hit — no audio touch
        chords = [c for s in model.get("sections", [])
                  for bar in s.get("bars", []) for c in bar]
        n = musx_suggestions(probs, chords)
        if dry_run:
            print(f"DRY   {p.name}: would write sug on {n}/{len(chords)} chords")
            continue
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
        print(f"OK    {p.name}: sug on {n}/{len(chords)} chords")
        n_done += 1
    print(f"\n{n_done} charts written, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(backfill(dry_run="--dry-run" in sys.argv[1:]))
