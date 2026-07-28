#!/usr/bin/env python3
"""Rescore specific JAAH slugs against their CURRENT pins and patch the scores
JSON in place. Recovery tool for repin_jaah.py when a mid-loop download 403s
before the final write (pins get updated incrementally; scores don't).

Retries a 403'd download after a short backoff (YouTube throttles rapid
re-downloads); on persistent failure the song is left unchanged and reported.

Usage:
    .venv/bin/python scripts/rescore_jaah_slugs.py cotton_tail mean_to_me ...
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from harmonia.data.yt_chord_corpus import download_audio          # noqa: E402
from harmonia.eval.accuracy_score import run_prediction           # noqa: E402
from scripts.build_jaah_corpus import load_lab, LABS_DIR          # noqa: E402
from scripts.build_jaah_benchmark import score_and_cards          # noqa: E402

RS = REPO / "docs" / "research_sessions"
PINS = RS / "jaah_source_pins.json"
SCORES = RS / "jaah_benchmark_scores.json"
AUDIO = REPO / "data" / "cache" / "jaah" / "audio"


def dl(vid, tries=3):
    for i in range(tries):
        try:
            return download_audio(vid, AUDIO)
        except Exception as e:
            if "403" in str(e) and i < tries - 1:
                print(f"    403 on {vid}, backoff {30*(i+1)}s…", flush=True)
                time.sleep(30 * (i + 1))
                continue
            raise


def main(argv):
    slugs = argv or ["cotton_tail", "mean_to_me", "maple_leaf_rag(hyman)",
                     "new_east_st_louis", "walkin_shoes"]
    pins = json.loads(PINS.read_text())
    scores = json.loads(SCORES.read_text())
    by = {s["slug"]: s for s in scores["scores"]}
    for slug in slugs:
        vid = pins.get(slug)
        rows = [(t0, t1, l) for t0, t1, l in load_lab(LABS_DIR / f"{slug}.lab")]
        print(f"[{slug}] pin={vid}", flush=True)
        try:
            wav = dl(vid)
        except Exception as e:
            print(f"    download failed permanently: {e} — leaving unchanged", flush=True)
            continue
        try:
            pred = run_prediction(wav)
            sc = score_and_cards(slug, rows, pred, wav, [])
        finally:
            Path(wav).unlink(missing_ok=True)
        old = by.get(slug, {})
        print(f"    root {old.get('root_acc')} -> {sc['root_acc']}  "
              f"family {old.get('family_acc')} -> {sc['family_acc']}", flush=True)
        if slug in by:
            by[slug].update({k: sc[k] for k in
                             ("root_acc", "family_acc", "nc_acc", "gt_span_s",
                              "chord_dur_s", "n_gt_chords")})
    SCORES.write_text(json.dumps(scores, indent=1))
    rr = [s["root_acc"] for s in scores["scores"] if s["root_acc"] is not None]
    ff = [s["family_acc"] for s in scores["scores"] if s["family_acc"] is not None]
    print(f"\nJAAH MEAN: root {np.mean(rr):.3f} family {np.mean(ff):.3f} (n={len(rr)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
