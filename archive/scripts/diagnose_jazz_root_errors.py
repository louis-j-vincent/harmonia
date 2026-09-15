#!/usr/bin/env python3
"""WHY does musx miss the ROOT on jazz, even with perfect (oracle) boundaries?

The oracle-boundary study found a real ~33pp musx-root weakness on jazz harmony
(clean JAAH oracle root 0.62 vs 0.955 clean pop) that segmentation can't fix.
This diagnoses the mechanism: on GT spans where musx's majority root disagrees
with the GT functional root, what IS musx predicting?

Buckets (duration-weighted over the error mass), on clean JAAH songs (oracle
root >= 0.35, i.e. right take):
  - bass confusion  : musx root == the GT SOUNDING BASS (slash-chord / inversion
                      — the root-vs-bass target ambiguity this project already
                      knows, corpus_schema.sounding_bass_pc).
  - interval class  : (musx_root - gt_root) mod 12 — +7 fifth, +5 fourth, +3/-3
                      relative maj/min, +6 tritone (sub), +2 step, ...
A dominant bass bucket ⇒ fix is target/bass-aware decoding; a dominant
fifth/fourth/tritone bucket ⇒ functional-substitution confusion (needs harmonic
context, not just the frame model). Read-only, cached probs.

Usage:
    .venv/bin/python scripts/diagnose_jazz_root_errors.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402
from scripts.build_jaah_corpus import parse_jaah, load_lab, LABS_DIR   # noqa: E402
from harmonia.data.corpus_schema import sounding_bass_pc              # noqa: E402

RS = REPO / "docs" / "research_sessions"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
MUSX_LAT = 0.113
_TRIAD_FAM = ["maj", "min", "sus", "sus", "dim", "aug"]

IVL_NAME = {0: "same(qual?)", 1: "+1 semitone", 2: "+2 step", 3: "+3 rel(min->Maj)",
            4: "+4", 5: "+5 fourth", 6: "+6 tritone", 7: "+7 fifth",
            8: "+8", 9: "+9 rel(Maj->min)", 10: "+10", 11: "+11/-1"}


def triad_root(idx):
    return None if idx == 0 else (idx - 1) % 12


def main():
    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    scores = json.loads((RS / "jaah_benchmark_scores.json").read_text())["scores"]

    err_ivl = defaultdict(float)      # interval -> error duration
    bass_dur = same_root_diff = 0.0
    err_total = 0.0
    n_songs = 0
    for s in scores:
        slug = s["slug"]
        vid = pins.get(slug)
        if not vid or not (PROB_CACHE / f"{vid}.npz").exists():
            continue
        wav = JAAH_AUD / f"{vid}.wav"
        rows = [(t0, t1, lab) for t0, t1, lab in load_lab(LABS_DIR / f"{slug}.lab")]
        triad = frame_posteriors(wav)[0]
        am = triad.argmax(1)
        nf = len(am)
        # per-song oracle root, to keep only clean takes
        dur = rc = 0.0
        span_data = []
        for t0, t1, lab in rows:
            gr, gf, _ = parse_jaah(lab.split("/")[0])
            if gr is None:
                continue
            a = max(0, int(round((t0 + MUSX_LAT) / FRAME_DT)))
            b = min(nf, int(round((t1 + MUSX_LAT) / FRAME_DT)))
            roots = [triad_root(int(x)) for x in am[a:b] if x != 0]
            roots = [r for r in roots if r is not None]
            if not roots:
                continue
            vals, cnt = np.unique(roots, return_counts=True)
            pr = int(vals[cnt.argmax()])
            gb = sounding_bass_pc(lab, gr)
            d = t1 - t0
            dur += d
            if pr == gr:
                rc += d
            span_data.append((d, gr, pr, gb))
        if dur == 0 or rc / dur < 0.35:      # wrong take -> skip
            continue
        n_songs += 1
        for d, gr, pr, gb in span_data:
            if pr == gr:
                continue
            err_total += d
            if gb is not None and pr == gb and gb != gr:
                bass_dur += d
            else:
                err_ivl[(pr - gr) % 12] += d

    print(f"=== musx root-error taxonomy on clean JAAH (n={n_songs} songs) ===")
    print(f"total root-error mass: {err_total:.0f}s\n")
    if err_total == 0:
        return 0
    print(f"  BASS confusion (musx root = GT sounding bass): "
          f"{100*bass_dur/err_total:.0f}%  ({bass_dur:.0f}s)")
    print("  remaining errors by interval (musx_root - gt_root):")
    for ivl, d in sorted(err_ivl.items(), key=lambda kv: -kv[1]):
        print(f"    {IVL_NAME.get(ivl, str(ivl)):<20} {100*d/err_total:>4.0f}%  ({d:.0f}s)")
    print("\nRead: a big BASS share -> a target/bass-aware decode recovers it; big "
          "fifth/fourth/tritone -> functional-substitution confusion needing "
          "harmonic context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
