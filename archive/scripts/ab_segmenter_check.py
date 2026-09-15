#!/usr/bin/env python3
"""Validate the segmentation analysis against the NEW best segmenter (beat-grid
snap, HARMONIA_REAL_BEAT_GRID=snap — the /compare variant C the other session
built).

My oracle-boundary study predicted a segmentation HEADROOM of +3pp (GuitarSet)
to +10.6pp (JAAH) on root: if boundaries were perfect, root would rise that
much. This checks whether the ACTUAL new segmenter captures any of it:

  A  grid=off  : shipped pipeline (synthetic beat lattice) — my baseline.
  C  grid=snap : re-lay onsets onto DETECTED beats (the new best segmenter).
  oracle       : GT boundaries + musx labels — the ceiling I computed.

If C moves A toward oracle, the segmentation lever is real AND the snap realises
part of it (my analysis holds). If C == A, the snap doesn't capture the headroom
(the analysis of WHERE the gain is would need revisiting).

Runs on GuitarSet comp (local audio) by default; --jaah re-downloads pinned JAAH.
Read-only except stdout.

Usage:
    .venv/bin/python scripts/ab_segmenter_check.py --max 12
    .venv/bin/python scripts/ab_segmenter_check.py --jaah --max 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import run_prediction              # noqa: E402
from scripts.build_jaah_benchmark import score_and_cards             # noqa: E402
from scripts.build_guitarset_benchmark import (                      # noqa: E402
    load_jams_chords, clean_label, audio_for, pick_subset, ANN,
)
from scripts.oracle_boundary_headroom import oracle_labels           # noqa: E402
from scripts.build_jaah_corpus import load_lab, LABS_DIR             # noqa: E402

RS = REPO / "docs" / "research_sessions"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"


def score_grid(wav, rows, grid):
    os.environ["HARMONIA_REAL_BEAT_GRID"] = grid
    pred = run_prediction(wav)
    return score_and_cards("_", rows, pred, wav, [])


def gs_items(max_n):
    items = []
    for jp in pick_subset(sorted(ANN.glob("*_comp.jams")), sample=True):
        stem = jp.stem
        wav = audio_for(stem)
        if wav is None:
            continue
        rows = [(t0, t1, clean_label(l)) for t0, t1, l in load_jams_chords(jp)]
        items.append((stem, wav, rows))
        if max_n and len(items) >= max_n:
            break
    return items


def jaah_items(max_n):
    import json
    from harmonia.data.yt_chord_corpus import download_audio
    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    scores = json.loads((RS / "jaah_benchmark_scores.json").read_text())["scores"]
    items = []
    for s in scores:
        slug = s["slug"]; vid = pins.get(slug)
        if not vid:
            continue
        rows = [(t0, t1, l) for t0, t1, l in load_lab(LABS_DIR / f"{slug}.lab")]
        try:
            wav = download_audio(vid, JAAH_AUD)
        except Exception:
            continue
        items.append((slug, wav, rows, True))    # True = delete wav after
        if max_n and len(items) >= max_n:
            break
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaah", action="store_true")
    ap.add_argument("--max", type=int, default=12)
    args = ap.parse_args(argv)

    name = "JAAH (dense jazz)" if args.jaah else "GuitarSet comp (clean)"
    items = jaah_items(args.max) if args.jaah else \
        [(s, w, r, False) for s, w, r in gs_items(args.max)]

    rows_out = []
    for stem, wav, rows, *rest in items:
        delete = rest[0] if rest else False
        try:
            A = score_grid(wav, rows, "off")
            C = score_grid(wav, rows, "snap")
            orr, orf, _ = oracle_labels(wav, rows)
        except Exception as e:
            print(f"[{stem}] fail {e}", flush=True)
            if delete:
                Path(wav).unlink(missing_ok=True)
            continue
        if delete:
            Path(wav).unlink(missing_ok=True)
        rows_out.append((stem, A["root_acc"], C["root_acc"], orr,
                         A["family_acc"], C["family_acc"], orf))
        print(f"  {stem[:26]:<26} A={A['root_acc']:.3f} C(snap)={C['root_acc']:.3f} "
              f"oracle={orr if orr is None else round(orr,3)}", flush=True)

    if not rows_out:
        print("no songs scored")
        return 1
    a = np.mean([r[1] for r in rows_out]); c = np.mean([r[2] for r in rows_out])
    o = np.mean([r[3] for r in rows_out if r[3] is not None])
    af = np.mean([r[4] for r in rows_out]); cf = np.mean([r[5] for r in rows_out])
    print(f"\n=== {name}  (n={len(rows_out)}) — validating the segmentation analysis ===")
    print(f"  {'':26} {'root':>7} {'family':>7}")
    print(f"  A  grid=off (shipped)      {a:>7.3f} {af:>7.3f}")
    print(f"  C  grid=snap (NEW segmenter){c:>7.3f} {cf:>7.3f}   ({c-a:+.3f} root vs A)")
    print(f"  oracle-boundary ceiling    {o:>7.3f}")
    captured = (c - a) / (o - a) if o != a else float("nan")
    print(f"\n  snap captured {100*captured:.0f}% of the oracle segmentation headroom "
          f"({c-a:+.3f} of {o-a:+.3f} available).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
