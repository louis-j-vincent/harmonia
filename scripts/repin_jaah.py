#!/usr/bin/env python3
"""Re-pin the JAAH wrong-take songs to their best-matching audio.

The oracle-boundary study flagged 5 JAAH songs whose audio almost certainly
doesn't match the GT (oracle-boundary root < 0.35 = a sourcing/take error, not
difficulty). This tries several YouTube candidates per song, scores each by
chroma-fit against the timestamped GT (build_jaah_corpus.chroma_fit), and pins
the best take. Then it rescores just those songs and patches the benchmark
scores JSON in place (the 45-song page/artifacts are left untouched).

Read-mostly: writes only jaah_source_pins.json (updated pins) and
jaah_benchmark_scores.json (patched entries); deletes stale musx_probs for
replaced video ids so a later re-run re-infers on the new audio.

Usage:
    .venv/bin/python scripts/repin_jaah.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from harmonia.data.yt_chord_corpus import download_audio          # noqa: E402
from harmonia.eval.accuracy_score import run_prediction           # noqa: E402
from scripts.build_jaah_corpus import (                           # noqa: E402
    yt_search, chroma_fit, ann_meta, mb_length_ms, load_lab, LABS_DIR,
)
from scripts.build_jaah_benchmark import score_and_cards          # noqa: E402

RS = REPO / "docs" / "research_sessions"
PINS = RS / "jaah_source_pins.json"
SCORES = RS / "jaah_benchmark_scores.json"
AUDIO = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"

FLAGGED = ["cotton_tail", "mean_to_me", "maple_leaf_rag(hyman)",
           "new_east_st_louis", "walkin_shoes"]
DUR_TOL = 12.0        # candidate duration must be within this of target
N_CAND = 6            # search this many, try the duration-matched ones
FIT_MIN = 0.03        # accept a re-pin only if margin (fit - permuted) beats this


def fit_of(vid, rows, target):
    """Download vid, return (true_fit, margin vs permuted); wav deleted."""
    wav = download_audio(vid, AUDIO)
    try:
        true = chroma_fit(wav, rows, shift=0.0)
        perm = np.mean([chroma_fit(wav, rows, permute=True, seed=s) for s in range(3)])
    finally:
        Path(wav).unlink(missing_ok=True)
    return true, true - perm


def main():
    pins = json.loads(PINS.read_text())
    scores = json.loads(SCORES.read_text())
    score_by = {s["slug"]: s for s in scores["scores"]}

    for slug in FLAGGED:
        rows = [(t0, t1, l) for t0, t1, l in load_lab(LABS_DIR / f"{slug}.lab")]
        artist, title, mbid, jdur = ann_meta(slug)
        mb_ms, _ = mb_length_ms(mbid) if mbid else (None, None)
        target = (mb_ms / 1000.0) if mb_ms else jdur
        cur_vid = pins.get(slug)
        print(f"\n[{slug}] {artist} - {title}  target={target:.0f}s  cur_pin={cur_vid}",
              flush=True)

        cands = yt_search(f"{artist} {title}", n=N_CAND)
        tries = [(v, d, t) for v, d, t in cands
                 if d is not None and abs(d - target) <= DUR_TOL]
        # always include the current pin for a fair before/after
        results = []
        seen = set()
        for v, d, t in tries:
            if v in seen:
                continue
            seen.add(v)
            try:
                tf, mg = fit_of(v, rows, target)
            except Exception as e:
                print(f"    {v} fail: {e}", flush=True)
                continue
            results.append((v, d, tf, mg, t))
            print(f"    cand {v} dur={d:.0f} fit={tf:.3f} margin={mg:+.3f}  '{t[:44]}'",
                  flush=True)
        cur_fit = next((r for r in results if r[0] == cur_vid), None)

        if not results:
            print("    no duration-matched candidates — leaving as-is", flush=True)
            continue
        best = max(results, key=lambda r: r[3])       # best margin
        if best[0] == cur_vid:
            print(f"    best is the current pin — no change", flush=True)
            continue
        if best[3] < FIT_MIN:
            print(f"    best margin {best[3]:+.3f} < {FIT_MIN} — no confident take, "
                  f"leaving as-is", flush=True)
            continue
        # re-pin
        old = pins.get(slug)
        pins[slug] = best[0]
        PINS.write_text(json.dumps(pins, indent=1))
        if old and old != best[0]:
            (PROB_CACHE / f"{old}.npz").unlink(missing_ok=True)
        print(f"    RE-PINNED {old} -> {best[0]} (margin {best[3]:+.3f})", flush=True)

        # rescore on the new take, patch the scores entry
        wav = download_audio(best[0], AUDIO)
        try:
            pred = run_prediction(wav)
            sc = score_and_cards(slug, rows, pred, wav, [])
        finally:
            Path(wav).unlink(missing_ok=True)
        oldsc = score_by.get(slug, {})
        print(f"    rescored root {oldsc.get('root_acc')} -> {sc['root_acc']}  "
              f"family {oldsc.get('family_acc')} -> {sc['family_acc']}", flush=True)
        if slug in score_by:
            score_by[slug].update({k: sc[k] for k in
                                   ("root_acc", "family_acc", "nc_acc",
                                    "gt_span_s", "chord_dur_s", "n_gt_chords")})

    SCORES.write_text(json.dumps(scores, indent=1))
    rr = [s["root_acc"] for s in scores["scores"] if s["root_acc"] is not None]
    ff = [s["family_acc"] for s in scores["scores"] if s["family_acc"] is not None]
    print(f"\nJAAH MEAN after re-pin: root {np.mean(rr):.3f} family {np.mean(ff):.3f} "
          f"(n={len(rr)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
