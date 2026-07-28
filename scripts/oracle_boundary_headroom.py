#!/usr/bin/env python3
"""LEVER premise-check: how much headroom is in SEGMENTATION alone?

The segmentation diagnostic showed musx sees chord changes but times them loosely
and over-segments. Before touching the pipeline, quantify the ceiling: if the
chord BOUNDARIES were perfect (oracle = the GT spans) and we only let musx label
each span (majority of its frame posteriors), how much better than the shipped
pipeline would root/family accuracy be?

  current   : shipped pipeline root/family (read from the benchmark scores JSON).
  oracle-bnd: GT spans + musx per-span majority (root, family) from the 23 ms
              frame posteriors (triad head), latency-corrected. Same GT, same
              musx model — ONLY the boundaries are made perfect.

gap = oracle-bnd − current  ==  the segmentation headroom. Large gap ⇒ the lever
(beat-snap + merge) pays; small gap ⇒ the ceiling is labelling, not boundaries.

Runs on the songs that already have cached musx probs (GuitarSet comp + JAAH
pinned). Read-only.

Usage:
    .venv/bin/python scripts/oracle_boundary_headroom.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402
from scripts.build_jaah_corpus import parse_jaah, load_lab, LABS_DIR   # noqa: E402

RS = REPO / "docs" / "research_sessions"
GS_ANN = REPO / "data" / "cache" / "guitarset" / "annotation"
GS_AUD = REPO / "data" / "cache" / "guitarset" / "audio"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
MUSX_LAT = 0.113

# musx triad head: 0 = N; i>=1 -> root (i-1)%12, type (i-1)//12 in
# [maj, min, sus4, sus2, dim, aug]
_TRIAD_FAM = ["maj", "min", "sus", "sus", "dim", "aug"]


def triad_rf(idx):
    if idx == 0:
        return (None, None)
    return ((idx - 1) % 12, _TRIAD_FAM[(idx - 1) // 12])


def oracle_labels(wav, gt_rows):
    """Per GT span, the majority (root, family) of the musx triad argmax over the
    span's frames (latency-corrected). Returns duration-weighted root/family acc
    over non-N GT spans."""
    triad = frame_posteriors(wav)[0]
    am = triad.argmax(1)
    nf = len(am)
    dur = rootc = famc = 0.0
    for t0, t1, glab in gt_rows:
        gr, gf, _ = parse_jaah(glab)
        if gr is None:
            continue
        a = max(0, int(round((t0 + MUSX_LAT) / FRAME_DT)))
        b = min(nf, int(round((t1 + MUSX_LAT) / FRAME_DT)))
        if b <= a:
            continue
        # majority (root, fam) over the span
        rf = [triad_rf(int(x)) for x in am[a:b] if x != 0]
        if not rf:
            continue
        vals, counts = np.unique(np.array([f"{r}:{f}" for r, f in rf]), return_counts=True)
        best = vals[counts.argmax()]
        pr, pf = best.split(":")
        pr = int(pr)
        d = t1 - t0
        dur += d
        if pr == gr:
            rootc += d
            if pf == gf:
                famc += d
    return (rootc / dur, famc / dur, dur) if dur else (None, None, 0.0)


def gs_rows(stem):
    j = json.loads((GS_ANN / f"{stem}.jams").read_text())
    ch = [a for a in j["annotations"] if a["namespace"] == "chord"][-1]["data"]
    return [(d["time"], d["time"] + d["duration"], d["value"].split("/")[0]) for d in ch]


def jaah_rows(slug):
    return [(t0, t1, lab) for t0, t1, lab in load_lab(LABS_DIR / f"{slug}.lab")]


def run(name, scores_path, rows_fn, wav_fn, key_map=None):
    scores = json.loads(scores_path.read_text())["scores"]
    cur = {s["slug"]: s["root_acc"] for s in scores}
    curf = {s["slug"]: s["family_acc"] for s in scores}
    pairs = []
    for slug in cur:
        wav = wav_fn(slug)
        if wav is None or not (PROB_CACHE / f"{wav.stem}.npz").exists():
            continue
        try:
            orr, orf, d = oracle_labels(wav, rows_fn(slug))
        except Exception:
            continue
        if orr is None or cur[slug] is None:
            continue
        pairs.append((slug, cur[slug], curf[slug], orr, orf))
    if not pairs:
        print(f"[{name}] no matchable songs with cached probs")
        return

    # Wrong-take detector: a correctly-sourced song, however hard, should score
    # a decent ROOT under oracle boundaries (perfect spans + musx on the right
    # audio). If even oracle root is near-floor, the audio almost certainly
    # doesn't match the GT — a sourcing/take error, not model difficulty.
    WRONG_TAKE = 0.35
    bad = [p for p in pairs if p[3] < WRONG_TAKE]
    good = [p for p in pairs if p[3] >= WRONG_TAKE]

    def _summ(tag, ps):
        c_r = np.mean([p[1] for p in ps]); c_f = np.mean([p[2] for p in ps])
        o_r = np.mean([p[3] for p in ps]); o_f = np.mean([p[4] for p in ps])
        print(f"  {tag:<26} n={len(ps):<3} "
              f"cur root {c_r:.3f} | oracle root {o_r:.3f} "
              f"(headroom {o_r-c_r:+.3f})")

    print(f"\n=== {name} ===")
    _summ("ALL", pairs)
    _summ("clean (oracle root >=.35)", good)
    if bad:
        print(f"  flagged WRONG-TAKE (oracle root <{WRONG_TAKE}): "
              f"{[p[0][:22] for p in sorted(bad, key=lambda p: p[3])]}")
    worst = sorted(good, key=lambda p: p[3] - p[1], reverse=True)[:5]
    print("  biggest clean per-song root gains from perfect boundaries:")
    for slug, cr, cf, orr, orf in worst:
        print(f"    {slug[:30]:<30} {cr:.2f} -> {orr:.2f}  ({orr-cr:+.2f})")


def gs_wav(stem):
    p = GS_AUD / f"{stem}_mic.wav"
    return p if p.exists() else None


def jaah_wav(slug):
    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    vid = pins.get(slug)
    return (JAAH_AUD / f"{vid}.wav") if vid else None


def main():
    run("GuitarSet comp (clean)", RS / "guitarset_benchmark_scores.json",
        gs_rows, gs_wav)
    run("JAAH (dense jazz)", RS / "jaah_benchmark_scores.json",
        jaah_rows, jaah_wav)
    print("\nNote: oracle uses GT boundaries + musx-only labels — it isolates the "
          "boundary contribution. A large positive headroom means beat-snap + "
          "merge is worth building; a small one means the label head is the ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
