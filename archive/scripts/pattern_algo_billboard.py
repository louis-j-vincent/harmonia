"""Refutation only: does "raw beats centred" survive 285 Billboard tracks?

The per-song evidence is the deliverable (`/reports/pattern_algo.html`); this
exists to try to KNOCK IT DOWN, not to select anything. Same 285 bar-annotated
McGill tracks as `scripts/tiling_v2.py`, same bar grid.

Metric: run the algorithm's first entry (period from the aggregated first rows
-> first square -> slide -> prominence peaks) and ask how the occurrence starts
line up with the ANNOTATED section starts. Two trivial baselines are printed
next to it, because a cue that fires everywhere wins recall for free:
  * every multiple of L from the motif  (the periodic baseline)
  * every bar                            (the degenerate baseline)

    python scripts/pattern_algo_billboard.py [n_tracks]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_sweep import load                                      # noqa: E402
from tiling_v2 import bar_vecs                                    # noqa: E402
from pattern_algo_core import (first_pattern, pick_peaks, slide,  # noqa: E402
                               song_period)

STATS = ["raw", "cosine", "centered", "diag", "antidiag"]
PROM = 0.25


def prf(pred, ref, tol=0):
    pred, ref = sorted(set(pred)), sorted(set(ref))
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    used, hit = set(), 0
    for p in pred:
        c = [i for i, g in enumerate(ref) if abs(g - p) <= tol and i not in used]
        if c:
            used.add(min(c, key=lambda i: abs(ref[i] - p)))
            hit += 1
    P, R = hit / len(pred), hit / len(ref)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    songs = load(n)
    acc = {k: [] for k in STATS + ["period", "every_bar"]}
    nped = []
    for s in songs:
        V = bar_vecs(s["Hb"])
        S = V @ V.T
        nb = len(S)
        if nb < 24:
            continue
        mot = first_pattern(S)
        if mot is None:
            continue
        b0, L = mot["b0"], mot["L"]
        nped.append(L)
        gt = [int(x) for x in s["starts"] if 0 < int(x) < nb]
        if not gt:
            continue
        for st in STATS:
            ds, f = slide(S, b0, L, st)
            sel = sorted({b0} | {p["d"] for p in pick_peaks(
                ds, f, "prominence", PROM, min_sep=max(2, L // 2), exclude=b0)})
            acc[st].append(prf(sel, gt, tol=1))
        acc["period"].append(prf(list(range(b0 % L, nb - L + 1, L)), gt, tol=1))
        acc["every_bar"].append(prf(list(range(nb)), gt, tol=1))

    print(f"\n{len(acc['raw'])} tracks, motif length median "
          f"{int(np.median(nped))} bars  (peaks vs annotated section starts, "
          f"+-1 bar)\n")
    print(f"  {'cue':<24s} {'prec':>6s} {'rappel':>7s} {'F':>6s}")
    for k in ["every_bar", "period"] + STATS:
        v = np.array(acc[k])
        if not len(v):
            continue
        print(f"  {k:<24s} {v[:,0].mean():6.3f} {v[:,1].mean():7.3f} "
              f"{v[:,2].mean():6.3f}")

    # song-level bootstrap on the decisive pair
    a = np.array([x[2] for x in acc["raw"]])
    b = np.array([x[2] for x in acc["centered"]])
    rng = np.random.default_rng(0)
    d = [np.mean(a[i] - b[i]) for i in
         (rng.integers(0, len(a), len(a)) for _ in range(4000))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  raw - centered (F): {np.mean(a-b):+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({'significatif' if lo > 0 or hi < 0 else 'zero inside'})")


if __name__ == "__main__":
    main()
