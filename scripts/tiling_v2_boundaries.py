"""Part 2 of the tiling-v2 measurement: does the minimal-loop SNAP make the
RUN EDGES better section boundaries? (`scripts/tiling_v2.py` measured the same
idea as a placement PRIOR; this measures it as Louis actually stated it — the
square itself is corrected.)

Also prints the honest CHANCE baseline for the "GT start lands on our loop
phase" diagnostic: with loop length L, a random bar hits the phase 1/L of the
time, so 62.5% means nothing until the L distribution is on the table.

    python scripts/tiling_v2_boundaries.py [n_tracks]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_binary_ssm import boundary_prf, cont_ssm, run_edge_score  # noqa: E402
from bibar_sweep import load  # noqa: E402
from tiling_v2 import (TILE_MIN, _pad, _z, bar_vecs, nov_bar,  # noqa: E402
                       phase_map, tiling_runs_v2)

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")


def run_cuts(runs, nb, snapped=True):
    a, b = ("b0", "b1") if snapped else ("b0_v1", "b1_v1")
    c = set()
    for r in runs:
        for x in (r[a], r[b] + 1):
            if 0 < x < nb:
                c.add(int(x))
    return sorted(c)


def peaks(nv, nb, frac=0.5):
    """`sections.py`'s own peak rule, on BAR grain."""
    if nb < 20:
        return []
    inner = nv[4:nb - 4]
    if not len(inner):
        return []
    thr = max(frac * inner.max(), inner.mean() + 0.5 * inner.std())
    return [i for i in range(4, nb - 4)
            if nv[i] == nv[max(0, i - 2):i + 3].max() and nv[i] >= thr]


def main(n=None):
    songs = load(n)
    print(f"{len(songs)} Billboard tracks · "
          f"{sum(len(s['starts']) for s in songs)} annotated section starts\n")
    res = {}

    # ── the chance baseline the 62.5% needs ─────────────────────────────────
    Ls, hit, tot, exp = Counter(), 0, 0, 0.0
    for s in songs:
        Vb, nb = bar_vecs(s["Hb"]), s["nb"]
        nv = _pad(nov_bar(s["Hb"], 0.0), nb)
        Lm, Pm = phase_map(Vb, nb, nv, 0.0, TILE_MIN)
        for sb in s["starts"]:
            if sb < 8 or sb > nb - 9 or Lm[sb] < 2:
                continue
            Ls[int(Lm[sb])] += 1
            tot += 1
            exp += 1.0 / Lm[sb]
            hit += sb % Lm[sb] == Pm[sb]
    print("DIAGNOSTIC — 'the GT start sits on our detected loop phase'")
    print(f"    detected loop lengths at GT starts: "
          f"{dict(sorted(Ls.items()))}")
    print(f"    observed {100*hit/max(1,tot):.1f}%   vs CHANCE "
          f"{100*exp/max(1,tot):.1f}%  (= mean 1/L)   n={tot}")
    res["phase"] = dict(hit=hit, tot=tot, chance=exp / max(1, tot),
                        L=dict(sorted(Ls.items())))

    # ── does the SNAP improve the run edges as BOUNDARIES? ──────────────────
    print("\nBOUNDARY DETECTION from the run edges — F at bar tolerance")
    print("  (v1 = today's tiling_runs edges; v2 = same runs snapped to a")
    print("   multiple of their minimal loop, phase chosen by the peaks)")
    res["bound"] = {}
    for tol in (0, 1, 2):
        acc = {k: [] for k in ("v1", "v2", "pk", "v1+pk", "v2+pk")}
        for s in songs:
            Vb, nb = bar_vecs(s["Hb"]), s["nb"]
            nv = _pad(nov_bar(s["Hb"], 0.0), nb)
            runs = tiling_runs_v2(Vb, nb, nv, 0.0, TILE_MIN)
            c1, c2 = run_cuts(runs, nb, False), run_cuts(runs, nb, True)
            pk = peaks(nv, nb)
            gt = list(s["starts"])
            for k, c in (("v1", c1), ("v2", c2), ("pk", pk),
                         ("v1+pk", sorted(set(c1) | set(pk))),
                         ("v2+pk", sorted(set(c2) | set(pk)))):
                acc[k].append(boundary_prf(c, gt, tol)[2])
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        res["bound"][tol] = row
        print(f"    +-{tol} bar   " + "   ".join(
            f"{k} {v:.3f}" for k, v in row.items()))

    # ── how much do the runs actually MOVE? ─────────────────────────────────
    moved = same = 0
    dl = []
    for s in songs:
        Vb, nb = bar_vecs(s["Hb"]), s["nb"]
        nv = _pad(nov_bar(s["Hb"], 0.0), nb)
        for r in tiling_runs_v2(Vb, nb, nv, 0.0, TILE_MIN):
            if r["L"] < 2:
                continue
            d = abs(r["b0"] - r["b0_v1"]) + abs(r["b1"] - r["b1_v1"])
            moved += d > 0
            same += d == 0
            dl.append(d)
    print(f"\n    runs the snap MOVED: {moved}/{moved+same} "
          f"({100*moved/max(1,moved+same):.0f}%), median shift "
          f"{np.median(dl) if dl else 0:.0f} bars")
    res["moved"] = [moved, same]
    json.dump(res, open(CACHE / "tiling_v2_bound.json", "w"), indent=1,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print("saved", CACHE / "tiling_v2_bound.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
