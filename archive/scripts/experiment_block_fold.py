"""Screen the symbolic block-fold premise (docs/known_issues.md #1, fold layer).

Step A (this script, --gt): can the bar-level chord SSM recover a song's
repeat structure from CLEAN ground-truth chords? This is the ceiling — if the
method can't fold AABA from perfect input, it won't from decoded chords.
Metric: adjusted Rand index between inferred bar clusters and GT
section_per_bar, across all songs with audio.

Step B (--decoded, subset): rerun on infer_song's decoded chords to measure
degradation and whether inferred grouping recovers the GT-fold accuracy gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from harmonia.models.block_fold import Bar, fold_structure  # noqa: E402
from analyze_accomp_emission import parse_chord  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MAN = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"


def gt_bars(rec: dict) -> list[Bar]:
    """Per-bar cells from the GT chord timeline (may have 1-2 chords/bar)."""
    bpb = rec["beats_per_bar"]
    by_bar: dict[int, list[tuple[int, int, str]]] = {}
    for ev in rec["chord_timeline"]:
        p = parse_chord(ev["mma"])
        if p is None:
            continue
        by_bar.setdefault(ev["bar"] - 1, []).append((ev["beat"], p[0], p[1]))
    bars = []
    for b in range(rec["n_bars"]):
        cs = sorted(by_bar.get(b, []))
        bars.append(Bar(idx=b, roots=[r for _, r, _ in cs],
                        quals=[q for _, _, q in cs], conf=[1.0] * len(cs)))
    return bars


def run_gt(level="root"):
    recs = [json.loads(l) for l in open(DB)]
    man = {json.loads(l)["song_id"] for l in open(MAN)}
    avail = [r for r in recs if r["song_id"] in man]
    rows = []
    for rec in avail:
        spb = rec["section_per_bar"]
        if len(set(spb)) < 2:      # single-section songs: no structure to recover
            continue
        bars = gt_bars(rec)
        fr = fold_structure(bars, level=level)
        n = min(len(spb), len(fr.bar_cluster))
        gt = spb[:n]
        pred = fr.bar_cluster[:n]
        ari = adjusted_rand_score(gt, pred)
        # true section length = modal run length of GT labels
        true_secs = len(set(spb))
        rows.append((rec["title"], rec["form"], fr.period_bars, fr.stats["n_unique"],
                     true_secs, ari))
    rows.sort(key=lambda r: r[-1])
    print(f"{'title':30s} {'form':16s} {'P':>3s} {'nU':>3s} {'trueU':>5s} {'ARI':>6s}")
    for t, f, p, nu, tu, ari in rows:
        print(f"{t[:30]:30s} {f[:16]:16s} {p:3d} {nu:3d} {tu:5d} {ari:6.2f}")
    aris = np.array([r[-1] for r in rows])
    print(f"\nn={len(rows)}  meanARI={aris.mean():.2f}  medianARI={np.median(aris):.2f}"
          f"  frac>0.5={np.mean(aris>0.5):.0%}  frac>0.8={np.mean(aris>0.8):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", action="store_true", help="clean GT-chord ceiling (all songs)")
    ap.add_argument("--level", default="root", choices=["root", "seventh"])
    args = ap.parse_args()
    run_gt(level=args.level)
