#!/usr/bin/env python3
"""Step 3 groundwork: is the JAAH failure driven by HARMONIC DENSITY / tempo?

Reads jaah_benchmark_scores.json + the JAAH .lab GT and, per song, computes:
  - chord density  = n_gt_chords / gt_span_s   (chords per second)
  - mean chord dur = 1 / density
  - fast_frac      = fraction of GT chords shorter than 1.0 s
and correlates each against the shipped pipeline's root & family accuracy.

The mechanism question the correlation adjudicates: if dense tunes fail
because the pipeline can't RESOLVE sub-beat chord changes, family should
collapse faster than root as density rises (roots survive coarse segmentation;
qualities/boundaries don't) — i.e. the family/root gap should widen with
density. That gap-vs-density slope is the lever pointer.

Writes a scatter plot (density vs acc, sized by span) to
docs/plots/jaah_density_vs_acc.png. Read-only on inputs.

Usage:
    .venv/bin/python scripts/diagnose_jaah_density.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCORES = REPO / "docs" / "research_sessions" / "jaah_benchmark_scores.json"
LABS = REPO / "data" / "cache" / "jaah" / "labs"
PLOT = REPO / "docs" / "plots" / "jaah_density_vs_acc.png"


def lab_density(slug):
    p = LABS / f"{slug}.lab"
    rows = []
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        t0, t1, lab = parts
        rows.append((float(t0), float(t1), lab))
    if not rows:
        return None
    span = rows[-1][1] - rows[0][0]
    durs = np.array([t1 - t0 for t0, t1, _ in rows])
    return {
        "n": len(rows), "span": span, "density": len(rows) / span,
        "mean_dur": float(durs.mean()), "median_dur": float(np.median(durs)),
        "fast_frac": float((durs < 1.0).mean()),
    }


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    if not SCORES.exists():
        print("no scores yet — run build_jaah_benchmark.py first.")
        return 0
    data = json.loads(SCORES.read_text())["scores"]
    rows = []
    for s in data:
        if s["root_acc"] is None:
            continue
        d = lab_density(s["slug"])
        if not d:
            continue
        rows.append({**s, **d, "gap": s["root_acc"] - s["family_acc"]})

    rows.sort(key=lambda r: r["density"])
    print(f"=== JAAH density diagnostic (n={len(rows)} songs) ===\n")
    print(f"  {'slug':<20} {'dens':>6} {'mdur':>6} {'fast%':>6} "
          f"{'root':>6} {'fam':>6} {'gap':>6}")
    for r in rows:
        print(f"  {r['slug']:<20} {r['density']:>6.2f} {r['mean_dur']:>6.2f} "
              f"{100*r['fast_frac']:>5.0f}% {r['root_acc']:>6.3f} "
              f"{r['family_acc']:>6.3f} {r['gap']:>6.3f}")

    dens = [r["density"] for r in rows]
    fast = [r["fast_frac"] for r in rows]
    root = [r["root_acc"] for r in rows]
    fam = [r["family_acc"] for r in rows]
    gap = [r["gap"] for r in rows]
    print("\nPearson correlations (n small — treat as hypothesis, not proof):")
    print(f"  density   vs root : {pearson(dens, root):+.2f}")
    print(f"  density   vs family: {pearson(dens, fam):+.2f}")
    print(f"  density   vs gap  : {pearson(dens, gap):+.2f}   "
          f"(+ => family collapses faster than root as density rises)")
    print(f"  fast_frac vs root : {pearson(fast, root):+.2f}")
    print(f"  fast_frac vs family: {pearson(fast, fam):+.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        sizes = [r["span"] / 2 for r in rows]
        for a, (yy, lbl) in zip(ax, [(root, "root acc"), (fam, "family acc")]):
            a.scatter(dens, yy, s=sizes, alpha=0.7, edgecolor="k", linewidth=0.5)
            for r, y in zip(rows, yy):
                a.annotate(r["slug"][:10], (r["density"], y), fontsize=6,
                           xytext=(3, 3), textcoords="offset points")
            if len(rows) >= 3:
                m, b = np.polyfit(dens, yy, 1)
                xs = np.array([min(dens), max(dens)])
                a.plot(xs, m * xs + b, "r--", lw=1,
                       label=f"slope {m:+.2f}, r={pearson(dens, yy):+.2f}")
                a.legend(fontsize=8)
            a.set_xlabel("GT chord density (chords/sec)")
            a.set_ylabel(lbl)
            a.set_title(f"JAAH: {lbl} vs harmonic density")
            a.grid(alpha=0.3)
        fig.tight_layout()
        PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT, dpi=110)
        print(f"\nwrote {PLOT.relative_to(REPO)}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
