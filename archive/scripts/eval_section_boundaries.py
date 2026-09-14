"""Boundary-F for section-structure detection (issue #22) on held-out AABA
tunes.  GT boundaries = bars where the iReal `section_per_bar` label changes.

Two variants scored at +-1 bar tolerance:
  gmerge baseline : cut wherever the per-beat root changes (chord-level cuts) —
                    the production segmentation, which over-segments sections.
  chord-SSM       : detect_section_boundaries() on the symbolic chord SSM.

Run on GT chords (upper bound: is the *method* sound?) by default; --inferred
would swap in the pipeline's per-beat roots (slower, not needed to gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, QUALITY_MAP
from harmonia.models.section_structure import build_chord_ssm, detect_section_boundaries

DB = REPO / "data" / "accomp_db" / "db.jsonl"
QIDX = {q: i for i, q in enumerate(sorted(set(QUALITY_MAP.values())))}


def key_pc(k: str) -> int:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pc = base.get(k[0].upper(), 0)
    if len(k) > 1 and k[1] in "#b-":
        pc += 1 if k[1] == "#" else -1
    return pc % 12


def gt_chord_seq(rec) -> list[tuple[int, int]]:
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb
    ton = key_pc(rec["key"])
    slots = sorted(((e["bar"] - 1) * bpb + e["beat"], e["mma"]) for e in rec["chord_timeline"])
    seq: list[tuple[int, int]] = []
    cr, cq, si = -1, -1, 0
    for beat in range(n_beats):
        while si < len(slots) and slots[si][0] <= beat:
            p = parse_chord(slots[si][1])
            if p is not None:
                cr, cq = (p[0] - ton) % 12, QIDX.get(p[1], 0)
            si += 1
        seq.append((cr, cq))
    return seq


def gt_boundaries(rec) -> list[int]:
    bpb = rec["beats_per_bar"]
    spb = rec["section_per_bar"]
    return [b * bpb for b in range(1, len(spb)) if spb[b] != spb[b - 1]]


def gmerge_boundaries(seq) -> list[int]:
    roots = [r for r, _ in seq]
    return [b for b in range(1, len(roots)) if roots[b] != roots[b - 1]]


def boundary_f(est: list[int], gt: list[int], tol: int) -> tuple[float, float, float]:
    if not gt:
        return float("nan"), float("nan"), float("nan")
    if not est:
        return 0.0, 0.0, 0.0
    gt_left = list(gt)
    hits = 0
    for e in est:
        for j, g in enumerate(gt_left):
            if abs(e - g) <= tol:
                hits += 1
                gt_left.pop(j)
                break
    prec = hits / len(est)
    rec = hits / len(gt)
    f = 0.0 if hits == 0 else 2 * prec * rec / (prec + rec)
    return f, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=70)
    ap.add_argument("--form", default="A16 B8 A8")
    ap.add_argument("--tol-bars", type=int, default=1)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and r["form"] == args.form]
    held = jz[args.start:]
    print(f"held-out {args.form}: {len(held)} tunes (index >= {args.start})")

    tol = args.tol_bars * 4
    rows = {"gmerge": [], "chord-SSM": []}
    for rec in held:
        seq = gt_chord_seq(rec)
        gt = gt_boundaries(rec)
        ssm = build_chord_ssm(seq)
        est_ssm = detect_section_boundaries(ssm, beats_per_bar=rec["beats_per_bar"])
        est_gm = gmerge_boundaries(seq)
        rows["gmerge"].append(boundary_f(est_gm, gt, tol))
        rows["chord-SSM"].append(boundary_f(est_ssm, gt, tol))

    print(f"\n{'variant':<12} {'bF':>6} {'prec':>6} {'rec':>6} {'n':>4}  (tol +-{args.tol_bars} bar)")
    print("-" * 44)
    for name, rs in rows.items():
        arr = np.array([r for r in rs if not np.isnan(r[0])])
        print(f"{name:<12} {arr[:,0].mean():>6.3f} {arr[:,1].mean():>6.3f} "
              f"{arr[:,2].mean():>6.3f} {len(arr):>4}")


if __name__ == "__main__":
    main()
