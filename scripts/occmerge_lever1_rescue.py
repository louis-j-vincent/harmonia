"""Levier 1, sonde n°2 — le réalignement sauve-t-il des folds REFUSÉS ?

La sonde n°1 (occmerge_lever1_probe.py) a montré que le modulo rigide est
déjà bon pour 95 % des occurrences, et que la seule occurrence gravement
désalignée d'un fold ACCEPTÉ était déjà écartée en variante par OUTLIER_Z.
Le coût réel candidat : des lettres entières refusées en
« stack incoherent » parce qu'une occurrence décalée tire la cohérence
sous le seuil 0.85 — la lettre perd alors tout le gain √N.

Ce script rejoue le gating de fold_letter_groups (outliers médiane+MAD
puis cohérence par position) sur chaque lettre ≥2 occurrences, avec et
sans décalage par occurrence (choisi contre le centroïde leave-one-out),
et liste les lettres qui passeraient le seuil une fois réalignées.

    python scripts/occmerge_lever1_rescue.py [--json out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.folding import (OUTLIER_Z, STACK_COHERENCE,  # noqa: E402
                                  _bar_vecs, section_period)
from harmonia_min.sections import halfbar_features  # noqa: E402

LIVE_CHARTS = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia"
                   "/harmonia_min/state/charts")
NNLS_CACHE = REPO / "data" / "cache" / "nnls_infer"


def stack_stats(Vb, ranges, P, shifts):
    """Rejoue le gating de fold_letter_groups avec un décalage par
    occurrence : renvoie (cohérence min, [cohérences], n variantes)."""
    pos_members = [[] for _ in range(P)]
    for (b0, b1), s in zip(ranges, shifts):
        for b in range(b0, b1 + 1):
            if b < len(Vb):
                pos_members[(b - b0 - s) % P].append(b)
    gated, n_var = [[] for _ in range(P)], 0
    for k in range(P):
        mem = pos_members[k]
        if len(mem) < 3:
            gated[k] = mem
            continue
        cen = np.mean([Vb[b] for b in mem], axis=0)
        cen /= max(np.linalg.norm(cen), 1e-9)
        d = {b: 1.0 - float(Vb[b] @ cen) for b in mem}
        med = float(np.median(list(d.values())))
        mad = float(np.median(np.abs(np.array(list(d.values())) - med))) + 1e-6
        for b in mem:
            if (d[b] - med) / mad <= OUTLIER_Z:
                gated[k].append(b)
            else:
                n_var += 1
    coh = []
    for k in range(P):
        g = gated[k]
        if len(g) < 2:
            continue
        pw = [float(Vb[a] @ Vb[b]) for i, a in enumerate(g) for b in g[i + 1:]]
        coh.append(float(np.median(pw)))
    if not coh or sum(len(g) for g in gated) < 2 * P:
        return None, [], n_var
    return min(coh), [round(c, 3) for c in coh], n_var


def best_shifts(Vb, ranges, P, rounds=2):
    """Décalage par occurrence contre le centroïde des autres (itéré)."""
    shifts = [0] * len(ranges)
    for _ in range(rounds):
        for i, (b0, b1) in enumerate(ranges):
            cen = [[] for _ in range(P)]
            for j, ((c0, c1), s) in enumerate(zip(ranges, shifts)):
                if j == i:
                    continue
                for b in range(c0, c1 + 1):
                    if b < len(Vb):
                        cen[(b - c0 - s) % P].append(Vb[b])
            cen = [np.mean(v, axis=0) if v else None for v in cen]
            cen = [c / max(np.linalg.norm(c), 1e-9) if c is not None else None
                   for c in cen]
            sc = []
            for s in range(P):
                num, den = 0.0, 0
                for b in range(b0, b1 + 1):
                    if b >= len(Vb):
                        continue
                    k = (b - b0 - s) % P
                    if cen[k] is not None:
                        num += float(Vb[b] @ cen[k])
                        den += 1
                sc.append(num / den if den else float("-inf"))
            shifts[i] = int(np.argmax(sc))
    return shifts


def main(json_out=None):
    rows = []
    for p in sorted(LIVE_CHARTS.glob("min_*.json")):
        d = json.loads(p.read_text())
        stem = d["file"].removeprefix("min_")
        cache = NNLS_CACHE / f"{stem}.npz"
        if not cache.exists():
            continue
        z = np.load(cache)
        grid = d["barGrid"]
        n_bars = len(grid) - 1
        Vb = _bar_vecs(halfbar_features(grid, z["arr"], z["times"]), n_bars)
        for sec in d["sections"]:
            ranges = [tuple(r) for r in sec["barRanges"]]
            if len(ranges) < 2:
                continue
            picks = []
            for b0, b1 in ranges:
                P, _ = section_period(Vb, b0, min(b1, n_bars - 1))
                if P is not None:
                    picks.append(P)
            if not picks:
                continue
            P = int(np.bincount(picks).argmax())
            rigid = [0] * len(ranges)
            coh0, cohs0, var0 = stack_stats(Vb, ranges, P, rigid)
            shifts = best_shifts(Vb, ranges, P)
            coh1, cohs1, var1 = stack_stats(Vb, ranges, P, shifts)
            fold_status = d["fold"].get(sec["label"], {})
            rows.append({
                "song": stem, "letter": sec["label"], "P": P,
                "occurrences": [list(r) for r in ranges],
                "shifts": shifts,
                "coh_rigid": None if coh0 is None else round(coh0, 3),
                "coh_aligned": None if coh1 is None else round(coh1, 3),
                "fold_reason": fold_status.get("reason"),
                "was_folded": "n_obs" in fold_status,
            })
    print(f"{len(rows)} lettres ≥2 occurrences avec P consensus\n")
    print(f"{'chanson':38s} {'let':4s} {'P':>2s} {'coh rigide':>10s} "
          f"{'coh realign':>11s} {'décalages':16s} statut fold live")
    for r in rows:
        c0 = "  --  " if r["coh_rigid"] is None else f"{r['coh_rigid']:.3f}"
        c1 = "  --  " if r["coh_aligned"] is None else f"{r['coh_aligned']:.3f}"
        interesting = (r["shifts"] and any(s != 0 for s in r["shifts"]))
        mark = " <<<" if interesting else ""
        status = "FOLDÉ" if r["was_folded"] else (r["fold_reason"] or "?")
        print(f"{r['song'][:38]:38s} {r['letter']:4s} {r['P']:2d} {c0:>10s} "
              f"{c1:>11s} {str(r['shifts']):16s} {status[:36]}{mark}")
    resc = [r for r in rows if r["coh_rigid"] is not None
            and r["coh_aligned"] is not None
            and r["coh_rigid"] < STACK_COHERENCE <= r["coh_aligned"]]
    print(f"\nlettres sauvées par le réalignement (coh {STACK_COHERENCE} "
          f"franchi) : {len(resc)}")
    for r in resc:
        print(f"  {r['song']} {r['letter']}: {r['coh_rigid']:.3f} → "
              f"{r['coh_aligned']:.3f} shifts={r['shifts']}")
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=1))
        print(f"→ {json_out}")


if __name__ == "__main__":
    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
    main(out)
