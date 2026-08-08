"""Levier 1 (merge d'occurrences) — test de falsification AVANT d'implémenter.

Hypothèse à tester (handoff 2026-08-08, §4 levier 1) : l'empilement
`(b - b0) % P` de fold_letter_groups aligne mal certaines occurrences ;
un balayage des décalages contre le centroïde des AUTRES occurrences
trouverait un meilleur alignement ailleurs qu'en 0.

Ce script ne modifie rien : pour chaque chart de la bibliothèque live,
chaque lettre à période consensus P et ≥2 occurrences, chaque occurrence,
il balaie le décalage cyclique s ∈ 0..P-1 (couvre ±1 mesure) et compare
le score d'alignement (cos moyen au centroïde leave-one-out) en s=0 vs au
meilleur s. Si le meilleur s est 0 partout, l'hypothèse tombe (pattern
d'erreur n°2 : falsifier au prix le plus bas d'abord).

Substrat identique au code réel : halfbar_features (NNLS bothchroma en
cache) → _bar_vecs, section_period pour P — mêmes fonctions importées.

    python scripts/occmerge_lever1_probe.py [--json out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.folding import _bar_vecs, section_period  # noqa: E402
from harmonia_min.sections import halfbar_features  # noqa: E402

# La bibliothèque LIVE (l'arbre principal), lue seulement.
LIVE_CHARTS = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia"
                   "/harmonia_min/state/charts")
NNLS_CACHE = REPO / "data" / "cache" / "nnls_infer"


def occ_positions(occ, P):
    """[(bar, position)] pour une occurrence, modulo rigide comme le code."""
    b0, b1 = occ
    return [(b, (b - b0) % P) for b in range(b0, b1 + 1)]


def probe_letter(Vb, ranges, P):
    """Pour chaque occurrence : score(s) pour s ∈ 0..P-1 contre le centroïde
    des autres occurrences ; renvoie la liste des sondes."""
    out = []
    for i, occ in enumerate(ranges):
        others = [o for j, o in enumerate(ranges) if j != i]
        if not others:
            continue
        cen = [[] for _ in range(P)]
        for o in others:
            for b, k in occ_positions(o, P):
                if b < len(Vb):
                    cen[k].append(Vb[b])
        cen = [np.mean(v, axis=0) if v else None for v in cen]
        cen = [c / max(np.linalg.norm(c), 1e-9) if c is not None else None
               for c in cen]
        b0, b1 = occ
        scores = []
        for s in range(P):
            num, den = 0.0, 0
            for b in range(b0, b1 + 1):
                if b >= len(Vb):
                    continue
                k = (b - b0 - s) % P
                if cen[k] is None:
                    continue
                num += float(Vb[b] @ cen[k])
                den += 1
            scores.append(num / den if den else float("-inf"))
        s_star = int(np.argmax(scores))
        out.append({"occ": list(occ), "scores": [round(x, 4) for x in scores],
                    "s0": round(scores[0], 4), "s_star": s_star,
                    "gain": round(scores[s_star] - scores[0], 4)})
    return out


def main(json_out=None):
    rows, songs = [], 0
    for p in sorted(LIVE_CHARTS.glob("min_*.json")):
        d = json.loads(p.read_text())
        stem = d["file"].removeprefix("min_")
        cache = NNLS_CACHE / f"{stem}.npz"
        if not cache.exists():
            print(f"-- {stem}: pas de cache NNLS, sauté")
            continue
        z = np.load(cache)
        arr, times = z["arr"], z["times"]
        grid = d["barGrid"]
        n_bars = len(grid) - 1
        F = halfbar_features(grid, arr, times)
        Vb = _bar_vecs(F, n_bars)
        songs += 1
        for sec in d["sections"]:
            letter = sec["label"]
            ranges = [tuple(r) for r in sec["barRanges"]]
            if len(ranges) < 2:
                continue
            picks = []
            for b0, b1 in ranges:
                P, sc = section_period(Vb, b0, min(b1, n_bars - 1))
                if P is not None:
                    picks.append(P)
            if not picks:
                continue
            P = int(np.bincount(picks).argmax())
            for pr in probe_letter(Vb, ranges, P):
                pr.update({"song": stem, "letter": letter, "P": P})
                rows.append(pr)
    shifted = [r for r in rows if r["s_star"] != 0]
    print(f"\n{songs} charts · {len(rows)} occurrences sondées "
          f"(lettres ≥2 occurrences, P consensus)")
    print(f"meilleur décalage = 0 : {len(rows) - len(shifted)}/{len(rows)}")
    if shifted:
        print(f"meilleur décalage ≠ 0 : {len(shifted)} — détail :")
        print(f"{'chanson':44s} {'lettre':6s} {'P':>2s} {'occ':>10s} "
              f"{'s*':>3s} {'s=0':>7s} {'gain':>7s}")
        for r in sorted(shifted, key=lambda r: -r["gain"]):
            print(f"{r['song'][:44]:44s} {r['letter']:6s} {r['P']:2d} "
                  f"{str(r['occ']):>10s} {r['s_star']:3d} {r['s0']:7.3f} "
                  f"{r['gain']:+7.3f}")
    gains = [r["gain"] for r in shifted]
    if gains:
        print(f"\ngain médian des décalées : {np.median(gains):+.3f} · "
              f"max {max(gains):+.3f}")
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=1))
        print(f"→ {json_out}")


if __name__ == "__main__":
    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
    main(out)
