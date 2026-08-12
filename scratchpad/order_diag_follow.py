"""Le « taux de suite » : un bloc de 8 est-il UNE unité, ou deux 4 collés ?

L'idée, en une phrase musicale : si la première moitié d'un bloc de 8 se rejoue
cinq fois dans le morceau mais que la deuxième moitié ne la SUIT qu'une fois sur
cinq, alors ces deux moitiés ne forment pas une unité — le bloc de 8 déborde sur
la section suivante.

    taux_de_suite(p) = |{c : H1 se rejoue en c ET H2 se rejoue en c+4}| / |{c : H1 se rejoue en c}|

Le test : pour chaque bloc de 8 posé par la passe actuelle, une frontière de
Louis tombe-t-elle en son MILIEU (p+4) ? Si oui le bloc déborde. Le taux de
suite le prédit-il ?
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

import order_bundle                     # noqa: E402
import order_search as OS               # noqa: E402
import harmonia_min.voice_sections as VS  # noqa: E402
from ssm_zoo import SONGS               # noqa: E402
from order_diag_support import gt_bounds  # noqa: E402


def sc_of(b, p, L):
    n, S, M, mute = b["n"], b["S"], b["M"], b["mute"]
    cm, ch = VS._slide(M, p, L, n), VS._slide(S, p, L, n)
    return VS.block_score(cm, ch, p, n, mute=mute, block=L)


def occ_free(b, p, L, thr):
    """Les reprises d'un bloc, SANS tenir compte de ce qui est déjà réclamé."""
    n = b["n"]
    free = np.zeros(n, bool)
    return [p] + VS._peaks(sc_of(b, p, L), p, n, thr, L, free)


def follow_rate(b, p, thr4=0.70, half=4):
    """Part des reprises de la 1re moitié que la 2e moitié suit."""
    n = b["n"]
    if p + 2 * half > n:
        return float("nan"), 0
    O1 = occ_free(b, p, half, thr4)
    if p + half + half > n:
        return float("nan"), len(O1)
    s2 = sc_of(b, p + half, half)
    ok = sum(1 for c in O1 if c + half < n and s2[c + half] >= thr4)
    return ok / max(1, len(O1)), len(O1)


def main():
    rows = []
    for stem, title in SONGS:
        b = order_bundle.get(stem)
        n = b["n"]
        gtb = set(gt_bounds(stem, n))
        runs, _ = OS._passes(b, b["hard"], [(8, 0.66)])
        print(f"{title}")
        for r in runs:
            fr, k1 = follow_rate(b, r["b0"])
            for c in [r["b0"]] + r["occ"]:
                mid_is_bound = any(abs(c + 4 - g) <= 1 for g in gtb)
                rows.append((fr, len(r["occ"]) + 1, mid_is_bound))
            n_split = sum(1 for c in [r["b0"]] + r["occ"]
                          if any(abs(c + 4 - g) <= 1 for g in gtb))
            print(f"    8@{r['b0'] + 1:3d} n_occ={len(r['occ']) + 1:2d} "
                  f"suite={fr:.2f} (|O1|={k1:2d})  "
                  f"milieux qui sont une frontière de Louis : "
                  f"{n_split}/{len(r['occ']) + 1}")
    a = np.array([[r[0], r[1], float(r[2])] for r in rows if not np.isnan(r[0])])
    print(f"\nn = {len(a)} occurrences de blocs de 8")
    print(f"corrélation taux-de-suite / « le milieu EST une frontière » : "
          f"{np.corrcoef(a[:, 0], a[:, 2])[0, 1]:.3f}")
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        s = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
        if len(s):
            print(f"  suite ∈ [{lo:.2f},{hi:.2f}) : {len(s):3d} blocs, "
                  f"{s[:, 2].mean():.0%} ont une frontière en leur milieu")


if __name__ == "__main__":
    main()
