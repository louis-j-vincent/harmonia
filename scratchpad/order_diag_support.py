"""Le pic corrobore-t-il un run de 4 ? Diagnostic avant d'écrire la règle.

Pour chaque run trouvé par une passe de 4 lancée EN PREMIER, on mesure :
  * appui-pics  : part de ses frontières (début et fin de chaque occurrence) qui
                  tombe à ±1 mesure d'un pic dur ;
  * appui-vérité: la même chose contre les frontières annotées par Louis.
Si l'appui-pics prédit l'appui-vérité, on tient la porte cherchée.
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
from ssm_zoo import SONGS               # noqa: E402


def bounds(run, n):
    out = []
    for c in [run["b0"]] + run["occ"]:
        out += [c, c + run["block"]]
    return sorted({x for x in out if 0 < x < n})


def support(bs, marks, tol=1):
    if not bs:
        return 0.0
    return sum(1 for x in bs if any(abs(x - m) <= tol for m in marks)) / len(bs)


def gt_bounds(stem, n):
    from section_metric import segs
    ref = OS.gt(stem)
    e = set()
    for b0, b1, _ in segs(ref, n):
        e.add(b0); e.add(b1 + 1)
    return sorted(x for x in e if 0 < x < n)


def main():
    pts = []
    for stem, title in SONGS:
        b = order_bundle.get(stem)
        n = b["n"]
        gtb = gt_bounds(stem, n)
        runs, _ = OS._passes(b, b["hard"], [(4, 0.70)])
        print(f"{title}   pics={[h + 1 for h in b['hard']]}")
        for r in runs:
            bs = bounds(r, n)
            sp = support(bs, b["hard"])
            sg = support(bs, gtb)
            pts.append((sp, sg, len(r["occ"]) + 1))
            print(f"    4@{r['b0'] + 1:3d}  n_occ={len(r['occ']) + 1:2d}  "
                  f"appui-pics {sp:.2f}   appui-vérité {sg:.2f}   "
                  f"occ={[o + 1 for o in r['occ']]}")
    a = np.array([[p, g] for p, g, _ in pts])
    print(f"\ncorrélation appui-pics / appui-vérité : "
          f"{np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.3f}  (n={len(pts)} runs)")
    for lo in (0.0, 0.2, 0.3, 0.4, 0.5):
        sel = a[a[:, 0] >= lo]
        print(f"  appui-pics >= {lo:.1f} : {len(sel):3d} runs, "
              f"appui-vérité moyen {sel[:, 1].mean():.3f}" if len(sel) else "")


if __name__ == "__main__":
    main()
