"""Le moteur de l'outil sections, testé contre les annotations de Louis.

On simule le geste exactement : on prend la PREMIÈRE occurrence d'une lettre
telle que Louis l'a marquée à la main (`harmonia_min/state/sections/*.json`,
18 morceaux), on la donne à `find_repeats`, et on regarde s'il retrouve les
AUTRES occurrences de cette lettre — qu'on ne lui a jamais montrées.

Ce n'est pas un score de modèle. C'est la question de l'outil : « quand le
doigt a désigné un couplet, est-ce que les autres couplets s'allument ? »
La seule vérité que ce projet accepte pour les sections est celle-là, et
elle vient de son oreille.

Deux chiffres, tous les deux nécessaires :
  retrouvées  occurrences annotées (hors celle donnée) qu'une trouvaille
              recouvre à plus de moitié — l'outil a-t-il fait le travail ;
  hors-lettre trouvailles qui ne touchent AUCUNE occurrence annotée de la
              lettre — ce que l'utilisateur devra désélectionner à la main.

    python scripts/section_tool_check.py [--stem X] [--margin 0.05]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx  # noqa: E402
from harmonia_min.section_tool import find_repeats  # noqa: E402

LIVE = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
CHARTS = LIVE / "harmonia_min" / "state" / "charts"
GT_DIR = LIVE / "harmonia_min" / "state" / "sections"
AUDIO = LIVE / "docs" / "audio"

SKIP_LABELS = {"intro", "outro"}


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def check_song(stem, gt, grid, triad, margin, melody=False):
    by_letter: dict[str, list[tuple[int, int]]] = {}
    for s in gt["sections"]:
        by_letter.setdefault(s["label"], []).append((s["b0"], s["b1"]))
    rows = []
    for letter, ranges in by_letter.items():
        if letter in SKIP_LABELS or len(ranges) < 2:
            continue
        ranges.sort()
        first, rest = ranges[0], ranges[1:]
        res = find_repeats(grid, triad, first[0], first[1],
                           audio=AUDIO / f"{stem}.m4a", melody=melody)
        found = [(o["b0"], o["b1"]) for o in res["occurrences"]]
        hit = sum(1 for r in rest
                  if any(overlap(r, f) > (r[1] - r[0] + 1) / 2 for f in found))
        noise = sum(1 for f in found
                    if not any(overlap(r, f) > 0 for r in ranges))
        rows.append({"stem": stem, "letter": letter, "sel": list(first),
                     "expected": len(rest), "hit": hit, "noise": noise,
                     "cell": res["cell_bars"], "n_inner": res["n_inner"],
                     "thr": res["thr"], "basis": res["parity"],
                     "tiling": res["agreement"]})
    return rows


def main(argv):
    only = argv[argv.index("--stem") + 1] if "--stem" in argv else None
    margin = float(argv[argv.index("--margin") + 1]) \
        if "--margin" in argv else 0.05
    rows, skipped = [], []
    for p in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(p.read_text())
        stem = gt["stem"]
        if only and stem != only:
            continue
        chart = CHARTS / f"min_{stem}.json"
        cache = AUDIO / f"{stem}.m4a"
        if not chart.exists() or not cache.exists():
            skipped.append(stem)
            continue
        m = json.loads(chart.read_text())
        if len(m["barGrid"]) - 1 != gt["n"]:
            skipped.append(f"{stem} (grille {len(m['barGrid'])-1} ≠ "
                           f"annotation {gt['n']})")
            continue
        triad = _musx.frame_posteriors(cache)[0]
        rows += check_song(stem, gt, m["barGrid"], triad, margin,
                           melody="--melody" in argv)
    exp = sum(r["expected"] for r in rows)
    hit = sum(r["hit"] for r in rows)
    noi = sum(r["noise"] for r in rows)
    print(f"\n{len({r['stem'] for r in rows})} morceaux · {len(rows)} lettres "
          f"· {hit}/{exp} occurrences retrouvées "
          f"({100 * hit / max(exp, 1):.0f} %) · {noi} trouvailles hors lettre "
          f"(marge {margin})")
    if skipped:
        print(f"sautés : {', '.join(skipped)}")
    print(f"\n{'morceau':30s} {'let':5s} {'sel':>9s} {'cell':>4s} "
          f"{'n_in':>4s} {'til':>5s} {'thr':>5s} {'base':>7s} "
          f"{'trouvé':>7s} {'bruit':>6s}")
    for r in sorted(rows, key=lambda r: (r["hit"] - r["expected"],
                                         -r["noise"])):
        print(f"{r['stem'][:30]:30s} {r['letter']:5s} {str(r['sel']):>9s} "
              f"{r['cell']:4d} {r['n_inner']:4d} {r['tiling']:5.2f} "
              f"{r['thr']:5.2f} {r['basis']:>7s} "
              f"{r['hit']:3d}/{r['expected']:<3d} {r['noise']:6d}")


if __name__ == "__main__":
    main(sys.argv[1:])
