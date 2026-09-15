"""À quelle granularité l'outil sections doit-il CHERCHER les reprises ?

Louis : « si on a sélectionné 16 barres, on peut partir du principe que ce
sont deux répétitions de 8 mesures, ou 4 de 4, et chercher les répétitions
à cette granularité-là ». Première mesure : chercher avec la petite cellule
effondre tout (0 occurrence retrouvée, jusqu'à 30 fausses par lettre). Ce
script arbitre proprement entre trois bras, sur les 18 morceaux annotés.

  L      on cherche avec la sélection ENTIÈRE (l'unité que le doigt a
         désignée), quoi qu'elle contienne ;
  cell   on cherche avec la cellule interne (la lecture littérale) ;
  garde  cellule, SAUF si elle inonde — plus de FLOOD× les trouvailles de
         la recherche entière : alors on retombe sur la sélection entière.

Métrique honnête : une occurrence annotée est retrouvée si l'UNION des
trouvailles la couvre à plus de moitié (deux cellules de 4 collées couvrent
une occurrence de 8 — les compter séparément les déclarait ratées, ce qui
punissait le bras `cell` pour une raison purement comptable).
`hors-lettre` = mesures trouvées qui ne tombent dans AUCUNE occurrence
annotée de la lettre : ce que l'utilisateur devra désélectionner.

    python scripts/section_tool_arms.py [--flood 3.0]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx                      # noqa: E402
from harmonia_min import section_tool as ST                 # noqa: E402

LIVE = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
CHARTS = LIVE / "harmonia_min" / "state" / "charts"
GT_DIR = LIVE / "harmonia_min" / "state" / "sections"
AUDIO = LIVE / "docs" / "audio"
SKIP = {"intro", "outro"}


def covered(bars: set, rng) -> float:
    L = rng[1] - rng[0] + 1
    return sum(1 for b in range(rng[0], rng[1] + 1) if b in bars) / max(L, 1)


def arms_for(grid, triad, b0, b1, flood):
    """Les trois bras, à partir des mêmes substrats calculés une fois."""
    out = {}
    full = ST.find_repeats(grid, triad, b0, b1, force_cell=b1 - b0 + 1)
    out["L"] = full
    cell = ST.find_repeats(grid, triad, b0, b1)
    out["cell"] = cell
    n_full = sum(1 for o in full["occurrences"] if o["source"] != "selection")
    n_cell = sum(1 for o in cell["occurrences"] if o["source"] != "selection")
    out["garde"] = full if (cell["cell_bars"] < full["cell_bars"]
                            and n_cell > flood * max(n_full, 1)) else cell
    return out


def main(argv):
    flood = float(argv[argv.index("--flood") + 1]) if "--flood" in argv else 3.0
    tot = {a: {"hit": 0, "noise_bars": 0} for a in ("L", "cell", "garde")}
    exp_total, rows = 0, []
    for p in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(p.read_text())
        stem = gt["stem"]
        chart, audio = CHARTS / f"min_{stem}.json", AUDIO / f"{stem}.m4a"
        if not chart.exists() or not audio.exists():
            continue
        m = json.loads(chart.read_text())
        if len(m["barGrid"]) - 1 != gt["n"]:
            continue
        triad = _musx.frame_posteriors(audio)[0]
        grid = m["barGrid"]
        by_letter: dict[str, list] = {}
        for s in gt["sections"]:
            by_letter.setdefault(s["label"], []).append((s["b0"], s["b1"]))
        for letter, ranges in by_letter.items():
            if letter in SKIP or len(ranges) < 2:
                continue
            ranges.sort()
            first, rest = ranges[0], ranges[1:]
            exp_total += len(rest)
            res = arms_for(grid, triad, first[0], first[1], flood)
            row = {"stem": stem[:26], "letter": letter, "exp": len(rest)}
            for arm, r in res.items():
                bars = set()
                for o in r["occurrences"]:
                    bars |= set(range(o["b0"], o["b1"] + 1))
                hit = sum(1 for q in rest if covered(bars, q) > 0.5)
                inside = set()
                for q in ranges:
                    inside |= set(range(q[0], q[1] + 1))
                noise = len(bars - inside)
                tot[arm]["hit"] += hit
                tot[arm]["noise_bars"] += noise
                row[arm] = (hit, noise, r["cell_bars"])
            rows.append(row)
    print(f"\n{len(rows)} lettres · {exp_total} occurrences à retrouver "
          f"(flood ×{flood})\n")
    print(f"{'bras':8s} {'retrouvées':>18s} {'mesures hors-lettre':>20s}")
    for arm in ("L", "cell", "garde"):
        h, nb = tot[arm]["hit"], tot[arm]["noise_bars"]
        print(f"{arm:8s} {h:7d}/{exp_total:<4d} ({100*h/exp_total:3.0f} %) "
              f"{nb:19d}")
    print(f"\n{'morceau':26s} {'let':5s} {'exp':>3s} "
          f"{'L (h/bruit/cell)':>18s} {'cell':>18s} {'garde':>18s}")
    for r in sorted(rows, key=lambda r: r["stem"]):
        f = lambda t: f"{t[0]}/{t[1]}/{t[2]}"          # noqa: E731
        print(f"{r['stem']:26s} {r['letter']:5s} {r['exp']:3d} "
              f"{f(r['L']):>18s} {f(r['cell']):>18s} {f(r['garde']):>18s}")


if __name__ == "__main__":
    main(sys.argv[1:])
