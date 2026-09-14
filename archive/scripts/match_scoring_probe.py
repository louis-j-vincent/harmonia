"""Le score d'appariement favorise-t-il les blocs LONGS ? (Louis, 2026-08-12)

« Lorsque j'identifie une première section elle est mal rematchée aux autres
endroits […] j'ai l'impression qu'on favorise le scoring de longues sections
même si elles ne sont pas bonnes ? »

Le score de prod (`voice_sections.block_score`) est une MOYENNE de
similarités mesure à mesure sur la longueur du bloc. Une moyenne dilue :
plus le bloc est long, plus une mesure qui ne colle pas se noie dans les
autres. Deux endroits qui n'ont rien à voir peuvent donc monter haut, et
c'est précisément ce qu'il décrit.

Ce script mesure le POUVOIR DE SÉPARATION, pas le score : pour chaque lettre
annotée à la main (≥2 occurrences), on prend la première occurrence comme
ancre et on regarde si les AUTRES occurrences annotées scorent au-dessus des
positions qui ne sont pas des occurrences. L'aire sous la courbe ROC (AUC)
dit exactement ça : 1,0 = les vraies reprises sont toujours devant, 0,5 = le
score ne sait rien.

Trois lois comparées, à substrat identique :
  moyenne   ce que fait la prod ;
  minimum   le bloc vaut sa PIRE mesure — une seule mesure qui diverge suffit
            à disqualifier ;
  q25       compromis : le premier quartile des mesures du bloc.

    python scripts/match_scoring_probe.py [--json out.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import harmonic_sections as HS   # noqa: E402
from harmonia_min import musx as _musx             # noqa: E402

CHARTS = REPO / "harmonia_min" / "state" / "charts"
GT = REPO / "harmonia_min" / "state" / "sections"
AUDIO = REPO / "docs" / "audio"
SKIP = {"intro", "outro"}


def slide_stats(S, b0, L, n):
    """Pour chaque position c : moyenne, minimum et q25 des similarités
    mesure à mesure entre le bloc [b0, b0+L) et [c, c+L)."""
    out = {}
    for name in ("moyenne", "minimum", "q25"):
        out[name] = np.zeros(n)
    for c in range(0, n - L + 1):
        v = np.array([S[b0 + i, c + i] for i in range(L)
                      if b0 + i < n and c + i < n])
        if not len(v):
            continue
        out["moyenne"][c] = v.mean()
        out["minimum"][c] = v.min()
        out["q25"][c] = np.quantile(v, 0.25)
    return out


def auc(pos, neg):
    """Probabilité qu'une vraie reprise score au-dessus d'une fausse."""
    if not len(pos) or not len(neg):
        return None
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def main(argv):
    rows = []
    for f in sorted(GT.glob("*.json")):
        gt = json.loads(f.read_text())
        stem = gt["stem"]
        chart, audio = CHARTS / f"min_{stem}.json", AUDIO / f"{stem}.m4a"
        if not chart.exists() or not audio.exists():
            continue
        m = json.loads(chart.read_text())
        grid = m["barGrid"]
        n = len(grid) - 1
        if n != gt.get("n"):
            continue
        V = HS.harmonic_vectors(_musx.frame_posteriors(audio)[0], grid)
        S = V @ V.T
        by = defaultdict(list)
        for s in gt["sections"]:
            by[s["label"]].append((s["b0"], s["b1"]))
        for lab, occ in by.items():
            if lab in SKIP or len(occ) < 2:
                continue
            occ.sort()
            (b0, b1), rest = occ[0], occ[1:]
            L = b1 - b0 + 1
            if L < 2 or b0 + L > n:
                continue
            st = slide_stats(S, b0, L, n)
            true_pos = [r[0] for r in rest]
            inside = set()
            for a, z in occ:
                inside |= set(range(a, z + 1))
            cand = [c for c in range(n - L + 1)
                    if c not in true_pos and not (set(range(c, c + L)) & inside)]
            for law, curve in st.items():
                a = auc(np.array([curve[c] for c in true_pos]),
                        np.array([curve[c] for c in cand]))
                if a is not None:
                    rows.append({"stem": stem, "letter": lab, "L": L,
                                 "law": law, "auc": round(a, 3)})
    if not rows:
        print("aucune lettre exploitable")
        return 1
    print(f"{len({(r['stem'], r['letter']) for r in rows})} lettres annotées "
          f"(≥2 occurrences), {len({r['stem'] for r in rows})} morceaux\n")
    print("POUVOIR DE SÉPARATION (AUC — 1,0 = parfait, 0,5 = ne sait rien)")
    print(f"{'loi':>9s} {'AUC moyenne':>12s} {'médiane':>9s} "
          f"{'lettres à AUC=1':>16s}")
    for law in ("moyenne", "minimum", "q25"):
        a = np.array([r["auc"] for r in rows if r["law"] == law])
        print(f"{law:>9s} {a.mean():12.3f} {np.median(a):9.3f} "
              f"{100 * (a == 1.0).mean():15.0f}%")
    print("\nPAR LONGUEUR DE BLOC (la question de Louis)")
    print(f"{'bloc':>6s} {'n':>3s} " +
          " ".join(f"{law:>9s}" for law in ("moyenne", "minimum", "q25")))
    for lo, hi in ((2, 3), (4, 7), (8, 8), (9, 32)):
        sel = [r for r in rows if lo <= r["L"] <= hi]
        if not sel:
            continue
        k = len({(r["stem"], r["letter"]) for r in sel})
        line = f"{lo}-{hi:<3d} {k:3d} "
        for law in ("moyenne", "minimum", "q25"):
            a = np.array([r["auc"] for r in sel if r["law"] == law])
            line += f"{a.mean():9.3f} "
        print(line)
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows,
                                                                  indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
