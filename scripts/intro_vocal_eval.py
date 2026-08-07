"""Le tableau : chaque descripteur vocal, TOUT SEUL, contre la vérité de Louis.

    python scripts/intro_vocal_eval.py

Deux mesures séparées, parce que Louis a demandé qu'elles le soient :

  POSITION  sur les neuf morceaux QUI ONT une intro, le descripteur pose-t-il la
            frontière à la bonne mesure ? (score exact, puis à ±1 mesure)
  PRÉSENCE  le descripteur sait-il dire « ici il n'y a pas d'intro » ?

Les mélanger cacherait exactement ce qu'il veut voir : un détecteur qui trouve
toujours une intro peut faire 9/9 en position et être inutilisable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import intro_vocal as IV                                          # noqa: E402


def predictions(D, kmax=IV.KMAX, win=IV.WIN):
    """{descripteur: {morceau: (mesure prédite, force, sens)}} — les deux sens."""
    out = {}
    for name in IV.CURVES:
        for sign, tag in ((IV.SIGNS[name], ""), (-IV.SIGNS[name], "⁻")):
            key = name + tag
            out[key] = {}
            for st, d in D.items():
                x = np.asarray(d["F"][name], float)
                k, s = IV.detect_step(x, sign, kmax, win)
                out[key][st] = (k, s, sign)
    # les détecteurs à SEUIL, qui n'ont de sens que pour « la voix entre »
    for name in ("voiced", "note_cov", "rms", "rms_p90"):
        for q in (0.05, 0.15, 0.30):
            key = f"{name}@{q:.2f}"
            out[key] = {}
            for st, d in D.items():
                x = np.asarray(d["F"][name], float)
                k, s = IV.detect_thresh(x, q, kmax)
                out[key][st] = (k, s, +1)
    return out


def score(D, P):
    """Une ligne par descripteur : position sur les 9, présence sur les 10."""
    withintro = [st for st in D if D[st]["truth"] > 0]
    rows = []
    for key, pred in P.items():
        ex = sum(pred[st][0] == D[st]["truth"] for st in withintro)
        t1 = sum(abs(pred[st][0] - D[st]["truth"]) <= 1 for st in withintro)
        err = float(np.mean([abs(pred[st][0] - D[st]["truth"]) for st in withintro]))
        # PRÉSENCE : la force de la marche sépare-t-elle « vraie intro (≥4) » de
        # « pas d'intro / une mesure (≤2) » ? Rangs, donc AUC.
        long_ = [pred[st][1] for st in D if D[st]["truth"] >= 4]
        short = [pred[st][1] for st in D if D[st]["truth"] <= 2]
        auc = float(np.mean([[1.0, 0.5, 0.0][(a < b) + (a <= b)]
                             for a in long_ for b in short]))
        rows.append({"key": key, "exact": ex, "tol1": t1, "err": err,
                     "auc": auc, "pred": {st: pred[st][0] for st in D},
                     "force": {st: pred[st][1] for st in D}})
    rows.sort(key=lambda r: (-r["exact"], -r["tol1"], r["err"]))
    return rows, withintro


def main():
    D = IV.collect()
    P = predictions(D)
    rows, wi = score(D, P)
    truth = {st: D[st]["truth"] for st in D}
    print("VÉRITÉ  ", {st[:14]: truth[st] for st in sorted(D)})
    print(f"règle livrée : {sum(D[s]['shipped'] == truth[s] for s in D)}/10 "
          f"(sur les 9 avec intro : "
          f"{sum(D[s]['shipped'] == truth[s] for s in wi)}/9)")
    print(f"« toujours 4 mesures » : {sum(truth[s] == 4 for s in D)}/10\n")
    print(f"{'descripteur':<22} {'exact/9':>8} {'±1/9':>6} {'err':>6} {'présence':>9}")
    for r in rows[:30]:
        print(f"{r['key']:<22} {r['exact']:>8} {r['tol1']:>6} "
              f"{r['err']:>6.1f} {r['auc']:>9.2f}")
    return D, rows


if __name__ == "__main__":
    main()
