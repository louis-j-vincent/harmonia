"""Le rappel des frontières, à budget de propositions fixé.

    .venv/bin/python scripts/section_recall.py

Louis, 2026-08-12, avant de partir :

  « je pense qu'on a largement tous les ingrédients pour faire marcher un modèle
    d'extraction de frontière, sachant qu'il ne nous les faut pas toutes, juste
    le + possible et pas de fausse frontière (0 faux positif) […] la notion de
    faux positif est floue car tu peux couper une section, donc pour l'instant je
    veux juste le + de frontières possible. »

**L'objectif change donc, et cette page de mesure existe pour ça** : ce qu'on
maximise est le RAPPEL des frontières, pas le F1. Couper une section en deux
n'est pas une faute — c'est réversible, un `merge` peut le défaire ; rater une
frontière ne l'est pas.

LE PROTOCOLE. Un budget de propositions par morceau, exprimé en « une frontière
toutes les K mesures » : à K = 8, un morceau de 80 mesures a droit à 10
propositions. On compare à budget ÉGAL — sinon un détecteur qui propose tout
gagne. Les propositions sont les meilleures probabilités, espacées d'au moins 2
mesures (deux frontières collées n'ont pas de sens). Une frontière est trouvée à
±1 mesure. Un-contre-tous par morceau : le modèle ne voit jamais le morceau
qu'on évalue.
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

from ssm_zoo import SONGS                      # noqa: E402
import section_features as SF                  # noqa: E402
from peak_profile import fused_profile         # noqa: E402

TOL = 1
MIN_GAP = 2          # deux propositions ne peuvent pas être à moins de 2 mesures
BUDGETS = (8, 4)     # une proposition par tranche de tant de mesures
PERIOD = 4           # la grille métrique : les sections commencent presque
                     # toujours sur une même phase modulo 4 mesures


def couverture(pred, n, tol=TOL) -> float:
    """La part du morceau que les propositions couvrent, tolérance comprise.

    LE GARDE-FOU, et il a déjà servi. À ±1 mesure, proposer une mesure sur
    quatre couvre 75 % du morceau : on peut alors afficher 95 % de rappel sans
    rien détecter du tout. Toute ligne de rappel doit être lue avec sa
    couverture — c'est l'erreur n°1 du projet (une mesure qui paraît bonne parce
    que son échelle est fausse).
    """
    cov = set()
    for b in pred:
        cov |= {b + d for d in range(-tol, tol + 1)}
    return len(cov & set(range(n))) / max(1, n)


def grille(score, n, k, period=PERIOD, gap=MIN_GAP):
    """La GRILLE MÉTRIQUE : on choisit la phase modulo `period` dont les k
    meilleures mesures totalisent le plus haut score, et on ne propose que des
    mesures de cette phase.

    Pourquoi ça marche : une section commence presque toujours sur la même
    phase hypermétrique. Le modèle n'a donc plus à trouver n frontières, mais
    UNE phase et n scores. Mesuré (±0 mesure, budget 1/8, couverture 12 %) :
    48 % de rappel médian contre 39 % pour le même modèle sans grille et 30 %
    pour le profil fusionné.
    """
    sc = np.asarray(score)
    def cand(p):
        return [i for i in range(p, n, period) if 0 < i < n]
    best = max(range(period),
               key=lambda p: float(np.sort(sc[cand(p)])[::-1][:k].sum()) if cand(p) else -1)
    idx = cand(best)
    idx.sort(key=lambda i: -sc[i])
    return sorted(idx[:k])


def topk(score: np.ndarray, k: int, gap=MIN_GAP) -> list[int]:
    """Les k meilleures mesures, espacées d'au moins `gap`."""
    out = []
    for i in np.argsort(-np.asarray(score)):
        i = int(i)
        if i == 0:
            continue
        if all(abs(i - j) >= gap for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return sorted(out)


def recall(pred, gt, tol=TOL):
    if not gt:
        return float("nan")
    return sum(1 for g in gt if any(abs(p - g) <= tol for p in pred)) / len(gt)


def precision(pred, gt, tol=TOL):
    if not pred:
        return float("nan")
    return sum(1 for p in pred if any(abs(p - g) <= tol for g in gt)) / len(pred)


def scores_loso():
    """Les scores par mesure de chaque concurrent, un-contre-tous."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    d = SF.load()
    stems = [s for s, _ in SONGS]
    X = {s: d[f"X:{s}"] for s in stems}
    Y = {s: d[f"y:{s}"] for s in stems}

    makers = {
        "linéaire": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")),
        "arbres": lambda: HistGradientBoostingClassifier(
            max_iter=250, max_depth=3, learning_rate=0.06,
            class_weight="balanced", l2_regularization=1.0),
    }
    out = {k: {} for k in makers}
    for s in stems:
        tr = [t for t in stems if t != s]
        Xtr = np.vstack([X[t] for t in tr]); ytr = np.concatenate([Y[t] for t in tr])
        for k, mk in makers.items():
            mdl = mk(); mdl.fit(Xtr, ytr)
            out[k][s] = mdl.predict_proba(X[s])[:, 1]
    # la référence : le profil fusionné, comme score par mesure
    out["profil"] = {}
    for s in stems:
        P, kept, _r, _l, n, _e = fused_profile(s)
        m = len(P); hb = m // n
        out["profil"][s] = np.asarray(P)[::hb][:n]
    return out, {s: [int(b) for b in np.flatnonzero(Y[s] > 0)] for s in stems}


def main():
    S, GT = scores_loso()
    stems = [s for s, _ in SONGS]
    titles = dict(SONGS)
    names = list(S)

    names = names + ["grille"]
    for tol in (0, 1):
        for K in BUDGETS:
            print(f"\n=== tolérance ±{tol} mesure · budget une proposition "
                  f"toutes les {K} mesures  (rappel · précision)")
            print(f"{'morceau':22s} " + " ".join(f"{n:>16s}" for n in names))
            agg = {n: [] for n in names}; cov = {n: [] for n in names}
            for s in stems:
                gt = GT[s]
                n_bars = len(S["profil"][s])
                k = max(1, int(round(n_bars / K)))
                row = ""
                for nm in names:
                    src = S["linéaire"][s] if nm == "grille" else S[nm][s]
                    pred = grille(src, n_bars, k) if nm == "grille" else topk(src, k)
                    r, p = recall(pred, gt, tol), precision(pred, gt, tol)
                    agg[nm].append((r, p)); cov[nm].append(couverture(pred, n_bars, tol))
                    row += f"{r:8.0%}{p:8.0%}"
                print(f"{titles[s][:22]:22s} " + row)
            print(f"{'MÉDIANE':22s} " + " ".join(
                f"{np.median([a[0] for a in agg[n]]):8.0%}"
                f"{np.median([a[1] for a in agg[n]]):8.0%}" for n in names))
            print(f"{'couverture du morceau':22s} " + " ".join(
                f"{np.median(cov[n]):16.0%}" for n in names))


if __name__ == "__main__":
    main()
