"""Le criblage : est-ce qu'un modèle APPRIS sur nos features bat la règle à la main ?

    .venv/bin/python scripts/section_screen.py

C'est le premier pas exigé par `docs/design_2026-08-12_section_model.md` §4, et
par la règle n°2 du projet : écrire le test le moins cher qui puisse tuer l'idée,
et le lancer avant d'écrire le modèle.

LA QUESTION, précisément : sur les mêmes 110 features par mesure
(`section_features.py`), un modèle linéaire — puis un modèle à arbres — trouve-t-il
les débuts de section MIEUX que notre profil fusionné pondéré par le contraste ?

Si non, l'information n'est pas dans les features et un réseau séquentiel ne la
fabriquera pas : il faudrait retourner aux substrats. Si oui, l'écart mesuré ici
est le plancher de ce que le modèle séquentiel doit dépasser.

LE PROTOCOLE, et il est le même pour les deux concurrents :

  * **un-contre-tous par morceau** (leave-one-song-out). Douze morceaux, douze
    entraînements, jamais une mesure du morceau évalué dans l'entraînement ;
  * la probabilité par mesure est **pic-piquée avec exactement la même règle** que
    le profil (maxima locaux à ≥ 4 mesures d'écart, proéminence ≥ 0,25 du plus
    fort) — sinon on comparerait deux sélecteurs et pas deux détecteurs ;
  * on compte juste à **±1 mesure**, la tolérance qu'on s'est fixée après avoir
    mesuré que nos pics ne sont exacts qu'une fois sur deux ;
  * et le rapport est **par morceau**, jamais une moyenne seule.
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

from ssm_zoo import SONGS                     # noqa: E402
import section_features as SF                 # noqa: E402
from peak_profile import fused_profile, select_peaks   # noqa: E402

TOL = 1          # une frontière comptée juste à une mesure près


def pick(prob: np.ndarray) -> list[int]:
    """Les pics de la probabilité, même règle que le profil fusionné."""
    kept, _ = select_peaks(np.asarray(prob, float))
    return sorted(kept)


def prf(pred: list[int], gt: list[int], tol=TOL):
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    tp_p = sum(1 for b in pred if any(abs(b - g) <= tol for g in gt))
    tp_r = sum(1 for g in gt if any(abs(b - g) <= tol for b in pred))
    p = tp_p / max(1, len(pred))
    r = tp_r / max(1, len(gt))
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    d = SF.load()
    stems = [s for s, _ in SONGS]
    titles = dict(SONGS)
    X = {s: d[f"X:{s}"] for s in stems}
    Y = {s: d[f"y:{s}"] for s in stems}

    # la référence : nos pics d'aujourd'hui, sur les mêmes morceaux
    base = {}
    for s in stems:
        P, kept, _r, _l, n, _e = fused_profile(s)
        m = len(P)
        base[s] = [h for h in sorted({int(round(c * n / m)) for c in kept})
                   if 0 < h < n]

    models = {
        "linéaire": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
        "arbres": lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.06,
            class_weight="balanced", l2_regularization=1.0),
    }

    res = {k: [] for k in ["profil", *models]}
    rows = []
    for s in stems:
        gt = [int(b) for b in np.flatnonzero(Y[s] > 0)]
        line = {"profil": prf(base[s], gt)}
        tr = [t for t in stems if t != s]
        Xtr = np.vstack([X[t] for t in tr])
        ytr = np.concatenate([Y[t] for t in tr])
        for name, mk in models.items():
            mdl = mk()
            mdl.fit(Xtr, ytr)
            prob = mdl.predict_proba(X[s])[:, 1]
            line[name] = prf(pick(prob), gt)
        for k in res:
            res[k].append(line[k])
        rows.append((titles[s], line))

    hdr = f"{'morceau':22s} " + " ".join(f"{k:>22s}" for k in ["profil", *models])
    print(hdr)
    print(f"{'':22s} " + " ".join(f"{'préc. rapp.  F1':>22s}" for _ in res))
    for title, line in rows:
        print(f"{title[:22]:22s} " + " ".join(
            f"{line[k][0]:6.2f}{line[k][1]:6.2f}{line[k][2]:7.2f}   " for k in res))
    print()
    for k in res:
        a = np.array(res[k])
        print(f"{k:10s} médiane  précision {np.median(a[:, 0]):.2f} · "
              f"rappel {np.median(a[:, 1]):.2f} · F1 {np.median(a[:, 2]):.2f}")

    # ce que le linéaire regarde : les 12 poids les plus forts
    mdl = models["linéaire"]()
    mdl.fit(np.vstack([X[s] for s in stems]), np.concatenate([Y[s] for s in stems]))
    w = mdl[-1].coef_[0]
    names = [str(x) for x in d["names"]]
    order = np.argsort(-np.abs(w))[:12]
    print("\nce que le linéaire regarde (poids les plus forts, tout le corpus) :")
    for i in order:
        print(f"   {names[i]:28s} {w[i]:+.2f}")


if __name__ == "__main__":
    main()
