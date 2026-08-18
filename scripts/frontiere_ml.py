#!/usr/bin/env python3
"""scripts/frontiere_ml.py — apprendre où tombe une frontière de section.

Louis, 2026-08-18 : « on ne regarde pas en bigramme, regarde tout en quatre, en
logique de quatre accords. Et plutôt que d'apprendre des règles par cœur, tu
vas me faire un modèle machine learning qui va essayer de prédire. Tu choisis
plusieurs architectures, tu me fais plusieurs tests, et je suis persuadé qu'il
y a une bonne réponse, donc si ça ne marche pas tu continues d'itérer. »

LE PROBLÈME. Pour chaque mesure du morceau : est-ce le DÉBUT d'une section ?
Ses 19 annotations à jour donnent 186 frontières sur ~1480 mesures — 13 % de
positifs, donc la précision brute ne veut rien dire et on mesure en aire sous
la courbe précision/rappel (AP).

LES ENTRÉES, en logique de quatre. Autour de la mesure candidate `b` :
les quatre degrés qui PRÉCÈDENT (b-4..b-1) et les quatre qui SUIVENT (b..b+3),
chacun relatif à la tonalité du morceau. C'est la fenêtre qu'il demande, et
elle voit donc la jonction entre deux blocs de quatre.

LA VALIDATION est LEAVE-ONE-SONG-OUT. Sans ça un modèle apprend la boucle d'un
morceau et se note lui-même : sur 19 morceaux, la fuite est massive.

    .venv/bin/python scripts/frontiere_ml.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.soudure import accords_par_mesure    # noqa: E402

CH = REPO / "harmonia_min" / "state" / "charts"
ETAT = REPO / "harmonia_min" / "state"
NDEG = 13          # 12 degres + « rien »
W = 4              # la fenetre, de chaque cote


def corpus():
    """[(stem, degres, qualites, frontieres, n)] pour chaque morceau annote."""
    out = []
    vus = set()
    for d in ("sections", "sections_draft"):
        for f in sorted((ETAT / d).glob("*.json")):
            if f.stem in vus:
                continue
            p = CH / f"min_{f.stem}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            ann = json.loads(f.read_text(encoding="utf-8"))
            if ann.get("n") != m.get("nBars"):
                continue                      # annotation perimee : on saute
            tonic = (m.get("key") or {}).get("tonic")
            if tonic is None:
                continue
            vus.add(f.stem)
            bars = accords_par_mesure(m)
            n = m["nBars"]
            deg, qual = [], []
            for b in range(n):
                c = next((c for c in bars[b] if not c.get("nc")), None)
                deg.append(12 if c is None else (c["root"] - tonic) % 12)
                q = "" if c is None else (c.get("q") or "")
                qual.append(0 if c is None else
                            (1 if q.startswith("-") else
                             2 if (q[:1].isdigit() or q.startswith("sus")) else 3))
            bounds = np.zeros(n, bool)
            for s in ann["sections"]:
                if 0 <= s["b0"] < n:
                    bounds[s["b0"]] = True
            out.append((f.stem, np.array(deg), np.array(qual), bounds, n))
    return out


def _bloc(deg, n, a):
    """Les quatre degres a partir de la mesure `a` (12 = hors morceau)."""
    return tuple(int(deg[a + k]) if 0 <= a + k < n else 12 for k in range(W))


def _ressemble(x, y):
    """Combien des quatre cases coincident, sur 1."""
    return sum(1 for u, v in zip(x, y) if u == v) / W


def traits(deg, qual, n, b, *, position=True, repet=True):
    """Le vecteur d'une mesure candidate.

    TROIS familles, toutes en logique de QUATRE accords :
      * l'HARMONIE — les quatre degres avant, les quatre apres, et leur qualite ;
      * la RUPTURE — de combien le bloc d'apres ressemble a celui d'avant ;
      * la REPETITION — le bloc d'apres reprend-il un bloc deja entendu 4, 8 ou
        16 mesures plus tot ? Et celui d'avant, se poursuit-il ?
    C'est la deuxieme et la troisieme qui manquaient au premier essai : une
    frontiere, c'est d'abord un endroit ou la repetition CHANGE de periode, et
    l'harmonie seule ne le voit pas.
    """
    v = []
    for k in range(-W, W):
        i = b + k
        d = deg[i] if 0 <= i < n else 12
        oh = [0.0] * NDEG
        oh[int(d)] = 1.0
        v += oh
        q = qual[i] if 0 <= i < n else 0
        oq = [0.0] * 4
        oq[int(q)] = 1.0
        v += oq
    if repet:
        apres, avant = _bloc(deg, n, b), _bloc(deg, n, b - W)
        v.append(_ressemble(apres, avant))                 # rupture immediate
        for lag in (4, 8, 12, 16, 24, 32):
            v.append(_ressemble(apres, _bloc(deg, n, b - lag)))   # deja entendu ?
            v.append(_ressemble(avant, _bloc(deg, n, b - W - lag)))
        # le bloc d'avant se POURSUIT-il au dela de b ? (si oui, b coupe dedans)
        v.append(_ressemble(_bloc(deg, n, b - 2), _bloc(deg, n, b - 2 - 4)))
        # combien d'accords differents dans chaque bloc
        v.append(len(set(apres)) / W)
        v.append(len(set(avant)) / W)
    if position:
        v += [1.0 if b % 4 == 0 else 0.0, 1.0 if b % 8 == 0 else 0.0,
              1.0 if b % 16 == 0 else 0.0, b / max(n - 1, 1)]
    return v


def jeu(data, *, position=True, repet=True):
    X, y, g = [], [], []
    for stem, deg, qual, bounds, n in data:
        for b in range(n):
            X.append(traits(deg, qual, n, b, position=position, repet=repet))
            y.append(bool(bounds[b]))
            g.append(stem)
    return np.array(X, float), np.array(y, bool), np.array(g)


def evalue(nom, faire_modele, X, y, g, seuils=True):
    """Leave-one-song-out : chaque morceau est predit par un modele qui ne l'a
    jamais vu."""
    from sklearn.metrics import average_precision_score
    p = np.zeros(len(y))
    for s in np.unique(g):
        tr, te = g != s, g == s
        if y[tr].sum() < 5:
            continue
        mdl = faire_modele()
        mdl.fit(X[tr], y[tr])
        p[te] = (mdl.predict_proba(X[te])[:, 1] if hasattr(mdl, "predict_proba")
                 else mdl.decision_function(X[te]))
    ap = average_precision_score(y, p)
    # F1 au meilleur seuil, pour dire quelque chose de lisible
    best = (0, 0, 0)
    for t in np.quantile(p, np.linspace(0.5, 0.999, 60)):
        pred = p >= t
        if pred.sum() == 0:
            continue
        prec = (pred & y).sum() / pred.sum()
        rec = (pred & y).sum() / y.sum()
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best[0]:
            best = (f1, prec, rec)
    return ap, best, p


def main() -> None:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = corpus()
    print(f"{len(data)} morceaux · "
          f"{sum(int(b.sum()) for _, _, _, b, _ in data)} frontieres · "
          f"{sum(n for *_, n in data)} mesures")
    print()

    essais = [(False, False, "harmonie seule"),
              (False, True, "harmonie + repetition"),
              (True, True, "harmonie + repetition + position")]
    for pos, rep, titre in essais:
        X, y, g = jeu(data, position=pos, repet=rep)
        print(f"=== {titre} — {X.shape[1]} traits")
        modeles = [
            ("hasard", lambda: DummyClassifier(strategy="stratified")),
            ("regression logistique", lambda: make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, C=0.5,
                                                     class_weight="balanced"))),
            ("foret aleatoire", lambda: RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                random_state=0, n_jobs=-1)),
            ("gradient boosting", lambda: GradientBoostingClassifier(
                n_estimators=200, max_depth=3, random_state=0)),
            ("MLP", lambda: make_pipeline(
                StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                                max_iter=800, random_state=0))),
        ]
        for nom, f in modeles:
            ap, (f1, prec, rec), _ = evalue(nom, f, X, y, g)
            print(f"   {nom:<24} AP {ap:.3f}   F1 {f1:.2f} "
                  f"(precision {prec:.0%}, rappel {rec:.0%})")
        print(f"   {'(base : tout positif)':<24} AP {y.mean():.3f}")
        print()





# ═══════════════════════════════════════════════════════════════════════════
# ITÉRATION 2 : décoder une SEGMENTATION, pas des mesures isolées
# ═══════════════════════════════════════════════════════════════════════════
# Classer chaque mesure séparément ignore ce qu'on sait le mieux du corpus :
# 81 % des longueurs de section sont des multiples de 4, et 8 est de loin la
# plus fréquente (94 sur 171, contre 40 pour 4). C'est l'idée de Sargent,
# Bimbot & Vincent (ISMIR 2011) : une contrainte de RÉGULARITÉ sur les
# longueurs, décodée en programmation dynamique, plutôt qu'un seuil par mesure.

def loi_longueurs(data, lisse=0.5):
    """P(longueur d'une section), apprise sur le corpus."""
    from collections import Counter
    c = Counter()
    for _stem, _d, _q, bounds, n in data:
        idx = list(np.where(bounds)[0]) + [n]
        for a, z in zip(idx, idx[1:]):
            if 1 <= z - a <= 40:
                c[int(z - a)] += 1
    tot = sum(c.values()) + lisse * 40
    return {L: (c.get(L, 0) + lisse) / tot for L in range(1, 41)}


def segmente(p, loi, n, penalite=0.0):
    """La suite de frontières la plus vraisemblable — programmation dynamique.

    On maximise  somme( log p(frontiere) + log P(longueur) )  sur toutes les
    découpes du morceau. Rend la liste des frontières.
    """
    NEG = -1e9
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    lq = np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))
    cum_q = np.concatenate([[0.0], np.cumsum(lq)])
    best = np.full(n + 1, NEG)
    prec = np.full(n + 1, -1, int)
    best[0] = 0.0
    for z in range(1, n + 1):
        for L in range(1, min(40, z) + 1):
            a = z - L
            if best[a] <= NEG / 2:
                continue
            # frontiere en a (sauf a=0, deja compte), pas de frontiere dans ]a,z[
            s = best[a] + np.log(loi.get(L, 1e-6)) - penalite
            s += (lp[a] if a > 0 else 0.0)
            s += cum_q[z] - cum_q[a + 1 if a > 0 else 0]
            if s > best[z]:
                best[z] = s
                prec[z] = a
    out, z = [], n
    while z > 0 and prec[z] >= 0:
        z = prec[z]
        if z > 0:
            out.append(z)
    return sorted(out)


def f1_frontieres(vrai, pred, n, tol=0):
    """F1 des frontières, avec une tolérance en mesures (usage MIREX)."""
    v = set(int(x) for x in np.where(vrai)[0] if x > 0)
    p = set(int(x) for x in pred)
    if not v or not p:
        return 0.0, 0.0, 0.0
    ok = sum(1 for x in p if any(abs(x - y) <= tol for y in v))
    okv = sum(1 for y in v if any(abs(x - y) <= tol for x in p))
    prec, rec = ok / len(p), okv / len(v)
    return (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)), prec, rec


def essai_segmentation(data):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y, g = jeu(data, position=True, repet=True)
    loi = loi_longueurs(data)
    print("   loi des longueurs apprise :",
          {L: f"{loi[L]:.0%}" for L in (2, 4, 6, 8, 12, 16) if loi[L] > 0.01})
    scores = {t: [] for t in (0, 1, 2)}
    base = {t: [] for t in (0, 1, 2)}
    for stem, deg, qual, bounds, n in data:
        tr = g != stem
        mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         random_state=0)
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[g == stem])[:, 1]
        loi_tr = loi_longueurs([d for d in data if d[0] != stem])
        pred = segmente(p, loi_tr, n)
        # base de comparaison : une frontiere toutes les 8 mesures
        pred8 = list(range(8, n, 8))
        for t in (0, 1, 2):
            scores[t].append(f1_frontieres(bounds, pred, n, tol=t)[0])
            base[t].append(f1_frontieres(bounds, pred8, n, tol=t)[0])
    for t in (0, 1, 2):
        print(f"   tolerance +-{t} mesure(s) : F1 modele {np.mean(scores[t]):.3f} "
              f"· F1 « une coupe toutes les 8 mesures » {np.mean(base[t]):.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# ITÉRATION 3 : la grille régulière est le squelette, le modèle choisit la PHASE
# ═══════════════════════════════════════════════════════════════════════════
# L'itération 2 a donné un résultat net et gênant : décoder librement une
# segmentation fait MOINS bien qu'une coupe toutes les 8 mesures (F1 0,27
# contre 0,47 à ±1). La régularité porte presque tout le signal, et les
# probabilités par mesure sont trop bruitées pour l'améliorer — elles éloignent
# le décodage de la grille au lieu de l'affiner.
#
# On garde donc la grille comme SQUELETTE et on ne demande au modèle que ce
# qu'une grille ne sait pas : où elle commence. C'est exactement la question de
# phase que Louis pose depuis le début, et c'est là que l'harmonie doit servir,
# puisque le comptage de répétitions, lui, est invariant par rotation.

def essai_phase(data):
    """Le modele choisit la phase d'une grille reguliere. Le reste est fixe."""
    from sklearn.ensemble import GradientBoostingClassifier
    X, y, g = jeu(data, position=False, repet=True)   # SANS position : elle
    res = {}                                          # donnerait la reponse
    for stem, deg, qual, bounds, n in data:
        tr = g != stem
        mdl = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         random_state=0)
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[g == stem])[:, 1]
        # Note d'une phase = la probabilite MOYENNE que le modele donne a ses
        # cases. Essaye aussi : le CONTRASTE (moyenne sur la grille moins
        # moyenne ailleurs), qui devait corriger le biais en faveur de la
        # periode 4 — mesure a 0,524 contre 0,534, donc rejete. La moyenne
        # simple reste.
        cand = {}
        for per in (4, 8):
            for ph in range(per):
                gr = [b for b in range(ph, n, per) if b > 0]
                if len(gr) < 3:
                    continue
                cand[(per, ph)] = float(np.mean([p[b] for b in gr]))
        (per, ph) = max(cand, key=cand.get)
        pred = [b for b in range(ph, n, per) if b > 0]
        # la meme grille, mais phase 0 : la base a battre
        base = [b for b in range(8, n, 8)]
        # et la MEILLEURE phase possible (oracle) : le plafond
        oracle = max(((pr, q) for pr in (4, 8) for q in range(pr)),
                     key=lambda k: f1_frontieres(
                         bounds, [b for b in range(k[1], n, k[0]) if b > 0],
                         n, tol=1)[0])
        res[stem] = (
            f1_frontieres(bounds, pred, n, tol=1)[0],
            f1_frontieres(bounds, base, n, tol=1)[0],
            f1_frontieres(bounds, [b for b in range(oracle[1], n, oracle[0])
                                   if b > 0], n, tol=1)[0],
            per, ph)
    mod = np.mean([v[0] for v in res.values()])
    bas = np.mean([v[1] for v in res.values()])
    ora = np.mean([v[2] for v in res.values()])
    print(f"   F1 a +-1 mesure — modele {mod:.3f} · grille fixe (8, phase 0) "
          f"{bas:.3f} · meilleure phase possible {ora:.3f}")
    juste = sum(1 for v in res.values() if v[0] >= v[2] - 1e-9)
    print(f"   le modele trouve la MEILLEURE phase sur {juste}/{len(res)} morceaux")
    return res


if __name__ == "__main__":
    main()
    d = corpus()
    print("=== ITERATION 2 : segmentation decodee (a priori de longueur)")
    essai_segmentation(d)
    print()
    print("=== ITERATION 3 : grille reguliere, le modele choisit la phase")
    essai_phase(d)
