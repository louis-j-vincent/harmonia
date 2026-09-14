"""D'abord le vocabulaire, ensuite le pavage. Les pics ne font qu'arbitrer.

    .venv/bin/python scripts/vocab_tiling.py

Louis, 2026-08-12, après le diagnostic de She Will Be Loved (deux C de 4 mesures
collés, qu'un bloc de 8 avale, et que l'harmonie ne peut pas séparer) :

  « on n'a pas besoin de les séparer. Quand on identifie les A, B et C, on remplit
    au bon endroit, et puis voilà. Les pics servent juste à caler quelques
    frontières dont on est sûrs afin de trancher sur plusieurs hypothèses de
    remplissage de la grille par section, après ça on continue la stratégie
    d'avant. »

CE QUE ÇA CHANGE, ET C'EST UN RENVERSEMENT. Jusqu'ici on avançait de gauche à
droite en réclamant des blocs : chaque décision était LOCALE et définitive, donc
un bloc de 8 posé sur deux C de 4 gagnait sur place, sans que la suite du morceau
ait son mot à dire. Ici :

  1. **le vocabulaire d'abord** — on cherche les motifs qui se répètent le plus
     dans TOUT le morceau, avec leur longueur. Le C de She Will Be Loved sort
     avec sept occurrences ; le faux bloc de 8 qui en avale deux n'en a que deux.
  2. **le pavage ensuite** — une programmation dynamique remplit la grille de
     mesures avec ces motifs, et choisit le pavage de coût minimal sur le
     morceau ENTIER. Deux C collés ne sont pas « une section à séparer » : ce
     sont deux poses du même motif, et la question ne se pose plus.
  3. **les pics arbitrent** — ils n'imposent rien tout seuls : ils interdisent
     qu'une pose les enjambe, ce qui élimine des pavages, et laisse le reste au
     coût global.

LE COÛT D'UNE POSE, et ses trois termes :

    − longueur × (ressemblance au motif − seuil)     ce que la pose explique
    + λ  par section                                  un pavage n'est pas un hachoir
    − β × log(occurrences du motif)                   un motif fréquent est préféré

Le dernier terme est celui qui règle le cas de Louis : à ressemblance égale, le
motif vu sept fois bat celui vu deux fois. C'est l'évidence GLOBALE qui tranche
là où l'évidence locale était muette.
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

import order_bundle                                   # noqa: E402
import order_search as OS                             # noqa: E402
from ssm_zoo import SONGS, gt_sections                # noqa: E402

LENGTHS = (4, 8, 12, 16)   # les longueurs de motif candidates
OCC_MIN = 0.80             # une occurrence compte si elle ressemble au motif au moins autant
THETA = 0.62               # ressemblance en deçà de laquelle une pose n'explique rien
LAMBDA = 0.55              # le prix d'une section (contre le hachoir)
BETA = 0.9                 # la prime à un motif fréquent
FREE_LEN = (4, 8)          # longueurs autorisées pour une section « inconnue »
FREE_COST = 1.15           # …et son prix, plus cher qu'une pose de motif
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _diag(S, a, b, L):
    n = S.shape[0]
    v = [S[a + i, b + i] for i in range(L) if a + i < n and b + i < n]
    return float(np.mean(v)) if v else 0.0


def _pool(S, n, L):
    """Toutes les ressemblances d'empans de longueur L du morceau, comme
    référence. On s'en sert pour convertir une ressemblance en RANG.

    Le rang, et pas un écart normalisé : en moyennant sur L mesures, l'écart-type
    rétrécit comme 1/racine(L), donc un même motif paraît « plus fort » à 16
    mesures qu'à 4 — c'est un artefact d'échantillonnage, pas de la musique. Le
    rang, lui, est comparable d'une longueur à l'autre par construction.
    """
    vals = []
    for a in range(0, n - L + 1, 2):
        for c in range(a + L, n - L + 1, 2):
            vals.append(_diag(S, a, c, L))
    return np.sort(np.asarray(vals)) if vals else np.zeros(1)


def _rank(pool, x):
    return float(np.searchsorted(pool, x) / max(1, len(pool)))


def _levels(S, n, L, q=(0.5, 0.9)):
    """Le niveau de ressemblance PROPRE au morceau, pour cette longueur.

    Indispensable, et c'est la leçon du seuil TILE (known_issues 2026-08-05) : sur
    une chanson bâtie sur une boucle de 4, n'importe quel empan de 16 mesures
    ressemble à n'importe quel autre. Une ressemblance brute de 0,90 ne veut donc
    rien dire tant qu'on ne sait pas ce que le morceau produit spontanément à
    cette longueur-là. La première version de ce fichier ne le faisait pas et son
    vocabulaire n'était fait que de motifs de 12 et 16 mesures (0,475 de médiane
    contre 0,799) — ils couvraient tout et ne disaient rien.
    """
    vals = []
    for a in range(0, n - L + 1, 2):
        for c in range(a + L, n - L + 1, 2):
            vals.append(_diag(S, a, c, L))
    if not vals:
        return 0.0, 1.0
    lo, hi = np.quantile(vals, q[0]), np.quantile(vals, q[1])
    return float(lo), float(max(hi - lo, 1e-6))


def vocabulary(b, lengths=LENGTHS, occ_min=OCC_MIN):
    """[{L, ref, occ, sim}] — les motifs du morceau, du plus couvrant au moins.

    Un motif est un empan (`ref`, `L`) ; ses occurrences sont les positions où le
    morceau rejoue la même chose. On ne garde que des ancres sur la grille de 2
    calée sur la première mesure chantée — c'est là que les sections commencent
    (mesuré : 90 % des frontières de Louis y sont).
    """
    S, n, start = b["S"], b["n"], b["start"]
    out = []
    for L in lengths:
        if n < 2 * L:
            continue
        lo, span = _levels(S, n, L)
        pool = _pool(S, n, L)
        for ref in range(start % 2, n - L + 1, 2):
            self_sim = _diag(S, ref, ref, L)
            if self_sim <= 0:
                continue
            occ, sims = [], []
            p = start % 2
            while p <= n - L:
                s = _diag(S, ref, p, L) / self_sim
                if s >= occ_min:
                    if occ and p - occ[-1] < L:      # pas deux poses qui se chevauchent
                        if s > sims[-1]:
                            occ[-1], sims[-1] = p, s
                    else:
                        occ.append(p); sims.append(s)
                p += 2
            if len(occ) >= 2:
                # la force du motif, en écarts au niveau spontané du morceau
                raw = float(np.mean([_diag(S, ref, p2, L) for p2 in occ]))
                rel = _rank(pool, raw)
                out.append({"L": L, "ref": ref, "occ": occ, "rel": rel,
                            "sim": float(np.mean(sims)),
                            "cover": len(occ) * L * max(0.0, rel)})
    # LE CLASSEMENT EST PAR FORCE, PAS PAR MESURES COUVERTES. Louis, 2026-08-12 :
    # « pourquoi la granularité est-elle de 16 mesures ? » — parce que je classais
    # par `force × longueur × occurrences`, qui est proportionnel à la longueur.
    # Vérifié sur She Will Be Loved : la force relative est la MÊME à toutes les
    # longueurs (1,22 à 4 mesures · 1,23 à 8 · 1,34 à 12 · 1,42 à 16), donc rien
    # dans la musique ne privilégie 16 ; c'était mon tri. À force égale on prend
    # le motif le PLUS COURT : il explique la même chose avec un vocabulaire plus
    # petit et il se pose plus librement (principe de simplicité du projet).
    # arrondi à 0,02 : deux motifs dont les rangs se tiennent sont réputés
    # ÉGAUX, et c'est alors le plus court qui gagne.
    out.sort(key=lambda m: (-round(m["rel"] / 0.02), m["L"], -len(m["occ"])))
    keep = []
    for m in out:
        span = {q for p in m["occ"] for q in range(p, p + m["L"])}
        redundant = False
        for m2 in keep:
            span2 = {q for p in m2["occ"] for q in range(p, p + m2["L"])}
            inter = len(span & span2)
            # un motif LONG dont les mesures sont déjà expliquées par un motif
            # plus court n'apporte rien : c'est le court, répété.
            if inter >= 0.6 * len(span) and m2["L"] <= m["L"]:
                redundant = True
                break
        if not redundant:
            keep.append(m)
        if len(keep) >= 10:
            break
    return keep


def tile(b, vocab, hard=None, theta=THETA, lam=LAMBDA, beta=BETA,
         free_cost=FREE_COST, hard_tol=1):
    """Le pavage de coût minimal, par programmation dynamique sur les mesures.

    Les pics `hard` ne notent rien : ils INTERDISENT qu'une pose les enjambe
    (à `hard_tol` mesures près). C'est le rôle que Louis leur donne — trancher
    entre des hypothèses de remplissage, pas en fabriquer.
    """
    S, n = b["S"], b["n"]
    hard = b["hard"] if hard is None else hard
    lev = {L: _levels(S, n, L) for L in sorted({m["L"] for m in vocab})}

    def crosses(p, L):
        return any(p + hard_tol < h < p + L - hard_tol for h in hard)

    INF = float("inf")
    dp = [INF] * (n + 1)
    back = [None] * (n + 1)
    dp[0] = 0.0
    for p in range(n):
        if dp[p] == INF:
            continue
        for mi, m in enumerate(vocab):
            L = m["L"]
            if p + L > n or crosses(p, L):
                continue
            # UNE POSE SEULEMENT LÀ OÙ LE MOTIF A ÉTÉ VU. Sans ça, un motif de 4
            # mesures « colle » partout sur une chanson en boucle et le pavage
            # dégénère en un mur de A — c'est ce que la page a montré. On remplit
            # aux endroits identifiés, ce qui est exactement la consigne de Louis :
            # « quand on identifie les A, B et C, on remplit au bon endroit ».
            if not any(abs(p - q) <= 1 for q in m["occ"]):
                continue
            lo, span = lev[L]
            rel = (_diag(S, m["ref"], p, L) - lo) / span
            c = dp[p] - L * max(0.0, rel) * 0.25 + lam - beta * np.log(len(m["occ"]))
            if c < dp[p + L]:
                dp[p + L], back[p + L] = c, (p, mi)
        for L in FREE_LEN:                            # une section « inconnue »
            if p + L > n or crosses(p, L):
                continue
            c = dp[p] + free_cost + lam
            if c < dp[p + L]:
                dp[p + L], back[p + L] = c, (p, -1)
        if p + 1 <= n and dp[p] + free_cost + lam < dp[p + 1]:
            dp[p + 1], back[p + 1] = dp[p] + free_cost + lam, (p, -1)

    if dp[n] == INF:
        return None
    out, q = [], n
    while q > 0:
        p, mi = back[q]
        out.append((p, q, mi))
        q = p
    return list(reversed(out))


def sections(stem_or_bundle, **kw):
    b = (order_bundle.get(stem_or_bundle) if isinstance(stem_or_bundle, str)
         else stem_or_bundle)
    vocab = vocabulary(b)
    lay = tile(b, vocab, **kw)
    if not lay:
        return None
    ren, k, out = {}, 0, []
    for p, q, mi in lay:
        if mi < 0:
            lab = "?"
        else:
            if mi not in ren:
                ren[mi] = LETTERS[k % len(LETTERS)]; k += 1
            lab = ren[mi]
        out.append({"b0": p, "b1": q - 1, "label": lab})
    # l'intro : tout ce qui précède le premier chant garde son nom
    for s in out:
        if s["b1"] < b["start"]:
            s["label"] = "intro"
    return out


def main():
    import order_lab as OL
    from section_metric import compare
    print(f"{'morceau':22s} {'agent':>7s} {'vocab+pavage':>13s}   vocabulaire")
    a, v = [], []
    for stem, title in SONGS:
        b = order_bundle.get(stem)
        gt = gt_sections(stem)["sections"]
        n = b["n"]
        x = compare(OL.new_sections(b)[0], gt, n)["score"]
        secs = sections(b)
        y = compare(secs, gt, n)["score"] if secs else float("nan")
        a.append(x); v.append(y)
        voc = vocabulary(b)[:4]
        vt = " ".join(f"{m['L']}×{len(m['occ'])}" for m in voc)
        mark = "+" if y > x + 0.005 else ("-" if y < x - 0.005 else "=")
        print(f"{title[:22]:22s} {x:7.3f} {y:13.3f} {mark}  {vt}")
    print(f"{'MÉDIANE':22s} {np.median(a):7.3f} {np.median(v):13.3f}")
    print(f"{'MOYENNE':22s} {np.mean(a):7.3f} {np.mean(v):13.3f}")


if __name__ == "__main__":
    main()
