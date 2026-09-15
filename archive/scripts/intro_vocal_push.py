"""« Ces mesures chantées sont-elles ENCORE l'intro ? » — l'arbitrage vocal.

    python scripts/intro_vocal_push.py

LE SEUL DÉFAUT QUI RESTE. La règle livrée (`voice_sections.sung_start`) fait
9/10 contre la vérité de Louis. Son unique erreur est The Walk : le chanteur y
chante QUATRE mesures d'intro, donc « l'intro finit quand on chante » place la
frontière quatre mesures trop tôt. Aucune règle à une mesure près ne peut
l'atteindre — c'est écrit dans la docstring de `sung_start`.

L'HYPOTHÈSE DE LOUIS, PRISE AU MOT. « Des fois, entre l'intro et le début de la
chanson, c'est la même harmonie, mais c'est le vocal qui change. » Donc : le
bloc chanté qui suit immédiatement le premier mot NE SE REPRODUIT PAS si c'est
encore l'intro, alors que le bloc d'après, lui, est un vrai couplet et se
reproduit. On compare les deux, dans la matrice de similarité de la voix.

CE QUI A DÉJÀ ÉTÉ ESSAYÉ ET ÉCARTÉ, et pourquoi ce n'est pas la même chose :
`voice_first` faisait GLISSER le bloc et regardait ses pics — « aucun contraste
sur The Walk, tous les blocs à 0.33–0.40 ». Un pic de glissement demande une
reprise ALIGNÉE. Ici on ne demande pas d'alignement : on prend la ressemblance
MOYENNE du bloc à tout le reste du morceau. Un couplet ressemble un peu à
beaucoup d'endroits ; une intro ne ressemble à rien, alignée ou pas.

Ce fichier mesure ça, et surtout mesure ce que ça vaut en VALIDATION CROISÉE :
avec dix morceaux et un seul qui a besoin du correctif, un seuil ajusté sur les
dix est de l'ajustement, pas un résultat.
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

BLK = 4        # le bloc mis en cause, en mesures
GAP = 8        # « le reste » commence après ça


MIN_SUNG = 0.5     # un bloc à moitié muet ne peut pas témoigner


def belong(M, n, a, k0, blk=BLK, gap=GAP):
    """Ressemblance moyenne du bloc [a, a+blk) au RESTE du morceau."""
    rest = [c for c in range(n) if c >= k0 + gap]
    v = [M[b, c] for b in range(a, min(n, a + blk)) for c in rest]
    return float(np.mean(v)) if v else np.nan


def ratio(d, tag, blk=BLK, gap=GAP, min_sung=MIN_SUNG):
    """Le bloc contesté contre le suivant. < 1 = « c'est encore l'intro ».

    LE TROU DANS LA MATRICE, et il m'a produit un faux positif franc. Une mesure
    sans note chantée a sa ligne mise à zéro dans la matrice de chant — c'est
    voulu, « muet » n'est pas « différent ». Mais un bloc ENTIÈREMENT muet a donc
    une ressemblance de zéro au reste, donc un rapport de zéro, donc il déclenche
    la poussée à coup sûr. Sur les 28 morceaux non annotés, le SEUL déclenchement
    était exactement ça (`Uw5OLnN7UvM`, rapport 0.000) : le chanteur y pose une
    levée à la fin de la mesure 0 puis se tait quatre mesures. Un silence n'est
    pas une intro qui ne se reproduit pas ; c'est une absence de témoignage.

    On exige donc que les deux blocs soient chantés à `min_sung` au moins, sinon
    on ne rend rien et la règle livrée garde la main.
    """
    M, n, k0 = d["F"][f"_M_{tag}"], d["n"], d["shipped"]
    sung = np.asarray(d["F"]["_mute"], float) < .5
    for a in (k0, k0 + blk):
        s = sung[a:min(n, a + blk)]
        if not s.size or float(np.mean(s)) < min_sung:
            return np.nan
    a = belong(M, n, k0, k0, blk, gap)
    b = belong(M, n, k0 + blk, k0, blk, gap)
    return a / b if b and np.isfinite(b) and abs(b) > 1e-6 else np.nan


def apply(d, r, thr, blk=BLK):
    """La règle livrée, poussée de `blk` mesures si le bloc ne se reproduit pas."""
    return d["shipped"] + blk if np.isfinite(r) and r < thr else d["shipped"]


def main():
    D = IV.collect()
    sts = sorted(D)
    print(f"{'morceau':<18} {'k0':>3} {'vrai':>5}  "
          f"{'rapport chant':>14} {'rapport timbre':>15}")
    R = {}
    for st in sts:
        d = D[st]
        R[st] = {t: ratio(d, t) for t in ("pitch", "timbre")}
        print(f"{st[:18]:<18} {d['shipped']:>3} {d['truth']:>5}  "
              f"{R[st]['pitch']:>14.3f} {R[st]['timbre']:>15.3f}")

    for tag in ("pitch", "timbre"):
        print(f"\n── rapport {tag} : balayage du seuil ──")
        best = None
        for thr in np.arange(0.30, 1.45, 0.05):
            ok = sum(apply(D[s], R[s][tag], thr) == D[s]["truth"] for s in sts)
            if best is None or ok > best[1]:
                best = (thr, ok)
        print(f"  meilleur seuil {best[0]:.2f} → {best[1]}/10 "
              f"(livré : 9/10)")
        # LA MARGE, pas seulement le compte : de combien le seuil peut-il bouger ?
        need = [R[s][tag] for s in sts if D[s]["truth"] > D[s]["shipped"]]
        keep = [R[s][tag] for s in sts if D[s]["truth"] == D[s]["shipped"]]
        print(f"  à pousser  : {sorted(np.round(need, 3))}")
        print(f"  à ne PAS pousser (le plus bas d'abord) : "
              f"{sorted(np.round(keep, 3))[:4]}")
        if need and keep:
            print(f"  MARGE = {min(keep) - max(need):+.3f} "
                  f"(l'écart entre le cas à corriger et le premier à casser)")
        # validation croisée : le seuil est réglé SANS le morceau qu'on teste
        hits = 0
        for st in sts:
            rest = [s for s in sts if s != st]
            b = None
            for thr in np.arange(0.30, 1.45, 0.05):
                ok = sum(apply(D[s], R[s][tag], thr) == D[s]["truth"] for s in rest)
                if b is None or ok > b[1]:
                    b = (thr, ok)
            hits += apply(D[st], R[st][tag], b[0]) == D[st]["truth"]
        print(f"  VALIDATION CROISÉE (laisser-un-dehors) : {hits}/10")
    return D, R


if __name__ == "__main__":
    main()
