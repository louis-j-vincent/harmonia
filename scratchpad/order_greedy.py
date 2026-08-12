"""« Le plus fort se sert le premier » — mais entre ANCRES, pas entre reprises.

`voice_sections._peaks` porte déjà la doctrine, et son commentaire dit pourquoi :
« LE PLUS FORT SE SERT LE PREMIER (sinon le premier tue le meilleur) ». Elle n'est
appliquée qu'AUX REPRISES d'une même ancre. Entre ancres, le parcours reste de
gauche à droite — et c'est exactement le mécanisme de l'échec de Blue Lights : le
bloc de 8 de la mesure 45 arrive avant le bloc de 4 de la mesure 13 parce qu'il
est à gauche, pas parce qu'il est meilleur.

Ce fichier applique la même règle un cran plus haut : à chaque tour, on construit
TOUS les candidats (toutes les ancres libres × toutes les longueurs) et on pose
LE MEILLEUR, puis on recommence. Aucun ordre n'est choisi — ni 8 d'abord ni 4
d'abord.
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

import order_search as OS                 # noqa: E402
import order_multi as MU                  # noqa: E402
import harmonia_min.voice_sections as VS  # noqa: E402


def greedy_runs(b, hard, lengths=(8, 4), thr=None, key="long", literal=None,
                stat="moy", max_blocks=40, hard_tol=1, fill=True):
    """`key` :
        "long"  la plus longue d'abord, puis le meilleur score  (= la prod, mais
                l'ordre des ancres est le score et non la position) ;
        "score" le meilleur score d'abord, la plus longue départage ;
        "cover" le plus de mesures réclamées d'abord.
    """
    n, start = b["n"], b["start"]
    thr = ({L: (VS.THR4 if L <= 4 else VS.THR8) for L in lengths}
           if thr is None else thr)
    hard = sorted(h for h in hard if 0 < h < n)
    claimed = np.zeros(n, bool)
    runs = []
    for _ in range(max_blocks):
        best = None
        for L in lengths:
            for p in range(start, n - L + 1, VS.UNIT):
                if claimed[p:p + L].any():
                    continue
                c = MU.candidate(b, hard, p, L, thr[L], claimed, hard_tol)
                if c is None:
                    continue
                if literal is not None and L == min(lengths) and c[stat] < literal:
                    continue
                if key == "long":
                    k = (L, c["moy"])
                elif key == "score":
                    k = (c["moy"], L)
                else:
                    k = ((len(c["occ"]) + 1) * L, c["moy"])
                if best is None or k > best[0]:
                    best = (k, c)
        if best is None:
            break
        c = best[1]
        runs.append({x: c[x] for x in ("b0", "occ", "block")})
        for q in [c["b0"]] + c["occ"]:
            claimed[q:min(n, q + c["block"])] = True
    if fill:
        r2, claimed = OS._passes(b, hard, [(min(lengths), thr[min(lengths)])],
                                 claimed=claimed)
        runs += r2
    return runs, claimed


def sections(b, hard=None, tail_unit=4, min_cut=2, **kw):
    hard = b["hard"] if hard is None else hard
    runs, _ = greedy_runs(b, hard, **kw)
    return OS.assemble(b, runs, hard, tail_unit=tail_unit, min_cut=min_cut)
