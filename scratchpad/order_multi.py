"""La concurrence généralisée : N longueurs proposées à chaque ancre, pas deux.

Suite directe de `order_mixed`. Louis annote beaucoup de sections de 12 mesures
(Blue Lights A[17-28], The Walk B[25-36]) que la recherche ne peut aujourd'hui
écrire que comme 8+4 sous deux lettres différentes. On teste donc si la
concurrence gagne à proposer aussi 12 et 16.

Le seuil est le même pour toutes les longueurs ≥ 8 : mesuré (journal, E2), le
score de bloc a la même échelle à 4 et à 8 mesures parce qu'il est relatif à
l'ancre.
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
import harmonia_min.voice_sections as VS  # noqa: E402


def candidate(b, hard, cursor, block, thr, claimed, hard_tol=1):
    n, S, M, mute = b["n"], b["S"], b["M"], b["mute"]

    def crosses(p, L):
        return any(p + hard_tol < h < p + L - hard_tol for h in hard)

    if cursor + block > n or crosses(cursor, block) \
            or claimed[cursor:cursor + block].any():
        return None
    cm, ch = VS._slide(M, cursor, block, n), VS._slide(S, cursor, block, n)
    sc = VS.block_score(cm, ch, cursor, n, mute=mute, block=block)
    occ = [p for p in VS._peaks(sc, cursor, n, thr, block, claimed)
           if not crosses(p, block)]
    if not occ and block >= 8:
        occ = [p for p in VS._peaks(sc, cursor, n, thr + VS.ODD_BONUS, block,
                                    claimed, par=1) if not crosses(p, block)]
    if not occ:
        return None
    v = [sc[p] for p in occ]
    return {"b0": cursor, "occ": occ, "block": block, "sc": sc,
            "moy": float(np.mean(v)), "min": float(np.min(v))}


def multi_runs(b, hard, lengths=(8, 4), thr=None, literal=0.88, stat="moy",
               fill=True, max_blocks=40, hard_tol=1, stride=4):
    """Les longueurs sont essayées de la plus longue à la plus courte.

    La plus longue disponible gagne, SAUF si une plus courte se rejoue à
    l'identique (≥ `literal`) alors qu'aucune plus longue n'existe — c'est la
    règle mesurée dans `order_rule`. `literal=None` désactive la règle (toute
    longueur disponible est prise, la plus longue d'abord).
    """
    n = b["n"]
    thr = ({L: (VS.THR4 if L <= 4 else VS.THR8) for L in lengths}
           if thr is None else thr)
    lens = sorted(lengths, reverse=True)
    short = min(lens)
    hard = sorted(h for h in hard if 0 < h < n)
    claimed = np.zeros(n, bool)

    def crosses(p, L):
        return any(p + hard_tol < h < p + L - hard_tol for h in hard)

    runs, cursor, k = [], b["start"], 0
    while cursor + short <= n and k < max_blocks:
        if claimed[cursor:cursor + short].any():
            cursor += VS.UNIT
            continue
        if crosses(cursor, short):
            nxt = next((h for h in hard if h > cursor), None)
            if nxt is None:
                break
            cursor = nxt
            continue
        pick = None
        for L in lens:
            c = candidate(b, hard, cursor, L, thr[L], claimed, hard_tol)
            if c is None:
                continue
            if L == short and literal is not None and c[stat] < literal:
                continue          # la boucle du morceau : elle attend son tour
            pick = c
            break
        if pick is not None:
            runs.append({x: pick[x] for x in ("b0", "occ", "block")})
            k += 1
            for c in [pick["b0"]] + pick["occ"]:
                claimed[c:min(n, c + pick["block"])] = True
            cursor += pick["block"]
        else:
            cursor += stride
    if fill:
        r2, claimed = OS._passes(b, hard, [(short, thr[short])], claimed=claimed)
        runs += r2
    return runs, claimed


def sections(b, hard=None, tail_unit=OS.TAIL_UNIT, **kw):
    hard = b["hard"] if hard is None else hard
    runs, _ = multi_runs(b, hard, **kw)
    return OS.assemble(b, runs, hard, tail_unit=tail_unit)
