"""La concurrence 4 contre 8 : à chaque ancre, les deux longueurs sont proposées.

Piste 3 du brief — « ne pas choisir un ordre du tout : proposer les blocs de 4 ET
de 8 en concurrence et arbitrer par score ». Le curseur avance comme aujourd'hui ;
à chaque ancre libre on construit LES DEUX runs (bloc de 8 et bloc de 4 avec
leurs reprises), et `decide(ctx) -> 4 | 8` tranche. La passe de comblement en 4
suit, inchangée.

`oracle_best` énumère tous les choix possibles et rend le meilleur score : c'est
le PLAFOND de cet espace de décisions. S'il est bas, aucun arbitre ne sauvera la
piste, et il faut chercher ailleurs.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

import order_search as OS               # noqa: E402
import harmonia_min.voice_sections as VS  # noqa: E402

THR = {8: VS.THR8, 4: VS.THR4}


def _candidate(b, hard, cursor, block, thr, claimed, hard_tol=1):
    n, S, M, mute = b["n"], b["S"], b["M"], b["mute"]

    def crosses(p, L):
        return any(p + hard_tol < h < p + L - hard_tol for h in hard)

    if cursor + block > n or crosses(cursor, block):
        return None
    cm, ch = VS._slide(M, cursor, block, n), VS._slide(S, cursor, block, n)
    sc = VS.block_score(cm, ch, cursor, n, mute=mute, block=block)
    occ = [p for p in VS._peaks(sc, cursor, n, thr, block, claimed)
           if not crosses(p, block)]
    if not occ and block == 8:
        occ = [p for p in VS._peaks(sc, cursor, n, thr + VS.ODD_BONUS, block,
                                    claimed, par=1) if not crosses(p, block)]
    if not occ:
        return None
    return {"b0": cursor, "occ": occ, "block": block, "sc": sc,
            "mean": float(np.mean([sc[p] for p in occ]))}


def mixed_runs(b, hard, decide, fill=True, max_blocks=40, hard_tol=1,
               stride=4, thr=None):
    """decide(c8, c4, ctx) -> le run choisi (ou None pour ne rien poser).

    Le parcours reprend celui de `voice_sections._pass` : on saute ce qui est
    déjà réclamé de `UNIT` en `UNIT`, on saute au pic suivant quand un pic barre
    les DEUX longueurs, et on avance de la longueur posée. Seule différence : à
    chaque ancre les deux longueurs sont construites et `decide` tranche.
    """
    n = b["n"]
    thr = dict(THR) if thr is None else thr
    hard = sorted(h for h in hard if 0 < h < n)
    claimed = np.zeros(n, bool)

    def crosses(p, L):
        return any(p + hard_tol < h < p + L - hard_tol for h in hard)

    runs, cursor, k = [], b["start"], 0
    while cursor + 4 <= n and k < max_blocks:
        if claimed[cursor:cursor + 4].any():
            cursor += VS.UNIT
            continue
        if crosses(cursor, 4):                    # même le petit bloc est barré
            nxt = next((h for h in hard if h > cursor), None)
            if nxt is None:
                break
            cursor = nxt
            continue
        c8 = (None if (cursor + 8 > n or claimed[cursor:cursor + 8].any())
              else _candidate(b, hard, cursor, 8, thr[8], claimed, hard_tol))
        c4 = _candidate(b, hard, cursor, 4, thr[4], claimed, hard_tol)
        pick = None
        if c8 is not None or c4 is not None:
            pick = decide(c8, c4, {"b": b, "cursor": cursor, "hard": hard,
                                   "claimed": claimed, "runs": runs})
        if pick is not None:
            runs.append({k2: pick[k2] for k2 in ("b0", "occ", "block")})
            k += 1
            for c in [pick["b0"]] + pick["occ"]:
                claimed[c:min(n, c + pick["block"])] = True
            cursor += pick["block"]
        else:
            cursor += stride
    if fill:
        r2, claimed = OS._passes(b, hard, [(4, thr[4])], claimed=claimed)
        runs += r2
    return runs, claimed


def prefer_long(c8, c4, ctx):
    return c8 if c8 is not None else c4


def prefer_short(c8, c4, ctx):
    return c4 if c4 is not None else c8


def sections(b, decide=prefer_long, hard=None, tail_unit=OS.TAIL_UNIT, **kw):
    hard = b["hard"] if hard is None else hard
    runs, _ = mixed_runs(b, hard, decide, **kw)
    return OS.assemble(b, runs, hard, tail_unit=tail_unit)


# ── le plafond ──────────────────────────────────────────────────────────────

def oracle_best(b, stem, max_dec=12):
    """Le meilleur score atteignable en choisissant 4 ou 8 à chaque ancre.

    On énumère les séquences de décisions : à la k-ième ancre où les deux
    longueurs existent, on prend 8 ou 4. C'est le plafond de la piste 3.
    """
    n = b["n"]
    best = (-1.0, None)
    # combien d'ancres ont réellement le choix ? (on le découvre en jouant
    # « toujours 8 » puis en variant)
    seen = []

    def probe(bits):
        idx = [0]

        def dec(c8, c4, ctx):
            if c8 is not None and c4 is not None:
                i = idx[0]; idx[0] += 1
                if i < len(bits) and bits[i] == 1:
                    return c4
                return c8
            return c8 if c8 is not None else c4
        secs = sections(b, dec)
        return secs, idx[0]

    _, ndec = probe(())
    ndec = min(ndec, max_dec)
    for bits in itertools.product((0, 1), repeat=ndec):
        secs, _ = probe(bits)
        s = OS.score(secs, stem, n)
        if s > best[0]:
            best = (s, bits)
    return best[0], best[1], ndec


# ── le point de décision qui compte vraiment ────────────────────────────────

def decide_only4(place_bits):
    """Quand SEUL le bloc de 4 existe à cette ancre : le poser, ou le laisser à
    la passe de comblement ? C'est LA décision qui sépare Blue Lights de
    The Walk (mesuré : ce n'est pas « 4 ou 8 », les deux ne coexistent presque
    jamais à la même ancre)."""
    idx = [0]

    def dec(c8, c4, ctx):
        if c8 is not None:
            return c8
        i = idx[0]; idx[0] += 1
        # PAR DÉFAUT ON DIFFÈRE. Le contraire (ma 1re version) faisait que « tout
        # différé » posait quand même tout dès que la liste de bits était plus
        # courte que le nombre de décisions — et différer EN CRÉE de nouvelles.
        # Les deux colonnes de la table sortaient identiques : c'était le bug.
        return c4 if i < len(place_bits) and place_bits[i] == 1 else None
    dec.count = idx
    return dec


def oracle_only4(b, stem, max_dec=14):
    """Plafond de la décision « poser ce 4 tout de suite ou le différer »."""
    n = b["n"]

    def probe(bits):
        dec = decide_only4(bits)
        secs = sections(b, dec, stride=4)
        return secs, dec.count[0]

    # Le nombre de points de décision dépend des décisions elles-mêmes (différer
    # en crée d'autres) : on prend le maximum observé entre les deux extrêmes.
    _, k0 = probe(())                       # tout différé
    _, k1 = probe((1,) * 40)                # tout posé
    k = min(max(k0, k1), max_dec)
    best = (-1.0, None)
    for bits in itertools.product((0, 1), repeat=k):
        secs, _ = probe(bits)
        s = OS.score(secs, stem, n)
        if s > best[0]:
            best = (s, bits)
    return best[0], best[1], k
