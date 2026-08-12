"""La recherche de sections, isolée de tout I/O, pour pouvoir la balayer.

Copie fidèle de `hard_prior_sections.voice_with_hard` (règle du dépôt : on ne
touche pas à `harmonia_min/`), avec les paramètres sortis en arguments :

    ordre des passes, seuil de chaque passe, tolérance aux pics, unité de queue.

`search(b, **kw)` prend un bundle de `order_bundle.get(stem)` et rend la liste
de sections. `table(fn)` la mesure sur les douze morceaux annotés.
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

import harmonia_min.voice_sections as VS   # noqa: E402  (lecture seule)

LETTERS = "ABCDEFGHIJKLMNOP"
TAIL_UNIT = 4
HARD_TOL = 1


# ── la recherche ────────────────────────────────────────────────────────────

def _passes(b, hard, passes, hard_tol=HARD_TOL, odd_bonus=None,
            max_blocks=20, claimed=None, gate=None):
    """`passes` = [(block, thr), ...] dans l'ordre d'exécution.

    `gate(run, ctx) -> bool` : appelé avant de verrouiller un run. Faux = le run
    est abandonné et ses mesures restent libres pour la passe suivante. `ctx`
    porte le bundle, les pics, le score du bloc et le tableau `claimed`.
    """
    S, M, mute, n, start = b["S"], b["M"], b["mute"], b["n"], b["start"]
    odd = VS.ODD_BONUS if odd_bonus is None else odd_bonus
    hard = sorted(h for h in hard if 0 < h < n)

    def crosses(p, L, tol=hard_tol):
        return any(p + tol < h < p + L - tol for h in hard)

    if claimed is None:
        claimed = np.zeros(n, bool)
    runs = []
    for block, thr in passes:
        cursor, k = start, 0
        while cursor + block <= n and k < max_blocks:
            if claimed[cursor:cursor + block].any():
                cursor += VS.UNIT
                continue
            if crosses(cursor, block):
                nxt = next((h for h in hard if h > cursor), None)
                if nxt is None:
                    break
                cursor = nxt
                continue
            cm = VS._slide(M, cursor, block, n)
            ch = VS._slide(S, cursor, block, n)
            sc = VS.block_score(cm, ch, cursor, n, mute=mute, block=block)
            occ = [p for p in VS._peaks(sc, cursor, n, thr, block, claimed)
                   if not crosses(p, block)]
            if not occ and block == VS.BLOCK:
                occ = [p for p in VS._peaks(sc, cursor, n, thr + odd, block,
                                            claimed, par=1)
                       if not crosses(p, block)]
            if occ:
                run = {"b0": cursor, "occ": occ, "block": block}
                ok = True if gate is None else gate(
                    run, {"b": b, "hard": hard, "sc": sc, "claimed": claimed,
                          "thr": thr, "n": n})
                if ok:
                    runs.append(run)
                    k += 1
                    for c in [cursor] + occ:
                        claimed[c:min(n, c + block)] = True
            cursor += block
    return runs, claimed


def assemble(b, runs, hard, tail_unit=TAIL_UNIT, min_cut=0):
    """Le lettrage + l'écriture, identiques à `voice_with_hard`.

    `min_cut` > 0 : une coupure sur pic qui laisserait moins de `min_cut` mesures
    d'un côté À L'INTÉRIEUR D'UN MÊME BLOC est annulée. Motif mesuré sur Chain of
    Fools : le pic tombe mesure 9, la frontière de Louis est mesure 10, et la
    coupure exacte écrit `A[2-8] | A[9-9]` — une section d'UNE mesure. La
    tolérance de ±1 mesure existait déjà pour la RECHERCHE (`crosses`) et pas
    pour l'ÉCRITURE. Les queues (`owner < 0`) ne sont pas concernées : elles sont
    courtes exprès.
    """
    n, start, S, V = b["n"], b["start"], b["S"], b["V"]
    owner, occid, k = np.full(n, -1), np.full(n, -1), 0
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + r["block"])] = i
            occid[c:min(n, c + r["block"])] = k
            k += 1

    cut = set(h for h in hard if 0 < h < n)
    # 1. les segments bruts : un (owner, occid) constant, coupé sur chaque pic
    raw, bb = [], 0
    if start > 0:
        raw.append({"b0": 0, "b1": start - 1, "label": "intro", "own": -2})
        bb = start
    while bb < n:
        z = bb
        while (z + 1 < n and owner[z + 1] == owner[bb] and occid[z + 1] == occid[bb]
               and (z + 1) not in cut):
            z += 1
        raw.append({"b0": bb, "b1": z, "label": owner[bb],
                    "own": (int(owner[bb]), int(occid[bb]))})
        bb = z + 1

    # 2. une coupure sur pic qui laisse un moignon est annulée
    if min_cut:
        m = []
        for s in raw:
            if (m and m[-1]["own"] == s["own"] and s["own"] != -2
                    and min(s["b1"] - s["b0"] + 1,
                            m[-1]["b1"] - m[-1]["b0"] + 1) < min_cut):
                m[-1]["b1"] = s["b1"]
            else:
                m.append(dict(s))
        raw = m

    # 3. cœur pair + queue amovible, sur les segments que personne n'a réclamés
    out = []
    for s in raw:
        L = s["b1"] - s["b0"] + 1
        if (tail_unit and not isinstance(s["label"], str) and s["label"] < 0
                and L % tail_unit and L > tail_unit):
            core = L - (L % tail_unit)
            out.append({"b0": s["b0"], "b1": s["b0"] + core - 1, "label": s["label"]})
            out.append({"b0": s["b0"] + core, "b1": s["b1"], "label": "queue"})
        else:
            out.append({k2: s[k2] for k2 in ("b0", "b1", "label")})

    tails = {(s["b0"], s["b1"]) for s in out if s["label"] == "queue"}
    ren, k = {}, 0
    for s in out:
        if isinstance(s["label"], (int, np.integer)):
            if s["label"] not in ren:
                ren[s["label"]] = LETTERS[k]; k += 1
            s["label"] = ren[s["label"]]
    VS.merge_letters(S, out, V=V)
    for s in out:
        if (s["b0"], s["b1"]) in tails:
            s["label"] = "queue"
    return out


def search(b, hard=None, passes=((8, VS.THR8), (4, VS.THR4)), **kw):
    hard = b["hard"] if hard is None else hard
    tail_unit = kw.pop("tail_unit", TAIL_UNIT)
    runs, _ = _passes(b, hard, list(passes), **kw)
    return assemble(b, runs, hard, tail_unit=tail_unit)


def search_staged(b, hard=None, stages=(), tail_unit=TAIL_UNIT, **kw):
    """Plusieurs `_passes` successives qui PARTAGENT le tableau `claimed`.

    `stages` = [(passes, gate), ...]. C'est ce qu'il faut pour « la passe de 4
    passe en premier mais ne garde que ce qu'elle sait défendre » : les mesures
    qu'elle n'a pas prises restent libres pour la passe de 8 qui suit.
    """
    import numpy as _np
    hard = b["hard"] if hard is None else hard
    claimed = _np.zeros(b["n"], bool)
    runs = []
    for ps, gate in stages:
        r, claimed = _passes(b, hard, list(ps), claimed=claimed, gate=gate, **kw)
        runs += r
    return assemble(b, runs, hard, tail_unit=tail_unit)


# ── la mesure ───────────────────────────────────────────────────────────────

def gt(stem):
    from ssm_zoo import gt_sections
    g = gt_sections(stem)
    return g["sections"] if g else None


def score(secs, stem, n):
    from section_metric import compare
    ref = gt(stem)
    if ref is None:
        return float("nan")
    return float(compare(secs, ref, n)["score"])


def table(fn, stems=None, quiet=False):
    """fn(bundle) -> sections. Rend {stem: score} + la médiane."""
    from ssm_zoo import SONGS
    import order_bundle
    rows = {}
    for stem, title in SONGS:
        if stems and stem not in stems:
            continue
        b = order_bundle.get(stem)
        try:
            secs = fn(b)
            s = score(secs, stem, b["n"])
        except Exception as e:                       # pragma: no cover
            if not quiet:
                print(f"   !! {title}: {e}")
            s = float("nan")
        rows[stem] = s
    med = float(np.nanmedian(list(rows.values())))
    return rows, med


TITLES = None


def show(rows, med, label=""):
    from ssm_zoo import SONGS
    global TITLES
    TITLES = dict(SONGS)
    print(f"  {label}   médiane {med:.3f}")
    for s, v in rows.items():
        print(f"    {TITLES.get(s, s)[:26]:28s} {v:.3f}")
