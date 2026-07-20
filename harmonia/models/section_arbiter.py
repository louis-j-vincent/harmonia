"""Importable prototype: section clustering = harmony (pos-agree) + distinctive-chord
veto + energy-confirmer arbitration. Kept out of chart_model.py (concurrent WIP);
wire in later. Deterministic, no audio deps here (energy passed in as per-block scalar)."""
from collections import Counter
import numpy as np

MATCH, PHASE_STRICT, MAXLAG = 0.6, 0.80, 1

def _om(a, b):
    k = min(len(a), len(b))
    return sum(1 for x, y in zip(a[:k], b[:k]) if x == y) / k if k else 0.0

def sim(a, b):
    if not a or not b:
        return 0.0
    base = (sum(1 for x, y in zip(a, b) if x == y) / len(a)
            if len(a) == len(b) else _om(a, b))
    best = base
    for lag in range(1, MAXLAG + 1):
        for m in (_om(a[lag:], b), _om(a, b[lag:])):
            if m >= PHASE_STRICT:
                best = max(best, m)
    return best

def veto(a, b, min_recur=2, min_frac=0.2):
    ca, cb = Counter(a), Counter(b)
    sa, sb = set(a), set(b)
    for r, n in ca.items():
        if n >= min_recur and n >= min_frac * len(a) and r not in sb:
            return True
    for r, n in cb.items():
        if n >= min_recur and n >= min_frac * len(b) and r not in sa:
            return True
    return False

def cluster(block_roots, block_energy=None, use_veto=False, use_energy=False,
            e_same=0.5, e_diff=1.2):
    """single-linkage; returns per-block integer cluster id.
    energy arbitration (calibrated evidence, not blind override):
      - override veto (allow merge) when energy is SIMILAR  (|dz|<e_same) -> same section varied
      - block a harmony-merge when energy is VERY DIFFERENT (|dz|>e_diff) -> diff section, harmony silent
    """
    nb = len(block_roots)
    z = None
    if use_energy and block_energy is not None and nb > 1:
        e = np.asarray(block_energy, float)
        z = (e - e.mean()) / (e.std() + 1e-9)
    parent = list(range(nb))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def dz(i, j):
        return abs(z[i] - z[j]) if z is not None else 0.0
    for i in range(nb):
        for j in range(i + 1, nb):
            s = sim(block_roots[i], block_roots[j])
            if s < MATCH:
                continue                      # harmony says different -> no merge
            merge = True
            if use_veto and veto(block_roots[i], block_roots[j]):
                merge = False                 # distinctive-chord veto blocks
                if use_energy and dz(i, j) < e_same:
                    merge = True               # ...unless energy says same section (varied)
            if merge and use_energy and dz(i, j) > e_diff:
                merge = False                 # harmony-same but energy strongly differs -> split
            if merge:
                parent[find(i)] = find(j)
    # normalize ids
    ids = {}
    out = []
    for i in range(nb):
        r = find(i)
        out.append(ids.setdefault(r, len(ids)))
    return out
