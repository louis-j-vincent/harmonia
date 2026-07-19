"""candidate_algos.py — shared candidate-GROUPING algorithms, factored out
of clustering_bakeoff.py so the SAME code that was benchmarked corpus-scale
on iReal (scratchpad/clustering_bakeoff.py, 2026-07-18 bar-merge bakeoff) is
what actually runs in production candidate generation
(scratchpad/bar_merge_candidates.py) — no reimplementation drift between
"the algorithm we measured" and "the algorithm we shipped."

Each gen_* function takes an (m,m) similarity matrix S (values expected
roughly in [-1,1], diag ~1) and returns a boolean (m,m) symmetric adjacency
of "predicted merge" pairs (diagonal False). Group extraction
(`groups_from_adjacency`) turns that adjacency into connected-component
candidate groups for UI presentation, independent of which algorithm
produced the adjacency.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def gen_threshold(S, tau):
    m = S.shape[0]
    out = S >= tau
    np.fill_diagonal(out, False)
    return out


def gen_knn_cc(S, k, floor):
    m = S.shape[0]
    adj = np.zeros((m, m), dtype=bool)
    for i in range(m):
        order = np.argsort(-S[i])
        cnt = 0
        for j in order:
            if j == i:
                continue
            if S[i, j] < floor:
                break
            adj[i, j] = adj[j, i] = True
            cnt += 1
            if cnt >= k:
                break
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(m):
            if adj[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    comp = np.array([find(i) for i in range(m)])
    out = comp[:, None] == comp[None, :]
    np.fill_diagonal(out, False)
    return out


def gen_agglomerative(S, linkage, dist_thresh):
    m = S.shape[0]
    if m < 2:
        return np.zeros((m, m), dtype=bool)
    D = 1.0 - np.clip(S, -1, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    try:
        cl = AgglomerativeClustering(metric="precomputed", linkage=linkage,
                                      distance_threshold=dist_thresh, n_clusters=None)
        labels = cl.fit_predict(D)
    except Exception:
        return np.zeros((m, m), dtype=bool)
    out = labels[:, None] == labels[None, :]
    np.fill_diagonal(out, False)
    return out


def gen_dbscan(S, eps, min_samples):
    m = S.shape[0]
    if m < 2:
        return np.zeros((m, m), dtype=bool)
    D = 1.0 - np.clip(S, -1, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    db = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = db.fit_predict(D)
    out = labels[:, None] == labels[None, :]
    noise = labels == -1
    out[noise, :] = False
    out[:, noise] = False
    np.fill_diagonal(out, False)
    return out


def eigengap_k(S, kmax=None):
    m = S.shape[0]
    kmax = kmax or max(1, m - 1)
    A = np.clip(S, 0, None).copy()
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    deg_safe = np.clip(deg, 1e-9, None)
    Dm12 = np.diag(1.0 / np.sqrt(deg_safe))
    L = np.eye(m) - Dm12 @ A @ Dm12
    evals = np.sort(np.linalg.eigvalsh(L))
    gaps = np.diff(evals[:kmax + 1])
    k = int(np.argmax(gaps)) + 1
    return max(2, min(k, m))


def gen_spectral_eigengap(S, floor_prune=0.0):
    m = S.shape[0]
    if m < 3:
        return np.zeros((m, m), dtype=bool)
    A = np.clip(S, 0, None).copy()
    A[A < floor_prune] = 0.0
    np.fill_diagonal(A, 0.0)
    if A.sum() < 1e-9:
        return np.zeros((m, m), dtype=bool)
    k = eigengap_k(A)
    from sklearn.cluster import SpectralClustering
    try:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                 assign_labels="kmeans", random_state=0)
        labels = sc.fit_predict(A + 1e-9 * np.eye(m))
    except Exception:
        return np.zeros((m, m), dtype=bool)
    out = labels[:, None] == labels[None, :]
    np.fill_diagonal(out, False)
    return out


def groups_from_adjacency(adj, min_size=2):
    """Connected components of a boolean adjacency -> list of index lists,
    size >= min_size only."""
    m = adj.shape[0]
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            if adj[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    comps = {}
    for i in range(m):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [sorted(v) for v in comps.values() if len(v) >= min_size]
