"""clustering_bakeoff.py — multi-algorithm candidate-generation bakeoff for
bar-merge chord-robustness pooling (2026-07-18 night, continuation call).

Reuses the EXACT evaluation protocol established in merge_criterion.py:
corpus-scale iReal (FILES from chord_distance_eval), grain=8 nuclear blocks,
same-GT-section-label = positive pair, song-level train/val/test split
(never block-level), FPR-gated operating point selection on val, multi-seed
before citing any margin. Ground truth pairs and block_sim/nuclear_spans
come directly from chord_distance_eval.py (position-aligned, no pool-then-
dot regression) and noise_calibrate.load_corpus_registers (bass/treble
proxy vectors), i.e. NOTHING here recomputes a similarity metric from
scratch — every algorithm operates on the SAME block_sim Gram matrix per
tune, only the candidate-GROUPING logic differs.

Algorithms compared (all take the same per-tune (m,m) similarity matrix S
and same-GT-label pairs as ground truth):
  1. threshold_pairs   — current baseline (bar_merge_candidates.py's method,
                          without the max_pairs_per_bar dedup which only
                          affects UI list length, not the eval metric)
  2. knn_cc            — for each block, k nearest neighbors above a floor
                          sim; connected components of the resulting graph
                          are candidate groups; predicted-positive pair =
                          any two blocks in the same component
  3. agglomerative     — hierarchical clustering (distance=1-sim), linkage
                          in {single, complete, average}, cut at a swept
                          distance threshold
  4. dbscan            — density-based on the same distance matrix, sweep
                          eps and min_samples; noise points (label -1)
                          contribute no positive pairs at all (principled
                          "no suggestion" for a block that doesn't cluster)
  5. spectral          — affinity = S directly (clipped >=0), k chosen via
                          the eigengap heuristic on the normalized Laplacian
                          per song (not fixed corpus-wide)

Evaluation: pool all (tune, pair) rows across the val split per operating
point, pick the operating point with highest recall subject to FPR<=TARGET
FPR (0.05, matching merge_criterion.py's target), then report the SAME
operating point's test-set precision/recall/FPR. Repeat over multiple
song-level splits (seeds) and report mean+-std. This is a strictly harder
protocol than tune-level averaging (pooling changes the effective weight
per tune) but matches merge_criterion.py's own pooled-dataset convention
exactly, so results are numerically comparable to the existing baseline
numbers already in known_issues.md.
"""
from __future__ import annotations
import sys, json, random, time, warnings
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
warnings.filterwarnings("ignore", category=UserWarning)

from noise_calibrate import load_corpus_registers, N_CORPUS_SAMPLE, GRAIN
from chord_distance_eval import nuclear_spans, block_sim

OUT_DIR = Path(__file__).resolve().parent
TARGET_FPR = 0.05
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------
# Per-tune similarity matrix + GT pair labels (shared by every algorithm)
# ---------------------------------------------------------------------

def tune_blocks(tune, grain=GRAIN):
    n = len(tune["labels"])
    spans = nuclear_spans(n, grain)
    m = len(spans)
    if m < 2:
        return None
    bass = [tune["bass"][s:e] for (s, e) in spans]
    treb = [tune["treble"][s:e] for (s, e) in spans]
    labels = [Counter(tune["labels"][s:e]).most_common(1)[0][0] for (s, e) in spans]
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            sb = block_sim(bass[i], bass[j])
            st = block_sim(treb[i], treb[j])
            sc = 0.5 * (sb + st)
            S[i, j] = S[j, i] = sc
    gt = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            gt[i, j] = (labels[i] == labels[j])
    return {"S": S, "gt": gt, "m": m}


def prep_corpus(corpus):
    out = []
    for c in corpus:
        b = tune_blocks(c)
        if b is not None:
            out.append(b)
    return out


def pair_indices(m):
    idx = [(i, j) for i in range(m) for j in range(i + 1, m)]
    return idx


# ---------------------------------------------------------------------
# Candidate generators. Each takes S (m,m) + a knob value, returns a
# boolean (m,m) "predicted merge" adjacency (symmetric, diag ignored).
# ---------------------------------------------------------------------

def gen_threshold(S, tau):
    return S >= tau


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
    # connected components -> full pairwise closure within each component
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
    """Normalized-Laplacian eigengap heuristic: k* = argmax gap between
    consecutive sorted eigenvalues of L_sym, restricted to k in [1, kmax]."""
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


# ---------------------------------------------------------------------
# Aggregate pooled precision/recall/FPR over a split, for a knob value
# ---------------------------------------------------------------------

def pooled_prf(blocks, gen_fn):
    tp = fp = fn = tn = 0
    for b in blocks:
        S, gt, m = b["S"], b["gt"], b["m"]
        pred = gen_fn(S)
        idx = pair_indices(m)
        for (i, j) in idx:
            p, g = bool(pred[i, j]), bool(gt[i, j])
            if p and g: tp += 1
            elif p and not g: fp += 1
            elif not p and g: fn += 1
            else: tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return prec, rec, fpr, dict(tp=tp, fp=fp, fn=fn, tn=tn)


def fpr_gate_select(val_blocks, gen_fn_factory, knobs, target_fpr=TARGET_FPR):
    """knobs: list of knob-values; gen_fn_factory(knob) -> gen_fn(S)->bool
    matrix. Returns knob achieving highest recall s.t. FPR<=target on val,
    falling back to the lowest-FPR knob if none qualifies."""
    best = None
    fallback = None
    for kn in knobs:
        gen_fn = gen_fn_factory(kn)
        prec, rec, fpr, _ = pooled_prf(val_blocks, gen_fn)
        row = (kn, prec, rec, fpr)
        if fallback is None or fpr < fallback[3]:
            fallback = row
        if fpr <= target_fpr:
            if best is None or rec > best[2]:
                best = row
    return best if best is not None else fallback


# ---------------------------------------------------------------------
# Per-seed run: split corpus, sweep each algorithm's knob(s) on val,
# report test numbers at the FPR-gated operating point.
# ---------------------------------------------------------------------

def split_corpus(blocks, seed):
    ids = list(range(len(blocks)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    tr = [blocks[i] for i in ids[:n_tr]]
    val = [blocks[i] for i in ids[n_tr:n_tr + n_val]]
    test = [blocks[i] for i in ids[n_tr + n_val:]]
    return tr, val, test


def run_seed(blocks, seed):
    tr, val, test = split_corpus(blocks, seed)
    results = {}

    # 1. threshold_pairs
    taus = np.round(np.arange(0.30, 0.995, 0.01), 3)
    op = fpr_gate_select(val, lambda t: (lambda S: gen_threshold(S, t)), taus)
    kn, _, _, _ = op
    p, r, f, cnt = pooled_prf(test, lambda S: gen_threshold(S, kn))
    results["threshold_pairs"] = {"knob": {"tau": float(kn)}, "precision": p, "recall": r, "fpr": f, **cnt}

    # 2. knn_cc: sweep k in {1,2,3,5}, floor in {0.3,0.5,0.7}
    best_overall = None
    for k in (1, 2, 3, 5):
        for floor in (0.3, 0.5, 0.7):
            def factory(kk=k, ff=floor):
                return lambda S: gen_knn_cc(S, kk, ff)
            gen_fn = factory()
            p_, r_, f_, _ = pooled_prf(val, gen_fn)
            if f_ <= TARGET_FPR and (best_overall is None or r_ > best_overall[0]):
                best_overall = (r_, k, floor, p_, f_)
    if best_overall is None:
        # fallback: lowest-FPR combo
        best_fpr = None
        for k in (1, 2, 3, 5):
            for floor in (0.3, 0.5, 0.7, 0.9):
                gen_fn = (lambda S, kk=k, ff=floor: gen_knn_cc(S, kk, ff))
                p_, r_, f_, _ = pooled_prf(val, gen_fn)
                if best_fpr is None or f_ < best_fpr[4]:
                    best_fpr = (r_, k, floor, p_, f_)
        best_overall = best_fpr
    _, k, floor, _, _ = best_overall
    gen_fn = (lambda S: gen_knn_cc(S, k, floor))
    p, r, f, cnt = pooled_prf(test, gen_fn)
    results["knn_cc"] = {"knob": {"k": k, "floor": floor}, "precision": p, "recall": r, "fpr": f, **cnt}

    # 3. agglomerative x3 linkages
    dist_threshs = np.round(np.arange(0.02, 0.85, 0.02), 3)
    for linkage in ("single", "complete", "average"):
        op = fpr_gate_select(val, lambda d, lk=linkage: (lambda S: gen_agglomerative(S, lk, d)), dist_threshs)
        kn, _, _, _ = op
        gen_fn = (lambda S, lk=linkage, d=kn: gen_agglomerative(S, lk, d))
        p, r, f, cnt = pooled_prf(test, gen_fn)
        results["agglomerative_%s" % linkage] = {"knob": {"dist_thresh": float(kn)},
                                                   "precision": p, "recall": r, "fpr": f, **cnt}

    # 4. dbscan: sweep eps, min_samples in {1,2}
    epss = np.round(np.arange(0.02, 0.85, 0.02), 3)
    best_overall = None
    for ms in (1, 2):
        for eps in epss:
            gen_fn = (lambda S, e=eps, m_=ms: gen_dbscan(S, e, m_))
            p_, r_, f_, _ = pooled_prf(val, gen_fn)
            if f_ <= TARGET_FPR and (best_overall is None or r_ > best_overall[0]):
                best_overall = (r_, eps, ms, p_, f_)
    if best_overall is None:
        best_fpr = None
        for ms in (1, 2):
            for eps in epss:
                gen_fn = (lambda S, e=eps, m_=ms: gen_dbscan(S, e, m_))
                p_, r_, f_, _ = pooled_prf(val, gen_fn)
                if best_fpr is None or f_ < best_fpr[4]:
                    best_fpr = (r_, eps, ms, p_, f_)
        best_overall = best_fpr
    _, eps, ms, _, _ = best_overall
    gen_fn = (lambda S: gen_dbscan(S, eps, ms))
    p, r, f, cnt = pooled_prf(test, gen_fn)
    results["dbscan"] = {"knob": {"eps": float(eps), "min_samples": ms}, "precision": p, "recall": r, "fpr": f, **cnt}

    # 5. spectral + eigengap (no knob to FPR-gate on besides a similarity
    # floor prune before building the affinity graph — sweep that)
    floors = np.round(np.arange(0.0, 0.9, 0.05), 3)
    op = fpr_gate_select(val, lambda fl: (lambda S: gen_spectral_eigengap(S, fl)), floors)
    kn, _, _, _ = op
    gen_fn = (lambda S: gen_spectral_eigengap(S, kn))
    p, r, f, cnt = pooled_prf(test, gen_fn)
    results["spectral_eigengap"] = {"knob": {"floor_prune": float(kn)}, "precision": p, "recall": r, "fpr": f, **cnt}

    return results


def main():
    t0 = time.time()
    print("Loading iReal corpus (bass/treble proxies)...")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE * 3)
    print("  corpus: %d multi-section tunes" % len(corpus))
    print("Precomputing per-tune block_sim matrices (grain=%d)..." % GRAIN)
    blocks = prep_corpus(corpus)
    print("  %d tunes usable (>=2 blocks)" % len(blocks))
    print("  elapsed %.1fs" % (time.time() - t0))

    all_results = defaultdict(list)
    for seed in SEEDS:
        t1 = time.time()
        r = run_seed(blocks, seed)
        for algo, row in r.items():
            all_results[algo].append(row)
        print("\n=== seed=%d (%.1fs) ===" % (seed, time.time() - t1))
        for algo, row in sorted(r.items()):
            print("  %-22s P=%.3f R=%.3f FPR=%.3f  knob=%s" %
                  (algo, row["precision"], row["recall"], row["fpr"], row["knob"]))

    print("\n=== SUMMARY (%d seeds, target_fpr<=%.2f) ===" % (len(SEEDS), TARGET_FPR))
    summary = {}
    for algo, rows in all_results.items():
        recs = [x["recall"] for x in rows]
        precs = [x["precision"] for x in rows]
        fprs = [x["fpr"] for x in rows]
        summary[algo] = {
            "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
            "precision_mean": float(np.mean(precs)), "precision_std": float(np.std(precs)),
            "fpr_mean": float(np.mean(fprs)), "fpr_std": float(np.std(fprs)),
        }
        print("  %-22s recall=%.3f+-%.3f  precision=%.3f+-%.3f  fpr=%.3f+-%.3f" %
              (algo, summary[algo]["recall_mean"], summary[algo]["recall_std"],
               summary[algo]["precision_mean"], summary[algo]["precision_std"],
               summary[algo]["fpr_mean"], summary[algo]["fpr_std"]))

    out = {"seeds": SEEDS, "target_fpr": TARGET_FPR, "per_seed": {k: v for k, v in all_results.items()},
           "summary": summary, "n_tunes": len(blocks), "grain": GRAIN}
    (OUT_DIR / "clustering_bakeoff_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote clustering_bakeoff_results.json")
    print("total elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
