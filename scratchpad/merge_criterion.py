"""merge_criterion.py — Step 2 (mandatory, per the 2026-07-18 continuation
brief): learned bar-merge criterion trained on CLEAN, non-noised iReal
directly (GT only exists here — real audio has none, so the criterion MUST
be learned on iReal one way or another). This is the primary/baseline
approach regardless of how Step 1's noise calibration goes; the floor-blend
recipe from Step 1 (alpha=0.40 combined) is layered on top as an explicit
robustness variant in this same script, not a replacement.

Setup: at grain=8 nuclear blocks (same spans as chord_distance_eval.py /
noise_calibrate.py), for every pair of blocks within a song, feature =
[sim_bass, sim_treble, sim_combined, block_distance_norm, size_ratio].
Label = 1 iff the two blocks share a majority GT section label (i.e. SHOULD
be merged). Song-level train/val/test split (never block-level — avoids
leakage per project convention).

User's explicit priority: MINIMIZE FALSE POSITIVES (merging bars that are
actually different) — so the reported operating point is chosen at a fixed
LOW false-positive-rate target on the val set, not at max-F1. Compared:
(a) simple threshold on sim_combined alone (the existing default), (b)
logistic regression over the 5-d feature vector. Multi-seed (project
integrity rule) before citing any margin.
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression

from noise_calibrate import load_corpus_registers, N_CORPUS_SAMPLE, GRAIN
from chord_distance_eval import nuclear_spans, block_sim

OUT_DIR = Path(__file__).resolve().parent
FLOOR_ALPHA = 0.40  # Step 1's calibrated combined-register alpha


def corpus_generic_vector(corpus, reg):
    acc = np.zeros(12); n = 0
    for c in corpus:
        for v in c[reg]:
            acc += v; n += 1
    acc /= max(n, 1)
    nn = np.linalg.norm(acc)
    return acc / nn if nn > 1e-9 else acc


def blend(vecs, alpha, generic):
    if alpha <= 0:
        return vecs
    out = []
    for v in vecs:
        nv = (1 - alpha) * v + alpha * generic
        n = np.linalg.norm(nv)
        out.append(nv / n if n > 1e-9 else nv)
    return out


def build_pairs(tune, grain=GRAIN, use_floor=False, generic=None, alpha=FLOOR_ALPHA):
    n = len(tune["labels"])
    spans = nuclear_spans(n, grain)
    if len(spans) < 2:
        return []
    bass, treb = tune["bass"], tune["treble"]
    if use_floor:
        bass = blend(bass, alpha, generic["bass"])
        treb = blend(treb, alpha, generic["treble"])
    block_bass = [bass[s:e] for (s, e) in spans]
    block_treb = [treb[s:e] for (s, e) in spans]
    block_labels = []
    for (s, e) in spans:
        block_labels.append(Counter(tune["labels"][s:e]).most_common(1)[0][0])
    m = len(spans)
    rows = []
    for i in range(m):
        for j in range(i + 1, m):
            sb = block_sim(block_bass[i], block_bass[j])
            st = block_sim(block_treb[i], block_treb[j])
            sc = 0.5 * (sb + st)
            dist = (j - i) / m
            size_i = spans[i][1] - spans[i][0]
            size_j = spans[j][1] - spans[j][0]
            size_ratio = min(size_i, size_j) / max(size_i, size_j)
            label = 1 if block_labels[i] == block_labels[j] else 0
            rows.append((sb, st, sc, dist, size_ratio, label))
    return rows


def make_dataset(corpus, use_floor=False, generic=None):
    X, y = [], []
    for c in corpus:
        rows = build_pairs(c, use_floor=use_floor, generic=generic)
        for r in rows:
            X.append(r[:5]); y.append(r[5])
    return np.array(X), np.array(y)


def fpr_gated_threshold(scores, y, target_fpr=0.05):
    """Highest threshold (on a single score dim) such that FPR <= target on
    this set; return (threshold, recall_at_threshold, fpr_at_threshold)."""
    neg = scores[y == 0]
    pos = scores[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return None, None, None
    # threshold = the (1-target_fpr) quantile of negative scores -> FPR<=target
    thr = float(np.quantile(neg, 1 - target_fpr))
    fpr = float(np.mean(neg >= thr))
    rec = float(np.mean(pos >= thr))
    return thr, rec, fpr


def eval_threshold_only(Xte, yte, thr):
    pred = (Xte[:, 2] >= thr).astype(int)  # sim_combined = column 2
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    tn = np.sum((pred == 0) & (yte == 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return prec, rec, fpr


def eval_logreg(clf, Xte, yte, thr):
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= thr).astype(int)
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    tn = np.sum((pred == 0) & (yte == 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return prec, rec, fpr


def run_one_seed(corpus, seed, target_fpr=0.05, use_floor=False, generic=None):
    ids = list(range(len(corpus)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    tr_ids, val_ids, te_ids = ids[:n_tr], ids[n_tr:n_tr + n_val], ids[n_tr + n_val:]
    train = [corpus[i] for i in tr_ids]
    val = [corpus[i] for i in val_ids]
    test = [corpus[i] for i in te_ids]

    Xtr, ytr = make_dataset(train, use_floor=use_floor, generic=generic)
    Xval, yval = make_dataset(val, use_floor=use_floor, generic=generic)
    Xte, yte = make_dataset(test, use_floor=use_floor, generic=generic)

    # --- Baseline: threshold on sim_combined alone, FPR-gated on val ---
    thr_base, _, _ = fpr_gated_threshold(Xval[:, 2], yval, target_fpr)
    p_b, r_b, f_b = eval_threshold_only(Xte, yte, thr_base)

    # --- Logistic regression on 5-d feature vector ---
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba_val = clf.predict_proba(Xval)[:, 1]
    thr_lr, _, _ = fpr_gated_threshold(proba_val, yval, target_fpr)
    p_l, r_l, f_l = eval_logreg(clf, Xte, yte, thr_lr)

    return {
        "n_train_pairs": len(ytr), "n_val_pairs": len(yval), "n_test_pairs": len(yte),
        "threshold_only": {"thr": thr_base, "precision": p_b, "recall": r_b, "fpr": f_b},
        "logreg": {"thr": thr_lr, "precision": p_l, "recall": r_l, "fpr": f_l,
                   "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])},
    }


def main():
    target_fpr = 0.05  # low-false-positive operating point, per user priority
    print("Loading iReal corpus (bass/treble proxies)...")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE * 3)  # bigger pool, still capped
    print("  corpus: %d multi-section tunes" % len(corpus))
    generic = {"bass": corpus_generic_vector(corpus, "bass"),
               "treble": corpus_generic_vector(corpus, "treble")}

    results = {"clean": [], "floor_blend": []}
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        print("\n=== seed=%d (clean) ===" % seed)
        r = run_one_seed(corpus, seed, target_fpr=target_fpr, use_floor=False)
        results["clean"].append(r)
        print("  threshold-only: P=%.3f R=%.3f FPR=%.3f (thr=%.3f)" %
              (r["threshold_only"]["precision"], r["threshold_only"]["recall"],
               r["threshold_only"]["fpr"], r["threshold_only"]["thr"]))
        print("  logreg:         P=%.3f R=%.3f FPR=%.3f (thr=%.3f)" %
              (r["logreg"]["precision"], r["logreg"]["recall"],
               r["logreg"]["fpr"], r["logreg"]["thr"]))

    for seed in seeds:
        print("\n=== seed=%d (floor-blend alpha=%.2f train-time augmentation) ===" %
              (seed, FLOOR_ALPHA))
        r = run_one_seed(corpus, seed, target_fpr=target_fpr, use_floor=True, generic=generic)
        results["floor_blend"].append(r)
        print("  threshold-only: P=%.3f R=%.3f FPR=%.3f (thr=%.3f)" %
              (r["threshold_only"]["precision"], r["threshold_only"]["recall"],
               r["threshold_only"]["fpr"], r["threshold_only"]["thr"]))
        print("  logreg:         P=%.3f R=%.3f FPR=%.3f (thr=%.3f)" %
              (r["logreg"]["precision"], r["logreg"]["recall"],
               r["logreg"]["fpr"], r["logreg"]["thr"]))

    def summarize(key, variant):
        recalls = [r[variant]["recall"] for r in results[key]]
        precs = [r[variant]["precision"] for r in results[key]]
        fprs = [r[variant]["fpr"] for r in results[key]]
        print("  [%s/%s] recall mean=%.3f std=%.3f | precision mean=%.3f std=%.3f | fpr mean=%.3f" %
              (key, variant, np.mean(recalls), np.std(recalls),
               np.mean(precs), np.std(precs), np.mean(fprs)))

    print("\n=== SUMMARY (5 seeds, target_fpr=%.2f) ===" % target_fpr)
    for key in ("clean", "floor_blend"):
        for variant in ("threshold_only", "logreg"):
            summarize(key, variant)

    (OUT_DIR / "merge_criterion_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote merge_criterion_results.json")


if __name__ == "__main__":
    main()
