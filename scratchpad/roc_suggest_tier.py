"""roc_suggest_tier.py — 2026-07-18 (overnight autonomous call, task 1 of 3):
re-tune the SUGGEST-tier bar-merge threshold (tau_suggest, currently 0.93,
originally selected at FPR<=0.05 giving only ~22% recall — see
`docs/known_issues.md` "Multi-algorithm bar-merge candidate-generation
BAKEOFF") around a LOW false-negative-rate operating point instead.

User's reasoning (relayed via the orchestrating session): the SUGGEST tier
is human-reviewed before it does anything (a wasted tap is cheap), so the
old FPR<=0.05 selection criterion was optimizing the wrong thing — a real
merge opportunity that never gets SHOWN to the human (a false negative) is
a permanent loss, whereas a bad suggestion the human taps and rejects costs
nothing. This script re-derives the operating point around recall targets
(60/75/85/90%) and reports the FPR/precision cost at each, using the
CORRECTED chord-identity GT label (same-chord-identity (root_pc, qbucket),
NOT same-GT-section — see `tau_auto_search.py`'s module docstring for the
full story of why the section label is wrong at bar grain: a single bar's
harmony does not determine section identity, verified there via a ~50%
error rate even at similarity==1.0 under the wrong label).

Reuses `tau_auto_search.py`'s corpus loader, bar-pair builder (min_gap=4,
matching `bar_merge_candidates.py`'s own min_gap), and song-level
train/val/test 3-way split machinery verbatim — no reimplementation, same
corpus (full iReal, 2399 usable tunes), same feature (bt_concat-equivalent
bass+treble proxy cosine similarity averaged 50/50), same corrected label.

**Methodology (nested, nothing selected and validated on the same data):**
for each of 5 song-level seeds (60/20/20 train/val/test):
  - a threshold/recall-target is CHOSEN by scanning the TRAIN+VAL pool
    (80%) for the lowest tau achieving >= the target recall on that pool;
  - the resulting operating point's FPR/precision is then measured ONCE on
    the untouched 20% TEST fold — this is the honestly-reported number.
ROC/PR curves themselves are reported two ways: (a) a GLOBAL full-corpus
curve (diagnostic only, matches tau_auto_search.py's own "global curve"
convention — NOT a split-safe generalization estimate, but useful for
visual shape/AUC context and dense enough for a good chart); (b) per-fold
TEST-only curves + AUCs (5 seeds) for split-safe AUC estimates, reported as
mean+-std.

Output: scratchpad/roc_suggest_tier_results.json — dense ROC/PR points
(global + per-fold), AUCs, and the recall-target operating-point table.
No chart built here (explicit brief: hand the JSON to the orchestrating
session for visualization, don't build a polished chart in this call).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from tau_auto_search import (
    load_corpus_bar_chords, build_bar_pairs, split_songs_3way, flatten,
    TAU_SUGGEST as OLD_TAU_SUGGEST, MIN_GAP, SEEDS,
)

OUT_DIR = Path(__file__).resolve().parent
RECALL_TARGETS = [0.60, 0.75, 0.85, 0.90]
N_GRID = 400  # dense but JSON-manageable threshold grid for curves

# numpy>=2.0 renamed trapz->trapezoid; keep this script working across
# whichever numpy the venv has (CLAUDE.md: numpy<2.5 pinned for numba, but
# the exact minor version drifts).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def roc_pr_curve(sims, labels, n_grid=N_GRID):
    """Dense ROC/PR curve over a quantile-spaced threshold grid spanning the
    full observed similarity range (not just >=TAU_SUGGEST — this call's
    whole point is to look BELOW the old floor too, since a lower
    tau_suggest is exactly the candidate change under consideration).
    Returns dict with parallel arrays: thresholds, tpr, fpr, precision,
    recall (==tpr), plus roc_auc (trapezoidal on (fpr,tpr)) and pr_auc
    (average-precision definition: sum of (recall_k-recall_{k-1})*precision_k,
    the standard step-function AP, not naive trapz-on-PR which is known to
    be optimistically biased under class imbalance)."""
    n = len(sims)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    order = np.argsort(-sims)  # descending sim
    s_sorted = sims[order]
    l_sorted = labels[order]
    cum_tp = np.cumsum(l_sorted == 1)
    cum_fp = np.cumsum(l_sorted == 0)
    tpr_full = cum_tp / max(n_pos, 1)
    fpr_full = cum_fp / max(n_neg, 1)
    prec_full = cum_tp / np.arange(1, n + 1)

    # thresholds = quantile grid over s_sorted so points concentrate where
    # the curve actually bends (high-sim region), not wasted on the flat
    # near-zero-similarity tail that dominates by raw count.
    idx_grid = np.unique(np.linspace(0, n - 1, n_grid).astype(int))
    thr = s_sorted[idx_grid]
    tpr = tpr_full[idx_grid]
    fpr = fpr_full[idx_grid]
    prec = prec_full[idx_grid]

    # ROC AUC: trapezoidal over (fpr,tpr) sorted ascending by fpr.
    ord2 = np.argsort(fpr_full)
    roc_auc = float(_trapz(tpr_full[ord2], fpr_full[ord2]))

    # PR AUC (average precision): standard step definition over the FULL
    # (non-subsampled) curve for accuracy, using recall breakpoints.
    recall_full = tpr_full
    delta_recall = np.diff(np.concatenate([[0.0], recall_full]))
    pr_auc = float(np.sum(delta_recall * prec_full))

    return {
        "thresholds": thr.tolist(), "tpr": tpr.tolist(), "fpr": fpr.tolist(),
        "precision": prec.tolist(), "recall": tpr.tolist(),
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "n_pairs": n, "n_pos": n_pos, "n_neg": n_neg,
    }


def find_tau_for_recall(sims, labels, target_recall):
    """Lowest tau achieving recall>=target on (sims,labels) — i.e. scan
    thresholds descending, find largest selected-set (lowest tau) with
    TPR>=target. Returns None if even tau=-inf (select everything) doesn't
    reach the target (shouldn't happen since recall->1 as tau->-inf, but
    guard anyway)."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-sims)
    l_sorted = labels[order]
    s_sorted = sims[order]
    cum_tp = np.cumsum(l_sorted == 1)
    tpr = cum_tp / n_pos
    idx = np.where(tpr >= target_recall)[0]
    if len(idx) == 0:
        return None
    k = idx[0]  # first index (highest tau) reaching target recall
    return float(s_sorted[k])


def evaluate_at_tau(sims, labels, tau):
    mask = sims >= tau
    n_sel = int(mask.sum())
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tp = int(np.sum(labels[mask] == 1))
    fp = int(np.sum(labels[mask] == 0))
    fn = n_pos - tp
    recall = tp / n_pos if n_pos else None
    precision = tp / n_sel if n_sel else None
    fpr = fp / n_neg if n_neg else None
    return {"tau": tau, "n_selected": n_sel, "tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "fpr": fpr}


def main():
    t0 = time.time()
    print("Loading FULL iReal corpus (shared with tau_auto_search.py)...")
    corpus = load_corpus_bar_chords(max_tunes=None)
    print("  %d tunes, elapsed %.1fs" % (len(corpus), time.time() - t0))

    per_tune = build_bar_pairs(corpus, min_gap=MIN_GAP)
    total_pairs = sum(len(r) for r in per_tune)
    print("  %d tunes usable, %d total bar-pairs, elapsed %.1fs" %
          (len(per_tune), total_pairs, time.time() - t0))

    # ---- (a) GLOBAL diagnostic curve (full corpus, no held-out split) ----
    sims_all, labels_all = flatten(per_tune)
    print("Computing global ROC/PR curve (%d pairs, %d positive)..." %
          (len(sims_all), int(labels_all.sum())))
    global_curve = roc_pr_curve(sims_all, labels_all)
    print("  GLOBAL: ROC-AUC=%.4f  PR-AUC(AP)=%.4f" % (global_curve["roc_auc"], global_curve["pr_auc"]))

    # ---- (b) per-fold TEST-only curves + nested recall-target selection ----
    fold_results = []
    for seed in SEEDS:
        train, val, test = split_songs_3way(per_tune, seed)
        sims_tr, labels_tr = flatten(train)
        sims_va, labels_va = flatten(val)
        sims_te, labels_te = flatten(test)
        # pool = train+val (selection side); test = held-out (reporting side)
        sims_pool = np.concatenate([sims_tr, sims_va])
        labels_pool = np.concatenate([labels_tr, labels_va])

        test_curve = roc_pr_curve(sims_te, labels_te, n_grid=200)
        print("\n=== seed=%d ===  test pairs=%d (pos=%d)  ROC-AUC=%.4f  PR-AUC=%.4f" %
              (seed, len(sims_te), int(labels_te.sum()), test_curve["roc_auc"], test_curve["pr_auc"]))

        recall_table = []
        for target in RECALL_TARGETS:
            tau = find_tau_for_recall(sims_pool, labels_pool, target)
            if tau is None:
                recall_table.append({"target_recall": target, "tau": None, "held_out_test": None})
                continue
            pool_eval = evaluate_at_tau(sims_pool, labels_pool, tau)
            test_eval = evaluate_at_tau(sims_te, labels_te, tau)
            recall_table.append({
                "target_recall": target, "tau": tau,
                "pool_selection": pool_eval, "held_out_test": test_eval,
            })
            print("  target_recall=%.2f -> tau=%.4f | pool: recall=%s fpr=%s prec=%s | "
                  "TEST(blind): recall=%s fpr=%s prec=%s n_sel=%d" %
                  (target, tau,
                   "%.3f" % pool_eval["recall"] if pool_eval["recall"] is not None else None,
                   "%.4f" % pool_eval["fpr"] if pool_eval["fpr"] is not None else None,
                   "%.3f" % pool_eval["precision"] if pool_eval["precision"] is not None else None,
                   "%.3f" % test_eval["recall"] if test_eval["recall"] is not None else None,
                   "%.4f" % test_eval["fpr"] if test_eval["fpr"] is not None else None,
                   "%.3f" % test_eval["precision"] if test_eval["precision"] is not None else None,
                   test_eval["n_selected"]))

        fold_results.append({
            "seed": seed, "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_pool_pairs": len(sims_pool), "n_test_pairs": len(sims_te),
            "test_curve": test_curve, "recall_table": recall_table,
        })

    roc_aucs = [f["test_curve"]["roc_auc"] for f in fold_results]
    pr_aucs = [f["test_curve"]["pr_auc"] for f in fold_results]
    print("\n=== 5-FOLD TEST-ONLY AUC SUMMARY ===")
    print("  ROC-AUC: mean=%.4f std=%.4f  (per-fold: %s)" %
          (np.mean(roc_aucs), np.std(roc_aucs), ["%.4f" % a for a in roc_aucs]))
    print("  PR-AUC:  mean=%.4f std=%.4f  (per-fold: %s)" %
          (np.mean(pr_aucs), np.std(pr_aucs), ["%.4f" % a for a in pr_aucs]))

    # ---- aggregate recall-target table across folds (mean +- std of the
    # blind-test FPR/precision at each target, since tau itself varies
    # slightly per fold) ----
    agg_table = []
    for i, target in enumerate(RECALL_TARGETS):
        taus, fprs, precs, recalls, n_sels = [], [], [], [], []
        for f in fold_results:
            row = f["recall_table"][i]
            if row["tau"] is None or row["held_out_test"] is None:
                continue
            taus.append(row["tau"])
            te = row["held_out_test"]
            if te["fpr"] is not None:
                fprs.append(te["fpr"])
            if te["precision"] is not None:
                precs.append(te["precision"])
            if te["recall"] is not None:
                recalls.append(te["recall"])
            n_sels.append(te["n_selected"])
        agg_table.append({
            "target_recall": target,
            "tau_mean": float(np.mean(taus)) if taus else None,
            "tau_min": float(np.min(taus)) if taus else None,
            "tau_max": float(np.max(taus)) if taus else None,
            "blind_test_fpr_mean": float(np.mean(fprs)) if fprs else None,
            "blind_test_fpr_max": float(np.max(fprs)) if fprs else None,
            "blind_test_precision_mean": float(np.mean(precs)) if precs else None,
            "blind_test_recall_mean": float(np.mean(recalls)) if recalls else None,
            "n_folds": len(taus),
        })
    print("\n=== AGGREGATE (5-fold) recall-target table ===")
    for row in agg_table:
        print("  target_recall=%.2f  tau_mean=%.4f (range %.4f-%.4f)  "
              "blind_FPR mean=%.4f max=%.4f  precision_mean=%.3f  actual_recall_mean=%.3f  (n_folds=%d)" %
              (row["target_recall"], row["tau_mean"], row["tau_min"], row["tau_max"],
               row["blind_test_fpr_mean"], row["blind_test_fpr_max"],
               row["blind_test_precision_mean"], row["blind_test_recall_mean"], row["n_folds"]))

    out = {
        "min_gap": MIN_GAP, "old_tau_suggest": OLD_TAU_SUGGEST,
        "recall_targets": RECALL_TARGETS, "seeds": SEEDS,
        "n_tunes": len(per_tune), "n_total_pairs": total_pairs,
        "global_curve": global_curve,
        "fold_results": fold_results,
        "auc_summary": {
            "roc_auc_mean": float(np.mean(roc_aucs)), "roc_auc_std": float(np.std(roc_aucs)),
            "pr_auc_mean": float(np.mean(pr_aucs)), "pr_auc_std": float(np.std(pr_aucs)),
        },
        "aggregate_recall_table": agg_table,
    }
    (OUT_DIR / "roc_suggest_tier_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote roc_suggest_tier_results.json, total elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
