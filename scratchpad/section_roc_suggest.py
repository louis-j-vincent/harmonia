"""section_roc_suggest.py — 2026-07-18, section-level suggestion tool, task 2:
corpus-scale ROC/AUC retune at grain=8 (standard) and grain=4 (comparison),
targeting HIGH RECALL (peu de faux negatifs) per the user's corrected
framing, with a precision/FPR story reported alongside.

Reuses roc_suggest_tier.py's exact methodology (same functions, imported
not reimplemented): dense ROC/PR curves (global diagnostic + 5-seed
song-level train/val-pool -> held-out-test nested recall-target selection),
just swapping the bar-level pair source for section_pairs.build_section_pairs.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tau_auto_search import load_corpus_bar_chords, split_songs_3way
from section_pairs import build_section_pairs
from roc_suggest_tier import roc_pr_curve, find_tau_for_recall, evaluate_at_tau

OUT_DIR = Path(__file__).resolve().parent
RECALL_TARGETS = [0.70, 0.80, 0.85, 0.90, 0.95]
SEEDS = [0, 1, 2, 3, 4]


def flatten(tune_rows_list):
    sims, labels = [], []
    for rows in tune_rows_list:
        for s, l in rows:
            sims.append(s); labels.append(l)
    return np.array(sims), np.array(labels)


def run_grain(corpus, grain):
    print("\n########## GRAIN=%d ##########" % grain)
    t0 = time.time()
    per_tune = build_section_pairs(corpus, grain)
    total_pairs = sum(len(r) for r in per_tune)
    print("  %d tunes usable, %d total block-pairs, %.1fs" %
          (len(per_tune), total_pairs, time.time() - t0))

    sims_all, labels_all = flatten(per_tune)
    print("Computing global ROC/PR curve (%d pairs, %d positive)..." %
          (len(sims_all), int(labels_all.sum())))
    global_curve = roc_pr_curve(sims_all, labels_all)
    print("  GLOBAL: ROC-AUC=%.4f  PR-AUC=%.4f" % (global_curve["roc_auc"], global_curve["pr_auc"]))

    fold_results = []
    for seed in SEEDS:
        train, val, test = split_songs_3way(per_tune, seed)
        sims_tr, labels_tr = flatten(train)
        sims_va, labels_va = flatten(val)
        sims_te, labels_te = flatten(test)
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
            recall_table.append({"target_recall": target, "tau": tau,
                                  "pool_selection": pool_eval, "held_out_test": test_eval})
            print("  target_recall=%.2f -> tau=%.4f | pool: recall=%.3f fpr=%.4f prec=%.3f | "
                  "TEST(blind): recall=%s fpr=%s prec=%s n_sel=%d" %
                  (target, tau, pool_eval["recall"], pool_eval["fpr"], pool_eval["precision"],
                   "%.3f" % test_eval["recall"] if test_eval["recall"] is not None else None,
                   "%.4f" % test_eval["fpr"] if test_eval["fpr"] is not None else None,
                   "%.3f" % test_eval["precision"] if test_eval["precision"] is not None else None,
                   test_eval["n_selected"]))

        fold_results.append({"seed": seed, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                              "n_pool_pairs": len(sims_pool), "n_test_pairs": len(sims_te),
                              "test_curve": test_curve, "recall_table": recall_table})

    roc_aucs = [f["test_curve"]["roc_auc"] for f in fold_results]
    pr_aucs = [f["test_curve"]["pr_auc"] for f in fold_results]
    print("\n=== 5-FOLD TEST-ONLY AUC SUMMARY (grain=%d) ===" % grain)
    print("  ROC-AUC: mean=%.4f std=%.4f" % (np.mean(roc_aucs), np.std(roc_aucs)))
    print("  PR-AUC:  mean=%.4f std=%.4f" % (np.mean(pr_aucs), np.std(pr_aucs)))

    agg_table = []
    for i, target in enumerate(RECALL_TARGETS):
        taus, fprs, precs, recalls = [], [], [], []
        for f in fold_results:
            row = f["recall_table"][i]
            if row["tau"] is None or row["held_out_test"] is None:
                continue
            taus.append(row["tau"])
            te = row["held_out_test"]
            if te["fpr"] is not None: fprs.append(te["fpr"])
            if te["precision"] is not None: precs.append(te["precision"])
            if te["recall"] is not None: recalls.append(te["recall"])
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
    print("\n=== AGGREGATE (5-fold) recall-target table (grain=%d) ===" % grain)
    for row in agg_table:
        print("  target_recall=%.2f  tau_mean=%.4f (range %.4f-%.4f)  "
              "blind_FPR mean=%.4f max=%.4f  precision_mean=%s  actual_recall_mean=%.3f  (n_folds=%d)" %
              (row["target_recall"], row["tau_mean"], row["tau_min"], row["tau_max"],
               row["blind_test_fpr_mean"], row["blind_test_fpr_max"],
               "%.3f" % row["blind_test_precision_mean"] if row["blind_test_precision_mean"] is not None else None,
               row["blind_test_recall_mean"], row["n_folds"]))

    return {
        "grain": grain, "n_tunes": len(per_tune), "n_total_pairs": total_pairs,
        "global_curve": global_curve, "fold_results": fold_results,
        "auc_summary": {"roc_auc_mean": float(np.mean(roc_aucs)), "roc_auc_std": float(np.std(roc_aucs)),
                         "pr_auc_mean": float(np.mean(pr_aucs)), "pr_auc_std": float(np.std(pr_aucs))},
        "aggregate_recall_table": agg_table,
    }


def main():
    t0 = time.time()
    print("Loading FULL iReal corpus...")
    corpus = load_corpus_bar_chords(max_tunes=None)
    print("  %d tunes, %.1fs" % (len(corpus), time.time() - t0))

    out = {"recall_targets": RECALL_TARGETS, "seeds": SEEDS, "grains": {}}
    for grain in (8, 4):
        out["grains"][str(grain)] = run_grain(corpus, grain)

    (OUT_DIR / "section_roc_suggest_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote section_roc_suggest_results.json, total elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
