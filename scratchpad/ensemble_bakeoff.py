"""ensemble_bakeoff.py — 2026-07-18 continuation, follow-up item 1: does the
UNION of k-NN's and agglomerative_complete's candidate edges beat either
algorithm alone, at a matched FPR<=0.05 operating point?

Reuses clustering_bakeoff.py's exact harness (same corpus load, same
per-tune block_sim matrices, same pooled precision/recall/FPR protocol, same
5 song-level splits) — imports its helpers rather than reimplementing them,
per CLAUDE.md "don't let eval code and shipped code drift" discipline (same
spirit extended to eval-vs-eval here).

Method: for each of k-NN and agglomerative_complete, take their ALREADY
FPR-gated (val, target_fpr=0.05) knob from clustering_bakeoff's own
selection (single independent operating point each), union their two
predicted-adjacency matrices, then check the UNION's FPR on val. Two cases:
  (a) if the naive union of both individually-safe operating points is
      itself still <=0.05 FPR on val, that's a free win to report directly.
  (b) if not (very likely — combining two candidate sources roughly adds
      their false-positive rates), re-derive a joint operating point: sweep
      BOTH knobs jointly (not independently) and pick the joint combo with
      highest val recall s.t. union FPR<=0.05, exactly analogous to how
      each single algorithm was gated. This is the "re-threshold the union
      appropriately" the brief asked for, not a naive unsafe union.
Report both, and the honest test-set numbers at the properly-gated joint
operating point, against each algorithm's own solo number.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from noise_calibrate import load_corpus_registers, N_CORPUS_SAMPLE, GRAIN
from clustering_bakeoff import (
    prep_corpus, split_corpus, pair_indices, pooled_prf,
    gen_knn_cc, gen_agglomerative, SEEDS, TARGET_FPR, OUT_DIR,
)

KNN_KS = (1, 2, 3, 5)
KNN_FLOORS = (0.3, 0.5, 0.7)
AGG_THRESHES = np.round(np.arange(0.02, 0.85, 0.02), 3)


def union_adj(S, knn_knob, agg_thresh):
    k, floor = knn_knob
    a1 = gen_knn_cc(S, k, floor)
    a2 = gen_agglomerative(S, "complete", agg_thresh)
    return a1 | a2


def pooled_prf_union(blocks, knn_knob, agg_thresh):
    return pooled_prf(blocks, lambda S: union_adj(S, knn_knob, agg_thresh))


def solo_best(val_blocks, gen_fn_factory, knobs, target_fpr=TARGET_FPR):
    """Same selection rule as clustering_bakeoff.fpr_gate_select but knobs
    can be composite (tuples) — returns (knob, prec, rec, fpr)."""
    best = None
    fallback = None
    for kn in knobs:
        gen_fn = gen_fn_factory(kn)
        p, r, f, _ = pooled_prf(val_blocks, gen_fn)
        row = (kn, p, r, f)
        if fallback is None or f < fallback[3]:
            fallback = row
        if f <= target_fpr and (best is None or r > best[2]):
            best = row
    return best if best is not None else fallback


def knn_solo_op(val_blocks):
    """EXACT replica of clustering_bakeoff.run_seed's knn_cc selection: a
    primary grid (floor in {0.3,0.5,0.7}) picking max-recall-at-FPR<=target,
    falling back to a WIDER grid (floor in {0.3,0.5,0.7,0.9}) picking
    min-FPR only if the primary grid finds nothing safe. Reimplemented here
    (not imported) because clustering_bakeoff.run_seed inlines this logic
    rather than exposing it as a standalone function — kept byte-for-byte
    equivalent so this script's knn_solo reference number reproduces the
    already-published 0.217+-0.031 recall baseline as a harness sanity
    check before trusting the ensemble comparison built on top of it."""
    best_overall = None
    for k in KNN_KS:
        for floor in KNN_FLOORS:
            gen_fn = (lambda S, kk=k, ff=floor: gen_knn_cc(S, kk, ff))
            p_, r_, f_, _ = pooled_prf(val_blocks, gen_fn)
            if f_ <= TARGET_FPR and (best_overall is None or r_ > best_overall[0]):
                best_overall = (r_, k, floor, p_, f_)
    if best_overall is None:
        best_fpr = None
        for k in KNN_KS:
            for floor in (0.3, 0.5, 0.7, 0.9):
                gen_fn = (lambda S, kk=k, ff=floor: gen_knn_cc(S, kk, ff))
                p_, r_, f_, _ = pooled_prf(val_blocks, gen_fn)
                if best_fpr is None or f_ < best_fpr[4]:
                    best_fpr = (r_, k, floor, p_, f_)
        best_overall = best_fpr
    _, k, floor, _, _ = best_overall
    return (k, floor)


def run_seed(blocks, seed):
    tr, val, test = split_corpus(blocks, seed)

    # --- each algorithm's OWN solo FPR-gated operating point (for reference) ---
    knn_knob = knn_solo_op(val)
    agg_op = solo_best(val, lambda t: (lambda S: gen_agglomerative(S, "complete", t)), AGG_THRESHES)
    agg_thresh, _, _, _ = agg_op
    p_knn, r_knn, f_knn, _ = pooled_prf(test, lambda S: gen_knn_cc(S, knn_knob[0], knn_knob[1]))
    p_agg, r_agg, f_agg, _ = pooled_prf(test, lambda S: gen_agglomerative(S, "complete", agg_thresh))

    # --- (a) naive union of the two INDEPENDENTLY-safe solo operating points ---
    p_naive, r_naive, f_naive, _ = pooled_prf_union(val, knn_knob, agg_thresh)
    naive_safe = f_naive <= TARGET_FPR
    if naive_safe:
        p_naive_te, r_naive_te, f_naive_te, _ = pooled_prf_union(test, knn_knob, agg_thresh)
    else:
        p_naive_te = r_naive_te = f_naive_te = None

    # --- (b) joint re-threshold: sweep BOTH knobs together, pick best val
    # recall s.t. union FPR<=target. Coarser grids than each solo sweep to
    # keep the joint grid (knn_knobs x agg_threshes) tractable. ---
    joint_knn_knobs = [(k, f) for k in KNN_KS for f in (0.5, 0.7, 0.9)]
    joint_agg_threshes = np.round(np.arange(0.02, 0.85, 0.05), 3)
    best = None
    fallback = None
    for kn in joint_knn_knobs:
        for at in joint_agg_threshes:
            p, r, f, _ = pooled_prf_union(val, kn, at)
            row = (kn, at, p, r, f)
            if fallback is None or f < fallback[4]:
                fallback = row
            if f <= TARGET_FPR and (best is None or r > best[3]):
                best = row
    joint_op = best if best is not None else fallback
    j_knn, j_agg, _, _, _ = joint_op
    p_joint_te, r_joint_te, f_joint_te, cnt_joint = pooled_prf_union(test, j_knn, j_agg)

    return {
        "knn_solo": {"knob": {"k": knn_knob[0], "floor": knn_knob[1]},
                     "precision": p_knn, "recall": r_knn, "fpr": f_knn},
        "agg_complete_solo": {"knob": {"dist_thresh": float(agg_thresh)},
                               "precision": p_agg, "recall": r_agg, "fpr": f_agg},
        "union_naive": {"val_fpr": f_naive, "safe_at_target": bool(naive_safe),
                         "test_precision": p_naive_te, "test_recall": r_naive_te,
                         "test_fpr": f_naive_te},
        "union_joint_rethreshold": {
            "knn_knob": {"k": j_knn[0], "floor": j_knn[1]},
            "agg_thresh": float(j_agg),
            "precision": p_joint_te, "recall": r_joint_te, "fpr": f_joint_te, **cnt_joint,
        },
    }


def main():
    t0 = time.time()
    print("Loading iReal corpus (bass/treble proxies)...")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE * 3)
    print("  corpus: %d multi-section tunes" % len(corpus))
    blocks = prep_corpus(corpus)
    print("  %d tunes usable (>=2 blocks); elapsed %.1fs" % (len(blocks), time.time() - t0))

    all_rows = defaultdict(list)
    for seed in SEEDS:
        t1 = time.time()
        r = run_seed(blocks, seed)
        for k, v in r.items():
            all_rows[k].append(v)
        print("\n=== seed=%d (%.1fs) ===" % (seed, time.time() - t1))
        print("  knn_solo:              R=%.3f P=%.3f FPR=%.3f knob=%s" %
              (r["knn_solo"]["recall"], r["knn_solo"]["precision"], r["knn_solo"]["fpr"], r["knn_solo"]["knob"]))
        print("  agg_complete_solo:     R=%.3f P=%.3f FPR=%.3f knob=%s" %
              (r["agg_complete_solo"]["recall"], r["agg_complete_solo"]["precision"], r["agg_complete_solo"]["fpr"], r["agg_complete_solo"]["knob"]))
        un = r["union_naive"]
        print("  union_naive:           val_fpr=%.3f safe=%s test_R=%s test_P=%s test_fpr=%s" %
              (un["val_fpr"], un["safe_at_target"], un["test_recall"], un["test_precision"], un["test_fpr"]))
        uj = r["union_joint_rethreshold"]
        print("  union_joint_rethresh:  R=%.3f P=%.3f FPR=%.3f knn=%s agg=%.3f" %
              (uj["recall"], uj["precision"], uj["fpr"], uj["knn_knob"], uj["agg_thresh"]))

    print("\n=== SUMMARY (%d seeds, target_fpr<=%.2f) ===" % (len(SEEDS), TARGET_FPR))
    summary = {}
    for key in ("knn_solo", "agg_complete_solo", "union_joint_rethreshold"):
        recs = [x["recall"] for x in all_rows[key]]
        precs = [x["precision"] for x in all_rows[key]]
        fprs = [x["fpr"] for x in all_rows[key]]
        summary[key] = {"recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
                         "precision_mean": float(np.mean(precs)), "precision_std": float(np.std(precs)),
                         "fpr_mean": float(np.mean(fprs)), "fpr_std": float(np.std(fprs))}
        print("  %-24s recall=%.3f+-%.3f  precision=%.3f+-%.3f  fpr=%.3f+-%.3f" %
              (key, summary[key]["recall_mean"], summary[key]["recall_std"],
               summary[key]["precision_mean"], summary[key]["precision_std"],
               summary[key]["fpr_mean"], summary[key]["fpr_std"]))
    n_naive_safe = sum(1 for x in all_rows["union_naive"] if x["safe_at_target"])
    print("  union_naive safe-at-target in %d/%d seeds" % (n_naive_safe, len(SEEDS)))

    out = {"seeds": SEEDS, "target_fpr": TARGET_FPR, "per_seed": {k: v for k, v in all_rows.items()},
           "summary": summary, "n_naive_safe_seeds": n_naive_safe, "n_tunes": len(blocks), "grain": GRAIN}
    (OUT_DIR / "ensemble_bakeoff_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote ensemble_bakeoff_results.json")
    print("total elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
