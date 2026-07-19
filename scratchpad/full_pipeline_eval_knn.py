"""full_pipeline_eval_knn.py — 2026-07-18 continuation, follow-up item 3
(lower priority per the brief, quick recheck only): does swapping the bar-
merge candidate generator in the full recommended pipeline
(full_pipeline_eval.py: intro-trim + section-merge clustering) from plain
threshold+union-find to the bakeoff-winning k-NN(k=1, floor=0.9)+connected-
components change the earlier finding that the full pipeline statistically
TIES flat block8 (+0.0050, full_pipeline_eval_results.json)?

Reuses full_pipeline_eval.py's corpus load, intro-trim logic, and V-measure
scoring UNCHANGED — only `predict_union_combined` (plain threshold+CC) is
swapped for `predict_union_knn` (k-NN edge selection + CC via
candidate_algos.gen_knn_cc/groups_from_adjacency — the SAME shared module
production candidate generation uses, per the "don't let eval and shipped
code drift" rule). k=1, floor=0.9 is the modal winning operating point from
clustering_bakeoff.py's own 5-seed sweep at grain=8 (matches 4/5 seeds
exactly; the 5th used floor=0.7) — reused directly rather than re-swept
here, since this is explicitly a quick lower-priority recheck.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from full_pipeline_eval import (
    load_corpus, GRAIN, INTRO_EDGE, INTRO_TAU, SEEDS, OUT_DIR,
)
from symstruct import vmeasure, predict_blockmatch
from intro_outro import edge_scores
from chord_distance_eval import nuclear_spans, block_sim
from candidate_algos import gen_knn_cc, groups_from_adjacency
import random

KNN_K = 1
KNN_FLOOR = 0.9


def predict_union_knn(bass, treble, n, size, k, floor):
    spans = nuclear_spans(n, size)
    bb = [bass[s:e] for (s, e) in spans]
    tb = [treble[s:e] for (s, e) in spans]
    m = len(spans)
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            sc = 0.5 * (block_sim(bb[i], bb[j]) + block_sim(tb[i], tb[j]))
            S[i, j] = S[j, i] = sc
    adj = gen_knn_cc(S, k, floor)
    groups = groups_from_adjacency(adj, min_size=2)
    lab = ["A"] * n
    gid = {}
    for gi, grp in enumerate(groups):
        for k_idx in grp:
            gid[k_idx] = gi
    next_solo = len(groups)
    for k_idx, (s, e) in enumerate(spans):
        if k_idx in gid:
            tag = "S%d" % gid[k_idx]
        else:
            tag = "S%d" % next_solo
            next_solo += 1
        for t in range(s, e):
            lab[t] = tag
    return lab


def full_pipeline_predict_knn(c):
    n = len(c["labels"])
    cluster = predict_union_knn(c["bass"], c["treble"], n, GRAIN, KNN_K, KNN_FLOOR)
    if n >= INTRO_EDGE * 4:
        lead_score, _ = edge_scores(c["vecs"], INTRO_EDGE)
        if lead_score is not None and lead_score >= INTRO_TAU:
            for i in range(min(INTRO_EDGE, n)):
                cluster[i] = "INTRO"
    return cluster


def main():
    print("Loading iReal corpus...")
    corpus = load_corpus()
    print("  corpus: %d multi-section tunes" % len(corpus))

    per_seed = []
    for seed in SEEDS:
        ids = list(range(len(corpus)))
        random.Random(seed).shuffle(ids)
        ntest = len(ids) // 5
        test = [corpus[i] for i in ids[:ntest]]

        vs_knn = [vmeasure(c["labels"], full_pipeline_predict_knn(c))[0] for c in test]
        vs_block8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0] for c in test]

        r = {"seed": seed, "n_test": len(test),
             "knn_pipeline_VF": float(np.mean(vs_knn)),
             "block8_VF": float(np.mean(vs_block8))}
        per_seed.append(r)
        print("  seed=%d  n_test=%d  knn_pipeline=%.4f  block8=%.4f" %
              (seed, len(test), r["knn_pipeline_VF"], r["block8_VF"]))

    knn_vs = [r["knn_pipeline_VF"] for r in per_seed]
    block8_vs = [r["block8_VF"] for r in per_seed]
    delta = np.mean(knn_vs) - np.mean(block8_vs)
    print("\n=== SUMMARY (5 seeds, %d-tune corpus, k=%d floor=%.2f) ===" % (len(corpus), KNN_K, KNN_FLOOR))
    print("  k-NN pipeline (intro-trim + knn_cc merge):  V_F=%.4f +- %.4f" % (np.mean(knn_vs), np.std(knn_vs)))
    print("  flat block8:                                V_F=%.4f +- %.4f" % (np.mean(block8_vs), np.std(block8_vs)))
    print("  delta (knn pipeline - block8): %+.4f  ->  %s" %
          (delta, "BEATS block8" if delta > 0.005 else
                  ("TIES block8 (within noise)" if abs(delta) >= -0.005 and delta <= 0.005 else "LOSES to block8")))

    out = {"n_corpus": len(corpus), "knn_k": KNN_K, "knn_floor": KNN_FLOOR, "per_seed": per_seed,
           "summary": {"knn_pipeline_VF_mean": float(np.mean(knn_vs)), "knn_pipeline_VF_std": float(np.std(knn_vs)),
                       "block8_VF_mean": float(np.mean(block8_vs)), "block8_VF_std": float(np.std(block8_vs)),
                       "delta_knn_vs_block8": float(delta)}}
    (OUT_DIR / "full_pipeline_eval_knn_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote full_pipeline_eval_knn_results.json")


if __name__ == "__main__":
    main()
