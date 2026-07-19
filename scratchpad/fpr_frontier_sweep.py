"""fpr_frontier_sweep.py — Follow-up 2 of the 3rd 2026-07-18 continuation
call. The prior call only reported two operating points for the section
detector (V-measure-optimal tau=0.78 -> V_F=0.682, and Step 2's low-FP
tau=0.973 -> V_F=0.638, a single FIXED comparison). This sweeps the actual
FPR-gate TARGET itself (the knob a deployer would turn), not just those two
endpoints, so there's a real precision/recall-vs-V-measure frontier to
choose from.

For each target_fpr in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}:
  1. Compute the FPR-gated bar-merge threshold on iReal train+val (Step 2's
     protocol, reused verbatim from merge_criterion.py).
  2. Evaluate that threshold's SECTION-LEVEL V-measure on held-out iReal
     test (Step 4's protocol, reused verbatim from section_detector.py).
  3. Report V_F, and the underlying bar-pair P/R/FPR that produced the tau,
     so the tradeoff is visible end to end (not just V_F in isolation).

Multi-seed (5 seeds) for every target_fpr — this is a corpus-scale sweep,
not a single-song result (rule #5).
"""
from __future__ import annotations
import sys, io, json, random
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import chord_vector_binary
from chord_distance_eval import nuclear_spans, block_sim, FILES
from symstruct import vmeasure, qbucket
from merge_criterion import fpr_gated_threshold, eval_threshold_only

OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8
FPR_TARGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SEEDS = [0, 1, 2, 3, 4]


def root_onehot(root_pc):
    v = np.zeros(12)
    if root_pc is not None and root_pc >= 0:
        v[root_pc % 12] = 1.0
    return v


def load_corpus():
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(Path("data/ireal") / (f + ".txt"))
        except Exception:
            continue
        for t in tunes:
            try:
                mc = tune_to_mma(t)
            except Exception:
                continue
            shift = 0
            if mc.key:
                pc = chord_root_pc(mc.key.rstrip("-"))
                shift = (-pc % 12) if pc is not None else 0
            bass_vecs, treb_vecs, labels = [], [], []
            for bar_no, section, slots in mc.timeline:
                bass_accum, treb_accum = None, None
                for (_, _, mma) in slots:
                    pc = chord_root_pc(mma)
                    if pc is None:
                        continue
                    rpc = (pc + shift) % 12
                    q = qbucket(mma)
                    bv = root_onehot(rpc)
                    tv = chord_vector_binary(rpc, q)
                    bass_accum = bv if bass_accum is None else bass_accum + bv
                    treb_accum = tv if treb_accum is None else treb_accum + tv
                bass_vecs.append(bass_accum if bass_accum is not None else np.zeros(12))
                treb_vecs.append(treb_accum if treb_accum is not None else np.zeros(12))
                labels.append(section)
            if len(labels) < GRAIN * 2 or len(set(labels)) < 2:
                continue
            out.append({"title": mc.title, "bass": bass_vecs, "treble": treb_vecs, "labels": labels})
    return out


def build_pairs_for_song(c, grain=GRAIN):
    n = len(c["labels"])
    spans = nuclear_spans(n, grain)
    if len(spans) < 2:
        return []
    bass, treb = c["bass"], c["treble"]
    block_bass = [bass[s:e] for (s, e) in spans]
    block_treb = [treb[s:e] for (s, e) in spans]
    from collections import Counter
    block_labels = [Counter(c["labels"][s:e]).most_common(1)[0][0] for (s, e) in spans]
    m = len(spans)
    rows = []
    for i in range(m):
        for j in range(i + 1, m):
            sb = block_sim(block_bass[i], block_bass[j])
            st = block_sim(block_treb[i], block_treb[j])
            sc = 0.5 * (sb + st)
            label = 1 if block_labels[i] == block_labels[j] else 0
            rows.append((sc, label))
    return rows


def make_bar_pair_dataset(ids, corpus):
    X, y = [], []
    for i in ids:
        for sc, label in build_pairs_for_song(corpus[i]):
            X.append(sc); y.append(label)
    return np.array(X), np.array(y)


def predict_union_combined(bass, treble, n, size, tau):
    spans = nuclear_spans(n, size)
    bb = [bass[s:e] for (s, e) in spans]
    tb = [treble[s:e] for (s, e) in spans]
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            sim = 0.5 * (block_sim(bb[i], bb[j]) + block_sim(tb[i], tb[j]))
            if sim >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def run_one_seed_one_fpr(corpus, seed, target_fpr):
    ids = list(range(len(corpus)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    tr_ids, val_ids, te_ids = ids[:n_tr], ids[n_tr:n_tr + n_val], ids[n_tr + n_val:]

    Xval, yval = make_bar_pair_dataset(val_ids, corpus)
    thr, _, _ = fpr_gated_threshold(Xval, yval, target_fpr)

    Xte, yte = make_bar_pair_dataset(te_ids, corpus)
    pred = (Xte >= thr).astype(int)
    tp = np.sum((pred == 1) & (yte == 1)); fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1)); tn = np.sum((pred == 0) & (yte == 0))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)

    test_songs = [corpus[i] for i in te_ids]
    vs = [vmeasure(c["labels"], predict_union_combined(
        c["bass"], c["treble"], len(c["labels"]), GRAIN, thr))[0] for c in test_songs]
    v_f = float(np.mean(vs))

    return {"thr": thr, "bar_pair_precision": prec, "bar_pair_recall": rec,
            "bar_pair_fpr": fpr, "section_V_F": v_f, "n_test_songs": len(test_songs)}


def main():
    print("Loading iReal corpus...")
    corpus = load_corpus()
    print("  corpus: %d multi-section tunes" % len(corpus))

    all_results = {}
    for target_fpr in FPR_TARGETS:
        print("\n=== target_fpr=%.2f ===" % target_fpr)
        seed_results = []
        for seed in SEEDS:
            r = run_one_seed_one_fpr(corpus, seed, target_fpr)
            seed_results.append(r)
            print("  seed=%d thr=%.4f  bar-pair P=%.3f R=%.3f FPR=%.3f | section V_F=%.4f" %
                  (seed, r["thr"], r["bar_pair_precision"], r["bar_pair_recall"],
                   r["bar_pair_fpr"], r["section_V_F"]))
        vfs = [r["section_V_F"] for r in seed_results]
        thrs = [r["thr"] for r in seed_results]
        recs = [r["bar_pair_recall"] for r in seed_results]
        precs = [r["bar_pair_precision"] for r in seed_results]
        fprs = [r["bar_pair_fpr"] for r in seed_results]
        summary = {"mean_thr": float(np.mean(thrs)), "mean_V_F": float(np.mean(vfs)),
                   "std_V_F": float(np.std(vfs)), "mean_bar_pair_recall": float(np.mean(recs)),
                   "mean_bar_pair_precision": float(np.mean(precs)), "mean_bar_pair_fpr": float(np.mean(fprs)),
                   "per_seed": seed_results}
        all_results[str(target_fpr)] = summary
        print("  SUMMARY target_fpr=%.2f: mean thr=%.4f  V_F=%.4f+-%.4f  bar-pair R=%.3f P=%.3f actual-FPR=%.3f" %
              (target_fpr, summary["mean_thr"], summary["mean_V_F"], summary["std_V_F"],
               summary["mean_bar_pair_recall"], summary["mean_bar_pair_precision"], summary["mean_bar_pair_fpr"]))

    print("\n=== FULL FRONTIER (target_fpr -> V_F) ===")
    for target_fpr in FPR_TARGETS:
        s = all_results[str(target_fpr)]
        print("  target_fpr=%.2f  ->  V_F=%.4f (thr=%.4f, bar-recall=%.3f)" %
              (target_fpr, s["mean_V_F"], s["mean_thr"], s["mean_bar_pair_recall"]))

    (OUT_DIR / "fpr_frontier_sweep_results.json").write_text(json.dumps(all_results, indent=2))
    print("\nwrote fpr_frontier_sweep_results.json")


if __name__ == "__main__":
    main()
