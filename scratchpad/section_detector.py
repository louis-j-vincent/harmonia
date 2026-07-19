"""section_detector.py — Step 4: section-level (8-bar) detector built
DIRECTLY on Step 2's merge criterion (combined bass+treble block_sim,
union-find clustering), evaluated the same way as everything else tonight
(V-measure, corpus-scale, against flat block8 0.68-0.70 as the bar to beat
or honestly fail to beat).

Two operating points reported side by side, because they answer different
questions:
  (a) V-measure-OPTIMAL tau (val-tuned to maximize V_F directly) — "how good
      can this similarity be as a structure detector if we don't care about
      false positives at all." This is directly comparable to the existing
      V1/V2/V3 CHORD-TONE-DISTANCE entry and flat block8.
  (b) Step 2's LOW-FP-gated tau (chosen to minimize false merges, the user's
      explicit deployment priority, reused directly from
      merge_criterion_results.json, NOT re-tuned for V-measure) — "what do
      we actually get if we deploy the criterion the user asked for." This
      quantifies the V-measure COST of prioritizing low false-positive rate,
      which is the honest number that matters for a real deployment
      decision, not just the best-case number.

Same 1992-tune corpus / song-level split protocol as chord_distance_eval.py
and symstruct.py, size=8 nuclear blocks.
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
from merge_criterion import fpr_gated_threshold

OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8


def root_onehot(root_pc):
    v = np.zeros(12)
    if root_pc is not None and root_pc >= 0:
        v[root_pc % 12] = 1.0
    return v


def load_corpus(max_tunes=None):
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
    if max_tunes:
        random.Random(0).shuffle(out)
        out = out[:max_tunes]
    return out


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


def main():
    print("Loading iReal corpus (bass/treble, combined-sim criterion)...")
    corpus = load_corpus()
    print("  corpus: %d multi-section tunes" % len(corpus))

    ids = list(range(len(corpus)))
    random.Random(0).shuffle(ids)
    nval = len(ids) // 5
    val_ids, test_ids = ids[:nval], ids[nval:]
    val = [corpus[i] for i in val_ids]
    test = [corpus[i] for i in test_ids]
    print("  val=%d test=%d" % (len(val), len(test)))

    # (a) V-measure-optimal tau
    print("\n=== (a) V-measure-OPTIMAL tau ===")
    taus = np.round(np.arange(0.3, 0.99, 0.03), 2)
    best_tau, best_v = None, -1
    for tau in taus:
        vs = [vmeasure(c["labels"], predict_union_combined(
            c["bass"], c["treble"], len(c["labels"]), GRAIN, tau))[0] for c in val]
        mv = np.mean(vs)
        if mv > best_v:
            best_v, best_tau = mv, tau
    vs_test = [vmeasure(c["labels"], predict_union_combined(
        c["bass"], c["treble"], len(c["labels"]), GRAIN, best_tau))[0] for c in test]
    print("  val_tau*=%.2f  TEST V_F=%.4f (n=%d)" % (best_tau, np.mean(vs_test), len(vs_test)))

    # (b) Step 2's low-FP-gated tau, reused verbatim (not re-tuned)
    print("\n=== (b) Step 2's LOW-FP-gated tau (deployment operating point) ===")
    mc_path = OUT_DIR / "merge_criterion_results.json"
    mc = json.loads(mc_path.read_text())
    thrs = [r["threshold_only"]["thr"] for r in mc["clean"]]
    mean_thr = float(np.mean(thrs))
    print("  Step 2's per-seed low-FP thresholds: %s  mean=%.4f" %
          ([round(t, 3) for t in thrs], mean_thr))
    vs_test_fp = [vmeasure(c["labels"], predict_union_combined(
        c["bass"], c["treble"], len(c["labels"]), GRAIN, mean_thr))[0] for c in test]
    print("  tau=%.4f (low-FP)  TEST V_F=%.4f (n=%d)" % (mean_thr, np.mean(vs_test_fp), len(vs_test_fp)))
    print("  cost of prioritizing low-FP over V-measure-optimal: %.4f - %.4f = %.4f" %
          (np.mean(vs_test), np.mean(vs_test_fp), np.mean(vs_test) - np.mean(vs_test_fp)))

    # (c) FRESH CHECK, consolidation call 2026-07-18: Follow-up 2 (fpr_frontier_sweep.py)
    # found target_fpr=0.10 is an INTERIOR OPTIMUM (V_F=0.6851+-0.0151 corpus-scale,
    # 5 seeds, its OWN independent train/val/test split machinery). This block re-derives
    # the same operating point using THIS script's own val/test split (different seed/split
    # logic than fpr_frontier_sweep.py) as an independent cross-check before adopting it as
    # the deployed default -- not a repeat of the same code path.
    print("\n=== (c) FRESH CHECK: target_fpr=0.10 (Follow-up 2's newly-found optimum) ===")
    def bar_pairs(c):
        n = len(c["labels"]); spans = nuclear_spans(n, GRAIN)
        if len(spans) < 2:
            return []
        bb = [c["bass"][s:e] for (s, e) in spans]; tb = [c["treble"][s:e] for (s, e) in spans]
        from collections import Counter
        bl = [Counter(c["labels"][s:e]).most_common(1)[0][0] for (s, e) in spans]
        rows = []
        m = len(spans)
        for i in range(m):
            for j in range(i + 1, m):
                sc = 0.5 * (block_sim(bb[i], bb[j]) + block_sim(tb[i], tb[j]))
                rows.append((sc, 1 if bl[i] == bl[j] else 0))
        return rows
    Xval, yval = [], []
    for c in val:
        for sc, lab in bar_pairs(c):
            Xval.append(sc); yval.append(lab)
    thr010, _, _ = fpr_gated_threshold(np.array(Xval), np.array(yval), target_fpr=0.10)
    vs_test_010 = [vmeasure(c["labels"], predict_union_combined(
        c["bass"], c["treble"], len(c["labels"]), GRAIN, thr010))[0] for c in test]
    print("  tau=%.4f (target_fpr=0.10, this script's own split)  TEST V_F=%.4f (n=%d)" %
          (thr010, np.mean(vs_test_010), len(vs_test_010)))
    print("  cross-check vs fpr_frontier_sweep.py's 5-seed corpus number (0.6851+-0.0151): %s" %
          ("CONFIRMS (within noise)" if abs(np.mean(vs_test_010) - 0.6851) < 0.03 else "DIVERGES, investigate"))

    out = {"n_corpus": len(corpus), "n_val": len(val), "n_test": len(test),
           "vmeasure_optimal": {"tau": best_tau, "test_VF": float(np.mean(vs_test))},
           "low_fp_gated": {"tau": mean_thr, "test_VF": float(np.mean(vs_test_fp))},
           "fpr010_fresh_check": {"tau": thr010, "test_VF": float(np.mean(vs_test_010))}}
    (OUT_DIR / "section_detector_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote section_detector_results.json")


if __name__ == "__main__":
    main()
