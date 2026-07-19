"""full_pipeline_eval.py — Consolidation call (2026-07-18, 4th continuation),
brief item 3: THE number that actually answers "what's the best approach" —
not the section detector alone (section_detector.py, already reported), but
the FULL recommended pipeline: Step 3's validated intro detector (edge=2,
target_fpr=0.05, tau=0.6662, precision 2.5x base rate) trims/labels the
leading edge, then Follow-up 2's FPR=0.10 interior-optimum bar-merge
threshold (tau=0.7759, mean over 5 seeds, fpr_frontier_sweep_results.json)
clusters the remaining bars via the same union-find criterion as
section_detector.py — evaluated end-to-end against iReal per-bar section GT
with V-measure, corpus-scale (~1989 multi-section tunes), 5 seeds, and
compared directly against flat block8 (symstruct.py's predict_blockmatch,
the standing 0.68-0.70 reference all night) ON THE SAME SONGS/SPLITS in the
same script (not a cross-script number comparison).

Outro trimming is DELIBERATELY OMITTED: intro_outro.py already established
no reliable outro/coda marker survives iReal's sectionizer (checked and
logged 2026-07-18), so an "outro-trimmed" pipeline stage would be
unvalidated by construction — including it would silently violate rule #3
(ground truth is a measurement, don't build on a signal that isn't there).
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
from symstruct import vmeasure, qbucket, bar_features, predict_blockmatch
from intro_outro import edge_scores

OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8
INTRO_EDGE = 2
INTRO_TAU = 0.6662     # Step 3, target_fpr=0.05, edge_size=2
SECTION_TAU = 0.7759   # Follow-up 2's interior optimum, target_fpr=0.10, 5-seed mean
SEEDS = [0, 1, 2, 3, 4]


def root_onehot(root_pc):
    v = np.zeros(12)
    if root_pc is not None and root_pc >= 0:
        v[root_pc % 12] = 1.0
    return v


def load_corpus():
    """One pass over the corpus building all 3 representations needed
    (bass/treble for the merge criterion + intro detector's combined vecs,
    feat for block8) from the SAME tune object, so every method is compared
    on identical songs."""
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
            feat, feat_labels = bar_features(mc)
            if len(feat_labels) != len(labels):
                continue  # alignment guard, drop mismatches rather than silently mis-pairing
            out.append({"title": mc.title, "bass": bass_vecs, "treble": treb_vecs,
                        "vecs": treb_vecs, "labels": labels, "feat": feat})
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


def full_pipeline_predict(c):
    n = len(c["labels"])
    cluster = predict_union_combined(c["bass"], c["treble"], n, GRAIN, SECTION_TAU)
    if n >= INTRO_EDGE * 4:
        lead_score, _ = edge_scores(c["vecs"], INTRO_EDGE)
        if lead_score is not None and lead_score >= INTRO_TAU:
            for i in range(min(INTRO_EDGE, n)):
                cluster[i] = "INTRO"
    return cluster


def main():
    print("Loading iReal corpus (full-pipeline eval: intro-trim + FPR=0.10 merge)...")
    corpus = load_corpus()
    print("  corpus: %d multi-section tunes" % len(corpus))

    n_intro_gt = sum(1 for c in corpus if c["labels"][0] == "i")
    print("  (%d/%d tunes have a real GT intro label on bar 0)" % (n_intro_gt, len(corpus)))

    per_seed = []
    for seed in SEEDS:
        ids = list(range(len(corpus)))
        random.Random(seed).shuffle(ids)
        ntest = len(ids) // 5
        test_ids = ids[:ntest]
        test = [corpus[i] for i in test_ids]

        vs_full = [vmeasure(c["labels"], full_pipeline_predict(c))[0] for c in test]
        vs_block8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0] for c in test]
        vs_section_only = [vmeasure(c["labels"], predict_union_combined(
            c["bass"], c["treble"], len(c["labels"]), GRAIN, SECTION_TAU))[0] for c in test]

        r = {"seed": seed, "n_test": len(test),
             "full_pipeline_VF": float(np.mean(vs_full)),
             "block8_VF": float(np.mean(vs_block8)),
             "section_only_VF": float(np.mean(vs_section_only))}
        per_seed.append(r)
        print("  seed=%d  n_test=%d  full_pipeline=%.4f  block8=%.4f  section_only(no intro trim)=%.4f" %
              (seed, len(test), r["full_pipeline_VF"], r["block8_VF"], r["section_only_VF"]))

    full_vs = [r["full_pipeline_VF"] for r in per_seed]
    block8_vs = [r["block8_VF"] for r in per_seed]
    section_only_vs = [r["section_only_VF"] for r in per_seed]
    print("\n=== SUMMARY (5 seeds, %d-tune corpus) ===" % len(corpus))
    print("  full pipeline (intro-trim + FPR=0.10 merge):  V_F=%.4f +- %.4f" %
          (np.mean(full_vs), np.std(full_vs)))
    print("  flat block8 (standing reference):              V_F=%.4f +- %.4f" %
          (np.mean(block8_vs), np.std(block8_vs)))
    print("  section-only (FPR=0.10 merge, NO intro trim):  V_F=%.4f +- %.4f" %
          (np.mean(section_only_vs), np.std(section_only_vs)))
    delta = np.mean(full_vs) - np.mean(block8_vs)
    print("  delta (full pipeline - block8): %+.4f  ->  %s" %
          (delta, "BEATS block8" if delta > 0.005 else
                  ("TIES block8 (within noise)" if abs(delta) <= 0.005 else "LOSES to block8")))

    out = {"n_corpus": len(corpus), "n_intro_gt": n_intro_gt, "per_seed": per_seed,
           "summary": {"full_pipeline_VF_mean": float(np.mean(full_vs)), "full_pipeline_VF_std": float(np.std(full_vs)),
                       "block8_VF_mean": float(np.mean(block8_vs)), "block8_VF_std": float(np.std(block8_vs)),
                       "section_only_VF_mean": float(np.mean(section_only_vs)), "section_only_VF_std": float(np.std(section_only_vs)),
                       "delta_full_vs_block8": float(delta)}}
    (OUT_DIR / "full_pipeline_eval_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote full_pipeline_eval_results.json")


if __name__ == "__main__":
    main()
