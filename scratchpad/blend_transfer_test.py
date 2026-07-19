"""blend_transfer_test.py — Follow-up 1 of the 2026-07-18 continuation call
(3rd call tonight). Tests the one connection between tonight's two positive
results (Step 1 RETRY's floor-blend calibration; Step 2's clean-iReal merge
criterion) that was flagged as unexploited in the morning summary.

Question: does TRAINING the bar-merge threshold on floor-blend-calibrated
iReal (alpha=0.40, the value Step 1 found matches real audio's SSM
statistics) produce a threshold that transfers better to real audio than the
clean-iReal-trained one — i.e. does it need Step 6's adaptive-percentile
patch less badly?

Design (proper this time, unlike the same-call "sanity pass" that blended
train+val+test together, which the morning summary explicitly flagged as
NOT a real test):
  1. TRAIN + VAL blended at alpha=0.40 (matches how you'd actually deploy:
     you calibrate against real-audio-like statistics, but you don't get to
     "cheat" by blending the test set too).
  2. TEST stays CLEAN — this checks the blended-trained threshold doesn't
     destroy iReal-native performance (the other honest failure mode: a
     threshold tuned to an artificially-elevated floor might be too
     permissive on real (clean) structure).
  3. Apply the resulting FIXED low-FP threshold to the exact same 3 real
     songs from real_transfer.py (same bar_ssm_rawchroma_*.json inputs,
     same union-find code), and compare section counts against the
     ALREADY-LOGGED clean-trained-tau failure modes (0.78 -> 1 section
     collapse on aretha; 0.973 -> 41/41 zero-merge on autumn_leaves).
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from noise_calibrate import load_corpus_registers, N_CORPUS_SAMPLE
from merge_criterion import (corpus_generic_vector, blend, build_pairs,
                              fpr_gated_threshold, eval_threshold_only, GRAIN)
from real_transfer import union_find_labels, runs_from_labels, SONGS, OUT_DIR as RT_DIR

OUT_DIR = Path(__file__).resolve().parent
FLOOR_ALPHA = 0.40
TARGET_FPR = 0.05


def make_dataset_mixed(corpus_ids, corpus, blend_flag, generic):
    X, y = [], []
    for i in corpus_ids:
        c = corpus[i]
        rows = build_pairs(c, use_floor=blend_flag, generic=generic, alpha=FLOOR_ALPHA)
        for r in rows:
            X.append(r[:5]); y.append(r[5])
    return np.array(X), np.array(y)


def run_one_seed(corpus, seed, generic):
    ids = list(range(len(corpus)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    tr_ids, val_ids, te_ids = ids[:n_tr], ids[n_tr:n_tr + n_val], ids[n_tr + n_val:]

    # TRAIN+VAL blended, TEST clean (the proper asymmetric design)
    Xtr, ytr = make_dataset_mixed(tr_ids, corpus, True, generic)
    Xval, yval = make_dataset_mixed(val_ids, corpus, True, generic)
    Xte_clean, yte_clean = make_dataset_mixed(te_ids, corpus, False, generic)

    thr_blend, _, _ = fpr_gated_threshold(Xval[:, 2], yval, TARGET_FPR)
    # eval on CLEAN test -> does it hurt iReal-native performance?
    p, r, f = eval_threshold_only(Xte_clean, yte_clean, thr_blend)

    # also eval the CLEAN-trained threshold (reproduced from merge_criterion.py's
    # own protocol) on the same clean test split, for an apples-to-apples delta
    Xtr_c, ytr_c = make_dataset_mixed(tr_ids, corpus, False, generic)
    Xval_c, yval_c = make_dataset_mixed(val_ids, corpus, False, generic)
    thr_clean, _, _ = fpr_gated_threshold(Xval_c[:, 2], yval_c, TARGET_FPR)
    p_c, r_c, f_c = eval_threshold_only(Xte_clean, yte_clean, thr_clean)

    return {
        "thr_blend_trained": thr_blend, "eval_on_clean_test": {"precision": p, "recall": r, "fpr": f},
        "thr_clean_trained": thr_clean, "clean_trained_eval_on_clean_test": {"precision": p_c, "recall": r_c, "fpr": f_c},
    }


def apply_tau_to_real_songs(tau, tag):
    out = {}
    for song in SONGS:
        d = json.loads((RT_DIR / ("bar_ssm_rawchroma_%s.json" % song)).read_text())
        n_bars = d["n_bars"]
        sim_b8 = np.array(d["grains_bass"]["8"]["similarity_matrix"])
        sim_t8 = np.array(d["grains_treble"]["8"]["similarity_matrix"])
        sim_c8 = 0.5 * (sim_b8 + sim_t8)
        labels = union_find_labels(sim_c8, tau)
        n_sections = len(set(labels))
        n_blocks = len(labels)
        out[song] = {"tau": tau, "n_sections": n_sections, "n_blocks": n_blocks,
                     "degenerate": (n_sections == 1 or n_sections == n_blocks)}
        print("  [%s] tau=%.4f song=%-24s -> %d sections / %d blocks  degenerate=%s" %
              (tag, tau, song, n_sections, n_blocks, out[song]["degenerate"]))
    return out


def main():
    print("Loading iReal corpus...")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE * 3)
    print("  corpus: %d multi-section tunes" % len(corpus))
    generic = {"bass": corpus_generic_vector(corpus, "bass"),
               "treble": corpus_generic_vector(corpus, "treble")}

    seeds = [0, 1, 2, 3, 4]
    results = []
    for seed in seeds:
        r = run_one_seed(corpus, seed, generic)
        results.append(r)
        print("seed=%d  blend-trained thr=%.4f -> P=%.3f R=%.3f FPR=%.3f (on CLEAN test) | "
              "clean-trained thr=%.4f -> P=%.3f R=%.3f FPR=%.3f" %
              (seed, r["thr_blend_trained"], r["eval_on_clean_test"]["precision"],
               r["eval_on_clean_test"]["recall"], r["eval_on_clean_test"]["fpr"],
               r["thr_clean_trained"], r["clean_trained_eval_on_clean_test"]["precision"],
               r["clean_trained_eval_on_clean_test"]["recall"], r["clean_trained_eval_on_clean_test"]["fpr"]))

    thr_blend_mean = float(np.mean([r["thr_blend_trained"] for r in results]))
    thr_clean_mean = float(np.mean([r["thr_clean_trained"] for r in results]))
    rec_blend_mean = float(np.mean([r["eval_on_clean_test"]["recall"] for r in results]))
    rec_clean_mean = float(np.mean([r["clean_trained_eval_on_clean_test"]["recall"] for r in results]))
    prec_blend_mean = float(np.mean([r["eval_on_clean_test"]["precision"] for r in results]))
    prec_clean_mean = float(np.mean([r["clean_trained_eval_on_clean_test"]["precision"] for r in results]))

    print("\n=== SUMMARY (5 seeds) ===")
    print("  blend-trained (alpha=%.2f) mean thr=%.4f  | on clean test: P=%.3f R=%.3f" %
          (FLOOR_ALPHA, thr_blend_mean, prec_blend_mean, rec_blend_mean))
    print("  clean-trained             mean thr=%.4f  | on clean test: P=%.3f R=%.3f" %
          (thr_clean_mean, prec_clean_mean, rec_clean_mean))

    print("\n=== Apply FIXED thresholds to the 3 real songs (grain=8, combined sim) ===")
    print("--- clean-trained threshold (known collapse/under-merge failure, reused for comparison) ---")
    real_clean = apply_tau_to_real_songs(thr_clean_mean, "clean-trained")
    print("--- blend-trained threshold (this call's hypothesis) ---")
    real_blend = apply_tau_to_real_songs(thr_blend_mean, "blend-trained")

    out = {"floor_alpha": FLOOR_ALPHA, "target_fpr": TARGET_FPR,
           "per_seed": results,
           "summary": {"thr_blend_trained_mean": thr_blend_mean, "thr_clean_trained_mean": thr_clean_mean,
                       "clean_test_eval": {"blend_trained": {"precision": prec_blend_mean, "recall": rec_blend_mean},
                                            "clean_trained": {"precision": prec_clean_mean, "recall": rec_clean_mean}}},
           "real_audio_transfer": {"clean_trained_tau": real_clean, "blend_trained_tau": real_blend}}
    (OUT_DIR / "blend_transfer_test_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote blend_transfer_test_results.json")


if __name__ == "__main__":
    main()
