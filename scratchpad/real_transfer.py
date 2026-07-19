"""real_transfer.py — Step 5: transfer Steps 2/3/4's criteria (bar-merge
low-FP threshold, section detector, intro detector) to the 3 real-audio
songs, WITHOUT GT (none exists in this repo for real audio), qualitative
only. Reuses the ALREADY-COMPUTED bar_ssm_rawchroma_<song>.json matrices
(grain=1/2/4/8/16, bass_only/treble_only, verified `nuclear_spans`/
`block_sim` convention identical to the iReal-side scripts — same
similarity function, same block-span construction, so a tau learned on
iReal is directly comparable / applicable here, modulo the known
distribution-shift caveat logged in the noise-calibration entries).

Three things computed per song:
1. SECTION clustering at grain=8, combined (bass+treble averaged) sim,
   at BOTH Step 4's operating points (V-measure-optimal tau=0.78, and
   Step 2's low-FP-gated tau=0.973) — side by side, so a human can see how
   much the deployment-priority threshold under- or over-merges relative to
   the best-case one on REAL audio specifically.
2. INTRO detection at grain=2 (Step 3's winning edge size), combined sim,
   score(block0) = mean sim to all other non-adjacent same-size blocks,
   predict intro if score >= Step 3's low-FP threshold (0.666, edge_size=2,
   target_fpr=0.05, reused verbatim from known_issues.md's logged table).
3. Raw combined-sim summary stats (min/mean/p90 off-diagonal) for a sanity
   cross-check against the already-logged real-audio floor numbers.

No GT scoring anywhere in this file — outputs are inspected by a human via
the deployed debug page, per the brief.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
SONGS = ["aretha_chain_of_fools", "autumn_leaves", "abba_chiquitita"]

TAU_VMEASURE_OPTIMAL = 0.78   # Step 4 (a)
TAU_LOW_FP = 0.9728           # Step 4 (b) / Step 2 (target_fpr=0.05), reused verbatim,
                               # kept as failure-case context (see consolidation call below)
TAU_FPR010 = 0.7759           # CONSOLIDATION call 2026-07-18: Follow-up 2's interior-optimum
                               # threshold (target_fpr=0.10, mean over 5 seeds,
                               # fpr_frontier_sweep_results.json), now the new
                               # "clean-trained" REFERENCE point in place of TAU_LOW_FP,
                               # per this call's brief item 2.
INTRO_TAU_LOW_FP = 0.6662     # Step 3, edge_size=2, target_fpr=0.05
ADAPTIVE_PERCENTILE = 90      # Step 6 error-analysis fix, see adaptive_tau().
                               # Swept 80-96 across all 3 songs first (doctrine:
                               # don't pick blind) — no single percentile hits
                               # every song's earlier learned-encoder reference
                               # count (aretha~2, autumn_leaves~18, abba~17)
                               # exactly; p=90 is the best available middle
                               # ground (aretha=5, autumn_leaves=12, abba=10 —
                               # all non-degenerate, vs the fixed-tau collapse
                               # to 1 or 41). Logged as an honest compromise,
                               # not a perfect fix.


def union_find_labels(sim, tau):
    m = sim.shape[0]
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            if sim[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    remap = {}
    labels = []
    for k in range(m):
        r = find(k)
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return labels


def runs_from_labels(labels, block_size, n_bars):
    runs = []
    for k, lab in enumerate(labels):
        bar_start = k * block_size
        bar_end = min(bar_start + block_size, n_bars)
        if runs and runs[-1]["label_id"] == lab:
            runs[-1]["bar_end"] = bar_end
        else:
            runs.append({"label_id": lab, "label": "S%d" % lab,
                         "bar_start": bar_start, "bar_end": bar_end})
    return runs


def intro_score(sim2, edge_size_blocks=1):
    """sim2 = grain-2 similarity matrix (m x m, block index = 2-bar block).
    Block 0's mean similarity to every OTHER block except its immediate
    neighbor (index 1), matching Step 3's stride-1-window methodology
    approximately (here: nuclear-block granularity, the finest available
    from the pre-computed matrices)."""
    m = sim2.shape[0]
    if m < 4:
        return None
    others = [j for j in range(1, m) if j != 0]  # exclude self; block 1 is adjacent but kept (matches nuclear, not stride-1)
    # exclude immediate neighbor block 1 specifically, mirroring the min_gap=2 convention used in stat_B
    others = [j for j in range(2, m)]
    if not others:
        return None
    return float(np.mean(sim2[0, others]))


def adaptive_tau(sim, percentile, min_gap=2):
    """Per-song threshold: the P-th percentile of THIS song's own
    off-diagonal (non-adjacent) similarity distribution, instead of a fixed
    global constant carried over from iReal. Error-analysis fix (Step 6):
    the two fixed iReal-calibrated taus both fail on real audio in OPPOSITE
    directions (0.78 over-merges to 1 cluster on aretha; 0.973 under-merges
    to near-zero merges on autumn_leaves/aretha) — the diagnosed cause is
    real audio's per-song-VARYING elevated similarity floor (already
    established in the Step 1 noise-calibration entries: aretha's floor
    ~0.89 mean / abba's ~0.67 mean — almost a 0.22 gap between songs), which
    no single fixed constant can straddle. A percentile-of-own-distribution
    threshold adapts to each song's floor automatically."""
    m = sim.shape[0]
    vals = [sim[i, j] for i in range(m) for j in range(m) if abs(i - j) >= min_gap]
    if not vals:
        return None
    return float(np.percentile(vals, percentile))


def main():
    results = {}
    for song in SONGS:
        d = json.loads((OUT_DIR / ("bar_ssm_rawchroma_%s.json" % song)).read_text())
        n_bars = d["n_bars"]
        tempo = d["tempo_bpm"]
        tonic = d["est_tonic_pc"]

        sim_b8 = np.array(d["grains_bass"]["8"]["similarity_matrix"])
        sim_t8 = np.array(d["grains_treble"]["8"]["similarity_matrix"])
        sim_c8 = 0.5 * (sim_b8 + sim_t8)

        labels_opt = union_find_labels(sim_c8, TAU_VMEASURE_OPTIMAL)
        labels_fp = union_find_labels(sim_c8, TAU_LOW_FP)
        labels_010 = union_find_labels(sim_c8, TAU_FPR010)
        runs_opt = runs_from_labels(labels_opt, 8, n_bars)
        runs_fp = runs_from_labels(labels_fp, 8, n_bars)
        runs_010 = runs_from_labels(labels_010, 8, n_bars)

        # Step 6 error-analysis fix: per-song adaptive percentile threshold
        tau_adapt = adaptive_tau(sim_c8, ADAPTIVE_PERCENTILE)
        labels_adapt = union_find_labels(sim_c8, tau_adapt)
        runs_adapt = runs_from_labels(labels_adapt, 8, n_bars)

        sim_b2 = np.array(d["grains_bass"]["2"]["similarity_matrix"])
        sim_t2 = np.array(d["grains_treble"]["2"]["similarity_matrix"])
        sim_c2 = 0.5 * (sim_b2 + sim_t2)
        intro_s = intro_score(sim_c2)
        is_intro = bool(intro_s is not None and intro_s >= INTRO_TAU_LOW_FP)

        # off-diagonal sanity stats (cross-check vs noise_calibrate_results.json)
        m = sim_c8.shape[0]
        offdiag = [sim_c8[i, j] for i in range(m) for j in range(m) if abs(i - j) >= 2]

        print("=== %s ===  tempo=%.1f  n_bars=%d  tonic_pc=%d" % (song, tempo, n_bars, tonic))
        print("  V-measure-optimal tau=%.2f -> %d sections (n_blocks=%d)" %
              (TAU_VMEASURE_OPTIMAL, len(set(labels_opt)), len(labels_opt)))
        print("  low-FP (target_fpr=0.05) tau=%.4f -> %d sections (n_blocks=%d)" %
              (TAU_LOW_FP, len(set(labels_fp)), len(labels_fp)))
        print("  FPR=0.10 interior-optimum tau=%.4f -> %d sections (n_blocks=%d)" %
              (TAU_FPR010, len(set(labels_010)), len(labels_010)))
        print("  ADAPTIVE (p%d of own offdiag dist) tau=%.4f -> %d sections (n_blocks=%d)" %
              (ADAPTIVE_PERCENTILE, tau_adapt, len(set(labels_adapt)), len(labels_adapt)))
        print("  intro score(block0, grain=2)=%s  predicted_intro=%s (thr=%.4f)" %
              ("%.4f" % intro_s if intro_s is not None else "NA", is_intro, INTRO_TAU_LOW_FP))
        print("  offdiag sim: mean=%.3f min=%.3f p90=%.3f" %
              (np.mean(offdiag), np.min(offdiag), np.percentile(offdiag, 90)))

        results[song] = {
            "tempo_bpm": tempo, "n_bars": n_bars, "est_tonic_pc": tonic,
            "section_vmeasure_optimal": {"tau": TAU_VMEASURE_OPTIMAL,
                "n_sections": len(set(labels_opt)), "runs": runs_opt},
            "section_low_fp": {"tau": TAU_LOW_FP,
                "n_sections": len(set(labels_fp)), "runs": runs_fp},
            "section_fpr010": {"tau": TAU_FPR010,
                "n_sections": len(set(labels_010)), "runs": runs_010},
            "section_adaptive": {"tau": tau_adapt, "percentile": ADAPTIVE_PERCENTILE,
                "n_sections": len(set(labels_adapt)), "runs": runs_adapt},
            "intro": {"score": intro_s, "predicted_intro": is_intro,
                      "threshold": INTRO_TAU_LOW_FP},
            "offdiag_sanity": {"mean": float(np.mean(offdiag)),
                                "min": float(np.min(offdiag)),
                                "p90": float(np.percentile(offdiag, 90))},
        }

    (OUT_DIR / "real_transfer_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote real_transfer_results.json")


if __name__ == "__main__":
    main()
