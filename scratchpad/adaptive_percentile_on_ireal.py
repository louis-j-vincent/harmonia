"""adaptive_percentile_on_ireal.py — Follow-up 3 of the 3rd 2026-07-18
continuation call. Step 6 (prior call) introduced a per-song adaptive
threshold (tau = P-th percentile of THAT SONG's own off-diagonal
similarity distribution) as a real-audio-transfer fix, validated only as
"non-degenerate + roughly plausible" (no GT exists for real audio). Since
GT DOES exist for iReal, this stress-tests whether the adaptive-percentile
IDEA is a generally good thresholding strategy (should not hurt iReal) or a
real-audio-specific patch (would hurt iReal, because iReal's off-diagonal
floor is much less song-dependent — Step 1 established iReal's own floor is
LOWER and more uniform than real audio's, so pegging tau to "the 90th
percentile of MY OWN song" is a much bigger relative move on iReal, where
most pairs are already dissimilar, than on real audio, where most pairs are
already very similar).

Method: for each test-set iReal tune (same corpus/split protocol as Step 4 /
Follow-up 2), compute grain=8 combined bass+treble block_sim matrix (same
function, same convention as everywhere else tonight), then for a sweep of
percentiles P in {50,60,70,75,80,85,88,90,92,94,96,98}, set
tau = percentile-P of that song's own off-diagonal (non-adjacent, min_gap=2)
similarity values, cluster via union-find, and score V-measure against GT
section labels. Report per-P mean V_F and degeneracy rate (fraction of
songs collapsing to 1 section or to all-singletons), and compare the BEST
adaptive-P V_F against the FIXED-tau references already established
(V-measure-optimal fixed tau=0.78 -> V_F=0.682 single-tune; Follow-up 2's
best fixed-tau frontier point, target_fpr=0.10 -> V_F=0.6851).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from fpr_frontier_sweep import load_corpus, build_pairs_for_song  # reuse loader
from chord_distance_eval import nuclear_spans, block_sim
from symstruct import vmeasure
import random

OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8
PERCENTILES = [50, 60, 70, 75, 80, 85, 88, 90, 92, 94, 96, 98]
SEEDS = [0, 1, 2, 3, 4]


def block_sim_matrix(c, grain=GRAIN):
    n = len(c["labels"])
    spans = nuclear_spans(n, grain)
    bass, treb = c["bass"], c["treble"]
    block_bass = [bass[s:e] for (s, e) in spans]
    block_treb = [treb[s:e] for (s, e) in spans]
    m = len(spans)
    sim = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            sb = block_sim(block_bass[i], block_bass[j])
            st = block_sim(block_treb[i], block_treb[j])
            v = 0.5 * (sb + st)
            sim[i, j] = sim[j, i] = v
    return sim, spans


def adaptive_tau(sim, percentile, min_gap=2):
    m = sim.shape[0]
    vals = [sim[i, j] for i in range(m) for j in range(m) if abs(i - j) >= min_gap]
    if not vals:
        return None
    return float(np.percentile(vals, percentile))


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


def expand_block_labels_to_bars(block_labels, spans, n_bars):
    lab = ["A"] * n_bars
    for k, (s, e) in enumerate(spans):
        for t in range(s, e):
            lab[t] = "S%d" % block_labels[k]
    return lab


def main():
    print("Loading iReal corpus...")
    corpus = load_corpus()
    print("  corpus: %d multi-section tunes" % len(corpus))

    # use same held-out TEST split as Follow-up 2 / section_detector (seed=0 for
    # single-seed inspection, then multi-seed for the summary numbers)
    all_seed_results = {p: [] for p in PERCENTILES}
    all_seed_degen = {p: [] for p in PERCENTILES}

    for seed in SEEDS:
        ids = list(range(len(corpus)))
        random.Random(seed).shuffle(ids)
        n = len(ids)
        n_tr, n_val = int(n * 0.6), int(n * 0.2)
        te_ids = ids[n_tr + n_val:]
        test_songs = [corpus[i] for i in te_ids]

        # precompute sim matrices once per song (reused across percentiles)
        sims = []
        for c in test_songs:
            sim, spans = block_sim_matrix(c)
            if sim.shape[0] < 2:
                continue
            sims.append((c, sim, spans))

        for p in PERCENTILES:
            vs = []
            n_degen = 0
            for c, sim, spans in sims:
                tau = adaptive_tau(sim, p)
                if tau is None:
                    continue
                block_labels = union_find_labels(sim, tau)
                n_sec = len(set(block_labels))
                n_blocks = len(block_labels)
                if n_sec == 1 or n_sec == n_blocks:
                    n_degen += 1
                bar_labels = expand_block_labels_to_bars(block_labels, spans, len(c["labels"]))
                v, _, _ = vmeasure(c["labels"], bar_labels)
                vs.append(v)
            mean_v = float(np.mean(vs)) if vs else None
            all_seed_results[p].append(mean_v)
            all_seed_degen[p].append(n_degen / max(len(sims), 1))
        print("  seed=%d done (n_test_songs=%d)" % (seed, len(sims)))

    print("\n=== ADAPTIVE-PERCENTILE ON iREal: mean V_F by percentile (5 seeds) ===")
    summary = {}
    for p in PERCENTILES:
        vfs = [v for v in all_seed_results[p] if v is not None]
        degens = all_seed_degen[p]
        mean_vf = float(np.mean(vfs))
        std_vf = float(np.std(vfs))
        mean_degen = float(np.mean(degens))
        summary[str(p)] = {"mean_V_F": mean_vf, "std_V_F": std_vf, "mean_degenerate_rate": mean_degen}
        print("  percentile=%3d  V_F=%.4f +- %.4f   degenerate_rate=%.3f" %
              (p, mean_vf, std_vf, mean_degen))

    best_p = max(summary, key=lambda k: summary[k]["mean_V_F"])
    print("\nBEST adaptive percentile: p=%s  V_F=%.4f" % (best_p, summary[best_p]["mean_V_F"]))
    print("Reference fixed-tau results (from prior calls / Follow-up 2):")
    print("  V-measure-optimal fixed tau=0.78 -> V_F=0.682 (single-seed, section_detector.py)")
    print("  Follow-up 2 best fixed-tau (target_fpr=0.10) -> V_F=0.6851 +- 0.0151 (5-seed)")
    print("  Step 2 low-FP fixed tau=0.973 -> V_F=0.638 (single-seed, section_detector.py)")

    (OUT_DIR / "adaptive_percentile_on_ireal_results.json").write_text(json.dumps({
        "percentile_sweep": summary,
        "best_percentile": best_p,
        "reference_fixed_tau": {
            "vmeasure_optimal_078": 0.682,
            "followup2_best_fixed_target_fpr_010": 0.6851,
            "low_fp_0973": 0.638,
        },
    }, indent=2))
    print("\nwrote adaptive_percentile_on_ireal_results.json")


if __name__ == "__main__":
    main()
