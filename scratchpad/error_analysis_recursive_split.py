"""error_analysis_recursive_split.py — mandatory error-analysis loop (Step
6 continuation) on the currently-deployed adaptive-percentile fix
(`real_transfer.py`, p90), 3rd continuation call tonight.

FAILURE PATTERN observed by inspecting real_transfer_results.json's
adaptive-tau runs directly (qualitative, no GT — matches the established
convention that real audio has none):

  autumn_leaves: one cluster label ("S4") accounts for 4 separate RUNS
  totalling 240 of 330 bars (32-112 = 80 bars in one contiguous run alone).
  abba_chiquitita: one cluster ("S3") accounts for 5 runs totalling 152 of
  232 bars (72-120 and 144-192 = 48 bars each in one contiguous run).

HYPOTHESIS: this is the same chord-only-similarity ceiling already logged
earlier tonight ("GRAMMAR-INDUCTION... chord-only matching is precision/
recall-capped even on clean GT" — some real section boundaries carry no
harmonic signal, e.g. a chorus repeated with different lyrics over
identical changes). At the WITHIN-RUN level specifically: an 80-bar
contiguous run under one cluster label is almost certainly actually
several repeats of an underlying phrase/chorus (Autumn Leaves is a 32-bar
form; 80 bars ~= 2.5 choruses) that share harmony closely enough to clear
the global adaptive tau, but that doesn't mean they're internally
undifferentiated — a LOCAL (within-run) re-clustering at a stricter
threshold might recover phrase-level sub-boundaries the global threshold
missed, without needing a lower global tau (which Step 6 already showed
causes collapse elsewhere in the same song).

FIX TRIED: recursive local re-split. Any run of >= MIN_RUN_BLOCKS nuclear
blocks (default 4 blocks = 32 bars) gets its own LOCAL adaptive tau
(percentile of only the within-run pairwise similarities, at a HIGHER
percentile than the global one, since the global run is already the "high
similarity" residue) and is re-clustered via union-find restricted to that
run's blocks only. Applied once (not fully recursive) to avoid runaway
over-splitting eating the whole budget on tuning a recursion depth.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from real_transfer import (union_find_labels, runs_from_labels, adaptive_tau,
                            SONGS, OUT_DIR, ADAPTIVE_PERCENTILE)

MIN_RUN_BLOCKS = 4          # only re-split runs of >=4 nuclear blocks (32 bars)
LOCAL_PERCENTILE = 75       # stricter LOCAL percentile within the run's own
                             # pairwise similarities (lower than the global
                             # P90 because the run's internal distribution is
                             # already the high-similarity residue -- P90 of
                             # THAT would barely split anything; P75 forces a
                             # real interior cut if one exists)


def recursive_local_resplit(sim, global_labels, min_run_blocks=MIN_RUN_BLOCKS,
                             local_percentile=LOCAL_PERCENTILE):
    """global_labels: block-level cluster ids from the global adaptive tau.
    Returns new block-level labels where any contiguous RUN (same label,
    consecutive blocks) of length >= min_run_blocks gets locally re-split."""
    m = len(global_labels)
    # find contiguous runs
    runs = []
    i = 0
    while i < m:
        j = i
        while j + 1 < m and global_labels[j + 1] == global_labels[i]:
            j += 1
        runs.append((i, j))  # inclusive
        i = j + 1

    new_labels = list(global_labels)
    next_id = max(global_labels) + 1
    n_resplit = 0
    for (s, e) in runs:
        length = e - s + 1
        if length < min_run_blocks:
            continue
        # local sim submatrix for this run
        idx = list(range(s, e + 1))
        sub = sim[np.ix_(idx, idx)]
        vals = [sub[a, b] for a in range(len(idx)) for b in range(len(idx))
                if abs(a - b) >= 1]  # min_gap=1 locally (blocks are already coarse)
        if not vals:
            continue
        local_tau = float(np.percentile(vals, local_percentile))
        sub_labels = union_find_labels(sub, local_tau)
        n_sub = len(set(sub_labels))
        if n_sub <= 1:
            continue  # no internal structure found, leave as-is
        n_resplit += 1
        for k, bi in enumerate(idx):
            new_labels[bi] = next_id + sub_labels[k]
        next_id += n_sub
    return new_labels, n_resplit


def main():
    results = {}
    for song in SONGS:
        d = json.loads((OUT_DIR / ("bar_ssm_rawchroma_%s.json" % song)).read_text())
        n_bars = d["n_bars"]
        sim_b8 = np.array(d["grains_bass"]["8"]["similarity_matrix"])
        sim_t8 = np.array(d["grains_treble"]["8"]["similarity_matrix"])
        sim_c8 = 0.5 * (sim_b8 + sim_t8)

        tau_global = adaptive_tau(sim_c8, ADAPTIVE_PERCENTILE)
        labels_global = union_find_labels(sim_c8, tau_global)
        runs_global = runs_from_labels(labels_global, 8, n_bars)

        labels_split, n_resplit = recursive_local_resplit(sim_c8, labels_global)
        runs_split = runs_from_labels(labels_split, 8, n_bars)

        print("=== %s === n_bars=%d" % (song, n_bars))
        print("  BEFORE (global adaptive p%d, tau=%.4f): %d sections, %d runs" %
              (ADAPTIVE_PERCENTILE, tau_global, len(set(labels_global)), len(runs_global)))
        print("  AFTER  (+ local re-split, %d runs locally re-split): %d sections, %d runs" %
              (n_resplit, len(set(labels_split)), len(runs_split)))
        max_run_before = max((r["bar_end"] - r["bar_start"]) for r in runs_global)
        max_run_after = max((r["bar_end"] - r["bar_start"]) for r in runs_split)
        print("  max contiguous run length: %d bars -> %d bars" % (max_run_before, max_run_after))
        for r in runs_split:
            print("    ", r)

        results[song] = {
            "n_bars": n_bars,
            "before": {"tau": tau_global, "n_sections": len(set(labels_global)),
                       "n_runs": len(runs_global), "max_run_bars": max_run_before,
                       "runs": runs_global},
            "after": {"n_sections": len(set(labels_split)), "n_runs": len(runs_split),
                      "max_run_bars": max_run_after, "n_runs_resplit": n_resplit,
                      "runs": runs_split},
        }

    (OUT_DIR / "error_analysis_recursive_split_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote error_analysis_recursive_split_results.json")


if __name__ == "__main__":
    main()
