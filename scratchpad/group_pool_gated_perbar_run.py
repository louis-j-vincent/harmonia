"""GATED + PER_BAR_POSITION combined variant: the stricter-gate mitigation
(drop within-cluster similarity outliers) applied on TOP of the
per-bar-position encoding that's needed for the merges to actually apply
at all (see group_pool_section_clusters_results.json's full_cluster/
gated_cluster rows: 0-1 of 10 groups apply with the naive whole-block
encoding). Answers task 4's proposed stricter gate directly, on the ONLY
encoding that actually exercises pool_beat_evidence for N>2 groups."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from group_pool_section_clusters import (
    SONGS, OUT_DIR, load_clusters, within_cluster_similarity_report,
    build_gated_letters, build_per_bar_position_groups, measure,
)


def main():
    out_path = OUT_DIR / "group_pool_section_clusters_results.json"
    results = json.loads(out_path.read_text())

    for slug in SONGS:
        letters, blocks, block_times_s, audio_matrix, symbolic_matrix = load_clusters(slug)
        sim_report = within_cluster_similarity_report(letters, audio_matrix, symbolic_matrix)
        gated_letters = build_gated_letters(sim_report, letters)
        groups_spans, group_letters = build_per_bar_position_groups(
            slug, gated_letters, blocks, block_times_s)
        res = measure(slug, "GATED_PER_BAR_POSITION", gated_letters, blocks, block_times_s,
                       groups_spans_override=groups_spans, group_letters_override=group_letters)
        results["songs"][slug]["gated_per_bar_position"] = res

    all_deltas, total_bars, total_changed, total_regressed, total_groups = [], 0, 0, 0, 0
    for slug in SONGS:
        r = results["songs"][slug]["gated_per_bar_position"]
        if r is None:
            continue
        all_deltas.extend([x["confidence_delta"] for x in r["per_bar"]])
        total_bars += r["n_bars_touched"]
        total_changed += r["n_label_changed"]
        total_regressed += r["n_regressions"]
        total_groups += r["n_groups_applied"]
    all_deltas = np.array(all_deltas)
    agg = {
        "total_groups": total_groups, "total_bars_touched": total_bars,
        "total_label_changes": total_changed, "total_regressions": total_regressed,
        "regression_rate": (total_regressed / total_bars) if total_bars else None,
        "pooled_confidence_delta_mean": float(all_deltas.mean()) if len(all_deltas) else None,
        "pooled_confidence_delta_std": float(all_deltas.std()) if len(all_deltas) else None,
    }
    results.setdefault("aggregate", {})["gated_per_bar_position"] = agg
    print("\n=== AGGREGATE [gated_per_bar_position] ===")
    print(json.dumps(agg, indent=2))

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nupdated {out_path}")


if __name__ == "__main__":
    main()
