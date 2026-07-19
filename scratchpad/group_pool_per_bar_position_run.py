"""Runs the PER_BAR_POSITION mitigation variant (see
group_pool_section_clusters.py::build_per_bar_position_groups's docstring)
for all 3 songs and merges into the existing results JSON, instead of
rerunning the already-complete FULL/GATED variants."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from group_pool_section_clusters import (
    SONGS, OUT_DIR, load_clusters, within_cluster_similarity_report,
    build_per_bar_position_groups, measure,
)


def main():
    out_path = OUT_DIR / "group_pool_section_clusters_results.json"
    results = json.loads(out_path.read_text())

    for slug in SONGS:
        letters, blocks, block_times_s, audio_matrix, symbolic_matrix = load_clusters(slug)
        groups_spans, group_letters = build_per_bar_position_groups(
            slug, letters, blocks, block_times_s)
        res = measure(slug, "PER_BAR_POSITION", letters, blocks, block_times_s,
                       groups_spans_override=groups_spans, group_letters_override=group_letters)
        results["songs"][slug]["per_bar_position"] = res

    all_deltas, total_bars, total_changed, total_regressed, total_groups = [], 0, 0, 0, 0
    for slug in SONGS:
        r = results["songs"][slug]["per_bar_position"]
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
    results.setdefault("aggregate", {})["per_bar_position"] = agg
    print("\n=== AGGREGATE [per_bar_position] ===")
    print(json.dumps(agg, indent=2))

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nupdated {out_path}")


if __name__ == "__main__":
    main()
