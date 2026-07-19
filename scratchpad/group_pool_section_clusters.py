"""group_pool_section_clusters.py — 2026-07-19 (research-loop call, budget
2.5h): N-WAY section-cluster group pooling — MEASUREMENT ONLY, not shipped.

Extends the validated PAIR-level `pool_beat_evidence` mechanism (see
docs/known_issues.md "Three follow-ups... a real multi-merge
ORDER-DEPENDENCE bug found+fixed" and "AUTO-tier auto-apply WIRED and
MEASURED", 2026-07-18) from 2-span merge groups to N-span groups spanning
an ENTIRE k<=5 section cluster (2 to ~10 8-bar blocks sharing a cluster
letter), using the k-prior-selected k per song (autumn_leaves k=5, abba
k=4, aretha k=3 — `scratchpad/k_prior_results.json` /
`scratchpad/section_structure_clusters_grain8.json`, both already at the
SAME k, confirmed by direct comparison in this call).

**Mechanism**: `pool_beat_evidence` (harmonia/models/user_constraints.py)
maps each merge's `spans` to beat ranges and pools beat-by-beat OFFSET
across all spans — already generalizes to whole-block (32-beat, 8-bar)
spans without code changes, per its own docstring ("equal MUSICAL length
=> equal beat COUNT, so corresponding beats align by offset"). This script
submits ONE merge group per cluster letter, spans = the block-level
[t0,t1] time ranges from `dual_matrix_grain8_results.json`'s
`block_times_s` (NOT individual per-bar spans) — i.e. beat 0 of every
block in the cluster pools together, beat 1 of every block pools together,
etc. Both a real HTTP round-trip against the live `/api/reinfer/<file>`
endpoint (production path) AND a direct `infer_chords_v1` call (for full
before/after confidence per bar, which the HTTP diff doesn't expose) are
run, same dual-verification pattern as `auto_apply_merges.py`.

**Stricter-gate variant** (added per a coordinator mid-task addition
reflecting the user's own musical caution, "des fois le A a deux
variations differentes" — a nominal same-letter cluster can contain a
legitimate internal variation, e.g. first/second-ending A-sections in an
AABA form): also builds a GATED group per cluster that drops any member
block whose mean pairwise similarity to the REST of the cluster sits
below (cluster_mean - 1 std) of all pairwise sims — i.e. an outlier
detector on the within-cluster similarity submatrix, not just trusting
cluster co-membership. Both FULL-cluster and GATED-cluster groups are
measured so the two can be compared directly.
"""
from __future__ import annotations
import sys, json, shutil, subprocess, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

REPO = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO / "docs" / "audio"
OUT_DIR = Path(__file__).resolve().parent

SONGS = {
    "aretha_franklin_chain_of_fools_official_lyric_video": {
        "chart_file": "inferred_aretha_franklin_chain_of_fools_official_lyric_video.html",
        "audio_name": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
        "k": 3,
    },
    "autumn_leaves": {
        "chart_file": "inferred_autumn_leaves.html",
        "audio_name": "autumn_leaves.m4a",
        "k": 5,
    },
    "abba_chiquitita_official_lyric_video": {
        "chart_file": "inferred_abba_chiquitita_official_lyric_video.html",
        "audio_name": "abba_chiquitita_official_lyric_video.m4a",
        "k": 4,
    },
}


def load_clusters(slug):
    """-> dict of {letter: [block_idx, ...]}, blocks list, block_times_s,
    audio_matrix, symbolic_matrix (grain=8)."""
    clusters_data = json.loads((OUT_DIR / "section_structure_clusters_grain8.json").read_text())[slug]
    dual = json.loads((OUT_DIR / "dual_matrix_grain8_results.json").read_text())[slug]
    assert clusters_data["k"] == SONGS[slug]["k"], (
        f"{slug}: deployed clusters k={clusters_data['k']} != k-prior-selected k={SONGS[slug]['k']}")
    blocks = clusters_data["blocks"]
    letters = {}
    for b in blocks:
        letters.setdefault(b["section"], []).append(b["block"])
    block_times_s = dual["block_times_s"]
    audio_matrix = np.array(dual["audio_matrix"])
    symbolic_matrix = np.array(dual["symbolic_matrix"])
    return letters, blocks, block_times_s, audio_matrix, symbolic_matrix


def within_cluster_similarity_report(letters, audio_matrix, symbolic_matrix):
    """For each multi-block letter, report the within-cluster pairwise
    similarity submatrix (audio+symbolic averaged 50/50) and flag outlier
    members (mean sim to rest of cluster < cluster_mean - 1*std of all
    pairwise sims in the cluster) — the stricter-gate check the user asked
    for: does a nominal 'A' cluster actually contain 2 sub-variations?"""
    joint = 0.5 * audio_matrix + 0.5 * symbolic_matrix
    report = {}
    for letter, members in letters.items():
        if len(members) < 2:
            continue
        members = sorted(members)
        n = len(members)
        sub = np.zeros((n, n))
        for i, bi in enumerate(members):
            for j, bj in enumerate(members):
                sub[i, j] = joint[bi, bj]
        offdiag = sub[~np.eye(n, dtype=bool)]
        mean_all = float(offdiag.mean())
        std_all = float(offdiag.std())
        per_member_mean = []
        for i in range(n):
            others = [sub[i, j] for j in range(n) if j != i]
            per_member_mean.append(float(np.mean(others)))
        thresh = mean_all - std_all
        outliers = [members[i] for i in range(n) if per_member_mean[i] < thresh]
        report[letter] = {
            "members": members,
            "n_members": n,
            "pairwise_mean": mean_all,
            "pairwise_std": std_all,
            "pairwise_min": float(offdiag.min()),
            "pairwise_max": float(offdiag.max()),
            "per_member_mean_sim_to_rest": dict(zip(members, per_member_mean)),
            "outlier_threshold": thresh,
            "outlier_members": outliers,
            "submatrix": sub.tolist(),
        }
    return report


def transcode(audio_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "22050", str(wav_path)],
        check=True, capture_output=True, timeout=180,
    )


def chord_at(chords, t):
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


def per_bar_spans_within_blocks(slug, blocks, block_idxs, block_times_s):
    """Subdivide each block's [t0,t1] span into `n_bars_in_block` equal
    per-bar spans for BAR-LEVEL reporting granularity (the merge itself
    uses the whole-block span, not these — these are for the before/after
    table only, same table shape as the pairwise validation work)."""
    out = {}
    for bidx in block_idxs:
        b = next(x for x in blocks if x["block"] == bidx)
        bar0, bar1 = b["bars"]
        n_bars = bar1 - bar0
        t0, t1 = block_times_s[bidx]
        edges = np.linspace(t0, t1, n_bars + 1)
        for i in range(n_bars):
            out[bar0 + i] = (float(edges[i]), float(edges[i + 1]))
    return out


def run_direct_pipeline(slug, groups_spans):
    """groups_spans: list of list-of-[t0,t1] (one group per cluster/variant).
    Direct call into infer_chords_v1, same production code path /api/reinfer
    uses when merges are present (see harmonia_server.py::api_reinfer)."""
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    audio_path = AUDIO_DIR / SONGS[slug]["audio_name"]
    tmp = Path(tempfile.mkdtemp(prefix="harmonia_grouppool_"))
    try:
        wav = tmp / "a.wav"
        transcode(audio_path, wav)
        cache = tmp
        base = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=0.0)
        merges = [{"spans": spans} for spans in groups_spans]
        cons = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=0.0,
                                user_constraints={"confirms": [], "merges": merges})
        return base, cons, merges
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def curl_verify_reinfer(chart_file, merges, port=7771):
    import urllib.request
    url = f"http://localhost:{port}/api/reinfer/{chart_file}"
    body = json.dumps({"confirms": [], "merges": merges}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def build_per_bar_position_groups(slug, letters_subset, blocks, block_times_s):
    """MITIGATION variant for the equal-beat-count blocker found in the
    whole-block-span encoding: instead of ONE merge per cluster with N
    32-beat (8-bar) spans (which needs all N blocks to share the EXACT
    beat count end-to-end -- shown empirically to fail for nearly every
    cluster size >2, see session log), decompose each cluster into up to 8
    merges, one per WITHIN-BLOCK BAR OFFSET (bar 0 of every member block
    pooled together, bar 1 of every member block pooled together, ...).
    Each sub-merge's spans are ~1 bar (~4 beats) long instead of ~32 beats,
    so the same absolute beat-grid quantization drift is far less likely to
    flip a span's beat COUNT (the failure mode is drift accumulating over
    the span's length, not a fixed per-span offset)."""
    groups_spans, group_letters = [], []
    for letter, members in sorted(letters_subset.items()):
        if len(members) < 2:
            continue
        members = sorted(members)
        n_bars_list = [next(x for x in blocks if x["block"] == m)["bars"] for m in members]
        min_len = min(b1 - b0 for b0, b1 in n_bars_list)
        bar_spans_per_block = [per_bar_spans_within_blocks(slug, blocks, [m], block_times_s)
                                for m in members]
        for off in range(min_len):
            spans = []
            for m, bsb in zip(members, bar_spans_per_block):
                bar0 = next(x for x in blocks if x["block"] == m)["bars"][0]
                spans.append(bsb[bar0 + off])
            groups_spans.append(spans)
            group_letters.append(f"{letter}_bar{off}")
    return groups_spans, group_letters


def measure(slug, variant_name, letters_subset, blocks, block_times_s, do_curl=True,
            groups_spans_override=None, group_letters_override=None):
    """letters_subset: {letter: [block_idx,...]} — only multi-block letters
    intended to be pooled this variant (FULL or GATED). If
    groups_spans_override is given, use it directly (PER_BAR_POSITION
    variant) instead of deriving whole-block spans from letters_subset."""
    if groups_spans_override is not None:
        groups_spans, group_letters = groups_spans_override, group_letters_override
        all_bar_spans = {}
        for letter, members in letters_subset.items():
            if len(members) < 2:
                continue
            all_bar_spans.update(per_bar_spans_within_blocks(slug, blocks, members, block_times_s))
    else:
        groups_spans = []
        group_letters = []
        all_bar_spans = {}
        for letter, members in sorted(letters_subset.items()):
            if len(members) < 2:
                continue
            spans = [block_times_s[b] for b in sorted(members)]
            groups_spans.append(spans)
            group_letters.append(letter)
            all_bar_spans.update(per_bar_spans_within_blocks(slug, blocks, members, block_times_s))

    print(f"\n=== {slug} [{variant_name}] ===")
    print(f"  groups: {list(zip(group_letters, [len(g) for g in groups_spans]))}")
    if not groups_spans:
        print("  (no multi-block clusters to pool in this variant)")
        return None

    t0 = time.time()
    base, cons, merges_payload = run_direct_pipeline(slug, groups_spans)
    print(f"  direct-pipeline calls done in {time.time()-t0:.1f}s")

    base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
    cons_ch = [c for c in cons.chords if c["end_s"] > c["start_s"]]

    per_bar = []
    for bar, (t0b, t1b) in sorted(all_bar_spans.items()):
        mid = 0.5 * (t0b + t1b)
        bc = chord_at(base_ch, mid)
        cc = chord_at(cons_ch, mid)
        if bc is None or cc is None:
            continue
        per_bar.append({
            "bar": bar, "t0": t0b, "t1": t1b,
            "before_label": bc["label"], "after_label": cc["label"],
            "before_confidence": bc.get("confidence", 0.0),
            "after_confidence": cc.get("confidence", 0.0),
            "label_changed": bc["label"] != cc["label"],
            "confidence_delta": cc.get("confidence", 0.0) - bc.get("confidence", 0.0),
        })

    n_label_changed = sum(1 for r in per_bar if r["label_changed"])
    deltas = np.array([r["confidence_delta"] for r in per_bar]) if per_bar else np.array([])
    n_regressed = int(np.sum(deltas < -1e-9)) if len(deltas) else 0
    regressions = [r for r in per_bar if r["confidence_delta"] < -1e-9]

    print(f"  bars matched: {len(per_bar)}  label changed: {n_label_changed}/{len(per_bar)}")
    if len(deltas):
        print(f"  confidence delta: mean={deltas.mean():+.4f} std={deltas.std():.4f} "
              f"min={deltas.min():+.4f} max={deltas.max():+.4f}")
    print(f"  REGRESSIONS: {n_regressed}/{len(per_bar)}")

    curl_result = None
    if do_curl:
        try:
            curl_result = curl_verify_reinfer(SONGS[slug]["chart_file"], merges_payload)
            rejected = curl_result.get("rejected", [])
            n_changed_http = curl_result.get("n_changed")
            print(f"  CURL verify: HTTP 200, n_changed={n_changed_http}, rejected={rejected}")
        except Exception as e:
            print(f"  CURL verify FAILED: {e}")
            curl_result = {"error": str(e)}

    return {
        "slug": slug, "variant": variant_name,
        "groups": [{"letter": l, "n_blocks": len(g)} for l, g in zip(group_letters, groups_spans)],
        "n_groups_applied": len(groups_spans),
        "n_bars_touched": len(per_bar),
        "n_label_changed": n_label_changed,
        "confidence_delta_mean": float(deltas.mean()) if len(deltas) else None,
        "confidence_delta_std": float(deltas.std()) if len(deltas) else None,
        "n_regressions": n_regressed,
        "regression_rate": (n_regressed / len(per_bar)) if per_bar else None,
        "regressions": regressions,
        "per_bar": per_bar,
        "curl_verify": {k: v for k, v in (curl_result or {}).items() if k != "chords"},
    }


def build_gated_letters(sim_report, letters):
    """Drop outlier members flagged by within_cluster_similarity_report."""
    gated = {}
    for letter, members in letters.items():
        if letter in sim_report:
            outliers = set(sim_report[letter]["outlier_members"])
            kept = [m for m in members if m not in outliers]
            gated[letter] = kept
        else:
            gated[letter] = members
    return gated


def main():
    results = {"songs": {}}
    for slug in SONGS:
        letters, blocks, block_times_s, audio_matrix, symbolic_matrix = load_clusters(slug)
        sim_report = within_cluster_similarity_report(letters, audio_matrix, symbolic_matrix)
        gated_letters = build_gated_letters(sim_report, letters)

        full_res = measure(slug, "FULL_CLUSTER", letters, blocks, block_times_s)
        gated_res = measure(slug, "GATED_CLUSTER", gated_letters, blocks, block_times_s)

        results["songs"][slug] = {
            "k": SONGS[slug]["k"],
            "letters": {l: sorted(m) for l, m in letters.items()},
            "within_cluster_similarity": sim_report,
            "gated_letters": {l: sorted(m) for l, m in gated_letters.items()},
            "full_cluster": full_res,
            "gated_cluster": gated_res,
        }

    # pooled aggregate across songs, FULL variant only (matches the
    # pairwise-baseline table's own aggregate shape)
    for variant_key in ("full_cluster", "gated_cluster"):
        all_deltas, total_bars, total_changed, total_regressed, total_groups = [], 0, 0, 0, 0
        for slug in SONGS:
            r = results["songs"][slug][variant_key]
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
        results.setdefault("aggregate", {})[variant_key] = agg
        print(f"\n=== AGGREGATE [{variant_key}] ===")
        print(json.dumps(agg, indent=2))

    (OUT_DIR / "group_pool_section_clusters_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print("\nwrote scratchpad/group_pool_section_clusters_results.json")


if __name__ == "__main__":
    main()
