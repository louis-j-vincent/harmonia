"""symbolic_v3_ensemble_search.py — 2026-07-18, continuation of the joint
audio+symbolic auto-tier gate call. Picks up the explicit NEXT STEP handoff
in docs/known_issues.md ("Joint audio+symbolic auto-tier gate..."): (a)
diagnose why aretha's joint-gate precision (54.5%, n=11) lags
autumn_leaves/abba (94-100%); (b) test V3_tiv as an alternative/ensemble
symbolic feature at corpus scale (joint_threshold_search.py only
premise-checked V3, never swept it corpus-scale).

**Task (a) result, established BEFORE this script (see docs/known_issues.md
/ research session log for the full derivation), summarized here because it
changes how this script's own pseudo-GT should be read**: aretha's low
54.5% precision is overwhelmingly a PSEUDO-GT MEASUREMENT ARTIFACT, not a
real symbolic-gate failure. `joint_threshold_search.py` (and
`realaudio_threshold_check.py` before it) samples pseudo-GT with a single
midpoint timestamp per bar (`chord_at(base_ch, midpoint)`), while
`symbolic_sim` itself is computed from a MAJORITY-VOTE label over the whole
bar span (`bar_chord_majority`) -- an inconsistency between the measuring
stick and the thing being measured. Aretha's baseline decode has far more
sub-bar chord segments than the other two songs (avg 3.11 segments/bar vs
2.04 abba / 2.04 autumn_leaves) and a correspondingly high
majority-vote-vs-midpoint "flicker rate" (22/83 bars = 26.5%, vs abba's
9.9%, autumn_leaves' 0.0%). Recomputing aretha's pseudo-GT with the SAME
majority-vote convention `symbolic_sim` uses (apples-to-apples, still model
self-decode, still not real GT) raises the joint gate's measured precision
54.5%->90.9% (10/11) -- pooled 89.6%->98.7%, meeting the original "never a
false positive" design bar. The ONE genuine remaining miss (aretha bars
54/78, C:dim7 vs C:hdim7) is real vocabulary aliasing under V1_binary's
6-way QBUCKET, which folds diminished-triad and half-diminished-7th chords
into the same family (both share the [0,3,6] triad) while the independent
pseudo-GT bucket (`label_bucket`) treats them as different families --
confirms hypothesis 2 from the brief, but as a MINOR, not dominant, driver
(1/77 pooled pairs).

This script therefore reports BOTH pseudo-GT conventions side by side
(never silently picks the more flattering one) for the V3/ensemble
question, so the V3 vs V1 comparison isn't confounded by the measurement
issue found above.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from auto_apply_merges import SONGS
from realaudio_threshold_check import get_baseline_chords, label_bucket
from chord_distance import chord_vector_binary, chord_vector_tiv, cosine
from dual_matrix_correlation import label_to_root_qual, bar_chord_majority

OUT_DIR = Path(__file__).resolve().parent

SLUGS = [
    "aretha_franklin_chain_of_fools_official_lyric_video",
    "autumn_leaves",
    "abba_chiquitita_official_lyric_video",
]
SHORT = {
    "aretha_franklin_chain_of_fools_official_lyric_video": "aretha",
    "autumn_leaves": "autumn_leaves",
    "abba_chiquitita_official_lyric_video": "abba",
}

# --- premise check: 4 known hand-verified pairs (from dual_matrix_correlation
# _results.json / joint_threshold_search.py's docstring, reused verbatim,
# not recomputed) ---
KNOWN_PAIRS = {
    "abba_32_64_FALSE_POS": {"audio_sim": 0.978954165174635, "V1": 0.6666666666666667, "V3": 0.5265377208552913},
    "aretha_13_17_FALSE_POS": {"audio_sim": 0.9783348193575163, "V1": 0.8660254037844387, "V3": 0.7964870830354095},
    "aretha_53_69_TRUE_POS": {"audio_sim": 0.9891801790916748, "V1": 1.0000000000000002, "V3": 1.0},
    "abba_206_222_TRUE_POS": {"audio_sim": 0.9937504840722634, "V1": 1.0, "V3": 1.0000000000000002},
}


def premise_check():
    print("=== PREMISE CHECK: V3 alone + ensemble rules on 4 known pairs ===", file=sys.stderr)
    for tau in [0.80, 0.90]:
        print(f"\n-- tau_symbolic={tau} --", file=sys.stderr)
        for name, d in KNOWN_PAIRS.items():
            v1, v3 = d["V1"], d["V3"]
            avg = 0.5 * (v1 + v3)
            v3_pass = v3 >= tau
            and_pass = (v1 >= tau) and (v3 >= tau)
            or_pass = (v1 >= tau) or (v3 >= tau)
            avg_pass = avg >= tau
            print(f"  {name:28s} V1={v1:.4f} V3={v3:.4f} avg={avg:.4f} | "
                  f"V3_alone={'PASS' if v3_pass else 'reject'}  AND={'PASS' if and_pass else 'reject'}  "
                  f"OR={'PASS' if or_pass else 'reject'}  AVG={'PASS' if avg_pass else 'reject'}", file=sys.stderr)


def load_rows():
    """One row per candidate pair, pooled + per-song, carrying BOTH pseudo-GT
    conventions (midpoint = original joint_threshold_search.py methodology,
    majority = task-(a)-corrected methodology) and all 3 symbolic schemes."""
    per_song = {}
    all_rows = []
    for slug in SLUGS:
        short = SHORT[slug]
        census = json.loads((OUT_DIR / f"bar_merge_full_census_{slug}.json").read_text())
        base_ch = get_baseline_chords(slug)
        rows = []
        for c in census["candidates"]:
            (t0a, t1a), (t0b, t1b) = c["spans"]
            mid_a, mid_b = 0.5 * (t0a + t1a), 0.5 * (t0b + t1b)
            ca = next((x for x in base_ch if x["start_s"] <= mid_a < x["end_s"]), None)
            cb = next((x for x in base_ch if x["start_s"] <= mid_b < x["end_s"]), None)
            if ca is None or cb is None:
                continue
            ba_mid, bb_mid = label_bucket(ca["label"]), label_bucket(cb["label"])
            if ba_mid is None or bb_mid is None:
                continue
            agree_mid = int(ba_mid == bb_mid)

            label_a = bar_chord_majority(base_ch, t0a, t1a)
            label_b = bar_chord_majority(base_ch, t0b, t1b)
            ba_maj, bb_maj = label_bucket(label_a), label_bucket(label_b)
            agree_maj = int(ba_maj == bb_maj) if (ba_maj is not None and bb_maj is not None) else agree_mid

            pc_a, q_a = label_to_root_qual(label_a)
            pc_b, q_b = label_to_root_qual(label_b)
            if pc_a is None or pc_b is None:
                v1 = v3 = 0.0
            else:
                v1 = cosine(chord_vector_binary(pc_a, q_a), chord_vector_binary(pc_b, q_b))
                v3 = cosine(chord_vector_tiv(pc_a, q_a), chord_vector_tiv(pc_b, q_b))
            row = {"song": short, "audio_sim": c["confidence"], "V1": v1, "V3": v3,
                   "avg": 0.5 * (v1 + v3), "agree_mid": agree_mid, "agree_maj": agree_maj}
            rows.append(row)
            all_rows.append(row)
        per_song[short] = rows
    return all_rows, per_song


def sweep(rows, gt_key, sym_key_or_rule, tau_grid, tau_audio=0.96):
    audio = np.array([r["audio_sim"] for r in rows])
    agree = np.array([r[gt_key] for r in rows])
    out = []
    for tau in tau_grid:
        if callable(sym_key_or_rule):
            sym_mask = np.array([sym_key_or_rule(r, tau) for r in rows])
        else:
            sym = np.array([r[sym_key_or_rule] for r in rows])
            sym_mask = sym >= tau
        mask = (audio >= tau_audio) & sym_mask
        n = int(mask.sum())
        prec = float(agree[mask].mean()) if n else None
        out.append({"tau": tau, "n": n, "precision": prec})
    return out


def main():
    premise_check()
    all_rows, per_song = load_rows()

    tau_grid = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    rules = {
        "V1_alone": "V1",
        "V3_alone": "V3",
        "AND(V1,V3)": lambda r, t: (r["V1"] >= t) and (r["V3"] >= t),
        "OR(V1,V3)": lambda r, t: (r["V1"] >= t) or (r["V3"] >= t),
        "AVG(V1,V3)": "avg",
    }

    results = {"pooled": {}, "per_song": {}}
    for gt_key in ["agree_mid", "agree_maj"]:
        print(f"\n\n########## POOLED, pseudo-GT = {gt_key} ##########", file=sys.stderr)
        results["pooled"][gt_key] = {}
        for rname, rule in rules.items():
            sw = sweep(all_rows, gt_key, rule, tau_grid)
            results["pooled"][gt_key][rname] = sw
            print(f"\n-- {rname} --", file=sys.stderr)
            for row in sw:
                print(f"  tau={row['tau']:.2f}  n={row['n']:4d}  precision={row['precision']}", file=sys.stderr)

    # focused per-song comparison AT tau_symbolic=0.90 (the deployed value),
    # both pseudo-GT conventions, all 5 rules
    print("\n\n########## PER-SONG @ tau_symbolic=0.90, tau_audio=0.96 ##########", file=sys.stderr)
    for short in ["aretha", "autumn_leaves", "abba"]:
        rows = per_song[short]
        results["per_song"][short] = {}
        for gt_key in ["agree_mid", "agree_maj"]:
            results["per_song"][short][gt_key] = {}
            print(f"\n-- {short} / {gt_key} --", file=sys.stderr)
            for rname, rule in rules.items():
                sw = sweep(rows, gt_key, rule, [0.90])[0]
                results["per_song"][short][gt_key][rname] = sw
                print(f"  {rname:14s} n={sw['n']:3d}  precision={sw['precision']}", file=sys.stderr)

    (OUT_DIR / "symbolic_v3_ensemble_search_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote symbolic_v3_ensemble_search_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
