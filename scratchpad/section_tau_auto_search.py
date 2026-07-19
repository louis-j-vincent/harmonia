"""section_tau_auto_search.py — 2026-07-19, N-way grain=4-vs-8 auto-tier
threshold call: "quel niveau k de fusion marche le mieux (4 ou 8 bars) pour
la fusion AUTOMATIQUE" — i.e. build the grain=4 and grain=8 equivalent of
`tau_auto_search.py`'s bar-level auto-tier threshold, something that does
NOT currently exist (the section-suggestion tool, docs/known_issues.md
"SECTION-level (8-bar) repeat-detection suggestion tool", explicitly shipped
suggest-tier-only, "no auto-tier exists for this tool", because that call's
GT was "same named section" — a weak, AUC~0.59-0.67 signal). This call does
NOT repeat that GT choice. It applies tau_auto_search.py's OWN correction
(bar-level "same section" was diagnosed there as the wrong label for a
POOLING task; "same chord identity" is the right one) at BLOCK grain: two
blocks are safe to auto-fuse (pool their chord evidence) iff their
POSITION-ALIGNED per-bar chord identity matches at EVERY aligned bar
position, not iff they share a section letter. This is a stricter, more
literal target than the section tool's, and a more honest one for what
auto-fusion actually does mechanically (pool_beat_evidence-style evidence
pooling across two spans, exactly the bar-merge tool's mechanism extended
to longer spans).

Reuses, does not rebuild: `tau_auto_search.load_corpus_bar_chords` (full
iReal corpus, per-bar bass/treble L2-unit proxy vectors + per-bar
(root_pc,qbucket) chord identity via majority vote) and its threshold-
selection machinery (`sweep_thresholds`, `select_tau_nested`,
`validate_on_held_out`, `split_songs`, `split_songs_3way`,
`clopper_pearson_upper`, `rule_of_three`) VERBATIM — only the pair-builder
(GT + feature at block grain instead of bar grain) is new. Block audio-
analog feature = hierarchy_shortcut.py's diagonal-prefix-sum block_gram_sim
on the SAME per-bar bass/treble Gram matrices tau_auto_search already
builds (0.5*bass_blocksim + 0.5*treble_blocksim, matching tau_auto_search's
own sim_combined convention).

MIN_GAP_BLOCKS=0 (adjacent blocks ARE the primary use case at block grain
per the section-suggestion-tool call's own caught bug — do not repeat that
mistake here).
"""
from __future__ import annotations
import sys, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tau_auto_search import (
    load_corpus_bar_chords, sweep_thresholds, select_tau_nested,
    validate_on_held_out, split_songs, split_songs_3way, flatten,
    clopper_pearson_upper, rule_of_three, find_tau_strict_zero,
    find_tau_for_target, RELAXED_TARGET, SEEDS,
)
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans

OUT_DIR = Path(__file__).resolve().parent
MIN_GAP_BLOCKS = 0
GRAINS = [4, 8]
START_SIM = 0.50  # matches section_merge_candidates.py's AUDIO_FLOOR convention


def block_gram_sim(prefix, sq, n, i0, j0, L):
    d = j0 - i0
    num = diag_sum(prefix, n, d, i0, L)
    na = np.sqrt(float(np.sum(sq[i0:i0 + L])))
    nb = np.sqrt(float(np.sum(sq[j0:j0 + L])))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def build_block_pairs_for_tune(c, grain, min_gap_blocks=MIN_GAP_BLOCKS):
    """One tune's (sim_combined, label) rows at the given block grain.
    label=1 iff AT MOST ONE aligned bar position (of the >=1 with both
    chord_ids resolved) disagrees on chord identity (root_pc, qbucket) --
    "allow-one-mismatch" block-identity GT. NOT literal all-match: an
    earlier version of this script used strict all-match and, when ported
    to real audio (section_realaudio_autotier.py), produced pseudo_gt=0 for
    100% of pairs (n=0/165) including pairs in EXTERNALLY CONFIRMED
    identical passages (aretha's static vamp) -- root-caused as compounding
    per-bar baseline-decode noise (~10-27% single-bar flicker rate,
    docs/known_issues.md "Aretha's joint-gate... measurement artifact"
    entry) across L>=4 aligned positions, not a real precision failure.
    This mirrors that exact finding and the autumn_leaves premise-check's
    own worked example (the KNOWN-true grain=8 A/A repeat scores
    symbolic_sim=0.875, i.e. 7/8 bars match, due to one explainable
    phase-alignment leak bar) -- allow-one-mismatch is applied identically
    on the SYMBOLIC corpus here (for an apples-to-apples threshold) and on
    real audio (section_realaudio_autotier.py), not just patched on one
    side."""
    bass = np.array(c["bass"])
    treb = np.array(c["treble"])
    chord_ids = c["chord_ids"]
    n = len(chord_ids)
    spans = nuclear_spans(n, grain)
    m = len(spans)
    if m < 2:
        return []
    Gb = bass @ bass.T
    Gt = treb @ treb.T
    sqb = np.diag(Gb).copy()
    sqt = np.diag(Gt).copy()
    pb = diagonal_prefix_sums(Gb)
    pt = diagonal_prefix_sums(Gt)
    rows = []
    for i in range(m):
        si, ei = spans[i]
        for j in range(i + 1 + min_gap_blocks, m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            valid = 0
            mismatches = 0
            for t in range(L):
                ci, cj = chord_ids[si + t], chord_ids[sj + t]
                if ci is None or cj is None:
                    continue
                valid += 1
                if ci != cj:
                    mismatches += 1
            if valid == 0:
                continue
            sb = block_gram_sim(pb, sqb, n, si, sj, L)
            st = block_gram_sim(pt, sqt, n, si, sj, L)
            sim_combined = 0.5 * (sb + st)
            rows.append((sim_combined, 1 if mismatches <= 1 else 0))
    return rows


def build_all_block_pairs(corpus, grain):
    per_tune = []
    for c in corpus:
        rows = build_block_pairs_for_tune(c, grain)
        if rows:
            per_tune.append(rows)
    return per_tune


def run_grain(per_tune, grain, t0):
    total_pairs = sum(len(r) for r in per_tune)
    print(f"grain={grain}: {len(per_tune)} tunes usable, {total_pairs} block-pairs, "
          f"elapsed {time.time()-t0:.1f}s")

    sims_all, labels_all = flatten(per_tune)
    s_all, n_all, fp_all, tp_all = sweep_thresholds(sims_all, labels_all, start=START_SIM)
    print(f"  === GLOBAL curve (grain={grain}) ===")
    global_curve = []
    for probe_tau in [1.0, 0.99, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        idx = np.searchsorted(-s_all, -probe_tau, side="right")
        if idx == 0:
            continue
        row = {"tau": probe_tau, "n": int(idx), "fp": int(fp_all[idx - 1]),
               "error_rate": float(fp_all[idx - 1] / idx)}
        global_curve.append(row)
        print(f"    tau={probe_tau:.2f}  n={row['n']:6d}  fp={row['fp']:5d}  error_rate={row['error_rate']:.4f}")

    fold_results = []
    for seed in SEEDS:
        pool, test = split_songs(per_tune, seed)
        sims_pool, labels_pool = flatten(pool)
        sims_test, labels_test = flatten(test)

        s_sorted, n_sel, cum_fp, cum_tp = sweep_thresholds(sims_pool, labels_pool, start=START_SIM)
        if len(s_sorted) == 0:
            fold_results.append({"seed": seed, "strict": None, "relaxed_nested": None})
            continue
        strict = find_tau_strict_zero(s_sorted, n_sel, cum_fp, cum_tp)

        train3, val3, test3 = split_songs_3way(per_tune, seed)
        sims_tr, labels_tr = flatten(train3)
        sims_va, labels_va = flatten(val3)
        sims_te, labels_te = flatten(test3)
        nested = select_tau_nested(sims_tr, labels_tr, sims_va, labels_va, RELAXED_TARGET, start=START_SIM)

        row = {"seed": seed, "n_pool_pairs_above_start": int(len(s_sorted))}
        if strict is not None:
            tau_s, k_s, fp_s, n_s = strict
            val_s = validate_on_held_out(sims_test, labels_test, tau_s)
            row["strict"] = {"tau_auto": tau_s, "pool_n": n_s, "pool_fp": fp_s,
                              "pool_error_rate": fp_s / n_s,
                              "pool_cp_upper_95": clopper_pearson_upper(fp_s, n_s, 0.05),
                              "pool_rule_of_three": rule_of_three(n_s),
                              "held_out_validation": val_s}
        else:
            row["strict"] = None

        if nested is not None:
            tau_n = nested["tau_auto"]
            blind_test = validate_on_held_out(sims_te, labels_te, tau_n)
            row["relaxed_nested"] = {**nested, "blind_test_validation": blind_test}
        else:
            row["relaxed_nested"] = None

        fold_results.append(row)
        print(f"  seed={seed}: pool_pairs_above_{START_SIM}={len(s_sorted)}")
        if row["strict"]:
            st = row["strict"]
            v = st["held_out_validation"]
            print(f"    STRICT tau_auto={st['tau_auto']:.4f} pool_N={st['pool_n']} "
                  f"CP_upper95={st['pool_cp_upper_95']:.4f}  held_out: N={v['n']} err={v['error_rate']}")
        else:
            print("    STRICT: none found")
        if row["relaxed_nested"]:
            rn = row["relaxed_nested"]
            bt = rn["blind_test_validation"]
            print(f"    RELAXED-NESTED tau_auto={rn['tau_auto']:.4f} train_err={rn['train_error_rate']:.4f} "
                  f"val_err={rn['val_check']['error_rate']:.4f}  BLIND test: N={bt['n']} err={bt['error_rate']}")
        else:
            print("    RELAXED-NESTED: none found (no tau satisfies target on both train and val)")

    relaxed_taus = [r["relaxed_nested"]["tau_auto"] for r in fold_results if r["relaxed_nested"]]
    strict_taus = [r["strict"]["tau_auto"] for r in fold_results if r["strict"]]
    consensus = {
        "relaxed_nested_tau_auto_mean": float(np.mean(relaxed_taus)) if relaxed_taus else None,
        "relaxed_nested_tau_auto_max": float(np.max(relaxed_taus)) if relaxed_taus else None,
        "relaxed_nested_tau_auto_min": float(np.min(relaxed_taus)) if relaxed_taus else None,
        "n_folds_with_relaxed": len(relaxed_taus),
        "strict_tau_auto_mean": float(np.mean(strict_taus)) if strict_taus else None,
        "strict_tau_auto_max": float(np.max(strict_taus)) if strict_taus else None,
        "n_folds_with_strict": len(strict_taus),
    }
    print(f"  CONSENSUS grain={grain}: relaxed_nested mean={consensus['relaxed_nested_tau_auto_mean']} "
          f"max={consensus['relaxed_nested_tau_auto_max']} (n_folds={consensus['n_folds_with_relaxed']}/5)")

    return {
        "grain": grain, "n_tunes": len(per_tune), "n_total_pairs": total_pairs,
        "global_curve": global_curve, "fold_results": fold_results, "consensus": consensus,
    }


def main():
    t0 = time.time()
    print("Loading FULL iReal corpus with per-bar chord identity (reused from tau_auto_search)...")
    corpus = load_corpus_bar_chords(max_tunes=None)
    print(f"  {len(corpus)} tunes, elapsed {time.time()-t0:.1f}s")

    results = {"min_gap_blocks": MIN_GAP_BLOCKS, "start_sim": START_SIM,
               "relaxed_target": RELAXED_TARGET, "seeds": SEEDS, "n_tunes_loaded": len(corpus),
               "gt_definition": "label=1 iff AT MOST ONE position-aligned aligned-bar pair "
                                 "within the block disagrees on (root_pc,qbucket) chord "
                                 "identity ('allow-one-mismatch' block-identity match, not "
                                 "same-section-letter, not literal all-match -- see module "
                                 "docstring for why strict all-match was rejected)"}
    for grain in GRAINS:
        per_tune = build_all_block_pairs(corpus, grain)
        results[f"grain_{grain}"] = run_grain(per_tune, grain, t0)

    out_path = OUT_DIR / "section_tau_auto_search_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}, total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
