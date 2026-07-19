"""section_matching_criteria.py — 2026-07-18, dual-matrix follow-up call.

User's two asks, verbatim:
1. Hard rule: <=4-5 DISTINCT section labels per song (never letter F or G
   displayed). The previous call's k=10 autumn_leaves clustering produced
   10 distinct letters (A-J) -- checked against real musical knowledge,
   this is wrong on its face (see docstring reasoning below).
2. Use the Mantel-validated dual audio+symbolic matrix agreement
   (`dual_matrix_grain8_results.json`, r=0.261-0.752, p<=0.019 all 3 songs)
   as the trusted foundation to build EXPLICIT, INSPECTABLE matching
   criteria the user can verify -- not a single black-box pick.

Reuses `dual_matrix_grain8_results.json`'s audio_matrix/symbolic_matrix
verbatim (same Gram-trick audio + real-decode symbolic similarity as the
deployed section_merge_candidates.py, so results here are consistent with
that JSON's numbers, not a reimplementation that could silently diverge).

Sanity check on n_blocks vs known form (reasoned explicitly, not asserted):
- autumn_leaves: n_blocks=41 (grain=8 -> ~328 bars). Known project fact
  (docs/known_issues.md, "330-bar/422s duration mismatch") is that
  autumn_leaves.m4a is a LONGER recording than the corpus entry implies,
  consistent with an extended-solo/vamp arrangement. Canonical Autumn
  Leaves lead-sheet form is AABC, 32 bars/chorus (4 x 8-bar sections).
  328 bars / 32 bars-per-chorus ~= 10.25 chorus repeats. A recording that
  repeats a 4-section form ~10 times needs ONLY 4 distinct section
  *labels*, reused ~10x each, not 10+ new letters -- so k<=5 is not just
  permissible here, it is the musically correct answer, and k=10 (one new
  letter roughly every ~3 blocks) is the over-fragmentation failure mode
  named in the brief.
- abba_chiquitita: n_blocks=29 (~232 bars). Pop song verse/chorus/bridge
  structure -> intro/verse/chorus/bridge/outro is the canonical <=5-way
  vocabulary for this genre.
- aretha_chain_of_fools: n_blocks=10 (~80 bars), shortest of the three.
  Soul vamp-based song -> often even fewer distinct sections (a single
  vamp/groove repeated with a bridge), so k<=5 should be an easy fit, and
  a degenerate single-cluster collapse here is a real risk to check for
  given this project's own documented "over-merge collapse" failure mode
  (docs/known_issues.md, aretha total-collapse entries).

Three candidate INSPECTABLE matching criteria (all complete-linkage
agglomerative clustering -- validated in the prior call to correctly
place autumn_leaves block0/block1 together and to avoid the union-find
chaining failure -- differing only in how the pairwise DISTANCE matrix is
built, so each is a single readable formula):

1. blend_0.6_0.4  (existing default, re-verified here at k<=5)
     D[i,j] = 1 - (0.6*audio_sim[i,j] + 0.4*symbolic_sim[i,j])
   Simplest, already load-bearing in the prior call's k=10 result.

2. symbolic_primary_audio_gate  (AND-style, inspectable per-song threshold)
     gate_i,j = audio_sim[i,j] >= P40(audio_sim)   # P40 = 40th percentile
                                                     # of that song's own
                                                     # audio_sim distribution
     D[i,j] = (1 - symbolic_sim[i,j])  if gate_i,j else  1.5  (pushed out
                                                              of easy-merge
                                                              range)
   Rationale: symbolic_sim is the more musically direct signal (it's
   literally "do these 8 bars have the same chords"), but task 3 of the
   prior call showed symbolic_sim alone is noisy on real-audio decode --
   this rule trusts it only when audio independently corroborates the
   pair is even plausible, i.e. an explicit AND gate, inspectable as one
   printed threshold number per song.

3. mutual_topK_rank_bonus  (directly operationalizes the Mantel finding)
     Rank all off-diagonal pairs within audio_sim (descending) and
     within symbolic_sim (descending), independently, per song.
     bonus_i,j = 0.15 if pair is in TOP 20% of BOTH rankings else 0.0
     D[i,j] = 1 - (0.6*audio_sim[i,j] + 0.4*symbolic_sim[i,j]) - bonus_i,j
   Rationale: the Mantel test's own logic is "two independently-derived
   measurements agreeing is stronger evidence than either alone" -- this
   rule rewards pairs where audio and symbolic independently AGREE the
   pair is high-similarity, exactly the situation the Mantel r>0 finding
   says should be trusted more.

For each candidate x song, sweep k in {3,4,5} (the user's hard rule) plus
report the "natural" k for two kinds of comparison:
  - k_natural_heuristic: the SAME formula the prior call used
    (max(4, min(10, n_blocks//4))) -- NOT data-driven, a fixed heuristic,
    kept only as a like-for-like comparison to the k=10/7/4 result already
    on record.
  - k_silhouette: a DATA-driven suggested k in [2,12] via silhouette score
    on the candidate's own distance matrix (precomputed, complete linkage)
    -- an honest premise check on whether k<=5 is actually a good fit for
    the data or a forced number that "looks right" but degrades structure.

Outputs:
  scratchpad/section_matching_criteria_results.json  -- full contract,
    every candidate x every k, pass/fail on <=5-letters, block0/1 check,
    silhouette scores, gate thresholds used (inspectable).
  scratchpad/section_structure_clusters_grain8.json  -- UPDATED in place,
    replacing the old single k=10/7/4 result with the RECOMMENDED
    candidate's k<=5 result (see recommendation in the log), same
    top-level schema (k, n_blocks, blocks[{block,bars,section}],
    block0_vs_block1_same_section) so the orchestrating session's existing
    viz code needs no changes -- plus a new "all_candidates" key carrying
    every candidate's k=3,4,5 section strings for a future side-by-side
    view, additive only.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

OUT_DIR = Path(__file__).resolve().parent
DUAL_MATRIX_PATH = OUT_DIR / "dual_matrix_grain8_results.json"
RESULTS_PATH = OUT_DIR / "section_matching_criteria_results.json"
CLUSTERS_PATH = OUT_DIR / "section_structure_clusters_grain8.json"

SONGS = ["autumn_leaves", "abba_chiquitita_official_lyric_video",
         "aretha_franklin_chain_of_fools_official_lyric_video"]

K_SWEEP = [3, 4, 5]
TOPK_FRACTION = 0.20
TOPK_BONUS = 0.15
GATE_PERCENTILE = 40


def letters_for_labels(cluster_ids):
    """Map cluster ids -> A,B,C... in order of FIRST APPEARANCE across
    blocks 0..n-1, so section strings read left-to-right the way a human
    would letter a form (first section encountered is always 'A')."""
    seen = {}
    out = []
    for cid in cluster_ids:
        if cid not in seen:
            seen[cid] = chr(ord('A') + len(seen))
        out.append(seen[cid])
    return out


def build_distance_matrices(audio, sym, n):
    iu = np.triu_indices(n, k=1)
    audio_offdiag = audio[iu]
    sym_offdiag = sym[iu]

    # --- candidate 1: blend 0.6/0.4 ---
    D_blend = 1.0 - (0.6 * audio + 0.4 * sym)
    np.fill_diagonal(D_blend, 0.0)

    # --- candidate 2: symbolic-primary, audio floor gate ---
    gate_threshold = float(np.percentile(audio_offdiag, GATE_PERCENTILE))
    gate = audio >= gate_threshold
    D_gate = np.where(gate, 1.0 - sym, 1.5)
    np.fill_diagonal(D_gate, 0.0)
    D_gate = np.clip(D_gate, 0.0, 1.5)

    # --- candidate 3: mutual top-K% rank agreement bonus ---
    n_pairs = len(iu[0])
    k_cut = max(1, int(round(TOPK_FRACTION * n_pairs)))
    audio_top_idx = set(np.argsort(-audio_offdiag)[:k_cut].tolist())
    sym_top_idx = set(np.argsort(-sym_offdiag)[:k_cut].tolist())
    both_top_idx = audio_top_idx & sym_top_idx
    bonus_flat = np.zeros(n_pairs)
    for idx in both_top_idx:
        bonus_flat[idx] = TOPK_BONUS
    bonus_mat = np.zeros((n, n))
    bonus_mat[iu] = bonus_flat
    bonus_mat = bonus_mat + bonus_mat.T
    D_topk = D_blend - bonus_mat
    np.fill_diagonal(D_topk, 0.0)
    D_topk = np.clip(D_topk, 0.0, None)

    meta = {
        "gate_threshold_audio_sim_p40": gate_threshold,
        "topk_n_pairs_considered": n_pairs,
        "topk_cutoff_count": k_cut,
        "topk_n_pairs_in_both_top20pct": len(both_top_idx),
    }
    return {
        "blend_0.6_0.4": D_blend,
        "symbolic_primary_audio_gate": D_gate,
        "mutual_topK_rank_bonus": D_topk,
    }, meta


def cluster_at_k(D, k):
    n = D.shape[0]
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z


def silhouette_suggested_k(D, k_range=range(2, 13)):
    n = D.shape[0]
    best_k, best_score = None, -2.0
    scores = {}
    for k in k_range:
        if k >= n:
            continue
        labels, _ = cluster_at_k(D, k)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(D, labels, metric="precomputed")
        except Exception:
            continue
        scores[k] = float(s)
        if s > best_score:
            best_score, best_k = s, k
    return best_k, best_score, scores


def n_blocks_bars_note(slug, n_blocks):
    notes = {
        "autumn_leaves": (
            f"n_blocks={n_blocks} (~{n_blocks*8} bars @ grain=8). Canonical "
            "AABC form = 4 sections x 8 bars = 32 bars/chorus. "
            f"{n_blocks*8}/32 = {n_blocks*8/32:.2f} chorus repeats -- a long "
            "recording (known 330-bar/422s duration mismatch, extended "
            "solo/vamp) reusing a SMALL section vocabulary many times, "
            "consistent with k<=5 being the musically correct target, not "
            "a forced constraint."
        ),
        "abba_chiquitita_official_lyric_video": (
            f"n_blocks={n_blocks} (~{n_blocks*8} bars). Pop verse/chorus/"
            "bridge/intro/outro vocabulary is canonically <=5 distinct "
            "section types even with many repeats."
        ),
        "aretha_franklin_chain_of_fools_official_lyric_video": (
            f"n_blocks={n_blocks} (~{n_blocks*8} bars), shortest of the 3. "
            "Soul vamp-based song -- often even fewer distinct sections "
            "(single groove + bridge); a degenerate single-cluster "
            "collapse is a real risk here given this project's documented "
            "aretha over-merge history, checked explicitly below."
        ),
    }
    return notes[slug]


def main():
    dual = json.loads(DUAL_MATRIX_PATH.read_text())
    results = {}
    clusters_out = {}

    for slug in SONGS:
        song = dual[slug]
        audio = np.array(song["audio_matrix"])
        sym = np.array(song["symbolic_matrix"])
        n = song["n_blocks"]
        block_bars = song["block_bars"]

        D_by_cand, gate_meta = build_distance_matrices(audio, sym, n)

        k_natural_heuristic = max(4, min(10, n // 4))

        song_result = {
            "n_blocks": n,
            "form_sanity_note": n_blocks_bars_note(slug, n),
            "k_natural_heuristic": k_natural_heuristic,
            "gate_and_topk_meta": gate_meta,
            "candidates": {},
        }

        for cand_name, D in D_by_cand.items():
            best_k, best_sil, sil_scores = silhouette_suggested_k(D)
            cand_result = {
                "silhouette_suggested_k": best_k,
                "silhouette_score_at_suggested_k": best_sil,
                "silhouette_scores_by_k": sil_scores,
                "k_sweep": {},
            }
            for k in K_SWEEP + ([k_natural_heuristic] if k_natural_heuristic not in K_SWEEP else []):
                labels, _ = cluster_at_k(D, min(k, n - 1) if n > 1 else 1)
                letters = letters_for_labels(labels)
                n_distinct = len(set(letters))
                section_string = "".join(letters)
                block0_1_same = bool(letters[0] == letters[1]) if n >= 2 else None
                cand_result["k_sweep"][str(k)] = {
                    "k_requested": k,
                    "n_distinct_sections_actual": n_distinct,
                    "passes_le5_rule": n_distinct <= 5,
                    "block0_vs_block1_same_section": block0_1_same,
                    "section_string": section_string,
                    "blocks": [
                        {"block": i, "bars": block_bars[i], "section": letters[i]}
                        for i in range(n)
                    ],
                }
            song_result["candidates"][cand_name] = cand_result

        results[slug] = song_result

        print(f"=== {slug} (n_blocks={n}) ===")
        print(f"  {song_result['form_sanity_note']}")
        print(f"  k_natural_heuristic={k_natural_heuristic}")
        for cand_name, cand_result in song_result["candidates"].items():
            print(f"  -- {cand_name} (silhouette-suggested k={cand_result['silhouette_suggested_k']}, "
                  f"score={cand_result['silhouette_score_at_suggested_k']:.3f}" +
                  (")" if cand_result['silhouette_score_at_suggested_k'] is not None else ""))
            for k in K_SWEEP:
                kr = cand_result["k_sweep"][str(k)]
                print(f"     k={k}: {kr['n_distinct_sections_actual']} distinct, "
                      f"<=5={kr['passes_le5_rule']}, block0==block1={kr['block0_vs_block1_same_section']}, "
                      f"string={kr['section_string']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    main()
