"""matrix_intrinsic_k.py — 2026-07-19, follow-up to the k-prior work
(`k_prior_results.json`, 54.1% exact-match / 93.4% within-1 corpus-scale).

User's question, verbatim (translated): "if we take the similarity matrix
at grain=8 or grain=16, can we find a criterion to determine k from the
matrix itself (per-song), as a complement to the corpus length prior?"

This tests THREE matrix-intrinsic k-selection signals, done properly this
time (the earlier "spectral_eigengap" negative result in
`clustering_bakeoff_results.json` was on tiny per-tune bar-merge graphs,
m~4-15 blocks, for a totally different task — pairwise merge-candidate
generation, not section-count selection. Grain=8 section-level graphs are
comparably sized here too (autumn_leaves has 41 blocks, corpus tunes have
similar block counts once quantized) — so this is a genuine re-test, not
an assumption the old result transfers):

1. eigengap — normalized graph Laplacian of the block similarity matrix,
   look for the largest eigenvalue gap in k in {2,3,4,5}.
2. gap statistic (Tibshirani et al. 2001) — within-cluster dispersion Wk at
   each k, compared against a randomization null (B shuffles of the
   off-diagonal similarity values, symmetric, same clustering procedure
   applied) rather than raw silhouette.
3. singular-value knee — eigenvalue-drop elbow directly on the similarity
   matrix's spectrum (cheapest of the three, no clustering needed).

All three evaluated corpus-scale on the SAME 1992-tune iReal corpus and
SAME eval protocol as `k_prior_selection.py`'s `corpus_scale_validation`
(true_k = block-quantized majority-vote distinct section-label count,
reused verbatim via import — not reimplemented, so results are directly
comparable) — reused, not rebuilt, per CLAUDE.md rule #5 (single-tune
findings are hypotheses; corpus-scale is what counts as evidence here).

Then the best-performing matrix-intrinsic signal is combined with the
EXISTING corpus length prior into an updated 3-way rule and checked against
the existing 54.1%/93.4% combined-rule numbers.

Grain=16 is tested at corpus scale too (symbolic-only, same iReal chord
vectors, just larger blocks) and, for the 3 real songs specifically, using
the already-precomputed grain=16 AUDIO-ONLY matrix in
`bar_ssm_rawchroma_<song>.json` (`grains["16"]["similarity_matrix"]`, the
bt_concat combined-register Gram-trick matrix) since no grain=16 SYMBOLIC
(model-decode) matrix has been built for real audio yet — flagged
explicitly in the output as audio-only at that grain, not a full blend.

No reinfer calls, no writes to any scope-guarded file. Output:
`scratchpad/matrix_intrinsic_k_results.json`.
"""
from __future__ import annotations
import io
import json
import sys
import time
from pathlib import Path
from contextlib import redirect_stdout
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc  # noqa: E402
from chord_distance import chord_vector_binary  # noqa: E402
from chord_distance_eval import nuclear_spans, block_sim, key_pc  # noqa: E402
from symstruct import qbucket  # noqa: E402
from k_prior_selection import fit_prior, prior_pk_regression, score_and_pick  # noqa: E402 (READ-ONLY reuse)

FILES = ["jazz1460", "pop400", "blues50", "brazilian220",
         "country", "dixieland1", "latin_salsa50"]
K_RANGE = [2, 3, 4, 5]
OUT_PATH = REPO / "scratchpad" / "matrix_intrinsic_k_results.json"
RNG = np.random.default_rng(20260719)

REAL_SONGS_GRAIN8 = json.loads((REPO / "scratchpad" / "dual_matrix_grain8_results.json").read_text())
REAL_SONG_FILES_16 = {
    "autumn_leaves": "bar_ssm_rawchroma_autumn_leaves.json",
    "abba_chiquitita_official_lyric_video": "bar_ssm_rawchroma_abba_chiquitita.json",
    "aretha_franklin_chain_of_fools_official_lyric_video": "bar_ssm_rawchroma_aretha_chain_of_fools.json",
}
REAL_SONG_ADAPTIVE_K = {  # from k_prior_results.json / section_matching_criteria, for reference
    "autumn_leaves": 5,
    "abba_chiquitita_official_lyric_video": 4,
    "aretha_franklin_chain_of_fools_official_lyric_video": 3,
}


# ── shared: load one tune's grain-g symbolic block similarity matrix ───────

def build_tune_matrix(mc, grain, min_blocks=3):
    shift = (-key_pc(mc.key) % 12)
    bar_vecs, labels = [], []
    for bar_no, section, slots in mc.timeline:
        accum = None
        for (_, _, mma) in slots:
            pc = chord_root_pc(mma)
            if pc is None:
                continue
            rpc = (pc + shift) % 12
            q = qbucket(mma)
            v = chord_vector_binary(rpc, q)
            accum = v if accum is None else accum + v
        bar_vecs.append(accum if accum is not None else np.zeros(12))
        labels.append(section)
    n_bars = len(labels)
    if n_bars < grain * min_blocks or len(set(labels)) < 2:
        return None
    spans = nuclear_spans(n_bars, grain)
    m = len(spans)
    if m < min_blocks:
        return None
    block_bars = [bar_vecs[s:e] for (s, e) in spans]
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            S[i, j] = 1.0 if i == j else block_sim(block_bars[i], block_bars[j])
    S = np.clip(S, 0.0, 1.0)  # cosine-like sim on nonneg chord-tone vectors, clip stray negatives
    true_k, _ = block_level_true_k(labels, spans)
    return {"S": S, "n_blocks": m, "n_bars": n_bars, "true_k": true_k, "title": mc.title}


def block_level_true_k(labels, spans):
    block_labels = []
    for (s, e) in spans:
        c = Counter(labels[s:e])
        block_labels.append(c.most_common(1)[0][0])
    return len(set(block_labels)), block_labels


def load_corpus(grain, min_blocks=3, max_tunes=None):
    print(f"loading iReal corpus (grain={grain})...", file=sys.stderr)
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(REPO / "data" / "ireal" / f"{f}.txt")
        except Exception:
            continue
        for t in tunes:
            try:
                buf2 = io.StringIO()
                with redirect_stdout(buf2):
                    mc = tune_to_mma(t)
            except Exception:
                continue
            rec = build_tune_matrix(mc, grain, min_blocks)
            if rec is not None:
                out.append(rec)
            if max_tunes and len(out) >= max_tunes:
                print(f"  ({len(out)} tunes, stopping early: max_tunes)", file=sys.stderr)
                return out
    print(f"  {len(out)} tunes usable (n_blocks>={min_blocks}, >=2 sections)", file=sys.stderr)
    return out


# ── method 1: eigengap on normalized graph Laplacian ────────────────────────

def eigengap_scores(S, k_candidates):
    n = S.shape[0]
    A = S.copy()
    np.fill_diagonal(A, 0.0)  # affinity, no self-loops for the Laplacian
    deg = A.sum(axis=1)
    deg_safe = np.clip(deg, 1e-9, None)
    Dm12 = np.diag(1.0 / np.sqrt(deg_safe))
    L = np.eye(n) - Dm12 @ A @ Dm12
    eigvals = np.sort(np.linalg.eigvalsh((L + L.T) / 2))
    scores = {}
    for k in k_candidates:
        if k < n:
            scores[k] = float(eigvals[k] - eigvals[k - 1])
        else:
            scores[k] = -1.0
    return scores, eigvals[: max(k_candidates) + 1].tolist()


# ── method 2: gap statistic vs. randomization null ─────────────────────────

def within_dispersion(D, labels, k_set):
    Wk = 0.0
    for c in k_set:
        idx = np.where(labels == c)[0]
        nr = len(idx)
        if nr < 2:
            continue
        Dr = D[np.ix_(idx, idx)].sum() / 2.0  # each pair counted twice
        Wk += Dr / nr
    return Wk


def gap_statistic_scores(S, k_candidates, B=10, rng=RNG):
    n = S.shape[0]
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    condensed = squareform(D, checks=False)
    if condensed.size == 0 or not np.any(condensed > 1e-9):
        return {k: 0.0 for k in k_candidates}, {k: 0.0 for k in k_candidates}
    Z = linkage(condensed, method="complete")

    log_wk_obs = {}
    for k in k_candidates:
        if k >= n:
            continue
        labs = fcluster(Z, t=k, criterion="maxclust")
        wk = within_dispersion(D, labs, set(labs))
        log_wk_obs[k] = np.log(wk) if wk > 1e-12 else np.log(1e-12)

    # randomization null: shuffle off-diagonal similarity values (preserve
    # symmetry, zero diagonal, same marginal distribution, destroy any
    # block/cluster structure), same clustering pipeline applied
    iu = np.triu_indices(n, k=1)
    vals = S[iu].copy()
    null_log_wk = {k: [] for k in log_wk_obs}
    for _ in range(B):
        shuffled = rng.permutation(vals)
        Sn = np.eye(n)
        Sn[iu] = shuffled
        Sn[(iu[1], iu[0])] = shuffled
        Dn = 1.0 - Sn
        np.fill_diagonal(Dn, 0.0)
        Dn = np.clip(Dn, 0.0, None)
        condensed_n = squareform(Dn, checks=False)
        if not np.any(condensed_n > 1e-9):
            for k in log_wk_obs:
                null_log_wk[k].append(np.log(1e-12))
            continue
        Zn = linkage(condensed_n, method="complete")
        for k in log_wk_obs:
            labs = fcluster(Zn, t=k, criterion="maxclust")
            wk = within_dispersion(Dn, labs, set(labs))
            null_log_wk[k].append(np.log(wk) if wk > 1e-12 else np.log(1e-12))

    gap = {}
    sd = {}
    for k in log_wk_obs:
        nulls = np.array(null_log_wk[k])
        gap[k] = float(nulls.mean() - log_wk_obs[k])
        sd[k] = float(nulls.std() * np.sqrt(1 + 1.0 / B))
    return gap, sd


# ── method 3: singular-value knee on the raw similarity matrix ─────────────

def svd_knee_scores(S, k_candidates):
    n = S.shape[0]
    sv = np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]  # symmetric -> eigenvalues == "singular values" up to sign
    sv = np.clip(sv, 0.0, None)
    scores = {}
    for k in k_candidates:
        if k < n:
            scores[k] = float(sv[k - 1] - sv[k])
        else:
            scores[k] = -1.0
    return scores, sv[: max(k_candidates) + 1].tolist()


# ── corpus-scale evaluation harness (mirrors k_prior_selection's protocol) ─

def evaluate_method(tunes, score_fn, k_range=K_RANGE, **kwargs):
    results = []
    for rec in tunes:
        S, n, true_k = rec["S"], rec["n_blocks"], rec["true_k"]
        k_candidates = [k for k in k_range if 2 <= k < n]
        if len(k_candidates) < 2:
            continue
        scores, *_ = score_fn(S, k_candidates, **kwargs) if kwargs else score_fn(S, k_candidates)
        k_star = max(scores, key=lambda kk: scores[kk])
        results.append({
            "title": rec["title"], "n_bars": rec["n_bars"], "n_blocks": n,
            "true_k": true_k, "k_star": k_star,
            "match": int(k_star == true_k),
            "within1": int(abs(k_star - true_k) <= 1),
        })
    n_tot = len(results)
    if n_tot == 0:
        return {"n_tunes": 0}
    exact = sum(r["match"] for r in results) / n_tot
    within1 = sum(r["within1"] for r in results) / n_tot
    true_k_counts = Counter(r["true_k"] for r in results)
    mode_k, mode_n = true_k_counts.most_common(1)[0]
    return {
        "n_tunes": n_tot,
        "exact_match_rate": exact,
        "within1_rate": within1,
        "trivial_mode_baseline_exact_match_rate": mode_n / n_tot,
        "trivial_mode_baseline_k": mode_k,
        "predicted_k_distribution": dict(Counter(r["k_star"] for r in results)),
        "true_k_distribution": dict(true_k_counts),
        "sample": results[:15],
    }


def evaluate_gap_statistic(tunes, B=10):
    results = []
    t0 = time.time()
    for i, rec in enumerate(tunes):
        S, n, true_k = rec["S"], rec["n_blocks"], rec["true_k"]
        k_candidates = [k for k in K_RANGE if 2 <= k < n]
        if len(k_candidates) < 2:
            continue
        gap, sd = gap_statistic_scores(S, k_candidates, B=B)
        if not gap:
            continue
        k_star = max(gap, key=lambda kk: gap[kk])
        results.append({
            "title": rec["title"], "n_bars": rec["n_bars"], "n_blocks": n,
            "true_k": true_k, "k_star": k_star,
            "match": int(k_star == true_k),
            "within1": int(abs(k_star - true_k) <= 1),
        })
        if (i + 1) % 300 == 0:
            print(f"    gap-statistic: {i+1}/{len(tunes)} tunes, {time.time()-t0:.1f}s elapsed", file=sys.stderr)
    n_tot = len(results)
    if n_tot == 0:
        return {"n_tunes": 0}
    exact = sum(r["match"] for r in results) / n_tot
    within1 = sum(r["within1"] for r in results) / n_tot
    true_k_counts = Counter(r["true_k"] for r in results)
    mode_k, mode_n = true_k_counts.most_common(1)[0]
    return {
        "n_tunes": n_tot, "exact_match_rate": exact, "within1_rate": within1,
        "trivial_mode_baseline_exact_match_rate": mode_n / n_tot,
        "trivial_mode_baseline_k": mode_k,
        "predicted_k_distribution": dict(Counter(r["k_star"] for r in results)),
        "true_k_distribution": dict(true_k_counts),
        "sample": results[:15],
        "B_null_shuffles": B,
    }


# ── combined rule: prior + best matrix-intrinsic signal ────────────────────

def evaluate_combined(tunes, prior, matrix_score_fn, weight, k_floor=2, **kwargs):
    results = []
    for rec in tunes:
        S, n, true_k, n_bars = rec["S"], rec["n_blocks"], rec["true_k"], rec["n_bars"]
        k_candidates = [k for k in K_RANGE if k_floor <= k < n]
        if len(k_candidates) < 2:
            continue
        pk, _ = prior_pk_regression(n_bars, prior)
        mscores = matrix_score_fn(S, k_candidates, **kwargs)
        if isinstance(mscores, tuple):
            mscores = mscores[0]
        if not mscores:
            continue
        # normalize matrix score to comparable scale (z-score across candidates)
        vals = np.array(list(mscores.values()))
        if vals.std() > 1e-9:
            norm = {k: (v - vals.mean()) / vals.std() for k, v in mscores.items()}
        else:
            norm = {k: 0.0 for k in mscores}
        combo = {}
        for k in k_candidates:
            if k not in mscores:
                continue
            p = max(pk.get(k, 1e-6), 1e-6)
            combo[k] = np.log(p) + weight * norm[k]
        if not combo:
            continue
        k_star = max(combo, key=lambda kk: combo[kk])
        results.append({"true_k": true_k, "k_star": k_star,
                         "match": int(k_star == true_k), "within1": int(abs(k_star - true_k) <= 1)})
    n_tot = len(results)
    if n_tot == 0:
        return {"n_tunes": 0}
    return {
        "n_tunes": n_tot,
        "exact_match_rate": sum(r["match"] for r in results) / n_tot,
        "within1_rate": sum(r["within1"] for r in results) / n_tot,
    }


# ── real songs (grain=8, using existing dual-matrix data) ──────────────────

def real_songs_grain8():
    out = {}
    for slug, data in REAL_SONGS_GRAIN8.items():
        audio = np.array(data["audio_matrix"])
        sym = np.array(data["symbolic_matrix"])
        S = np.clip(0.6 * audio + 0.4 * sym, 0.0, 1.0)  # blend_0.6_0.4, the deployed default
        np.fill_diagonal(S, 1.0)
        n = S.shape[0]
        k_candidates = [k for k in K_RANGE if 2 <= k < n]
        eg, eigvals = eigengap_scores(S, k_candidates)
        gap, sd = gap_statistic_scores(S, k_candidates, B=200)
        svd, sv = svd_knee_scores(S, k_candidates)
        out[slug] = {
            "n_blocks": n,
            "adaptive_heuristic_k_for_reference": REAL_SONG_ADAPTIVE_K[slug],
            "eigengap": {"scores_by_k": eg, "k_star": max(eg, key=lambda kk: eg[kk]),
                         "first_6_laplacian_eigenvalues": eigvals},
            "gap_statistic": {"gap_by_k": gap, "se_by_k": sd, "k_star": max(gap, key=lambda kk: gap[kk])},
            "svd_knee": {"scores_by_k": svd, "k_star": max(svd, key=lambda kk: svd[kk]),
                         "top_6_singular_values": sv},
        }
    return out


def real_songs_grain16_audio_only():
    out = {}
    for slug, fname in REAL_SONG_FILES_16.items():
        d = json.loads((REPO / "scratchpad" / fname).read_text())
        S = np.array(d["grains"]["16"]["similarity_matrix"])
        S = np.clip(S, 0.0, 1.0)
        np.fill_diagonal(S, 1.0)
        n = S.shape[0]
        k_candidates = [k for k in K_RANGE if 2 <= k < n]
        if len(k_candidates) < 2:
            out[slug] = {"n_blocks": n, "note": "too few grain=16 blocks for k in [2,5] sweep"}
            continue
        eg, eigvals = eigengap_scores(S, k_candidates)
        gap, sd = gap_statistic_scores(S, k_candidates, B=200)
        svd, sv = svd_knee_scores(S, k_candidates)
        out[slug] = {
            "n_blocks": n, "note": "AUDIO-ONLY (bt_concat Gram-trick), no grain=16 symbolic matrix exists yet",
            "adaptive_heuristic_k_for_reference": REAL_SONG_ADAPTIVE_K[slug],
            "eigengap": {"scores_by_k": eg, "k_star": max(eg, key=lambda kk: eg[kk])},
            "gap_statistic": {"gap_by_k": gap, "k_star": max(gap, key=lambda kk: gap[kk])},
            "svd_knee": {"scores_by_k": svd, "k_star": max(svd, key=lambda kk: svd[kk])},
        }
    return out


def main():
    t_start = time.time()
    print("Fitting prior (reused from k_prior_selection.fit_prior)...", file=sys.stderr)
    prior = fit_prior()

    print("Loading corpus grain=8...", file=sys.stderr)
    tunes8 = load_corpus(grain=8)

    print("Method 1: eigengap (corpus grain=8)...", file=sys.stderr)
    eigengap_corpus = evaluate_method(tunes8, eigengap_scores)
    print(f"  exact={eigengap_corpus['exact_match_rate']:.3f} within1={eigengap_corpus['within1_rate']:.3f}", file=sys.stderr)

    print("Method 2: gap statistic (corpus grain=8, B=10)...", file=sys.stderr)
    gapstat_corpus = evaluate_gap_statistic(tunes8, B=10)
    print(f"  exact={gapstat_corpus['exact_match_rate']:.3f} within1={gapstat_corpus['within1_rate']:.3f}", file=sys.stderr)

    print("Method 3: SVD knee (corpus grain=8)...", file=sys.stderr)
    svdknee_corpus = evaluate_method(tunes8, svd_knee_scores)
    print(f"  exact={svdknee_corpus['exact_match_rate']:.3f} within1={svdknee_corpus['within1_rate']:.3f}", file=sys.stderr)

    print("Loading corpus grain=16...", file=sys.stderr)
    tunes16 = load_corpus(grain=16, min_blocks=3)
    eigengap16 = evaluate_method(tunes16, eigengap_scores)
    gapstat16 = evaluate_gap_statistic(tunes16, B=10)
    svdknee16 = evaluate_method(tunes16, svd_knee_scores)
    print(f"  grain16 n_tunes={eigengap16.get('n_tunes')} eigengap_exact={eigengap16.get('exact_match_rate')}", file=sys.stderr)

    # pick best matrix-intrinsic method @ grain=8 for the combined rule
    candidates_summary = {
        "eigengap": eigengap_corpus["exact_match_rate"],
        "gap_statistic": gapstat_corpus["exact_match_rate"],
        "svd_knee": svdknee_corpus["exact_match_rate"],
    }
    best_method_name = max(candidates_summary, key=lambda k: candidates_summary[k])
    best_fn = {"eigengap": eigengap_scores, "gap_statistic": None, "svd_knee": svd_knee_scores}[best_method_name]

    print(f"Best matrix-intrinsic method: {best_method_name} ({candidates_summary[best_method_name]:.3f} exact-match)", file=sys.stderr)

    combined_sweep = {}
    if best_method_name == "gap_statistic":
        # gap statistic needs its own combined-rule path (different signature)
        def gap_score_fn(S, k_candidates):
            gap, _ = gap_statistic_scores(S, k_candidates, B=6)
            return gap
        for w in [0.5, 1.0, 2.0, 5.0]:
            combined_sweep[str(w)] = evaluate_combined(tunes8, prior, gap_score_fn, w)
    else:
        for w in [0.5, 1.0, 2.0, 5.0]:
            combined_sweep[str(w)] = evaluate_combined(tunes8, prior, best_fn, w)
    print("Combined-rule (prior + best matrix signal) weight sweep:", file=sys.stderr)
    for w, r in combined_sweep.items():
        print(f"  weight={w}: exact={r.get('exact_match_rate'):.3f} within1={r.get('within1_rate'):.3f}", file=sys.stderr)

    print("Real songs, grain=8 (blend_0.6_0.4 matrix)...", file=sys.stderr)
    real8 = real_songs_grain8()
    print("Real songs, grain=16 (audio-only)...", file=sys.stderr)
    real16 = real_songs_grain16_audio_only()

    # existing baseline numbers for direct comparison (read-only reuse)
    k_prior_existing = json.loads((REPO / "scratchpad" / "k_prior_results.json").read_text())
    existing_baseline = {
        "combined_rule_exact_match_rate": k_prior_existing["corpus_scale_validation"]["combined_rule_exact_match_rate"],
        "combined_rule_within1_rate": k_prior_existing["corpus_scale_validation"]["combined_rule_within1_rate"],
        "prior_only_exact_match_rate": k_prior_existing["corpus_scale_validation"]["prior_only_exact_match_rate"],
        "silhouette_only_exact_match_rate": k_prior_existing["corpus_scale_validation"]["silhouette_only_exact_match_rate"],
        "trivial_mode_baseline_exact_match_rate": k_prior_existing["corpus_scale_validation"]["trivial_mode_baseline_exact_match_rate"],
    }

    out = {
        "meta": {
            "elapsed_s": time.time() - t_start,
            "corpus_n_tunes_grain8": eigengap_corpus.get("n_tunes"),
            "corpus_n_tunes_grain16": eigengap16.get("n_tunes"),
        },
        "existing_baseline_for_comparison": existing_baseline,
        "grain8_corpus_scale": {
            "eigengap": eigengap_corpus,
            "gap_statistic": gapstat_corpus,
            "svd_knee": svdknee_corpus,
        },
        "grain16_corpus_scale": {
            "eigengap": eigengap16,
            "gap_statistic": gapstat16,
            "svd_knee": svdknee16,
        },
        "best_matrix_intrinsic_method": best_method_name,
        "combined_rule_weight_sweep": combined_sweep,
        "real_songs_grain8_blend": real8,
        "real_songs_grain16_audio_only": real16,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, default=float))
    print(f"Wrote {OUT_PATH} ({time.time()-t_start:.1f}s total)", file=sys.stderr)


if __name__ == "__main__":
    main()
