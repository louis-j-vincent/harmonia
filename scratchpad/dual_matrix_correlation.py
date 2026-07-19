"""dual_matrix_correlation.py — 2026-07-18, dual-matrix cross-validation
(user's idea, translated from their "matrice a bar 1" framing).

Real audio has NO ground truth for structure. Every threshold derived on
symbolic iReal data has been shown NOT to transfer to real audio (tau_auto
=0.96, see docs/known_issues.md "AUTO-tier auto-apply WIRED and MEASURED"
and scratchpad/realaudio_threshold_check.py: pooled agreement only
39.4%-62.5% instead of the claimed 98-99%, because a threshold calibrated
on iReal's clean symbolic proxy feature was ported unchanged onto real
audio's continuous rawchroma.bt_concat feature space).

This script does NOT port any threshold across corpora or feature spaces.
Instead, for each of the 3 real songs, it builds TWO independent 1-bar
self-similarity matrices from THAT SAME SONG:

  1. SYMBOLIC matrix: the production pipeline's own decode
     (harmonia.models.chord_pipeline_v1.infer_chords_v1, same call already
     cached in scratchpad/baseline_chords_<slug>.json from tonight's
     realaudio_threshold_check.py run) -> one chord label per bar (majority
     vote over sub-bar chord segments, by total duration covered) -> a
     chord-tone-distance SSM using chord_distance.py's V1/V2/V3 schemes.
  2. AUDIO matrix: the existing untrained raw-chroma V4 SSM
     (scratchpad/bar_ssm_rawchroma_<song>.json, grain=1, bt_concat variant,
     already computed and validated tonight) -- reused as-is, NOT rebuilt.

Then it correlates the two matrices for the SAME song (never across songs,
never against a foreign corpus) and asks: do two independently-derived
views of "which bars are similar" agree more than chance? And: does that
agreement look different near the song boundaries (candidate intro/outro
padding) than in a matched-size interior window?

Validation is necessarily indirect (no real-audio structure GT exists) --
cross-matrix agreement is the only signal available, stated explicitly,
same epistemic caveat as the realaudio pseudo-GT check earlier tonight.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from chord_distance import (chord_vector_binary, chord_vector_weighted,
                             chord_vector_tiv, cosine)
from rawchroma import per_bar_rawchroma

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = REPO / "docs" / "audio"

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# short-name (bar_ssm_rawchroma_<X>.json) <-> slug (baseline_chords_<X>.json, audio file)
SONGS = {
    "aretha_chain_of_fools": {
        "slug": "aretha_franklin_chain_of_fools_official_lyric_video",
        "audio_name": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    },
    "autumn_leaves": {
        "slug": "autumn_leaves",
        "audio_name": "autumn_leaves.m4a",
    },
    "abba_chiquitita": {
        "slug": "abba_chiquitita_official_lyric_video",
        "audio_name": "abba_chiquitita_official_lyric_video.m4a",
    },
}

SCHEMES = {
    "V1_binary": chord_vector_binary,
    "V2_weighted": chord_vector_weighted,
    "V3_tiv": chord_vector_tiv,
}

N_PERM = 999
RNG_SEED = 0
K_SWEEP = [4, 8, 16]
N_RANDOM_WINDOWS = 20  # interior-window controls per K, per song


def label_to_root_qual(label):
    """Billboard-style label ('C#:maj7','D:min','N',...) -> (root_pc, qual_idx)
    where qual_idx indexes chord_distance.QBUCKET_INTERVALS (0 maj,1 min,
    2 dom7-family,3 dim,4 aug,5 sus4). This is a coarser bucket than the
    model's own label vocabulary (e.g. hdim7/m7b5 has no dedicated bucket in
    chord_distance.py's 6 families -- mapped to dim, since a half-diminished
    triad IS a diminished triad; only the 7th differs, and QBUCKET_INTERVALS
    is triad/tetrad-membership based, not degree-exact). Documented
    approximation, not a lossless remap."""
    if not label or label in ("N", "X"):
        return None, None
    if ":" in label:
        root, qual = label.split(":", 1)
    else:
        root, qual = label, "maj"
    try:
        pc = NOTE.index(root)
    except ValueError:
        return None, None
    q = qual.lower()
    if "hdim" in q or "m7b5" in q:
        qual_idx = 3       # half-dim triad == dim triad (chord_distance has no hdim family)
    elif q.startswith("dim") or q == "o":
        qual_idx = 3
    elif "aug" in q or q == "+":
        qual_idx = 4
    elif "sus" in q:
        qual_idx = 5
    elif q.startswith("min") or (q.startswith("m") and not q.startswith("maj")):
        qual_idx = 1
    elif "maj" in q:
        qual_idx = 0
    elif any(t in q for t in ("7", "9", "13", "11", "alt")):
        qual_idx = 2        # dominant/7th family
    else:
        qual_idx = 0
    return pc, qual_idx


def bar_chord_majority(chords, t0, t1):
    """Majority-by-duration chord label active within [t0,t1). Falls back to
    the label at the bar midpoint if no segment overlaps meaningfully."""
    best_label, best_dur = None, 0.0
    for c in chords:
        ov = min(t1, c["end_s"]) - max(t0, c["start_s"])
        if ov > best_dur:
            best_dur, best_label = ov, c["label"]
    if best_label is None:
        mid = 0.5 * (t0 + t1)
        for c in chords:
            if c["start_s"] <= mid < c["end_s"]:
                return c["label"]
    return best_label


def build_symbolic_matrices(bar_times, base_chords):
    n_bars = len(bar_times) - 1
    root_qual = []
    for b in range(n_bars):
        label = bar_chord_majority(base_chords, bar_times[b], bar_times[b + 1])
        pc, q = label_to_root_qual(label)
        root_qual.append((pc, q))
    mats = {}
    for name, fn in SCHEMES.items():
        vecs = [fn(pc, q) if pc is not None else
                (np.zeros(6, dtype=complex) if name == "V3_tiv" else np.zeros(12))
                for pc, q in root_qual]
        sim = np.zeros((n_bars, n_bars))
        for i in range(n_bars):
            for j in range(n_bars):
                sim[i, j] = cosine(vecs[i], vecs[j])
        mats[name] = sim
    return mats, root_qual


def pearson_upper(a_flat, b_flat):
    if a_flat.std() < 1e-12 or b_flat.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def mantel_test(sym_mat, audio_mat, n_perm=N_PERM, seed=RNG_SEED):
    """Permutation test for two n x n similarity matrices on the SAME
    objects (bars): permute node labels of one matrix jointly (rows+cols),
    recompute the naive upper-triangle correlation each time, and compare
    the OBSERVED correlation to this null distribution. This controls for
    the strong local/block autocorrelation structure that inflates a naive
    Pearson correlation's apparent significance for SSMs specifically."""
    n = sym_mat.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    sym_flat = sym_mat[iu, ju]
    obs = pearson_upper(sym_flat, audio_mat[iu, ju])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        a_perm = audio_mat[perm][:, perm]
        null[p] = pearson_upper(sym_flat, a_perm[iu, ju])
    p_value = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    z = float((obs - null.mean()) / (null.std() + 1e-12))
    return {
        "observed_r": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "z_score": z,
        "p_value_two_sided": p_value,
        "n_perm": n_perm,
    }


def boundary_window_stats(sym_mat, audio_mat, n, K, n_random=N_RANDOM_WINDOWS, seed=RNG_SEED):
    """Compare correlation over bar-pairs INVOLVING the first/last K bars vs
    a matched-size RANDOM interior window's bar-pairs. 'Involving' = at
    least one of the pair's two bars is in the boundary/window set (matches
    a same-size control, not just 'vs the rest')."""
    iu, ju = np.triu_indices(n, k=1)
    sym_flat_all = sym_mat[iu, ju]
    audio_flat_all = audio_mat[iu, ju]

    def pairs_involving(idx_set):
        mask = np.isin(iu, list(idx_set)) | np.isin(ju, list(idx_set))
        return mask

    boundary_idx = set(range(0, min(K, n))) | set(range(max(0, n - K), n))
    b_mask = pairs_involving(boundary_idx)
    boundary_r = pearson_upper(sym_flat_all[b_mask], audio_flat_all[b_mask])
    n_boundary_pairs = int(b_mask.sum())

    interior_lo, interior_hi = min(K, n), max(0, n - K)
    rng = np.random.default_rng(seed + K)
    random_rs = []
    if interior_hi - interior_lo >= 2 * K:
        for _ in range(n_random):
            start = rng.integers(interior_lo, interior_hi - 2 * K + 1)
            window = set(range(start, start + 2 * K))
            m = pairs_involving(window)
            random_rs.append(pearson_upper(sym_flat_all[m], audio_flat_all[m]))
    random_rs = np.array(random_rs) if random_rs else np.array([np.nan])

    # single-matrix boundary "signature": mean similarity of boundary bars
    # to the REST of the song (not to each other), per matrix.
    rest_idx = [i for i in range(n) if i not in boundary_idx]
    def mean_sim_to_rest(mat):
        if not rest_idx or not boundary_idx:
            return None
        vals = [mat[i, j] for i in boundary_idx for j in rest_idx if i != j]
        return float(np.mean(vals)) if vals else None
    overall_mean_sim = {
        "symbolic": float(np.mean([sym_mat[i, j] for i in range(n) for j in range(n) if i != j])),
        "audio": float(np.mean([audio_mat[i, j] for i in range(n) for j in range(n) if i != j])),
    }

    return {
        "K": K,
        "n_boundary_pairs": n_boundary_pairs,
        "boundary_correlation_r": boundary_r,
        "random_interior_window_r_mean": float(np.nanmean(random_rs)),
        "random_interior_window_r_std": float(np.nanstd(random_rs)),
        "random_interior_window_n_draws": int(np.sum(~np.isnan(random_rs))),
        "boundary_mean_sim_to_rest": {
            "symbolic": mean_sim_to_rest(sym_mat),
            "audio": mean_sim_to_rest(audio_mat),
        },
        "corpus_mean_sim_overall": overall_mean_sim,
    }


def process_song(short_name, cfg):
    print("=== %s ===" % short_name, file=sys.stderr)
    audio_path = AUDIO_DIR / cfg["audio_name"]
    ssm_path = OUT_DIR / ("bar_ssm_rawchroma_%s.json" % short_name)
    baseline_path = OUT_DIR / ("baseline_chords_%s.json" % cfg["slug"])
    audio_ssm_data = json.loads(ssm_path.read_text())
    base_chords = json.loads(baseline_path.read_text())
    audio_mat = np.array(audio_ssm_data["grains"]["1"]["similarity_matrix"])
    n_audio = audio_mat.shape[0]

    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    n_bar_pipeline = len(variants["bt_concat"])
    assert n_bar_pipeline == n_audio, (
        "bar-count mismatch: per_bar_rawchroma gives %d bars, stored SSM has %d -- "
        "cannot align indices, STOP" % (n_bar_pipeline, n_audio))
    n = n_audio

    sym_mats, root_qual = build_symbolic_matrices(bar_times, base_chords)

    result = {"song": short_name, "n_bars": n, "tempo_bpm": tempo}

    # --- cheap premise check first: does ANY scheme correlate above chance? ---
    premise = {}
    for scheme_name, sym_mat in sym_mats.items():
        mt = mantel_test(sym_mat, audio_mat)
        premise[scheme_name] = mt
        print("  [premise] %-12s observed_r=%.4f  null=%.4f+-%.4f  z=%.2f  p=%.4f" %
              (scheme_name, mt["observed_r"], mt["null_mean"], mt["null_std"],
               mt["z_score"], mt["p_value_two_sided"]), file=sys.stderr)
    result["global_mantel"] = premise

    # pick the best-correlating scheme for the boundary analysis (report all,
    # but sweep K only for the scheme with the strongest global signal, to
    # bound compute -- documented, not silently narrowed)
    best_scheme = max(premise, key=lambda k: premise[k]["observed_r"])
    result["boundary_scheme_used"] = best_scheme

    boundary = []
    for K in K_SWEEP:
        stats = boundary_window_stats(sym_mats[best_scheme], audio_mat, n, K)
        boundary.append(stats)
        print("  [boundary K=%2d] boundary_r=%.4f  random_interior_r=%.4f+-%.4f (n_draws=%d)" %
              (K, stats["boundary_correlation_r"], stats["random_interior_window_r_mean"],
               stats["random_interior_window_r_std"], stats["random_interior_window_n_draws"]),
              file=sys.stderr)
    result["boundary_sweep"] = boundary

    # hand-inspectable example: top-3 highest-audio-similarity bar pairs,
    # show what the two matrices + raw labels say about them
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(-audio_mat[iu, ju])[:5]
    examples = []
    for idx in order:
        i, j = int(iu[idx]), int(ju[idx])
        examples.append({
            "bars": [i, j],
            "audio_sim": float(audio_mat[i, j]),
            "symbolic_sim_by_scheme": {k: float(sym_mats[k][i, j]) for k in sym_mats},
            "labels": [bar_chord_majority(base_chords, bar_times[i], bar_times[i + 1]),
                       bar_chord_majority(base_chords, bar_times[j], bar_times[j + 1])],
        })
    result["hand_inspect_examples_top_audio_sim"] = examples

    result["matrices"] = {
        "audio_bt_concat": audio_mat.tolist(),
        **{("symbolic_%s" % k): v.tolist() for k, v in sym_mats.items()},
    }
    result["bar_root_qual"] = [[p, q] for p, q in root_qual]
    result["bar_times"] = bar_times.tolist()

    return result


def main():
    all_results = {}
    for short_name, cfg in SONGS.items():
        all_results[short_name] = process_song(short_name, cfg)

    out_path = OUT_DIR / "dual_matrix_correlation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print("\nwrote %s" % out_path, file=sys.stderr)

    print("\n=== SUMMARY (global Mantel, all schemes, all songs) ===", file=sys.stderr)
    for song, res in all_results.items():
        for scheme, mt in res["global_mantel"].items():
            sig = "***" if mt["p_value_two_sided"] < 0.01 else ("*" if mt["p_value_two_sided"] < 0.05 else "ns")
            print("  %-20s %-12s r=%+.4f  z=%+.2f  p=%.4f  %s" %
                  (song, scheme, mt["observed_r"], mt["z_score"], mt["p_value_two_sided"], sig),
                  file=sys.stderr)


if __name__ == "__main__":
    main()
