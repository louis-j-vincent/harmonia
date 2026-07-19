"""hierarchy_shortcut.py — Step 0 of the 2026-07-18 SSM-model brief: verify
the "coarser grains are derivable from the finest-grain Gram matrix" claim.

Cosine similarity does NOT compose across grains (it's normalized/nonlinear),
but block_sim's numerator IS a plain sum of per-bar-position raw dot
products (see chord_distance_eval.block_sim docstring), and squared norms
are sums of per-bar squared norms. So:

  1. Compute the 1-bar raw (unnormalized) dot-product Gram matrix G1[i,j] =
     dot(bar_i, bar_j) ONCE, plus per-bar squared norms sq[i] = G1[i,i].
  2. For ANY coarser grain (block size k, position-aligned pairing per
     nuclear_spans), the numerator for blocks (s_i..s_i+L) vs (s_j..s_j+L)
     (L = min block length, per block_sim's own truncation rule) is
     sum_{t=0}^{L-1} G1[s_i+t, s_j+t] — a sum along one diagonal of G1,
     offset d = s_j - s_i. Precomputing PREFIX SUMS along every diagonal of
     G1 turns this into an O(1) lookup per block pair instead of an O(k)
     re-walk, and — the actual point of this check — it means G1 is computed
     ONCE and every coarser grain matrix is DERIVED from it, never
     recomputed from raw per-bar vectors again.

This script builds the diagonal-prefix-sum structure from G1 for both the
bass_only and treble_only registers (per 2026-07-18 coordinator note: keep
registers separate, do not collapse to bt_concat) on the 3 real-audio songs,
derives sizes {2,4,8,16} from size=1, and checks the result against the
EXISTING full-recompute matrices in bar_ssm_rawchroma_<song>.json to float
precision. If this doesn't match, per the brief: stop, there's a bug, don't
build anything on top of it.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from chord_distance_eval import nuclear_spans
from rawchroma import per_bar_rawchroma

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
SIZES = [1, 2, 4, 8, 16]

REAL_SONGS = {
    "aretha_chain_of_fools": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    "autumn_leaves": "autumn_leaves.m4a",
    "abba_chiquitita": "abba_chiquitita_official_lyric_video.m4a",
}


def diagonal_prefix_sums(G):
    """G: (n,n) raw dot-product Gram matrix. Returns dict offset d ->
    1-D prefix-sum array over the diagonal {G[i, i+d]} for valid i, such
    that diag_sum(d, i0, i1) = sum_{i=i0}^{i1} G[i, i+d] is a lookup."""
    n = G.shape[0]
    prefix = {}
    for d in range(-(n - 1), n):
        if d >= 0:
            vals = np.array([G[i, i + d] for i in range(n - d)])
        else:
            vals = np.array([G[i, i + d] for i in range(-d, n)])
        # prefix[d][k] = sum of first k values of this diagonal, aligned so
        # that prefix[d] index 0 corresponds to the SAME i as vals index 0
        cs = np.concatenate([[0.0], np.cumsum(vals)])
        prefix[d] = cs
    return prefix


def diag_sum(prefix, n, d, i0, length):
    """sum_{t=0}^{length-1} G[i0+t, i0+t+d], using the precomputed diagonal
    prefix-sum table. i0 is the row index of the first element."""
    cs = prefix[d]
    # for d>=0, vals[k] corresponds to row k (k=0..n-d-1); i0 maps directly.
    # for d<0, vals[k] corresponds to row k-d (k=0..n+d-1); need offset.
    if d >= 0:
        start = i0
    else:
        start = i0 + d  # vals index = i0 - (-d) = i0+d
    return float(cs[start + length] - cs[start])


def derive_grain(G, sq, n, size):
    spans = nuclear_spans(n, size)
    m = len(spans)
    prefix = diagonal_prefix_sums(G)
    sim = np.zeros((m, m))
    lengths = [e - s for (s, e) in spans]
    block_sqnorm = [None] * m
    for k, (s, e) in enumerate(spans):
        block_sqnorm[k] = float(np.sum(sq[s:e]))
    for i in range(m):
        si, ei = spans[i]
        for j in range(m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            d = sj - si
            num = diag_sum(prefix, n, d, si, L)
            na = np.sqrt(float(np.sum(sq[si:si + L])))
            nb = np.sqrt(float(np.sum(sq[sj:sj + L])))
            if na < 1e-9 or nb < 1e-9:
                sim[i, j] = 0.0
            else:
                sim[i, j] = num / (na * nb)
    return sim


def verify_song(song, audio_path):
    print("=== %s ===" % song)
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    ref_path = OUT_DIR / ("bar_ssm_rawchroma_%s.json" % song)
    ref = json.loads(ref_path.read_text())

    results = {}
    for reg, key in (("bass_only", "grains_bass"), ("treble_only", "grains_treble")):
        v = variants[reg]  # (n_bars, 12), unit-norm per bar
        n = len(v)
        t0 = time.time()
        G = v @ v.T  # raw dot-product Gram matrix, computed ONCE
        sq = np.diag(G).copy()
        t_gram = time.time() - t0

        maxdiffs = {}
        t_derive_total = 0.0
        for size in SIZES:
            if size == 1:
                continue
            if str(size) not in ref[key]:
                continue
            t1 = time.time()
            derived = derive_grain(G, sq, n, size)
            t_derive_total += time.time() - t1
            existing = np.array(ref[key][str(size)]["similarity_matrix"])
            diff = np.abs(derived - existing).max()
            maxdiffs[size] = diff
        results[reg] = {"n_bars": n, "t_gram_s": t_gram,
                         "t_derive_all_sizes_s": t_derive_total,
                         "max_abs_diff_per_size": maxdiffs}
        print("  [%s] n_bars=%d  Gram build %.4fs, derive(2,4,8,16) total %.4fs" %
              (reg, n, t_gram, t_derive_total))
        for size, diff in maxdiffs.items():
            status = "OK" if diff < 1e-6 else "MISMATCH"
            print("    size=%2d  max_abs_diff=%.2e  [%s]" % (size, diff, status))
    return results


def main():
    all_results = {}
    for song, fname in REAL_SONGS.items():
        audio = REPO / "docs" / "audio" / fname
        all_results[song] = verify_song(song, audio)
    ok = all(diff < 1e-6
              for song_res in all_results.values()
              for reg_res in song_res.values()
              for diff in reg_res["max_abs_diff_per_size"].values())
    print("\nOVERALL: %s" % ("ALL MATCH (<1e-6)" if ok else "MISMATCH FOUND — STOP, DO NOT BUILD ON THIS"))
    (OUT_DIR / "hierarchy_shortcut_verify.json").write_text(json.dumps(all_results, indent=2))
    print("wrote hierarchy_shortcut_verify.json")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
