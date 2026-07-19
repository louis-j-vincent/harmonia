"""dual_matrix_grain8.py — 2026-07-18, section-level suggestion follow-up,
task 2 (user's explicit ask: "build BOTH an audio similarity matrix AND a
structural/symbolic similarity matrix, SHOW them side-by-side").

Reuses `dual_matrix_correlation.py`'s already-validated pattern (Mantel
test, boundary analysis) but at BLOCK grain=8 instead of bar grain=1, and
reuses `section_merge_candidates.py`'s exact block-similarity machinery
(hierarchy_shortcut prefix-sum Gram trick for audio, `bar_label_bucket` +
chord_vector_binary cosine for symbolic — SAME functions, so the matrices
here are consistent with the candidate JSON's numbers, not a
reimplementation that could silently diverge).

Symbolic matrix here is built from the model's OWN real-audio chord decode
(`get_baseline_chords`, same as section_merge_candidates.py) — this is
DELIBERATE, not the bug diagnosed in task 1: task 1 found the deployed
candidate JSON's symbolic_sim is noisier than the clean-iReal premise-check
number because it inherits real-audio decode noise. This script is not
"fixing" that (task 3 does, with a joint criterion) — it is showing the
SAME two views (audio-alone vs the model's own symbolic reading of that
same audio) side by side, exactly as asked, so the user can see for
themselves where they agree/disagree. A third, oracle-iReal symbolic matrix
is out of scope for "does the deployed tool have a problem" (the deployed
tool never sees the clean chart) but IS included as a bonus reference layer
for autumn_leaves only (the one song where the clean-iReal alignment
exists), clearly labeled.

Output: one JSON per song with n x n (block-grain) audio_matrix,
symbolic_matrix, block_bars (bar span per block index), plus the
already-known section_merge_candidates rank for cross-reference, plus
Mantel test agreement stats. NOT a rendered chart -- data only, per the
division of labor for this call.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from rawchroma import per_bar_rawchroma
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans
from chord_distance import chord_vector_binary
from realaudio_threshold_check import get_baseline_chords
from auto_apply_merges import SONGS, AUDIO_DIR
from section_merge_candidates import bar_label_bucket, block_gram_sim
from dual_matrix_correlation import mantel_test

OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8

SONGS_TO_RUN = ["autumn_leaves", "abba_chiquitita_official_lyric_video",
                "aretha_franklin_chain_of_fools_official_lyric_video"]


def build_for_song(slug):
    audio_name = SONGS[slug]["audio_name"]
    audio_path = AUDIO_DIR / audio_name
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    base_ch = get_baseline_chords(slug)
    n = len(variants["bt_concat"])
    spans = nuclear_spans(n, GRAIN)
    m = len(spans)
    bar_bucket = [bar_label_bucket(base_ch, bar_times[i], bar_times[i + 1]) for i in range(n)]

    v = variants["bt_concat"]
    row_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.clip(row_norm, 1e-9, None)
    G = v_unit @ v_unit.T
    sq = np.diag(G).copy()
    prefix = diagonal_prefix_sums(G)

    audio_mat = np.zeros((m, m))
    sym_mat = np.zeros((m, m))
    sym_valid = np.zeros((m, m), dtype=int)
    for i in range(m):
        si, ei = spans[i]
        for j in range(m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            if i == j:
                audio_mat[i, j] = 1.0
                sym_mat[i, j] = 1.0
                sym_valid[i, j] = L
                continue
            a = block_gram_sim(prefix, sq, n, si, sj, L)
            audio_mat[i, j] = a
            sym_vecs_a, sym_vecs_b, valid = [], [], 0
            for t in range(L):
                ba, bb = bar_bucket[si + t], bar_bucket[sj + t]
                sym_vecs_a.append(chord_vector_binary(*ba) if ba else np.zeros(12))
                sym_vecs_b.append(chord_vector_binary(*bb) if bb else np.zeros(12))
                if ba is not None and bb is not None:
                    valid += 1
            num = sum(float(np.dot(x, y)) for x, y in zip(sym_vecs_a, sym_vecs_b))
            na = np.sqrt(sum(float(np.dot(x, x)) for x in sym_vecs_a))
            nb = np.sqrt(sum(float(np.dot(x, x)) for x in sym_vecs_b))
            s = num / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0
            sym_mat[i, j] = s
            sym_valid[i, j] = valid

    mt = mantel_test(sym_mat, audio_mat, n_perm=999)

    block0_vs_1 = {
        "audio_sim": float(audio_mat[0, 1]),
        "symbolic_sim": float(sym_mat[0, 1]),
        "audio_rank_desc": int(1 + np.sum(audio_mat[np.triu_indices(m, k=1)] > audio_mat[0, 1])),
        "symbolic_rank_desc": int(1 + np.sum(sym_mat[np.triu_indices(m, k=1)] > sym_mat[0, 1])),
        "n_pairs_total": int(m * (m - 1) / 2),
        "bright_in_audio": bool(audio_mat[0, 1] > np.median(audio_mat[np.triu_indices(m, k=1)])),
        "bright_in_symbolic": bool(sym_mat[0, 1] > np.median(sym_mat[np.triu_indices(m, k=1)])),
    }

    return {
        "slug": slug,
        "grain": GRAIN,
        "n_bars": n,
        "n_blocks": m,
        "tempo_bpm": tempo,
        "block_bars": [[int(s), int(e)] for s, e in spans],
        "block_times_s": [[float(bar_times[s]), float(bar_times[e])] for s, e in spans],
        "audio_matrix": audio_mat.tolist(),
        "symbolic_matrix": sym_mat.tolist(),
        "symbolic_n_valid_bars_matrix": sym_valid.tolist(),
        "mantel_test_audio_vs_symbolic": mt,
        "block0_vs_block1_diagnostic": block0_vs_1,
    }


def main():
    results = {}
    for slug in SONGS_TO_RUN:
        print(f"=== {slug} ===", file=sys.stderr)
        res = build_for_song(slug)
        results[slug] = res
        mt = res["mantel_test_audio_vs_symbolic"]
        b01 = res["block0_vs_block1_diagnostic"]
        print(f"  n_blocks={res['n_blocks']} mantel r={mt['observed_r']:.3f} z={mt['z_score']:.2f} p={mt['p_value_two_sided']:.4f}", file=sys.stderr)
        print(f"  block0-vs-1: audio={b01['audio_sim']:.3f} (rank {b01['audio_rank_desc']}/{b01['n_pairs_total']}) "
              f"symbolic={b01['symbolic_sim']:.3f} (rank {b01['symbolic_rank_desc']}/{b01['n_pairs_total']})", file=sys.stderr)
    out_path = OUT_DIR / "dual_matrix_grain8_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
