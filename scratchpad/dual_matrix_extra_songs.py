"""dual_matrix_extra_songs.py — 2026-07-18, task 4 (time-permitting) of the
joint audio+symbolic auto-tier gate call.

The original DUAL-MATRIX call (`dual_matrix_correlation.py`) found the
boundary/intro-outro-padding analysis inconclusive at n=3 songs: abba
showed the hypothesized "boundary bars agree less across matrices"
direction (z=-2.3 to -2.9), aretha showed the OPPOSITE direction
(z=+7.4 to +14.9), autumn_leaves showed no effect — too few songs to tell
if abba-vs-aretha is a real distinction (e.g. static-vamp vs
harmonically-active intro) or small-n noise. This script reruns the EXACT
SAME boundary-vs-random-interior-window Mantel comparison
(`dual_matrix_correlation.py`'s `mantel_test`/`boundary_window_stats`,
imported and reused verbatim, not reimplemented) on 3 more real songs that
already have an analyzed chord chart (`docs/plots/inferred_*.html`
confirms this): elton_john_goodbye_yellow_brick_road_lyrics,
land_of_1000_dances, the_commodores_easy_1977.

Unlike the original 3 songs, this script does NOT reuse a precomputed
`bar_ssm_rawchroma_<song>.json` (none exists for these 3) — it calls
`per_bar_rawchroma` directly for the audio matrix (grain=1 only, matching
what the original script actually used from that file:
`grains["1"]["similarity_matrix"]`) and runs a fresh (cached)
`infer_chords_v1` baseline decode for the symbolic side, same
unconstrained-baseline methodology as `realaudio_threshold_check.py`.
"""
from __future__ import annotations
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from rawchroma import per_bar_rawchroma
from dual_matrix_correlation import (build_symbolic_matrices, mantel_test,
                                      boundary_window_stats, K_SWEEP)
from auto_apply_merges import transcode

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = REPO / "docs" / "audio"

EXTRA_SONGS = {
    "elton_john_goodbye_yellow_brick_road": "elton_john_goodbye_yellow_brick_road_lyrics.m4a",
    "land_of_1000_dances": "land_of_1000_dances.m4a",
    "commodores_easy": "the_commodores_easy_1977.m4a",
}


def get_baseline_chords_cached(song_key, audio_name):
    cache_path = OUT_DIR / f"baseline_chords_extra_{song_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    audio_path = AUDIO_DIR / audio_name
    tmp = Path(tempfile.mkdtemp(prefix="harmonia_extra_baseline_"))
    try:
        wav = tmp / "a.wav"
        transcode(audio_path, wav)
        base = infer_chords_v1(wav, cache_dir=tmp, joint_transition_weight=0.0)
        base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
        cache_path.write_text(json.dumps(base_ch))
        return base_ch
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def process_song(song_key, audio_name):
    print("=== %s ===" % song_key, file=sys.stderr)
    audio_path = AUDIO_DIR / audio_name
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    v = variants["bt_concat"]
    n = len(v)
    row_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.clip(row_norm, 1e-9, None)
    audio_mat = v_unit @ v_unit.T

    base_chords = get_baseline_chords_cached(song_key, audio_name)
    sym_mats, root_qual = build_symbolic_matrices(bar_times, base_chords)

    result = {"song": song_key, "n_bars": n, "tempo_bpm": tempo}

    premise = {}
    for scheme_name, sym_mat in sym_mats.items():
        mt = mantel_test(sym_mat, audio_mat)
        premise[scheme_name] = mt
        print("  [premise] %-12s observed_r=%.4f  null=%.4f+-%.4f  z=%.2f  p=%.4f" %
              (scheme_name, mt["observed_r"], mt["null_mean"], mt["null_std"],
               mt["z_score"], mt["p_value_two_sided"]), file=sys.stderr)
    result["global_mantel"] = premise

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
    result["bar_root_qual"] = [[p, q] for p, q in root_qual]
    result["bar_times"] = bar_times.tolist()
    return result


def main():
    all_results = {}
    for song_key, audio_name in EXTRA_SONGS.items():
        all_results[song_key] = process_song(song_key, audio_name)

    out_path = OUT_DIR / "dual_matrix_extra_songs_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print("\nwrote %s" % out_path, file=sys.stderr)

    # combine with the original 3 songs' already-computed boundary z-scores
    # for a pooled n=6 read (z at K=8, the middle of the original K_SWEEP,
    # picked as the single comparable statistic across all 6 songs)
    orig = json.loads((OUT_DIR / "dual_matrix_correlation_results.json").read_text())

    def boundary_r_gap(res, K=8):
        for b in res["boundary_sweep"]:
            if b["K"] == K:
                gap = b["boundary_correlation_r"] - b["random_interior_window_r_mean"]
                return gap
        return None

    print("\n=== POOLED (original 3 + extra 3), boundary_r - random_interior_r @ K=8 ===",
          file=sys.stderr)
    all_gaps = {}
    for song, res in orig.items():
        g = boundary_r_gap(res)
        all_gaps[song] = g
        print("  %-35s gap=%s" % (song, g), file=sys.stderr)
    for song, res in all_results.items():
        g = boundary_r_gap(res)
        all_gaps[song] = g
        print("  %-35s gap=%s" % (song, g), file=sys.stderr)

    (OUT_DIR / "dual_matrix_pooled_boundary_gaps.json").write_text(json.dumps(all_gaps, indent=2))
    print("\nwrote dual_matrix_pooled_boundary_gaps.json", file=sys.stderr)


if __name__ == "__main__":
    main()
