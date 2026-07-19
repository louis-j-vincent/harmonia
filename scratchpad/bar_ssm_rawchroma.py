"""bar_ssm_rawchroma.py — multi-granularity SSMs (1/2/4/8/16-bar) using the
TRUE untrained raw-chroma V4 (scratchpad/rawchroma.py), for 4 songs: the
existing iReal 'All Of Me' (unchanged, V1 scheme) + 3 real-audio songs
(aretha_chain_of_fools, autumn_leaves, abba_chiquitita — the 2 new ones are
autumn_leaves + abba_chiquitita, per the user's "2 more songs" request).

Primary variant: bt_concat (both registers, independently normalized before
concatenation per the user's correction). bass_only/treble_only computed too,
at size=8 only, for a quick register comparison (not full 5-grain, to keep
scope bounded as agreed).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from chord_distance_eval import nuclear_spans, block_sim
from rawchroma import per_bar_rawchroma

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
SIZES = [1, 2, 4, 8, 16]

REAL_SONGS = {
    "aretha_chain_of_fools": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    "autumn_leaves": "autumn_leaves.m4a",
    "abba_chiquitita": "abba_chiquitita_official_lyric_video.m4a",
}


def multigrain_matrices(bar_vecs):
    n = len(bar_vecs)
    out = {}
    for size in SIZES:
        if n < size:
            continue
        spans = nuclear_spans(n, size)
        block_bars = [bar_vecs[s:e] for (s, e) in spans]
        m = len(spans)
        sim = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                sim[i, j] = block_sim(block_bars[i], block_bars[j])
        out[str(size)] = {"n_blocks": m, "similarity_matrix": sim.tolist()}
    return out


def register_comparison(variants, size=8):
    """Quick bass-only vs treble-only vs bt_concat comparison at one grain:
    distinct-block-count + non-degenerate check, not a full matrix sweep."""
    out = {}
    for name, bar_vecs in variants.items():
        n = len(bar_vecs)
        if n < size:
            continue
        spans = nuclear_spans(n, size)
        block_bars = [bar_vecs[s:e] for (s, e) in spans]
        m = len(spans)
        # union-find at a fixed reasonable tau for a qualitative read
        for tau in (0.75, 0.85, 0.90):
            parent = list(range(m))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for i in range(m):
                for j in range(i + 1, m):
                    if block_sim(block_bars[i], block_bars[j]) >= tau:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
            n_clusters = len(set(find(i) for i in range(m)))
            out.setdefault(name, {})[str(tau)] = {"n_blocks": m, "n_clusters": n_clusters}
    return out


def main():
    for song, fname in REAL_SONGS.items():
        print("=== %s ===" % song)
        audio = REPO / "docs" / "audio" / fname
        variants, bar_times, tempo, tonic = per_bar_rawchroma(audio)
        print("  tempo=%.1f bpm  n_bars=%d  tonic_pc=%d" %
              (tempo, len(variants["bt_concat"]), tonic))

        mg_all = {}
        for variant_name in ("bass_only", "treble_only", "bt_concat"):
            mg = multigrain_matrices(variants[variant_name])
            mg_all[variant_name] = mg
            for size in SIZES:
                if str(size) in mg:
                    print("  %-11s size=%2d -> %d blocks" %
                          (variant_name, size, mg[str(size)]["n_blocks"]))

        # verify bt_concat == elementwise average of bass_only/treble_only
        # (both halves are independently unit-normalized per bar, so this
        # must hold exactly — see the worked-example check run before this
        # script; re-verified per-song here as a cheap guard)
        for size in SIZES:
            if str(size) not in mg_all["bt_concat"]:
                continue
            import numpy as _np
            c = _np.array(mg_all["bt_concat"][str(size)]["similarity_matrix"])
            b = _np.array(mg_all["bass_only"][str(size)]["similarity_matrix"])
            t = _np.array(mg_all["treble_only"][str(size)]["similarity_matrix"])
            diff = _np.abs(c - (b + t) / 2).max()
            assert diff < 1e-6, "bt_concat != avg(bass,treble) at size=%d (diff=%.2e)" % (size, diff)
        print("  verified: bt_concat == avg(bass_only, treble_only) at every grain")

        reg = register_comparison(variants, size=8)
        for name, taus in reg.items():
            print("  register=%-11s " % name +
                  "  ".join("tau=%s:%d/%d clusters" % (t, v["n_clusters"], v["n_blocks"])
                            for t, v in taus.items()))

        out = {"song": song, "source": "real_audio_rawchroma_untrained",
               "scheme": "bt_concat", "n_bars": len(variants["bt_concat"]),
               "tempo_bpm": tempo, "est_tonic_pc": tonic,
               "grains": mg_all["bt_concat"],
               "grains_bass": mg_all["bass_only"],
               "grains_treble": mg_all["treble_only"],
               "register_comparison": reg}
        outp = OUT_DIR / ("bar_ssm_rawchroma_%s.json" % song)
        outp.write_text(json.dumps(out))
        print("  wrote", outp.name)
        print()


if __name__ == "__main__":
    main()
