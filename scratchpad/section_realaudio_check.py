"""section_realaudio_check.py — 2026-07-18, section-level suggestion tool,
task 3: does grain=8 (and grain=4) real-audio-ONLY block similarity have the
SAME real-to-symbolic transfer failure the bar-level (grain=1) tau_auto=0.96
threshold had (calibrated on clean iReal symbolic proxy features, then found
to only reach 39.4% pooled baseline-label agreement on real audio at that
threshold — see realaudio_threshold_check.py / auto_apply_merges.py)?

Methodology (reuses realaudio_threshold_check.py's pseudo-GT approach,
extends it from bar-pairs to grain=8/4 BLOCK-pairs): for the 3 real songs,
build ALL block-pairs at the target grain (via rawchroma per-bar Gram +
hierarchy_shortcut's exact derive_grain shortcut — no re-derivation of the
similarity math), audio_sim = bt_concat block_sim; pseudo-GT "agree" =
whether the two blocks' MAJORITY baseline-decode (root_pc, quality-family)
bucket matches at >=50% of paired bar-positions (a block-level generalization
of the bar-level label_bucket check). Also computes symbolic_sim (V1_binary
chord-tone cosine between the two blocks' majority-vote labels, POSITION-
ALIGNED like block_sim, not pooled) as a candidate joint-gate signal,
directly reusing chord_distance.chord_vector_binary + the project's existing
joint-gate pattern (joint_threshold_search.py) if audio-only turns out to be
insufficient.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from rawchroma import per_bar_rawchroma
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans
from chord_distance import chord_vector_binary, cosine
from realaudio_threshold_check import get_baseline_chords, label_bucket
from auto_apply_merges import SONGS, AUDIO_DIR

OUT_DIR = Path(__file__).resolve().parent
# see section_pairs.py's identical fix note: adjacent blocks are the
# primary use case at block grain, not a trivial case to exclude.
MIN_GAP_BLOCKS = 0


def block_gram_sim(prefix, sq, n, i0, j0, L):
    """`prefix` must be precomputed ONCE per song (diagonal_prefix_sums(G))
    — see section_pairs.py's identical fix note; recomputing per-pair made
    the corpus-scale sibling script untractable."""
    d = j0 - i0
    num = diag_sum(prefix, n, d, i0, L)
    na = np.sqrt(float(np.sum(sq[i0:i0 + L])))
    nb = np.sqrt(float(np.sum(sq[j0:j0 + L])))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def bar_label_bucket(base_chords, t0, t1):
    """Majority baseline-decode bucket by duration overlap within [t0,t1)."""
    overlaps = {}
    for c in base_chords:
        s, e = max(c["start_s"], t0), min(c["end_s"], t1)
        if e <= s:
            continue
        b = label_bucket(c.get("label"))
        if b is None:
            continue
        overlaps[b] = overlaps.get(b, 0.0) + (e - s)
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda kv: kv[1])[0]


def analyze_song(slug, audio_name, grain):
    audio_path = AUDIO_DIR / audio_name
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    base_ch = get_baseline_chords(slug)
    n = len(variants["bt_concat"])
    spans = nuclear_spans(n, grain)
    m = len(spans)

    # per-bar majority bucket (for block majority + symbolic_sim)
    bar_bucket = [bar_label_bucket(base_ch, bar_times[i], bar_times[i + 1]) for i in range(n)]

    v = variants["bt_concat"]
    row_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.clip(row_norm, 1e-9, None)
    G = v_unit @ v_unit.T
    sq = np.diag(G).copy()
    prefix = diagonal_prefix_sums(G)

    rows = []
    for i in range(m):
        si, ei = spans[i]
        for j in range(i + 1 + MIN_GAP_BLOCKS, m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            audio_sim = block_gram_sim(prefix, sq, n, si, sj, L)

            # position-aligned pseudo-GT agree: fraction of paired bar
            # positions whose bucket matches; "agree"=1 if >=50% match
            # (mirrors build_section_pairs' majority-section >=50% rule)
            matches, valid = 0, 0
            sym_vecs_a, sym_vecs_b = [], []
            for t in range(L):
                ba, bb = bar_bucket[si + t], bar_bucket[sj + t]
                va = chord_vector_binary(*ba) if ba else np.zeros(12)
                vb = chord_vector_binary(*bb) if bb else np.zeros(12)
                sym_vecs_a.append(va)
                sym_vecs_b.append(vb)
                if ba is not None and bb is not None:
                    valid += 1
                    if ba == bb:
                        matches += 1
            if valid == 0:
                continue
            agree = 1 if matches / valid >= 0.5 else 0
            # symbolic_sim: position-aligned block_sim over per-bar binary
            # chord-tone vectors (same convention as chord_distance_eval.block_sim)
            num = sum(float(np.dot(a, b)) for a, b in zip(sym_vecs_a, sym_vecs_b))
            na = np.sqrt(sum(float(np.dot(a, a)) for a in sym_vecs_a))
            nb = np.sqrt(sum(float(np.dot(b, b)) for b in sym_vecs_b))
            symbolic_sim = num / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0

            rows.append({"song": slug, "grain": grain, "block_i": i, "block_j": j,
                         "bars": [si, sj], "audio_sim": audio_sim,
                         "symbolic_sim": symbolic_sim, "agree": agree})
    return rows


def sweep(rows, sim_key, taus):
    sims = np.array([r[sim_key] for r in rows])
    agrees = np.array([r["agree"] for r in rows])
    out = []
    for tau in taus:
        mask = sims >= tau
        n = int(mask.sum())
        prec = float(agrees[mask].mean()) if n else None
        out.append({"tau": tau, "n": n, "agree_rate": prec})
    return out


def main():
    all_rows = {4: [], 8: []}
    for slug, meta in SONGS.items():
        for grain in (4, 8):
            rows = analyze_song(slug, meta["audio_name"], grain)
            all_rows[grain].extend(rows)
            print(f"{slug} grain={grain}: n_pairs={len(rows)}", file=sys.stderr)

    out = {}
    for grain in (4, 8):
        rows = all_rows[grain]
        taus = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]
        audio_sweep = sweep(rows, "audio_sim", taus)
        print(f"\n=== grain={grain} AUDIO-ONLY sweep (pooled, n={len(rows)}) ===", file=sys.stderr)
        for r in audio_sweep:
            print(f"  tau={r['tau']:.2f}  n={r['n']:4d}  agree_rate={r['agree_rate']}", file=sys.stderr)

        sym_taus = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
        # joint sweep: anchor audio at a representative "high" floor per grain
        # (use 0.90 as a generic anchor -- the ROC retune script picks the
        # real operating point; this is a diagnostic transfer check only)
        anchor = 0.90
        audio_sims = np.array([r["audio_sim"] for r in rows])
        agrees = np.array([r["agree"] for r in rows])
        sym_sims = np.array([r["symbolic_sim"] for r in rows])
        auto_mask = audio_sims >= anchor
        n_auto_pool = int(auto_mask.sum())
        joint_sweep = []
        for ts in sym_taus:
            mask = auto_mask & (sym_sims >= ts)
            n = int(mask.sum())
            recall_of_pool = n / n_auto_pool if n_auto_pool else None
            prec = float(agrees[mask].mean()) if n else None
            joint_sweep.append({"tau_symbolic": ts, "n": n,
                                 "recall_of_audio_pool": recall_of_pool, "precision": prec})
        print(f"\n=== grain={grain} JOINT sweep: audio_sim>={anchor} AND symbolic_sim>=tau_sym "
              f"(n_audio_pool={n_auto_pool}) ===", file=sys.stderr)
        for r in joint_sweep:
            print(f"  tau_sym={r['tau_symbolic']:.2f}  n={r['n']:4d}  "
                  f"recall_of_pool={r['recall_of_audio_pool']}  precision={r['precision']}", file=sys.stderr)

        out[str(grain)] = {"n_pairs": len(rows), "audio_only_sweep": audio_sweep,
                            "joint_sweep_anchor_%.2f" % anchor: joint_sweep,
                            "rows": rows}

    (OUT_DIR / "section_realaudio_check_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote section_realaudio_check_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
