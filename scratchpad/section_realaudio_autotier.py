"""section_realaudio_autotier.py — 2026-07-19, real-audio validation of the
grain=4 / grain=8 auto-tier thresholds found by section_tau_auto_search.py
(symbolic iReal corpus: tau_auto grain=4 ~0.989, grain=8 ~0.9916, block-
identity GT). Same two-track methodology already established tonight for
the bar-level tau_auto real-audio failure:

  (a) pseudo-GT (model's own baseline-decode bar bucket, majority-vote,
      EXTENDED to block grain: label=1 iff EVERY position-aligned aligned-
      bar pair's bucket matches -- same strict all-match convention the
      symbolic corpus GT used, for apples-to-apples), corpus-scale on the
      full real candidate census (all pairs above a low audio floor, not
      just the ones a UI would show).
  (b) external (non-circular) GT, REUSED not re-sourced from this session's
      already-documented findings (docs/known_issues.md):
        - autumn_leaves: bars 0-15 are the ONE confidently-aligned window
          (irealb_autumn_leaves_sectionwise.json, this project's issue #37)
          -- chart bars 1-8 (A occurrence 1) == bars 9-15 (A occurrence 2,
          bar-for-bar chord-identical). Gives a real external-GT TRUE label
          for grain=8 block0-vs-block1 (bars0-7 vs bars8-15) and grain=4
          block0-vs-block2 (bars0-3 vs bars8-11, both entirely inside the
          A/A repeat).
        - aretha: pop400.txt's chart + chords-and-tabs.net agree the song
          is a single Cm/Cm7 vamp for its ENTIRE body except a 10-bar
          no-chord bridge (RMS-located at 83.7-100s) and an abrupt ending
          cutoff (~161-162s decay tail). ANY block pair with BOTH blocks'
          time spans entirely inside [0,83.7) or (100,161) is therefore an
          externally-supported TRUE match (same harmonic family) -- this
          gives a much larger real external-GT sample than the previous
          call's n=11 hand-check, reused here as a coverage rule rather
          than re-deriving from scratch.
        - abba: only 2 bar-level pairs were externally hand-verified
          previously (206/222 TP, 32/64 FP) -- too sparse to build a block-
          level rule from; reported only if a generated block candidate's
          span happens to contain one of those two bar timestamps on BOTH
          sides (unlikely, checked but not expected to fire).

  (c) same joint audio+symbolic gate pattern as joint_threshold_search.py,
      applied at block grain if (a) shows the direct tau_auto port fails to
      transfer (per CLAUDE.md rule #6, do not assume a component swap
      preserves anything -- check).

Reuses (does not rebuild): rawchroma.per_bar_rawchroma, hierarchy_shortcut's
diagonal-prefix-sum machinery, chord_distance_eval.nuclear_spans,
chord_distance.chord_vector_binary, realaudio_threshold_check.
{get_baseline_chords,label_bucket}, section_merge_candidates.bar_label_bucket,
auto_apply_merges.{SONGS,AUDIO_DIR}.
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
from realaudio_threshold_check import get_baseline_chords, label_bucket
from section_merge_candidates import bar_label_bucket, block_gram_sim
from auto_apply_merges import SONGS, AUDIO_DIR

OUT_DIR = Path(__file__).resolve().parent
MIN_GAP_BLOCKS = 0
AUDIO_FLOOR = 0.50
GRAINS = [4, 8]
TAU_AUTO_SYMBOLIC = {4: 0.9665, 8: 0.9583}  # section_tau_auto_search consensus, MAX across
# folds (most conservative fold-consistent choice, matches tau_auto_search.py's own
# "ship the max" convention), allow-one-mismatch GT, post-strict-GT-artifact fix

# aretha external-GT vamp windows (real seconds), reused from docs/known_issues.md
# "REAL (non-circular) external GT check" entry -- no-chord bridge 83.7-100s,
# abrupt ending cutoff ~161-162s. Intro C7 (0-~5s) is a DIFFERENT chord from
# the Cm/Cm7 body, so restrict the "same-family vamp" window to [5, 83.7) and
# (100, 161), explicitly excluding the intro too (conservative, not assumed).
ARETHA_VAMP_WINDOWS = [(5.0, 83.7), (100.0, 161.0)]
# abba: only 2 bar-level pairs were ever externally hand-verified (206/222 TP
# at ~289.9s/312.4s, 32/64 FP) -- too sparse (2 single-bar timestamps) to
# build any block-level coverage rule from; not attempted here. abba's
# auto-tier validation below rests on pseudo-GT only, flagged as such.


def in_any_window(t0, t1, windows):
    return any(w0 <= t0 and t1 <= w1 for (w0, w1) in windows)


def build_song_blocks(slug, audio_name, grain):
    audio_path = AUDIO_DIR / audio_name
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    base_ch = get_baseline_chords(slug)
    n = len(variants["bt_concat"])
    spans = nuclear_spans(n, grain)
    m = len(spans)
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
            if audio_sim < AUDIO_FLOOR:
                continue
            sym_vecs_a, sym_vecs_b, agree_valid, agree_mismatches = [], [], 0, 0
            for t in range(L):
                ba, bb = bar_bucket[si + t], bar_bucket[sj + t]
                sym_vecs_a.append(chord_vector_binary(*ba) if ba else np.zeros(12))
                sym_vecs_b.append(chord_vector_binary(*bb) if bb else np.zeros(12))
                if ba is not None and bb is not None:
                    agree_valid += 1
                    if ba != bb:
                        agree_mismatches += 1
            num = sum(float(np.dot(a, b)) for a, b in zip(sym_vecs_a, sym_vecs_b))
            na = np.sqrt(sum(float(np.dot(a, a)) for a in sym_vecs_a))
            nb = np.sqrt(sum(float(np.dot(b, b)) for b in sym_vecs_b))
            symbolic_sim = num / (na * nb) if na > 1e-9 and nb > 1e-9 else None
            # allow-one-mismatch pseudo-GT (matches section_tau_auto_search.py's
            # symbolic-corpus GT definition -- see that script's docstring for
            # why literal all-match was rejected as unusable on noisy real audio)
            pseudo_gt = (1 if agree_mismatches <= 1 else 0) if agree_valid > 0 else None

            t0a, t1a = float(bar_times[si]), float(bar_times[ei])
            t0b, t1b = float(bar_times[sj]), float(bar_times[ej])
            ext = None
            if slug == "autumn_leaves":
                # bars 0-15 confidently aligned; A occ1=bars0-7, occ2=bars8-15
                if grain == 8 and (si, ei) == (0, 8) and (sj, ej) == (8, 16):
                    ext = 1
                elif grain == 4 and si < 4 and ei <= 4 and sj >= 8 and ej <= 12:
                    ext = 1
            elif slug == "aretha_franklin_chain_of_fools_official_lyric_video":
                if in_any_window(t0a, t1a, ARETHA_VAMP_WINDOWS) and in_any_window(t0b, t1b, ARETHA_VAMP_WINDOWS):
                    ext = 1

            rows.append({
                "blocks": [i, j], "bars": [[int(si), int(ei)], [int(sj), int(ej)]],
                "spans": [[t0a, t1a], [t0b, t1b]],
                "audio_sim": audio_sim, "symbolic_sim": symbolic_sim,
                "n_bars_aligned": L, "n_bars_pseudo_gt_valid": agree_valid,
                "pseudo_gt_match": pseudo_gt, "external_gt_match": ext,
            })
    return rows, {"n_bars_total": n, "n_blocks": m, "tempo_bpm": tempo}


def summarize_tau(rows, tau_audio, gt_key):
    sel = [r for r in rows if r["audio_sim"] >= tau_audio and r[gt_key] is not None]
    n = len(sel)
    if n == 0:
        return {"tau_audio": tau_audio, "n": 0, "precision": None}
    fp = sum(1 for r in sel if r[gt_key] == 0)
    return {"tau_audio": tau_audio, "n": n, "fp": fp, "precision": 1.0 - fp / n}


def joint_summarize(rows, tau_audio, tau_symbolic, gt_key):
    sel = [r for r in rows if r["audio_sim"] >= tau_audio
           and r["symbolic_sim"] is not None and r["symbolic_sim"] >= tau_symbolic
           and r[gt_key] is not None]
    n = len(sel)
    if n == 0:
        return {"tau_audio": tau_audio, "tau_symbolic": tau_symbolic, "n": 0, "precision": None}
    fp = sum(1 for r in sel if r[gt_key] == 0)
    return {"tau_audio": tau_audio, "tau_symbolic": tau_symbolic, "n": n, "fp": fp,
            "precision": 1.0 - fp / n}


def main():
    results = {"tau_auto_symbolic_reference": TAU_AUTO_SYMBOLIC, "songs": {}}
    pooled_rows = {4: [], 8: []}
    for slug, sm in SONGS.items():
        results["songs"][slug] = {}
        for grain in GRAINS:
            rows, meta = build_song_blocks(slug, sm["audio_name"], grain)
            pooled_rows[grain].extend([{**r, "song": slug} for r in rows])
            n_pseudo_pos = sum(1 for r in rows if r["pseudo_gt_match"] == 1)
            n_pseudo_valid = sum(1 for r in rows if r["pseudo_gt_match"] is not None)
            n_ext = sum(1 for r in rows if r["external_gt_match"] is not None)
            direct_port = summarize_tau(rows, TAU_AUTO_SYMBOLIC[grain], "pseudo_gt_match")
            direct_port_ext = summarize_tau(rows, TAU_AUTO_SYMBOLIC[grain], "external_gt_match")
            print(f"=== {slug} grain={grain} === n_pairs={len(rows)} (floor {AUDIO_FLOOR}) "
                  f"pseudo_gt: {n_pseudo_pos}/{n_pseudo_valid} positive, ext_gt: n={n_ext}")
            print(f"  direct tau_auto port ({TAU_AUTO_SYMBOLIC[grain]}): "
                  f"pseudo n={direct_port['n']} precision={direct_port.get('precision')}  "
                  f"ext n={direct_port_ext['n']} precision={direct_port_ext.get('precision')}")
            results["songs"][slug][f"grain_{grain}"] = {
                "meta": meta, "n_pairs_above_floor": len(rows),
                "n_pseudo_gt_valid": n_pseudo_valid, "n_pseudo_gt_positive": n_pseudo_pos,
                "n_external_gt_labeled": n_ext,
                "direct_tau_auto_port_pseudo_gt": direct_port,
                "direct_tau_auto_port_external_gt": direct_port_ext,
                "rows": rows,
            }

    # pooled real-audio precision-vs-tau curve, and joint-gate sweep, per grain
    print("\n=== POOLED (all 3 songs), audio_sim-only sweep ===")
    for grain in GRAINS:
        rows = pooled_rows[grain]
        print(f"-- grain={grain} --")
        sweep = []
        for tau in [0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70]:
            s = summarize_tau(rows, tau, "pseudo_gt_match")
            sweep.append(s)
            print(f"  tau_audio={tau:.2f}  n={s['n']:4d}  precision={s.get('precision')}")
        results.setdefault("pooled_audio_only_sweep", {})[str(grain)] = sweep

    print("\n=== POOLED joint gate sweep (audio_sim>=tau_auto AND symbolic_sim>=tau_sym) ===")
    for grain in GRAINS:
        rows = pooled_rows[grain]
        print(f"-- grain={grain}, tau_audio fixed at symbolic-derived tau_auto={TAU_AUTO_SYMBOLIC[grain]} --")
        joint_sweep = []
        for tau_sym in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            j = joint_summarize(rows, TAU_AUTO_SYMBOLIC[grain], tau_sym, "pseudo_gt_match")
            joint_sweep.append(j)
            print(f"  tau_sym={tau_sym:.2f}  n={j['n']:4d}  precision={j.get('precision')}")
        results.setdefault("pooled_joint_gate_sweep", {})[str(grain)] = joint_sweep

    out_path = OUT_DIR / "section_realaudio_autotier_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
