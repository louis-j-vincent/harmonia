"""joint_threshold_search.py — 2026-07-18, dual-matrix follow-up (task 2 of
the "DUAL-MATRIX CROSS-VALIDATION" continuation call).

**Premise check (task 1, done first, cheaply) result, logged here for
provenance**: computed V1/V2/V3 symbolic chord-tone cosine similarity
(chord_distance.py) for the 2 known real-audio FALSE POSITIVES (abba
32/64, aretha 13/17) vs the 2 known TRUE POSITIVES (aretha 53/69, abba
206/222), reusing scratchpad/dual_matrix_correlation_results.json's
already-built matrices (no rebuild):

  pair                          audio_sim   V1_binary  V2_weighted  V3_tiv
  abba 32/64      (FALSE POS)   0.9790      0.6667     0.9286       0.5265
  aretha 13/17     (FALSE POS)  0.9783      0.8660     0.9522       0.7965
  aretha 53/69     (TRUE POS)   0.9892      1.0000     1.0000       1.0000
  abba 206/222     (TRUE POS)   0.9938      1.0000     1.0000       1.0000

V1 and V3 show a clean gap (FP max 0.87/0.80 vs TP=1.0 exactly);
V2_weighted is borderline (FP max 0.95 vs TP=1.0 -- role-weighting
compresses the gap, not a good discriminator here). Audio_sim alone shows
NO gap at all (0.978-0.994 for BOTH classes, confirming the whole reason
this thread exists). Premise PASSES for V1/V3: proceed to corpus-scale
joint-threshold derivation below, using V1_binary as primary (V3 given as
a secondary check; V2 excluded from the tau_symbolic sweep as a documented,
justified exclusion, not silently dropped).

**Methodology (reuses realaudio_threshold_check.py's pseudo-GT approach
verbatim, extends it to a second axis)**: for every candidate pair in the
full 3-song census (bar_merge_full_census_<slug>.json), pseudo-GT
"agree" = model's own UNCONSTRAINED baseline decode's (root_pc,
quality-family-5-way-bucket) match at the two bar midpoints -- IDENTICAL
definition/code to realaudio_threshold_check.py's label_bucket, reused
unchanged so the two scripts' agree-rate numbers are directly comparable.
audio_sim = the candidate's existing "confidence" field (bt_concat cosine,
unchanged). symbolic_sim = chord_distance V1_binary (or V3_tiv) cosine
similarity between the two bars' MAJORITY-VOTE chord label (by duration
overlap, chord_distance.label_to_root_qual-style bucketing borrowed from
dual_matrix_correlation.py), i.e. a courser feature computed from the SAME
baseline decode "agree" is computed from.

**Explicit circularity caveat (do not skip, this is the epistemically
important part)**: agree and symbolic_sim are BOTH derived from the same
underlying baseline chord labels -- symbolic_sim is not an independent
oracle. When two bars' pseudo-GT bucket matches exactly, symbolic_sim will
typically (not always -- QBUCKET's 6-way split differs from label_bucket's
5-way split, so there is real daylight) be high/1.0; when it doesn't
match, symbolic_sim is typically lower. This means "symbolic_sim alone
predicts agree extremely well" is expected and NOT the interesting
result -- it would be true almost by construction, and reporting it as a
discovery would be a methodological error. The ACTUAL question this script
answers, which is NOT circular: does adding a symbolic_sim>=tau_symbolic
requirement ON TOP OF the EXISTING audio_sim>=0.96 selection (the tau_auto
gate that is already deployed and already known to have only 39.4%-62.5%
agreement) raise that subset's precision, and at what recall cost (how
much of the audio-selected auto pool survives the extra filter)? This is a
legitimate joint-operating-point question even though symbolic_sim and
agree share a data source, because in a REAL deployment the joint filter
(audio high AND baseline-decode-symbolic high) is a concrete, computable
gate -- the question is whether it's a useful gate, not whether the two
signals are independent (they aren't, by design).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from auto_apply_merges import SONGS, AUDIO_DIR
from realaudio_threshold_check import get_baseline_chords, label_bucket
from chord_distance import chord_vector_binary, chord_vector_weighted, chord_vector_tiv, cosine
from dual_matrix_correlation import label_to_root_qual, bar_chord_majority

OUT_DIR = Path(__file__).resolve().parent

SCHEMES = {
    "V1_binary": chord_vector_binary,
    "V2_weighted": chord_vector_weighted,
    "V3_tiv": chord_vector_tiv,
}


def symbolic_sim_for_span(base_ch, t0, t1, other_t0, other_t1, scheme_fn, is_tiv):
    label_a = bar_chord_majority(base_ch, t0, t1)
    label_b = bar_chord_majority(base_ch, other_t0, other_t1)
    pc_a, q_a = label_to_root_qual(label_a)
    pc_b, q_b = label_to_root_qual(label_b)
    zero = np.zeros(6, dtype=complex) if is_tiv else np.zeros(12)
    va = scheme_fn(pc_a, q_a) if pc_a is not None else zero
    vb = scheme_fn(pc_b, q_b) if pc_b is not None else zero
    return cosine(va, vb)


def main():
    all_rows = []  # dict per candidate pair, pooled across songs
    per_song = {}
    for slug in SONGS:
        census = json.loads((OUT_DIR / f"bar_merge_full_census_{slug}.json").read_text())
        base_ch = get_baseline_chords(slug)
        rows = []
        for c in census["candidates"]:
            (t0a, t1a), (t0b, t1b) = c["spans"]
            mid_a, mid_b = 0.5 * (t0a + t1a), 0.5 * (t0b + t1b)
            ca = next((x for x in base_ch if x["start_s"] <= mid_a < x["end_s"]), None)
            cb = next((x for x in base_ch if x["start_s"] <= mid_b < x["end_s"]), None)
            if ca is None or cb is None:
                continue
            ba, bb = label_bucket(ca["label"]), label_bucket(cb["label"])
            if ba is None or bb is None:
                continue
            agree = int(ba == bb)
            row = {"audio_sim": c["confidence"], "agree": agree, "tier": c["tier"], "song": slug}
            for name, fn in SCHEMES.items():
                row[name] = symbolic_sim_for_span(base_ch, t0a, t1a, t0b, t1b, fn, name == "V3_tiv")
            rows.append(row)
            all_rows.append(row)
        per_song[slug] = rows
        print(f"=== {slug} === n={len(rows)}", file=sys.stderr)

    # --- audio-only baseline (reproduce realaudio_threshold_check's number as a sanity check) ---
    audio_sims = np.array([r["audio_sim"] for r in all_rows])
    agrees = np.array([r["agree"] for r in all_rows])
    audio_only_096 = float(agrees[audio_sims >= 0.96].mean())
    audio_only_099 = float(agrees[audio_sims >= 0.99].mean()) if (audio_sims >= 0.99).any() else None
    print(f"\nsanity-check reproduction: audio-only tau=0.96 agree_rate={audio_only_096:.4f} "
          f"(expect ~0.394); tau=0.99 agree_rate={audio_only_099}", file=sys.stderr)

    # --- joint 2D grid: tau_audio fixed at existing tau_auto=0.96 gate (the deployed
    # selection), sweep tau_symbolic on TOP of it for each scheme ---
    print("\n=== JOINT FILTER: audio_sim>=0.96 (deployed tau_auto) AND symbolic_sim>=tau_sym ===",
          file=sys.stderr)
    auto_mask = audio_sims >= 0.96
    n_auto_pool = int(auto_mask.sum())
    joint_sweep = {}
    for scheme in SCHEMES:
        sym = np.array([r[scheme] for r in all_rows])
        sweep = []
        for tau_sym in [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]:
            mask = auto_mask & (sym >= tau_sym)
            n = int(mask.sum())
            recall_of_auto_pool = n / n_auto_pool if n_auto_pool else None
            precision = float(agrees[mask].mean()) if n else None
            sweep.append({"tau_symbolic": tau_sym, "n": n,
                           "recall_of_auto_pool": recall_of_auto_pool, "precision": precision})
            print(f"  [{scheme}] tau_sym={tau_sym:.2f}  n={n:4d}  "
                  f"recall_of_auto_pool={recall_of_auto_pool}  precision={precision}", file=sys.stderr)
        joint_sweep[scheme] = sweep

    # --- full 2D precision surface: sweep BOTH tau_audio and tau_symbolic
    # (not just anchored at 0.96) for V1_binary, the premise-check winner ---
    print("\n=== FULL 2D SURFACE (V1_binary), audio_sim x tau_symbolic ===", file=sys.stderr)
    sym_v1 = np.array([r["V1_binary"] for r in all_rows])
    tau_audio_grid = [0.90, 0.93, 0.96, 0.97, 0.98, 0.99]
    tau_sym_grid = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
    surface = []
    for ta in tau_audio_grid:
        for ts in tau_sym_grid:
            mask = (audio_sims >= ta) & (sym_v1 >= ts)
            n = int(mask.sum())
            precision = float(agrees[mask].mean()) if n else None
            surface.append({"tau_audio": ta, "tau_symbolic": ts, "n": n, "precision": precision})
            print(f"  tau_audio={ta:.2f}  tau_sym={ts:.2f}  n={n:4d}  precision={precision}",
                  file=sys.stderr)

    out = {
        "n_total_pairs": len(all_rows),
        "audio_only_sanity_check": {"tau_0.96_agree_rate": audio_only_096,
                                     "tau_0.99_agree_rate": audio_only_099},
        "n_auto_pool_tau_audio_096": n_auto_pool,
        "joint_sweep_anchored_at_tau_auto_096": joint_sweep,
        "full_2d_surface_V1_binary": surface,
        "premise_check_4_known_pairs": {
            "abba_32_64_false_positive": {"audio_sim": 0.978954165174635,
                "V1_binary": 0.6666666666666667, "V2_weighted": 0.9285714285714286,
                "V3_tiv": 0.5265377208552913},
            "aretha_13_17_false_positive": {"audio_sim": 0.9783348193575163,
                "V1_binary": 0.8660254037844387, "V2_weighted": 0.9522267331341506,
                "V3_tiv": 0.7964870830354095},
            "aretha_53_69_true_positive": {"audio_sim": 0.9891801790916748,
                "V1_binary": 1.0000000000000002, "V2_weighted": 1.0, "V3_tiv": 1.0},
            "abba_206_222_true_positive": {"audio_sim": 0.9937504840722634,
                "V1_binary": 1.0, "V2_weighted": 1.0, "V3_tiv": 1.0000000000000002},
        },
    }
    (OUT_DIR / "joint_threshold_search_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote joint_threshold_search_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
