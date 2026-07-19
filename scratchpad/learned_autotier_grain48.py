"""learned_autotier_grain48.py -- 2026-07-19, LEARNED (not threshold) merge
criterion at section grain (k=4, k=8), architected REAL-AUDIO-FIRST per the
brief: "a k=4/8 en peut trouver un critere via ML... le but est de trouver
le bon compromis pour ne quasiment pas avoir de faux positif."

Direct continuation of three prior calls this session, all logged in
docs/known_issues.md (search "Section-grain (4-bar vs 8-bar) AUTO-tier",
"N-WAY section-cluster group pooling", "Matrix-intrinsic k-selection") plus
the ONE prior learned-classifier attempt at bar grain
(scratchpad/merge_criterion.py: logreg did NOT beat a threshold at
FPR<=0.05, recall 0.126 vs 0.187). The established failure mode this call
must not repeat: a criterion trained/tuned on the SYMBOLIC iReal corpus
that was never checked against real audio before being trusted -- the
grain=4/8 tau_auto port recovered just 0.9%/0% of real known-true pairs
despite 0.96-0.99 nested-CV symbolic validation.

**This call's different angle**: train DIRECTLY on real-audio pairs
(leave-one-song-out across the 3 available real songs), not on the
symbolic corpus, since the symbolic-corpus-transfer channel has now failed
independently 3 times (tau threshold, tau_auto nested-CV, joint gate all
derived symbolically). Real-audio full census already exists at both
grains for all 3 songs (section_realaudio_autotier_results.json,
AUDIO_FLOOR=0.50, MIN_GAP_BLOCKS=0) -- reused here, not rebuilt, except we
add 4 new features that need the raw audio Gram matrix (not saved in that
JSON) so this script recomputes per_bar_rawchroma once per song (cheap,
~seconds) rather than re-deriving from the huge results file.

New features (beyond audio_sim, symbolic_sim already used in the joint
gate): block_distance_norm (|i-j|/n_blocks), song_length_bars (n, as a
per-song noise-context knob), local_variance (mean of block i's and block
j's row-variance of audio_sim against ALL other blocks in the same song --
proxy for "is this whole neighborhood noisy"), abs_diff (|audio_sim -
symbolic_sim|, agreement-DIRECTION not just each value).

A LARGE symbolic-corpus run (grain 4/8, iReal, features that transfer:
audio-analog sim_bass/sim_treb/sim_combined/block_distance/size_ratio/
local_variance -- NOT symbolic_sim, since on iReal that would be built
directly from the GT chord identity and would be a near-tautological
perfect separator, not a real feature) is also run, for direct comparison
to the merge_criterion.py bar-level finding and to explicitly show
whether symbolic-corpus training material helps or hurts once ported to
real audio (expectation, given 3 independent prior failures: it does not
transfer; this call tests that expectation rather than assuming it).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from rawchroma import per_bar_rawchroma
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans, block_sim
from chord_distance import chord_vector_binary
from realaudio_threshold_check import get_baseline_chords
from section_merge_candidates import bar_label_bucket, block_gram_sim
from auto_apply_merges import SONGS, AUDIO_DIR
from section_tau_auto_search import load_corpus_bar_chords

OUT_DIR = Path(__file__).resolve().parent
MIN_GAP_BLOCKS = 0
AUDIO_FLOOR = 0.50
GRAINS = [4, 8]
TAU_AUTO_SYMBOLIC = {4: 0.9665, 8: 0.9583}
TARGET_FPRS = [0.01, 0.02, 0.05]

ARETHA_VAMP_WINDOWS = [(5.0, 83.7), (100.0, 161.0)]


def in_any_window(t0, t1, windows):
    return any(w0 <= t0 and t1 <= w1 for (w0, w1) in windows)


# ---------------------------------------------------------------------
# REAL AUDIO: richer feature build, per song, per grain
# ---------------------------------------------------------------------

def build_real_song_features(slug, audio_name, grain):
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

    # full block x block audio_sim matrix (cheap: m <= 83), for local-variance feature
    block_sim_mat = np.zeros((m, m))
    for i in range(m):
        si, ei = spans[i]
        for j in range(m):
            if i == j:
                continue
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            block_sim_mat[i, j] = block_gram_sim(prefix, sq, n, si, sj, L)
    row_var = np.array([np.var(block_sim_mat[i, [k for k in range(m) if k != i]])
                         if m > 1 else 0.0 for i in range(m)])

    rows = []
    for i in range(m):
        si, ei = spans[i]
        for j in range(i + 1 + MIN_GAP_BLOCKS, m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            audio_sim = block_sim_mat[i, j]
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
            symbolic_sim = num / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0
            pseudo_gt = (1 if agree_mismatches <= 1 else 0) if agree_valid > 0 else None

            t0a, t1a = float(bar_times[si]), float(bar_times[ei])
            t0b, t1b = float(bar_times[sj]), float(bar_times[ej])
            ext = None
            if slug == "autumn_leaves":
                if grain == 8 and (si, ei) == (0, 8) and (sj, ej) == (8, 16):
                    ext = 1
                elif grain == 4 and si < 4 and ei <= 4 and sj >= 8 and ej <= 12:
                    ext = 1
            elif slug == "aretha_franklin_chain_of_fools_official_lyric_video":
                if in_any_window(t0a, t1a, ARETHA_VAMP_WINDOWS) and in_any_window(t0b, t1b, ARETHA_VAMP_WINDOWS):
                    ext = 1

            block_distance_norm = (j - i) / max(m, 1)
            local_variance = 0.5 * (row_var[i] + row_var[j])
            abs_diff = abs(audio_sim - symbolic_sim)

            rows.append({
                "song": slug, "grain": grain, "blocks": [i, j],
                "audio_sim": float(audio_sim), "symbolic_sim": float(symbolic_sim),
                "block_distance_norm": float(block_distance_norm),
                "song_length_bars": int(n), "local_variance": float(local_variance),
                "abs_diff": float(abs_diff),
                "pseudo_gt_match": pseudo_gt, "external_gt_match": ext,
            })
    return rows, {"n_bars_total": n, "n_blocks": m, "tempo_bpm": tempo}


FEATURE_NAMES_REAL = ["audio_sim", "symbolic_sim", "block_distance_norm",
                       "local_variance", "abs_diff"]
# AUDIO-ONLY ablation: symbolic_sim and abs_diff are BOTH derived from the
# same baseline-decode bar buckets that pseudo_gt_match itself is defined
# from (mismatches<=1 of those buckets) -- using them as features to
# predict pseudo_gt_match risks being near-tautological (the exact
# circularity CLAUDE.md rule #3 / known_issues.md's joint-gate entry
# already flag for this project). This ablation checks whether any real,
# non-circular signal survives once those two features are dropped.
FEATURE_NAMES_AUDIO_ONLY = ["audio_sim", "block_distance_norm", "local_variance"]


def rows_to_xy(rows, gt_key):
    X, y, meta = [], [], []
    for r in rows:
        if r[gt_key] is None:
            continue
        X.append([r[f] for f in FEATURE_NAMES_REAL])
        y.append(r[gt_key])
        meta.append(r)
    return np.array(X, dtype=float), np.array(y, dtype=int), meta


# ---------------------------------------------------------------------
# SYMBOLIC iReal corpus: audio-analog features only (no decode-derived
# symbolic_sim -- would be near-tautological w.r.t. GT, see docstring)
# ---------------------------------------------------------------------

FEATURE_NAMES_SYM = ["sim_bass", "sim_treb", "sim_combined",
                      "block_distance_norm", "size_ratio", "local_variance"]


def build_symbolic_pairs_for_tune(c, grain):
    bass = np.array(c["bass"]); treb = np.array(c["treble"])
    chord_ids = c["chord_ids"]
    n = len(chord_ids)
    spans = nuclear_spans(n, grain)
    m = len(spans)
    if m < 2:
        return []
    Gb = bass @ bass.T; Gt = treb @ treb.T
    sqb = np.diag(Gb).copy(); sqt = np.diag(Gt).copy()
    pb = diagonal_prefix_sums(Gb); pt = diagonal_prefix_sums(Gt)

    sim_mat = np.zeros((m, m))
    for i in range(m):
        si, ei = spans[i]
        for j in range(m):
            if i == j:
                continue
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            sb = block_gram_sim(pb, sqb, n, si, sj, L)
            st = block_gram_sim(pt, sqt, n, si, sj, L)
            sim_mat[i, j] = 0.5 * (sb + st)
    row_var = np.array([np.var(sim_mat[i, [k for k in range(m) if k != i]])
                         if m > 1 else 0.0 for i in range(m)])

    rows = []
    for i in range(m):
        si, ei = spans[i]
        for j in range(i + 1, m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            valid = 0; mismatches = 0
            for t in range(L):
                ci, cj = chord_ids[si + t], chord_ids[sj + t]
                if ci is None or cj is None:
                    continue
                valid += 1
                if ci != cj:
                    mismatches += 1
            if valid == 0:
                continue
            sb = block_gram_sim(pb, sqb, n, si, sj, L)
            st = block_gram_sim(pt, sqt, n, si, sj, L)
            sc = 0.5 * (sb + st)
            size_i = ei - si; size_j = ej - sj
            size_ratio = min(size_i, size_j) / max(size_i, size_j)
            block_distance_norm = (j - i) / max(m, 1)
            local_variance = 0.5 * (row_var[i] + row_var[j])
            label = 1 if mismatches <= 1 else 0
            rows.append({"sim_bass": float(sb), "sim_treb": float(st),
                          "sim_combined": float(sc),
                          "block_distance_norm": float(block_distance_norm),
                          "size_ratio": float(size_ratio),
                          "local_variance": float(local_variance),
                          "label": label})
    return rows


def build_symbolic_dataset(grain, max_tunes=600):
    corpus = load_corpus_bar_chords(max_tunes=max_tunes)
    X, y = [], []
    for c in corpus:
        for r in build_symbolic_pairs_for_tune(c, grain):
            X.append([r[f] for f in FEATURE_NAMES_SYM]); y.append(r["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int), len(corpus)


# ---------------------------------------------------------------------
# Model zoo + FPR-gated evaluation
# ---------------------------------------------------------------------

def fpr_gated_threshold(scores, y, target_fpr):
    neg = scores[y == 0]
    if len(neg) == 0:
        return None
    thr = float(np.quantile(neg, 1 - target_fpr))
    return thr


def eval_at_threshold(scores, y, thr):
    pred = (scores >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {"n_flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "fpr": fpr}


def make_models():
    return {
        "threshold_audio_sim": ("score_col", "audio_sim"),
        "logreg": ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        "shallow_tree": ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                                           class_weight="balanced", random_state=0)),
        "gbm": ("model", GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                                      learning_rate=0.1, random_state=0)),
    }


def leave_one_song_out_real(rows, gt_key, feature_names, target_fprs):
    songs = sorted(set(r["song"] for r in rows))
    out = {"gt_key": gt_key, "songs": songs, "folds": []}
    for held in songs:
        train_rows = [r for r in rows if r["song"] != held and r[gt_key] is not None]
        test_rows = [r for r in rows if r["song"] == held and r[gt_key] is not None]
        if len(train_rows) < 10 or len(test_rows) < 3:
            out["folds"].append({"held_out": held, "skipped": True,
                                  "n_train": len(train_rows), "n_test": len(test_rows)})
            continue
        Xtr = np.array([[r[f] for f in feature_names] for r in train_rows], dtype=float)
        ytr = np.array([r[gt_key] for r in train_rows], dtype=int)
        Xte = np.array([[r[f] for f in feature_names] for r in test_rows], dtype=float)
        yte = np.array([r[gt_key] for r in test_rows], dtype=int)
        if len(set(ytr.tolist())) < 2:
            out["folds"].append({"held_out": held, "skipped": True, "reason": "single-class train",
                                  "n_train": len(train_rows), "n_test": len(test_rows)})
            continue

        fold = {"held_out": held, "n_train": len(train_rows), "n_test": len(test_rows),
                "n_test_pos": int(yte.sum()), "models": {}}
        for name, (kind, obj) in make_models().items():
            if kind == "score_col":
                col = feature_names.index(obj)
                sc_tr, sc_te = Xtr[:, col], Xte[:, col]
            else:
                clf = obj
                clf.fit(Xtr, ytr)
                sc_tr = clf.predict_proba(Xtr)[:, 1]
                sc_te = clf.predict_proba(Xte)[:, 1]
            fold["models"][name] = {}
            for tfpr in target_fprs:
                thr = fpr_gated_threshold(sc_tr, ytr, tfpr)
                if thr is None:
                    fold["models"][name][str(tfpr)] = None
                    continue
                res = eval_at_threshold(sc_te, yte, thr)
                res["thr"] = thr
                fold["models"][name][str(tfpr)] = res
        out["folds"].append(fold)
    return out


def pool_fold_metrics(los_result, target_fprs):
    """Pool tp/fp/fn/tn across LOSO folds for each model/fpr -> corpus-level
    real-audio recall/precision, not just per-fold (small-n per fold)."""
    pooled = {}
    for fold in los_result["folds"]:
        if fold.get("skipped"):
            continue
        for name, byfpr in fold["models"].items():
            for tfpr in target_fprs:
                key = (name, tfpr)
                r = byfpr.get(str(tfpr))
                if r is None:
                    continue
                acc = pooled.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_test": 0})
                acc["tp"] += r["tp"]; acc["fp"] += r["fp"]; acc["fn"] += r["fn"]; acc["tn"] += r["tn"]
                acc["n_test"] += fold["n_test"]
    out = {}
    for (name, tfpr), acc in pooled.items():
        tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        out.setdefault(name, {})[str(tfpr)] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "fpr": fpr, "n_test_pooled": acc["n_test"]}
    return out


def symbolic_trained_ported_to_real(real_rows, gt_key, grain, target_fprs, max_tunes=600):
    """Train each model on the FULL symbolic corpus (audio-analog features
    only), port DIRECTLY to real audio (no fine-tuning) -- replicates the
    established failure-mode test (3 prior calls found near-0% real
    recovery from symbolic-only training); this call checks whether a
    richer feature set + nonlinear model changes that conclusion."""
    Xsym, ysym, n_tunes = build_symbolic_dataset(grain, max_tunes=max_tunes)
    real_test = [r for r in real_rows if r[gt_key] is not None]
    if not real_test:
        return {"skipped": True}
    Xte = np.array([[r["audio_sim"], r["symbolic_sim"], r["block_distance_norm"],
                      r["local_variance"], r["abs_diff"]] for r in real_test], dtype=float)
    # symbolic corpus has no direct symbolic_sim/abs_diff analog (would be
    # tautological); approximate: sim_combined stands in for BOTH audio_sim
    # and symbolic_sim on the symbolic side (they coincide when there's no
    # decode noise), abs_diff=0, to keep the real-side feature vector the
    # same dimensionality a model trained on symbolic 3-feature core can use.
    Xsym_expanded = np.stack([
        Xsym[:, FEATURE_NAMES_SYM.index("sim_combined")],
        Xsym[:, FEATURE_NAMES_SYM.index("sim_combined")],
        Xsym[:, FEATURE_NAMES_SYM.index("block_distance_norm")],
        Xsym[:, FEATURE_NAMES_SYM.index("local_variance")],
        np.zeros(len(Xsym)),
    ], axis=1)
    yte = np.array([r[gt_key] for r in real_test], dtype=int)

    n_real_test_neg = int(len(yte) - yte.sum())
    out = {"n_symbolic_tunes": n_tunes, "n_symbolic_pairs": len(ysym),
           "n_real_test": len(yte), "n_real_test_pos": int(yte.sum()),
           "n_real_test_neg": n_real_test_neg,
           "precision_is_vacuous": n_real_test_neg == 0,
           "note": ("test set has ZERO confirmed-negative examples -- precision=1.0 whenever "
                     "n_flagged>0 is a tautology of the label design, not a real precision "
                     "measurement; only recall (fraction of externally-confirmed TRUE pairs "
                     "recovered) is informative here" if n_real_test_neg == 0 else None),
           "models": {}}
    for name, (kind, obj) in make_models().items():
        if kind == "score_col":
            sc_tr = Xsym_expanded[:, 0]
            sc_te = Xte[:, 0]
        else:
            clf = obj
            clf.fit(Xsym_expanded, ysym)
            sc_tr = clf.predict_proba(Xsym_expanded)[:, 1]
            sc_te = clf.predict_proba(Xte)[:, 1]
        out["models"][name] = {}
        for tfpr in target_fprs:
            thr = fpr_gated_threshold(sc_tr, ysym, tfpr)
            if thr is None:
                out["models"][name][str(tfpr)] = None
                continue
            res = eval_at_threshold(sc_te, yte, thr)
            res["thr"] = thr
            out["models"][name][str(tfpr)] = res
    return out


def main():
    t0 = time.time()
    results = {"feature_names_real": FEATURE_NAMES_REAL, "feature_names_symbolic": FEATURE_NAMES_SYM,
                "target_fprs": TARGET_FPRS, "grains": {}}

    print("Building real-audio feature rows (3 songs x 2 grains)...")
    real_rows_by_grain = {4: [], 8: []}
    song_meta = {}
    for slug, sm in SONGS.items():
        for grain in GRAINS:
            rows, meta = build_real_song_features(slug, sm["audio_name"], grain)
            real_rows_by_grain[grain].extend(rows)
            song_meta.setdefault(slug, {})[grain] = meta
            n_pseudo = sum(1 for r in rows if r["pseudo_gt_match"] is not None)
            n_pseudo_pos = sum(1 for r in rows if r["pseudo_gt_match"] == 1)
            n_ext = sum(1 for r in rows if r["external_gt_match"] is not None)
            n_ext_pos = sum(1 for r in rows if r["external_gt_match"] == 1)
            print(f"  {slug} grain={grain}: n_pairs={len(rows)} pseudo_gt {n_pseudo_pos}/{n_pseudo} "
                  f"ext_gt {n_ext_pos}/{n_ext}  elapsed={time.time()-t0:.1f}s")

    results["song_meta"] = {s: {str(g): m for g, m in gm.items()} for s, gm in song_meta.items()}

    for grain in GRAINS:
        print(f"\n########## GRAIN={grain} ##########")
        rows = real_rows_by_grain[grain]
        grain_out = {"n_pairs_total": len(rows)}

        for gt_key in ("pseudo_gt_match", "external_gt_match"):
            print(f"-- LOSO on real audio (full feature set incl. symbolic_sim), gt={gt_key} --")
            los = leave_one_song_out_real(rows, gt_key, FEATURE_NAMES_REAL, TARGET_FPRS)
            pooled = pool_fold_metrics(los, TARGET_FPRS)
            grain_out[f"loso_{gt_key}"] = {"folds": los["folds"], "pooled": pooled}
            for name, byfpr in pooled.items():
                for tfpr, r in byfpr.items():
                    print(f"   {name:22s} fpr<={tfpr}: n_flagged(tp+fp)={r['tp']+r['fp']:3d} "
                          f"recall={r['recall']:.3f} precision={r['precision']:.3f} "
                          f"realized_fpr={r['fpr']:.3f} (pooled test n={r['n_test_pooled']})")

            print(f"-- LOSO on real audio (AUDIO-ONLY ablation, no symbolic_sim/abs_diff -- circularity check), gt={gt_key} --")
            los_ao = leave_one_song_out_real(rows, gt_key, FEATURE_NAMES_AUDIO_ONLY, TARGET_FPRS)
            pooled_ao = pool_fold_metrics(los_ao, TARGET_FPRS)
            grain_out[f"loso_{gt_key}_audio_only"] = {"folds": los_ao["folds"], "pooled": pooled_ao}
            for name, byfpr in pooled_ao.items():
                for tfpr, r in byfpr.items():
                    print(f"   {name:22s} fpr<={tfpr}: n_flagged(tp+fp)={r['tp']+r['fp']:3d} "
                          f"recall={r['recall']:.3f} precision={r['precision']:.3f} "
                          f"realized_fpr={r['fpr']:.3f} (pooled test n={r['n_test_pooled']})")
            if gt_key == "external_gt_match" and not pooled:
                print("   [NOTE] external_gt_match, as currently built (aretha vamp-window + "
                      "autumn_leaves A/A repeat coverage rules), assigns label=1 or label=None "
                      "-- it NEVER assigns 0. LOSO training is impossible (single-class, no "
                      "negatives to FPR-gate against) -- this is a real GT-design gap, not a "
                      "training failure, flagged explicitly rather than silently reporting empty.")

        print(f"-- symbolic-corpus-trained, ported to real audio (replicate prior failure?), gt=pseudo_gt_match --")
        ported = symbolic_trained_ported_to_real(rows, "pseudo_gt_match", grain, TARGET_FPRS)
        grain_out["symbolic_trained_ported_pseudo_gt"] = ported
        if not ported.get("skipped"):
            for name, byfpr in ported["models"].items():
                for tfpr, r in byfpr.items():
                    if r is None:
                        continue
                    print(f"   {name:22s} fpr<={tfpr}: n_flagged={r['n_flagged']:3d} "
                          f"recall={r['recall']:.3f} precision={r['precision']:.3f} realized_fpr={r['fpr']:.3f}")

        print(f"-- symbolic-corpus-trained, ported to real audio, gt=external_gt_match --")
        ported_ext = symbolic_trained_ported_to_real(rows, "external_gt_match", grain, TARGET_FPRS)
        grain_out["symbolic_trained_ported_external_gt"] = ported_ext
        if not ported_ext.get("skipped"):
            for name, byfpr in ported_ext["models"].items():
                for tfpr, r in byfpr.items():
                    if r is None:
                        continue
                    print(f"   {name:22s} fpr<={tfpr}: n_flagged={r['n_flagged']:3d} "
                          f"recall={r['recall']:.3f} precision={r['precision']:.3f} realized_fpr={r['fpr']:.3f}")

        results["grains"][str(grain)] = grain_out

    out_path = OUT_DIR / "learned_autotier_grain48_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}, total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
