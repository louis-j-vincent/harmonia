"""tau_auto_search.py — 2026-07-18 (SCOPE-GUARDED continuation call): find
tau_auto, the similarity threshold above which bar-merge candidates can be
AUTO-APPLIED (no human tap), as the second tier above the existing
tau_suggest (k-NN k=1 floor~0.9-0.93, FPR<=0.05, the shipped suggestion-tier
operating point — see docs/known_issues.md "Multi-algorithm bar-merge
candidate-generation BAKEOFF").

User's original framing (French): "un seuil a partir duquel on merge en
n'ayant jamais de faux positifs" -- a threshold above which merging never
produces a false positive. RELAXED mid-task by the coordinator: a 1-2%
error rate in the auto tier is explicitly acceptable; the goal is now the
LOWEST tau_auto that keeps the auto-tier's error rate reliably <= ~2%, so
the auto tier is usefully large, not the most conservative threshold
findable. Both the strict (zero observed FP) and relaxed (~2% target)
versions are computed and reported; the relaxed one is what gets shipped
(tagged into the candidate JSON).

**Metric definition (stated explicitly, this matters):** the operationally
relevant quantity for auto-apply is NOT the classical global FPR (FP / all
negative pairs in the corpus) used by the suggestion-tier bakeoff -- at a
high similarity threshold, the global FPR is trivially near-zero simply
because almost no negative pair reaches high similarity at all, regardless
of threshold quality. What actually matters for "how often is an
auto-merge WRONG" is the error rate AMONG THE PAIRS THE THRESHOLD ACTUALLY
SELECTS: auto_error_rate(tau) = FP(tau) / (FP(tau) + TP(tau)) where
FP/TP are counted only among pairs with sim >= tau. This is 1 - precision
(false discovery rate) of the auto-tier decision rule, not the suggestion
tier's FP/(FP+TN). Both are reported per threshold for completeness, but
auto_error_rate is the one the search optimizes and the one that answers
the user's actual question ("of the things we merge without asking, how
many are wrong").

**Feature/level choice**: computed at BAR level (grain=1), not the
grain=8 nuclear-block level merge_criterion.py/clustering_bakeoff.py used
for the suggestion-tier bakeoff. Bar level matches what
bar_merge_candidates.py actually thresholds in production (per-bar
bt_concat cosine similarity on real audio) far more closely than an
8-bar-block aggregate would; iReal has no audio, so the closest available
proxy is the SAME bass-proxy (root one-hot) / treble-proxy (full chord-tone
binary vector) features noise_calibrate.py / merge_criterion.py already use
for GT-label pairs, evaluated per BAR pair instead of per BLOCK pair, with
MIN_GAP=4 bars applied (matching bar_merge_candidates.py's own min_gap, to
exclude trivial adjacent-sustain pairs from the search -- those aren't
representative of the real candidate pool, which already excludes them by
construction).

Corpus: FULL 1989-tune iReal corpus (all 7 playlists, no max_tunes cap --
this call's brief explicitly asks for the largest achievable negative-pair
pool, not the 900-tune sample the earlier bakeoff used for a different,
cheaper sweep). Song-level train/val-pool (80%) vs held-out test fold (20%)
split, 5 seeds -- tau_auto is SELECTED on the 80% pool only and VALIDATED
on the untouched 20% fold, never the reverse (CLAUDE.md "Honesty bar").

**GROUND-TRUTH LABEL CORRECTION (found during this call's premise check,
before trusting any threshold from the section-label GT the rest of this
project's bar-merge thread has used): "same GT section" is the WRONG label
for this task.** A first pass using the same-section label
(merge_criterion.py's/clustering_bakeoff.py's convention, inherited from
the original STRUCTURE-detection framing) found that even at sim==1.0
EXACTLY (bit-identical bass/treble feature vectors), ~50% of bar pairs
belong to DIFFERENT sections -- i.e. no threshold, however high, gets
anywhere near a low error rate under that label. Root cause: two bars can
carry the literal same one-bar chord (e.g. both "C") while structurally
living in different sections (verse vs. bridge both resting on the tonic
is extremely common) -- section identity is simply not determined by a
single bar's harmony, so "same section" was never going to be a clean
target for a BAR-level similarity threshold (grain=8 nuclear blocks, which
is what the rest of the bar-merge thread actually thresholds on, carry
enough context to make section-identity approximately decodable; a single
bar does not). But for THIS task -- auto-pooling a bar's chord evidence
with another bar's -- what actually matters is whether the two bars carry
the SAME UNDERLYING CHORD, not whether they're part of the same structural
section. Pooling two genuinely-same-chord bars from different sections
(e.g. two "C" bars, one in the verse and one in the bridge) is exactly the
intended, correct use of this mechanism, not a false positive. **GT
redefined for this script: label=1 iff the two bars' majority chord
identity (root_pc, qbucket -- the same 6-way coarse quality family
symstruct.qbucket already uses corpus-wide) matches; label=0 otherwise.**
This is computed directly from each bar's MMA chord symbol(s) (majority
vote across a bar's slots when a bar contains a chord change), not
inferred from the bass/treble proxy vectors, so it does not trivially
overlap with the similarity feature the threshold is being swept on.
"""
from __future__ import annotations
import sys, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from noise_calibrate import load_corpus_registers
from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance_eval import FILES
from symstruct import qbucket
from collections import Counter

OUT_DIR = Path(__file__).resolve().parent
MIN_GAP = 4  # matches bar_merge_candidates.py's MIN_GAP
TAU_SUGGEST = 0.93  # matches bar_merge_candidates.py's DEFAULT_TAU (current shipped suggest-tier floor)
SEEDS = [0, 1, 2, 3, 4]
RELAXED_TARGET = 0.02  # 1-2% band, upper edge -- coordinator's relaxed spec


def clopper_pearson_upper(x, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson upper bound on a true proportion
    p, given x 'successes' (here: false positives) observed in n trials.
    Exact closed form exists for x=0 (p_upper = 1 - alpha**(1/n)); for x>0
    use the Beta-quantile definition via scipy if available, else fall back
    to a normal-approximation-free bisection on the incomplete beta (so this
    has no hard scipy dependency)."""
    if n == 0:
        return 1.0
    if x == 0:
        return 1.0 - alpha ** (1.0 / n)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, x + 1, n - x))
    except ImportError:
        # bisection on the regularized incomplete beta via math.comb-free
        # binomial survival sum (n is small enough here, x+1..n term count
        # bounded, this is only a fallback path)
        from math import lgamma, exp, log
        def log_binom_pmf(k, n, p):
            if p <= 0:
                return -np.inf if k > 0 else 0.0
            if p >= 1:
                return -np.inf if k < n else 0.0
            return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
                    + k * log(p) + (n - k) * log(1 - p))
        def cdf_le_x(p):
            return sum(exp(log_binom_pmf(k, n, p)) for k in range(0, x + 1))
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            # P(X<=x | p=mid) decreasing in mid; want the mid where this
            # equals alpha (that's the CP upper bound definition)
            if cdf_le_x(mid) > alpha:
                lo = mid
            else:
                hi = mid
        return hi


def rule_of_three(n):
    return 3.0 / n if n > 0 else 1.0


def load_corpus_bar_chords(max_tunes=None):
    """Same parsing path as noise_calibrate.load_corpus_registers (bass-proxy
    root one-hot / treble-proxy chord-tone-binary vectors per bar), but ALSO
    records each bar's own chord identity (root_pc, qbucket) via majority
    vote across the bar's slots (handles mid-bar chord changes) -- this is
    the corrected GT for "should these two bars be pooled", see module
    docstring. Bars with no resolvable majority chord (all rests, or a tie
    with no single majority) get chord_id=None and are excluded from pairs
    at build time (can't judge whether an undefined chord matches another)."""
    import io
    from contextlib import redirect_stdout
    from noise_calibrate import root_onehot
    from chord_distance import chord_vector_binary
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(Path("data/ireal") / (f + ".txt"))
        except Exception:
            continue
        for t in tunes:
            try:
                mc = tune_to_mma(t)
            except Exception:
                continue
            shift = 0
            if mc.key:
                pc = chord_root_pc(mc.key.rstrip("-"))
                shift = (-pc % 12) if pc is not None else 0
            bass_vecs, treb_vecs, chord_ids, sections = [], [], [], []
            for bar_no, section, slots in mc.timeline:
                bass_accum, treb_accum = None, None
                slot_chords = []
                for (_, _, mma) in slots:
                    pc = chord_root_pc(mma)
                    if pc is None:
                        continue
                    rpc = (pc + shift) % 12
                    q = qbucket(mma)
                    bv = root_onehot(rpc, q)
                    tv = chord_vector_binary(rpc, q)
                    bass_accum = bv if bass_accum is None else bass_accum + bv
                    treb_accum = tv if treb_accum is None else treb_accum + tv
                    if q is not None and q >= 0:
                        slot_chords.append((rpc, q))
                bass_vecs.append(bass_accum if bass_accum is not None else np.zeros(12))
                treb_vecs.append(treb_accum if treb_accum is not None else np.zeros(12))
                sections.append(section)
                if slot_chords:
                    chord_ids.append(Counter(slot_chords).most_common(1)[0][0])
                else:
                    chord_ids.append(None)
            if len(sections) < 8 or all(c is None for c in chord_ids):
                continue

            def _l2(vecs):
                res = []
                for v in vecs:
                    n = np.linalg.norm(v)
                    res.append(v / n if n > 1e-9 else v)
                return res
            out.append({"title": mc.title, "bass": _l2(bass_vecs), "treble": _l2(treb_vecs),
                        "chord_ids": chord_ids, "sections": sections})
    if max_tunes:
        random.Random(0).shuffle(out)
        out = out[:max_tunes]
    return out


def build_bar_pairs(corpus, min_gap=MIN_GAP):
    """Per tune: all (sim_combined, label) pairs with |i-j)>=min_gap.
    label=1 means SAME CHORD IDENTITY (root_pc, qbucket) -- the corrected
    GT for this task, see module docstring. Pairs where either bar has no
    resolvable chord (chord_ids[i] is None) are excluded."""
    per_tune = []
    for c in corpus:
        bass, treb, chord_ids = c["bass"], c["treble"], c["chord_ids"]
        n = len(chord_ids)
        rows = []
        for i in range(n):
            if chord_ids[i] is None:
                continue
            bi, ti = bass[i], treb[i]
            ci = chord_ids[i]
            for j in range(i + min_gap, n):
                if chord_ids[j] is None:
                    continue
                sb = float(np.dot(bi, bass[j]))
                st = float(np.dot(ti, treb[j]))
                sc = 0.5 * (sb + st)
                rows.append((sc, 1 if ci == chord_ids[j] else 0))
        if rows:
            per_tune.append(rows)
    return per_tune


def split_songs(per_tune, seed, test_frac=0.20):
    ids = list(range(len(per_tune)))
    random.Random(seed).shuffle(ids)
    n_test = int(len(ids) * test_frac)
    test_ids = set(ids[:n_test])
    pool = [per_tune[i] for i in ids if i not in test_ids]
    test = [per_tune[i] for i in ids if i in test_ids]
    return pool, test


def split_songs_3way(per_tune, seed, val_frac=0.20, test_frac=0.20):
    """Song-level 60/20/20 train/val/test. tau is SELECTED on train, checked
    (and escalated if it breaches target) on val, and FINALLY blind-scored
    on test -- test is touched exactly once, at the very end."""
    ids = list(range(len(per_tune)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    train = [per_tune[i] for i in train_ids]
    val = [per_tune[i] for i in val_ids]
    test = [per_tune[i] for i in test_ids]
    return train, val, test


def flatten(tune_rows_list):
    sims, labels = [], []
    for rows in tune_rows_list:
        for s, l in rows:
            sims.append(s); labels.append(l)
    return np.array(sims), np.array(labels)


def sweep_thresholds(sims, labels, start=TAU_SUGGEST):
    """Sort descending by sim, compute cumulative (N_selected, FP, TP) for
    every prefix with sim>=start. Returns arrays (thr, n_sel, fp, tp) indexed
    by prefix length (thr[k] = the k-th highest sim among sim>=start)."""
    mask = sims >= start
    s = sims[mask]; l = labels[mask]
    order = np.argsort(-s)
    s_sorted = s[order]; l_sorted = l[order]
    is_fp = (l_sorted == 0).astype(int)
    is_tp = (l_sorted == 1).astype(int)
    cum_fp = np.cumsum(is_fp)
    cum_tp = np.cumsum(is_tp)
    n_sel = np.arange(1, len(s_sorted) + 1)
    return s_sorted, n_sel, cum_fp, cum_tp


def find_tau_for_target(s_sorted, n_sel, cum_fp, cum_tp, target, min_n=30):
    """Largest k (lowest tau) such that cumulative FP/N <= target AND N>=min_n.
    Returns (tau, k, fp, n) or None if no k qualifies."""
    err = cum_fp / n_sel
    ok = (err <= target) & (n_sel >= min_n)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    k = idx[-1]  # largest qualifying prefix -> lowest tau
    return float(s_sorted[k]), int(n_sel[k]), int(cum_fp[k]), int(n_sel[k])


def find_tau_strict_zero(s_sorted, n_sel, cum_fp, cum_tp, min_n=10):
    """Largest k with cum_fp==0 (and N>=min_n)."""
    ok = (cum_fp == 0) & (n_sel >= min_n)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    k = idx[-1]
    return float(s_sorted[k]), int(n_sel[k]), 0, int(n_sel[k])


def select_tau_nested(sims_train, labels_train, sims_val, labels_val, target,
                       start=TAU_SUGGEST, min_n=30):
    """Diagnosed instability (this call, see module docstring): picking the
    lowest tau meeting `target` on a SINGLE pool overfits -- the corpus has
    a steep, heterogeneous error cliff right above tau_suggest (many tunes'
    error rate jumps from ~2% to ~9-40% within a narrow band), so a
    single-split threshold choice can land on the wrong side of that cliff
    for an unseen split. This does proper nested selection: candidate taus
    are the sorted unique TRAIN sims (>=start) meeting the target on TRAIN;
    walk from the loosest (lowest tau) candidate upward (stricter) until the
    SAME tau also meets the target on the independent VAL split. Returns the
    first tau satisfying both, or None if none does even at the strictest
    (train-side) candidate."""
    s_sorted, n_sel, cum_fp, cum_tp = sweep_thresholds(sims_train, labels_train, start=start)
    if len(s_sorted) == 0:
        return None
    err = cum_fp / n_sel
    ok = (err <= target) & (n_sel >= min_n)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    # candidates from loosest (largest k / lowest tau) to strictest (k=idx[0])
    for k in idx[::-1]:
        tau = float(s_sorted[k])
        val_check = validate_on_held_out(sims_val, labels_val, tau)
        if val_check["n"] >= min_n and val_check["error_rate"] is not None and val_check["error_rate"] <= target:
            return {"tau_auto": tau, "train_n": int(n_sel[k]), "train_fp": int(cum_fp[k]),
                    "train_error_rate": float(cum_fp[k] / n_sel[k]), "val_check": val_check}
    return None


def validate_on_held_out(sims_test, labels_test, tau):
    mask = sims_test >= tau
    n = int(mask.sum())
    if n == 0:
        return {"tau": tau, "n": 0, "fp": 0, "tp": 0, "error_rate": None,
                "cp_upper_95": None, "rule_of_three": None}
    fp = int(np.sum((labels_test[mask] == 0)))
    tp = int(np.sum((labels_test[mask] == 1)))
    err = fp / n
    return {"tau": tau, "n": n, "fp": fp, "tp": tp, "error_rate": err,
            "cp_upper_95": clopper_pearson_upper(fp, n, 0.05),
            "rule_of_three": rule_of_three(n) if fp == 0 else None}


def main():
    t0 = time.time()
    print("Loading FULL iReal corpus (no max_tunes cap), with per-bar chord identity...")
    corpus = load_corpus_bar_chords(max_tunes=None)
    print("  %d tunes, elapsed %.1fs" % (len(corpus), time.time() - t0))

    print("Building per-bar pair pools (min_gap=%d, label=same chord identity)..." % MIN_GAP)
    per_tune = build_bar_pairs(corpus)
    total_pairs = sum(len(r) for r in per_tune)
    print("  %d tunes usable, %d total bar-pairs, elapsed %.1fs" %
          (len(per_tune), total_pairs, time.time() - t0))

    # ---- Global (full-corpus, no split) diagnostic curve first: this
    # exposed the fold-instability finding below (a steep, heterogeneous
    # error cliff just above tau_suggest), so report it up front for context.
    sims_all, labels_all = flatten(per_tune)
    s_all, n_all, fp_all, tp_all = sweep_thresholds(sims_all, labels_all, start=TAU_SUGGEST)
    print("\n=== GLOBAL full-corpus error-vs-tau curve (diagnostic, not a split-safe estimate) ===")
    global_curve = []
    for probe_tau in [1.0, 0.999, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]:
        idx = np.searchsorted(-s_all, -probe_tau, side="right")
        if idx == 0:
            continue
        row = {"tau": probe_tau, "n": int(idx), "fp": int(fp_all[idx - 1]),
               "error_rate": float(fp_all[idx - 1] / idx)}
        global_curve.append(row)
        print("  tau=%.3f  n=%d  fp=%d  error_rate=%.4f" % (probe_tau, row["n"], row["fp"], row["error_rate"]))

    fold_results = []
    for seed in SEEDS:
        pool, test = split_songs(per_tune, seed)
        sims_pool, labels_pool = flatten(pool)
        sims_test, labels_test = flatten(test)

        s_sorted, n_sel, cum_fp, cum_tp = sweep_thresholds(sims_pool, labels_pool, start=TAU_SUGGEST)
        if len(s_sorted) == 0:
            print("seed=%d: no pairs with sim>=%.2f in pool, skipping" % (seed, TAU_SUGGEST))
            continue

        strict = find_tau_strict_zero(s_sorted, n_sel, cum_fp, cum_tp)
        # naive single-split relaxed selection -- KEPT for comparison, this is
        # what exposed the fold-instability problem (see docstring); NOT what
        # gets shipped.
        relaxed_naive = find_tau_for_target(s_sorted, n_sel, cum_fp, cum_tp, RELAXED_TARGET)

        # nested (train/val/test) relaxed selection -- what gets shipped.
        train3, val3, test3 = split_songs_3way(per_tune, seed)
        sims_tr, labels_tr = flatten(train3)
        sims_va, labels_va = flatten(val3)
        sims_te, labels_te = flatten(test3)
        nested = select_tau_nested(sims_tr, labels_tr, sims_va, labels_va, RELAXED_TARGET, start=TAU_SUGGEST)

        row = {"seed": seed, "n_pool_pairs_above_tau_suggest": int(len(s_sorted)),
               "n_test_songs": len(test), "n_pool_songs": len(pool)}

        if strict is not None:
            tau_s, k_s, fp_s, n_s = strict
            val_s = validate_on_held_out(sims_test, labels_test, tau_s)
            row["strict"] = {"tau_auto": tau_s, "pool_n": n_s, "pool_fp": fp_s,
                              "pool_error_rate": fp_s / n_s,
                              "pool_cp_upper_95": clopper_pearson_upper(fp_s, n_s, 0.05),
                              "pool_rule_of_three": rule_of_three(n_s),
                              "held_out_validation": val_s}
        else:
            row["strict"] = None

        if relaxed_naive is not None:
            tau_r, k_r, fp_r, n_r = relaxed_naive
            val_r = validate_on_held_out(sims_test, labels_test, tau_r)
            row["relaxed_naive_single_split"] = {"tau_auto": tau_r, "pool_n": n_r, "pool_fp": fp_r,
                               "pool_error_rate": fp_r / n_r,
                               "pool_cp_upper_95": clopper_pearson_upper(fp_r, n_r, 0.05),
                               "held_out_validation": val_r}
        else:
            row["relaxed_naive_single_split"] = None

        if nested is not None:
            tau_n = nested["tau_auto"]
            blind_test = validate_on_held_out(sims_te, labels_te, tau_n)
            row["relaxed_nested"] = {**nested, "blind_test_validation": blind_test}
        else:
            row["relaxed_nested"] = None

        fold_results.append(row)
        print("\n=== seed=%d ===" % seed)
        print("  pool pairs above tau_suggest=%.2f: %d" % (TAU_SUGGEST, len(s_sorted)))
        if row["strict"]:
            st = row["strict"]
            print("  STRICT (0 observed FP): tau_auto=%.4f  pool N=%d  CP_upper95=%.4f  rule_of_3=%.4f" %
                  (st["tau_auto"], st["pool_n"], st["pool_cp_upper_95"], st["pool_rule_of_three"]))
            v = st["held_out_validation"]
            print("    held-out test: N=%d FP=%d error_rate=%s CP_upper95=%s" %
                  (v["n"], v["fp"], v["error_rate"], v["cp_upper_95"]))
        else:
            print("  STRICT: no threshold in [tau_suggest,1.0] achieves 0 FP with N>=10")
        if row["relaxed_naive_single_split"]:
            rl = row["relaxed_naive_single_split"]
            print("  RELAXED-NAIVE (single-split, target<=%.0f%%): tau_auto=%.4f  pool N=%d  pool_FP=%d  pool_err=%.4f  CP_upper95=%.4f" %
                  (RELAXED_TARGET * 100, rl["tau_auto"], rl["pool_n"], rl["pool_fp"],
                   rl["pool_error_rate"], rl["pool_cp_upper_95"]))
            v = rl["held_out_validation"]
            print("    held-out test: N=%d FP=%d error_rate=%s CP_upper95=%s   <-- fold instability shows up HERE" %
                  (v["n"], v["fp"], v["error_rate"], v["cp_upper_95"]))
        else:
            print("  RELAXED-NAIVE: no threshold in [tau_suggest,1.0] achieves <=%.0f%% with N>=30" %
                  (RELAXED_TARGET * 100))
        if row["relaxed_nested"]:
            rn = row["relaxed_nested"]
            print("  RELAXED-NESTED (train->val escalation, target<=%.0f%%): tau_auto=%.4f  train_err=%.4f  val_err=%.4f" %
                  (RELAXED_TARGET * 100, rn["tau_auto"], rn["train_error_rate"], rn["val_check"]["error_rate"]))
            bt = rn["blind_test_validation"]
            print("    BLIND test fold: N=%d FP=%d error_rate=%s CP_upper95=%s" %
                  (bt["n"], bt["fp"], bt["error_rate"], bt["cp_upper_95"]))
        else:
            print("  RELAXED-NESTED: no tau satisfied target on BOTH train and val")

    # consensus tau_auto (relaxed-nested, the one to ship): max across folds
    # (most conservative fold-consistent choice) -- report full distribution too
    relaxed_taus = [r["relaxed_nested"]["tau_auto"] for r in fold_results if r["relaxed_nested"]]
    strict_taus = [r["strict"]["tau_auto"] for r in fold_results if r["strict"]]
    print("\n=== CONSENSUS ACROSS %d FOLDS ===" % len(fold_results))
    if relaxed_taus:
        print("  RELAXED-NESTED tau_auto per fold:", ["%.4f" % t for t in relaxed_taus])
        print("  RELAXED-NESTED tau_auto: mean=%.4f  max=%.4f  min=%.4f" %
              (np.mean(relaxed_taus), np.max(relaxed_taus), np.min(relaxed_taus)))
    if strict_taus:
        print("  STRICT tau_auto per fold:", ["%.4f" % t for t in strict_taus])
        print("  STRICT tau_auto: mean=%.4f  max=%.4f  min=%.4f" %
              (np.mean(strict_taus), np.max(strict_taus), np.min(strict_taus)))

    out = {
        "min_gap": MIN_GAP, "tau_suggest": TAU_SUGGEST, "relaxed_target": RELAXED_TARGET,
        "seeds": SEEDS, "n_tunes": len(per_tune), "n_total_pairs": total_pairs,
        "global_curve": global_curve,
        "fold_results": fold_results,
        "consensus": {
            "relaxed_nested_tau_auto_mean": float(np.mean(relaxed_taus)) if relaxed_taus else None,
            "relaxed_nested_tau_auto_max": float(np.max(relaxed_taus)) if relaxed_taus else None,
            "relaxed_nested_tau_auto_min": float(np.min(relaxed_taus)) if relaxed_taus else None,
            "strict_tau_auto_mean": float(np.mean(strict_taus)) if strict_taus else None,
            "strict_tau_auto_max": float(np.max(strict_taus)) if strict_taus else None,
            "strict_tau_auto_min": float(np.min(strict_taus)) if strict_taus else None,
        },
    }
    (OUT_DIR / "tau_auto_search_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote tau_auto_search_results.json, total elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
