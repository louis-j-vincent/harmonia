"""noise_calibrate.py — Step 1 of the 2026-07-18 SSM-merge-criterion brief:
calibrate synthetic noise injected into iReal per-bar chord vectors so its
SSM statistics match the 3 real-audio SSMs already generated tonight.

Real audio has NO section ground truth (confirmed in the handoff / known
issues — no GT exists for these 3 songs), so the calibration TARGET
statistic must be computable identically on both corpora without labels.

Two statistics, both per register (bass/treble kept separate, per the
mid-session user design call):

  stat_B (GT-FREE, used for the actual calibration): at grain=8, for each
  block i, best_match(i) = max_{j: |j-i|>=2} sim(i,j) (excludes immediate
  neighbors, which are trivially locally-coherent and would inflate the
  "repeat found" signal without being a real distant repeat). background(i)
  = median of the same off-diagonal-excluding-neighbors row. stat_B =
  mean_i(best_match(i)) - mean_i(background(i)) — "how far the best distant
  match stands above the typical/noise-floor similarity." Large stat_B =
  clean, sharp repeats stand out. Small/zero stat_B = repeats don't stand
  out from noise (smeared-out SSM).

  stat_A (LABEL-based, iReal only, VALIDATION of stat_B, not the target
  itself): AUC of block_sim scores classifying position-matched same-GT-
  section-run pairs (positive) vs random different-label pairs (negative),
  reusing bar_distance_matrix.py's sanity_check() "known-repeat" logic,
  generalized to blocks instead of single bars and formalized as an AUC
  instead of an eyeballed number. Computed on CLEAN iReal only, to confirm
  stat_B (which throws away the labels) actually tracks the real,
  label-based separability signal before trusting it as a proxy for real
  audio's (label-free) separability.

Noise model: additive Gaussian on each per-bar 12-d vector (bass-proxy =
root-only one-hot, treble-proxy = full V1 binary chord-tone vector — the
natural iReal analogs of real audio's bass/treble registers), sigma swept,
re-L2-normalized after perturbation (same "renormalize per bar" discipline
as rawchroma.py). Sweep sigma, find the value whose corpus-mean stat_B (per
register) matches real audio's stat_B (mean over the 3 songs, per register).
"""
from __future__ import annotations
import sys, io, json, random
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import chord_vector_binary, cosine
from chord_distance_eval import nuclear_spans, block_sim, FILES
from symstruct import qbucket

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
GRAIN = 8
N_CORPUS_SAMPLE = 300  # corpus-scale, matches this project's usual TEST split size
SEEDS = [0, 1, 2]


def root_onehot(root_pc, qual):
    v = np.zeros(12)
    if root_pc is None or root_pc < 0:
        return v
    v[root_pc % 12] = 1.0
    return v


def load_corpus_registers(max_tunes=None):
    """Per tune: per-bar (bass_proxy, treble_proxy) 12-d vectors + labels."""
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
            bass_vecs, treb_vecs, labels = [], [], []
            for bar_no, section, slots in mc.timeline:
                bass_accum, treb_accum = None, None
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
                bass_vecs.append(bass_accum if bass_accum is not None else np.zeros(12))
                treb_vecs.append(treb_accum if treb_accum is not None else np.zeros(12))
                labels.append(section)
            if len(labels) < GRAIN * 2 or len(set(labels)) < 2:
                continue
            # L2-normalize per bar (parallel to rawchroma.py's per-bar unit-norm convention)
            def _l2(vecs):
                out = []
                for v in vecs:
                    n = np.linalg.norm(v)
                    out.append(v / n if n > 1e-9 else v)
                return out
            out.append({"title": mc.title, "bass": _l2(bass_vecs), "treble": _l2(treb_vecs),
                        "labels": labels})
    if max_tunes:
        random.Random(0).shuffle(out)
        out = out[:max_tunes]
    return out


def add_noise(vecs, sigma, rng):
    if sigma <= 0:
        return vecs
    out = []
    for v in vecs:
        n = np.linalg.norm(v)
        if n < 1e-9:
            out.append(v)
            continue
        noisy = v + rng.normal(0, sigma, size=v.shape)
        nn = np.linalg.norm(noisy)
        out.append(noisy / nn if nn > 1e-9 else v)
    return out


def block_matrix(bar_vecs, grain=GRAIN):
    n = len(bar_vecs)
    spans = nuclear_spans(n, grain)
    block_bars = [bar_vecs[s:e] for (s, e) in spans]
    m = len(spans)
    sim = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            sim[i, j] = block_sim(block_bars[i], block_bars[j])
    return sim, spans


def stat_B(sim, min_gap=2):
    """REVISED after premise-check failed (see log): the original "gap"
    formulation (mean_best - mean_background) correlates NEGATIVELY with
    the label-based stat_A AUC (r=-0.23, n=78) because mean_best and
    mean_background are themselves strongly positively correlated (both
    driven by shared per-song factors like chord density) and cancel in
    the subtraction. Diagnostic sweep over 4 candidate GT-free statistics
    found mean_p90 (mean over blocks of the 90th-percentile off-diagonal
    similarity, excluding immediate neighbors) has the best positive
    correlation with stat_A (r=0.50, n=78) — smoother/more robust than
    mean_best (r=0.44) since it's not a single noisy max. Using mean_p90
    as stat_B going forward; kept the same function name/call sites."""
    m = sim.shape[0]
    if m < min_gap + 2:
        return None
    p90 = []
    for i in range(m):
        others = [j for j in range(m) if abs(j - i) >= min_gap]
        if not others:
            continue
        row = sim[i, others]
        p90.append(np.percentile(row, 90))
    if not p90:
        return None
    return float(np.mean(p90))


def stat_A_auc(bar_vecs, labels, grain=GRAIN):
    """Label-based AUC: block_sim of position-matched same-run-repeat pairs
    (positive) vs random different-label block pairs (negative)."""
    n = len(bar_vecs)
    spans = nuclear_spans(n, grain)
    block_bars = [bar_vecs[s:e] for (s, e) in spans]
    block_labels = []
    for (s, e) in spans:
        # majority label in this block
        from collections import Counter
        block_labels.append(Counter(labels[s:e]).most_common(1)[0][0])
    m = len(spans)
    pos, neg = [], []
    for i in range(m):
        for j in range(i + 1, m):
            sim = block_sim(block_bars[i], block_bars[j])
            if block_labels[i] == block_labels[j]:
                pos.append(sim)
            else:
                neg.append(sim)
    if not pos or not neg:
        return None
    # AUC via Mann-Whitney U
    pos_a, neg_a = np.array(pos), np.array(neg)
    count = 0
    for p in pos_a:
        count += np.sum(p > neg_a) + 0.5 * np.sum(p == neg_a)
    auc = count / (len(pos_a) * len(neg_a))
    return float(auc)


def real_audio_stats():
    """Load already-saved bar_ssm_rawchroma_*.json grain=8 bass/treble
    matrices and compute stat_B for each register PLUS the combined signal
    (sim_combined = (sim_bass+sim_treble)/2, exactly == cosine(bt_concat)
    per the already-proven independent-unit-norm identity — this is now the
    PRIMARY target per the mid-session correction below; bass/treble kept
    as diagnostic breakdown, not discarded, since Step 0's per-register
    substrate is reused either way)."""
    songs = ["aretha_chain_of_fools", "autumn_leaves", "abba_chiquitita"]
    out = {}
    for song in songs:
        d = json.loads((OUT_DIR / ("bar_ssm_rawchroma_%s.json" % song)).read_text())
        sim_b = np.array(d["grains_bass"][str(GRAIN)]["similarity_matrix"])
        sim_t = np.array(d["grains_treble"][str(GRAIN)]["similarity_matrix"])
        sim_c = (sim_b + sim_t) / 2.0
        out.setdefault("bass", {})[song] = stat_B(sim_b)
        out.setdefault("treble", {})[song] = stat_B(sim_t)
        out.setdefault("combined", {})[song] = stat_B(sim_c)
    return out


def main():
    print("=== Real-audio stat_B (grain=8), per register ===")
    real_stats = real_audio_stats()
    real_target = {}
    for reg, songs in real_stats.items():
        vals = list(songs.values())
        real_target[reg] = float(np.mean(vals))
        print("  %-8s per-song: %s  mean=%.4f" %
              (reg, {k: round(v, 4) for k, v in songs.items()}, real_target[reg]))

    print("\n=== Loading iReal corpus (bass/treble proxies, keynorm) ===")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE)
    print("  corpus sample: %d tunes" % len(corpus))

    print("\n=== Validation: does GT-free stat_B track label-based stat_A on CLEAN iReal? ===")
    sample_for_auc = corpus[:80]
    pairs = []
    for c in sample_for_auc:
        sim, spans = block_matrix(c["treble"], GRAIN)
        sb = stat_B(sim)
        sa = stat_A_auc(c["treble"], c["labels"], GRAIN)
        if sb is not None and sa is not None:
            pairs.append((sb, sa))
    if len(pairs) >= 5:
        sb_arr = np.array([p[0] for p in pairs])
        sa_arr = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(sb_arr, sa_arr)[0, 1])
        print("  n=%d songs, Pearson corr(stat_B, stat_A AUC) = %.3f" % (len(pairs), corr))
        print("  clean-iReal stat_B: mean=%.4f  clean-iReal stat_A(AUC): mean=%.4f" %
              (sb_arr.mean(), sa_arr.mean()))
    else:
        print("  too few songs with valid stats:", len(pairs))

    print("\n=== Noise sweep (sigma), PRIMARY = combined signal (mid-session")
    print("    corrected back to bt_concat-equivalent scalar; bass/treble")
    print("    kept as diagnostic breakdown, not the primary target) ===")
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    sweep = {"bass": {}, "treble": {}, "combined": {}}
    for reg in ("bass", "treble", "combined"):
        print("  register=%s  target(real audio)=%.4f" % (reg, real_target[reg]))
        for sigma in sigmas:
            vals = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                for c in corpus:
                    if reg == "combined":
                        # perturb each register independently, re-normalize each
                        # half separately, THEN concatenate — mirrors rawchroma.py's
                        # per-half-independent-L2-norm discipline exactly.
                        nb = add_noise(c["bass"], sigma, rng)
                        nt = add_noise(c["treble"], sigma, rng)
                        noisy = [np.concatenate([b, t]) for b, t in zip(nb, nt)]
                    else:
                        noisy = add_noise(c[reg], sigma, rng)
                    sim, spans = block_matrix(noisy, GRAIN)
                    sb = stat_B(sim)
                    if sb is not None:
                        vals.append(sb)
            m = float(np.mean(vals)) if vals else None
            sweep[reg][sigma] = m
            print("    sigma=%.2f  mean stat_B=%s (n=%d obs)" %
                  (sigma, ("%.4f" % m) if m is not None else "NA", len(vals)))

    # find best sigma per register (closest to real-audio target)
    best = {}
    for reg in ("bass", "treble", "combined"):
        valid = {s: v for s, v in sweep[reg].items() if v is not None}
        best_sigma = min(valid, key=lambda s: abs(valid[s] - real_target[reg]))
        best[reg] = {"sigma": best_sigma, "stat_B": valid[best_sigma],
                     "target": real_target[reg]}
        print("  BEST sigma for %s: %.2f (stat_B=%.4f, target=%.4f, |gap|=%.4f)" %
              (reg, best_sigma, valid[best_sigma], real_target[reg],
               abs(valid[best_sigma] - real_target[reg])))

    out = {"grain": GRAIN, "n_corpus_sample": len(corpus),
           "real_audio_stats": real_stats, "real_target": real_target,
           "validation_corr_stat_B_vs_stat_A": corr if len(pairs) >= 5 else None,
           "sweep": sweep, "best_sigma": best}
    (OUT_DIR / "noise_calibrate_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote noise_calibrate_results.json")


if __name__ == "__main__":
    main()
