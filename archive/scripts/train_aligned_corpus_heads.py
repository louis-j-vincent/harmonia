"""train_aligned_corpus_heads.py — close-the-loop training on aligned_corpus.npz.

2026-07-21 mission: train->evaluate on real, non-circular audio->error analysis
->propose fix->retrain, comparing against the CURRENT PRODUCTION baseline
(musx bass/quality), not just the old RWC-only nnls24_heads.npz checkpoint.

SONG-LEVEL split on aligned_corpus (never chord-level, matches project
convention everywhere else). RWC (all rows, already has its own internal
song-level CV elsewhere) is used as additional training signal; the real,
non-circular held-out test set is a slice of aligned_corpus songs that NEVER
appear in training for either RWC-only or merged variants.

Variants trained/evaluated on the SAME held-out aligned_corpus songs:
  A. rwc_only     — reproduces the shipped nnls24_heads.npz recipe (RWC rows
                    only), i.e. "what's shipped today, in-house heads".
  B. merged       — RWC rows + aligned_corpus TRAIN rows.
  C. aligned_only — aligned_corpus TRAIN rows only (sanity: does real audio
                    alone, with no RWC, already beat A?).

Metrics: root accuracy, quality accuracy (strict 7-way exact), and
quality partial-credit (family-or-better: maj-ish{maj,dom,aug,sus} /
min-ish{min} / dim-ish{hdim,dim}).

Usage:
  .venv/bin/python scripts/train_aligned_corpus_heads.py --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from multihead_training import MLP, train_clf, predict_proba, rotate_by_root  # noqa: E402

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
KQ = 7
FAMILY_OF_Q7 = np.array([0, 1, 0, 2, 2, 0, 0])  # maj-ish/min-ish/dim-ish

RWC_NPZ = REPO / "data" / "cache" / "rwc" / "rwc_nnls24.npz"
ALIGNED_NPZ = REPO / "data" / "cache" / "aligned_corpus" / "aligned_corpus.npz"
SHIPPED = REPO / "harmonia" / "models" / "nnls24_heads.npz"


def load_rwc():
    d = np.load(RWC_NPZ, allow_pickle=True)
    return d["nnls24"].astype(np.float32), d["root"].astype(np.int64) % 12, \
        d["quality_idx"].astype(np.int64), d["song_id"]


def load_aligned():
    d = np.load(ALIGNED_NPZ, allow_pickle=True)
    assert list(map(str, d["qualities"])) == QUALITIES
    return d["feat24"].astype(np.float32), d["root"].astype(np.int64) % 12, \
        d["quality_idx"].astype(np.int64), d["song_id"]


def flag_suspect_songs(feat24, roots, song_ids, min_rows=6, frac_thresh=0.4):
    """Song-level (not per-section) transposition/misalignment quality gate.

    Error analysis (2026-07-21, docs/known_issues.md): a handful of
    aligned_corpus songs show near-ZERO root accuracy that is NOT random
    noise -- the (predicted - true) root offset has one dominant nonzero
    value covering a large fraction of that song's rows (e.g. "Ain't
    Misbehavin'": 50% of rows off by exactly +3 semitones, 0% off by 0).
    This is the signature of either a globally-transposed real performance
    (common for jazz vocalists) or a wrong/mismatched YouTube video slipping
    past the per-SECTION chroma-shape accept gate, which validates chord-
    CHANGE timing, not absolute pitch -- a transposed cover has the identical
    change-point pattern, so it passes undetected.

    This is deliberately AGGREGATED over every row in a song (avg ~30 rows/
    song here), not the per-section short-window (3-10 chord) check that was
    already tried and rejected for exactly this reason (known_issues.md
    "TRIED AND REJECTED: model-predicted root as an alignment safeguard" --
    short windows have too few chords for one wrong root to average out).
    Flagged there as worth trying at the AGGREGATE level; this is that.

    Uses the UNTRAINED bass-chroma argmax (not a trained model prediction)
    against the corpus's own GT root -- a data-quality filter, not a label
    correction (never overwrite iReal's GT with a model guess, per CLAUDE.md
    rule #3 trust ordering).
    """
    import collections
    bass_argmax = feat24[:, :12].argmax(1)
    suspect = set()
    for s in np.unique(song_ids):
        m = song_ids == s
        n = int(m.sum())
        if n < min_rows:
            continue
        diff = (bass_argmax[m] - roots[m]) % 12
        cnt = collections.Counter(diff.tolist())
        top_off, top_n = cnt.most_common(1)[0]
        agree0 = cnt.get(0, 0) / n
        if top_off != 0 and (top_n / n) >= frac_thresh and (top_n / n) > agree0:
            suspect.add(str(s))
    return suspect


def song_level_split(song_ids, seed, test_frac=0.2):
    songs = np.unique(song_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(songs)
    n_te = max(1, int(round(test_frac * len(songs))))
    test_songs = set(songs[:n_te].tolist())
    te = np.isin(song_ids, list(test_songs))
    return ~te, te, test_songs


def train_heads(nn24, roots, quals, epochs_root=50, epochs_qual=60, seed=0):
    """Train root + quality-cascade heads on the given rows (no further split
    -- caller passes exactly the training pool; internal val slice for early
    stopping only)."""
    rng = np.random.RandomState(seed + 100)
    n = len(nn24)
    va_pick = rng.choice(n, size=max(1, n // 8), replace=False)
    va = np.zeros(n, bool)
    va[va_pick] = True
    tr = ~va

    bass, treb = nn24[:, :12], nn24[:, 12:]
    cnt = np.bincount(quals, minlength=KQ)
    cw = (cnt.sum() / (KQ * np.maximum(cnt, 1))).astype(np.float32)

    rm = train_clf(nn24[tr], roots[tr], nn24[va], roots[va], 24, 12,
                   hid=(128, 64), epochs=epochs_root)
    root_proba = predict_proba(rm, nn24)
    pred_root = root_proba.argmax(1)

    Xc = np.concatenate([rotate_by_root(bass, pred_root),
                         rotate_by_root(treb, pred_root)], 1)
    qm = train_clf(Xc[tr], quals[tr], Xc[va], quals[va], 24, KQ,
                   hid=(128, 64), epochs=epochs_qual, cw=cw)
    return rm, qm


def eval_heads(rm, qm, nn24, roots, quals):
    bass, treb = nn24[:, :12], nn24[:, 12:]
    root_proba = predict_proba(rm, nn24)
    pred_root = root_proba.argmax(1)
    root_acc = float((pred_root == roots).mean())

    Xc = np.concatenate([rotate_by_root(bass, pred_root),
                         rotate_by_root(treb, pred_root)], 1)
    pred_q = predict_proba(qm, Xc).argmax(1)
    q_acc = float((pred_q == quals).mean())
    fam_acc = float((FAMILY_OF_Q7[pred_q] == FAMILY_OF_Q7[quals]).mean())
    # root-conditioned quality (oracle root) -- isolates the quality head
    Xc_oracle = np.concatenate([rotate_by_root(bass, roots),
                                rotate_by_root(treb, roots)], 1)
    pred_q_oracle = predict_proba(qm, Xc_oracle).argmax(1)
    q_acc_oracle = float((pred_q_oracle == quals).mean())
    return dict(root_acc=root_acc, qual_acc=q_acc, qual_family_acc=fam_acc,
                qual_acc_oracle_root=q_acc_oracle,
                pred_root=pred_root, pred_qual=pred_q)


def confusion(pred, true, k, names):
    cm = np.zeros((k, k), dtype=int)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    lines = ["true\\pred  " + "  ".join(f"{n:>5s}" for n in names)]
    for i, n in enumerate(names):
        lines.append(f"{n:>9s}  " + "  ".join(f"{cm[i, j]:5d}" for j in range(k)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--filter-suspect", action="store_true",
                    help="drop songs flagged by flag_suspect_songs (transposition/misalignment gate)")
    a = ap.parse_args()

    nn_rwc, r_rwc, q_rwc, sid_rwc = load_rwc()
    nn_al, r_al, q_al, sid_al = load_aligned()
    print(f"RWC: {len(nn_rwc)} rows / {len(np.unique(sid_rwc))} songs")
    print(f"aligned_corpus: {len(nn_al)} rows / {len(np.unique(sid_al))} songs")

    if a.filter_suspect:
        suspect = flag_suspect_songs(nn_al, r_al, sid_al)
        print(f"filter-suspect: dropping {len(suspect)} songs: {sorted(suspect)}")
        keep = ~np.isin(sid_al, list(suspect))
        nn_al, r_al, q_al, sid_al = nn_al[keep], r_al[keep], q_al[keep], sid_al[keep]
        print(f"aligned_corpus after filter: {len(nn_al)} rows / {len(np.unique(sid_al))} songs")

    tr_mask, te_mask, test_songs = song_level_split(sid_al, a.seed, a.test_frac)
    print(f"aligned_corpus split: train {tr_mask.sum()} rows / "
          f"{len(np.unique(sid_al[tr_mask]))} songs, "
          f"test {te_mask.sum()} rows / {len(test_songs)} songs")
    print("held-out songs:", sorted(test_songs))

    nn_al_tr, r_al_tr, q_al_tr = nn_al[tr_mask], r_al[tr_mask], q_al[tr_mask]
    nn_al_te, r_al_te, q_al_te = nn_al[te_mask], r_al[te_mask], q_al[te_mask]
    print("held-out quality dist:",
          dict(zip(QUALITIES, np.bincount(q_al_te, minlength=KQ).tolist())))

    results = {}

    # A. RWC-only (reproduces shipped recipe)
    print("\n=== A: rwc_only ===", flush=True)
    rm_a, qm_a = train_heads(nn_rwc, r_rwc, q_rwc, seed=a.seed)
    results["rwc_only"] = eval_heads(rm_a, qm_a, nn_al_te, r_al_te, q_al_te)

    # B. merged RWC + aligned-train
    print("=== B: merged ===", flush=True)
    nn_m = np.concatenate([nn_rwc, nn_al_tr], 0)
    r_m = np.concatenate([r_rwc, r_al_tr], 0)
    q_m = np.concatenate([q_rwc, q_al_tr], 0)
    rm_b, qm_b = train_heads(nn_m, r_m, q_m, seed=a.seed)
    results["merged"] = eval_heads(rm_b, qm_b, nn_al_te, r_al_te, q_al_te)

    # C. aligned-only (sanity)
    print("=== C: aligned_only ===", flush=True)
    if len(nn_al_tr) >= 50:
        rm_c, qm_c = train_heads(nn_al_tr, r_al_tr, q_al_tr, seed=a.seed,
                                 epochs_root=80, epochs_qual=100)
        results["aligned_only"] = eval_heads(rm_c, qm_c, nn_al_te, r_al_te, q_al_te)
    else:
        print("  skipped (too few rows)")

    # D. shipped checkpoint (production in-house heads, RWC-only, frozen)
    print("=== D: shipped nnls24_heads.npz ===", flush=True)
    if SHIPPED.exists():
        import torch
        d = np.load(SHIPPED, allow_pickle=True)
        hid = tuple(int(x) for x in d["hid"])

        def _build(prefix, din, dout):
            m = MLP(din, dout, hid)
            state = {k[len(prefix) + 2:]: torch.tensor(d[k])
                     for k in d.files if k.startswith(prefix + "__")}
            m.load_state_dict(state)
            m.eval()
            return m

        rm_d = _build("root", int(d["root_din"][0]), int(d["root_dout"][0]))
        qm_d = _build("qual", int(d["qual_din"][0]), int(d["qual_dout"][0]))
        results["shipped_checkpoint"] = eval_heads(rm_d, qm_d, nn_al_te, r_al_te, q_al_te)
    else:
        print("  shipped checkpoint missing, skipped")

    print("\n" + "=" * 70)
    print(f"{'variant':20s} {'root_acc':>9s} {'qual_acc':>9s} {'qual_fam':>9s} {'qual|oracle_root':>17s}")
    for name, r in results.items():
        print(f"{name:20s} {r['root_acc']:9.3f} {r['qual_acc']:9.3f} "
              f"{r['qual_family_acc']:9.3f} {r['qual_acc_oracle_root']:17.3f}")

    # confusion matrices for the best in-house variant (by qual_acc) -- error analysis
    best_name = max(results, key=lambda k: results[k]["qual_acc"])
    print(f"\n--- quality confusion matrix, best variant ({best_name}) ---")
    print(confusion(results[best_name]["pred_qual"], q_al_te, KQ, QUALITIES))
    print(f"\n--- root confusion (best variant), rows = true pc, cols = pred pc ---")
    root_names = [str(i) for i in range(12)]
    print(confusion(results[best_name]["pred_root"], r_al_te, 12, root_names))

    np.savez(REPO / "scratchpad" / "aligned_corpus_train_results.npz",
              seed=a.seed, test_songs=np.array(sorted(test_songs)),
              te_mask_song_id=sid_al[te_mask], te_root=r_al_te, te_qual=q_al_te,
              **{f"{k}__pred_root": v["pred_root"] for k, v in results.items()},
              **{f"{k}__pred_qual": v["pred_qual"] for k, v in results.items()})
    print("\nsaved scratchpad/aligned_corpus_train_results.npz")

    return results, sorted(test_songs)


if __name__ == "__main__":
    main()
