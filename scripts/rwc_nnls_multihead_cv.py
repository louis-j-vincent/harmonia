"""Multi-seed song-grouped CV: the ORIGINAL NNLS full recipe vs BP48 baseline,
on RWC-Popular, over the IDENTICAL chord blocks / splits (confound-clean).

Recipe provenance (verified, not reconstructed): imports the exact functions of
scratchpad/multihead_training.py that produced the 0.890 Billboard headline
(#31 Add-4, PHASE-0 AUDIT row 4):
  * root head  = MLP(din -> 128 -> 64 -> 12), nonlinear, val-early-stopped
  * quality    = root-relative rotation (rotate_by_root: candidate root -> idx 0)
                 + learned trigram context (6 neighbour root-posteriors, o in
                 {-3..-1,1..3}, each rotated into the target root frame, concat
                 as FEATURES). Root posteriors come from the trained root head.

Feature front-ends compared on the SAME rows/splits (only this differs):
  * NNLS-24 : data/cache/rwc/rwc_nnls24.npz  (real Mauch NNLS-Chroma VAMP,
              C-frame, L2-per-half; bass=[:12], treble=[12:])
  * BP48    : rwc_bp48_fixed.npz feat48_abs (root head) / feat48 root-relative
              (quality head) -- the established RWC BP48 baseline representation.

Quality is reported OACLE-root-frame for BOTH (the BP48 baseline's feat48 is
already GT-root-relative, so this is apples-to-apples), plus the realistic
predicted-root cascade for NNLS. 7-way quality (RWC vocab: maj/min/dom/hdim/dim/
aug/sus), balanced (macro-recall) accuracy.

Every number printed here comes from a completed run; the caller quotes the log
scratchpad/rwc_nnls_cv.log. No expected/should-be values.
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from harmonia.data.corpus_schema import load_corpus
from multihead_training import (
    train_clf, predict_proba, rotate_by_root, neighbor, balanced_recall,
)

QUALITIES = ['maj', 'min', 'dom', 'hdim', 'dim', 'aug', 'sus']
KQ = 7


def song_split(sid, seed, test_frac=0.2, val_frac=0.1):
    songs = np.unique(sid); rng = np.random.RandomState(seed); rng.shuffle(songs)
    n = len(songs)
    nte = max(1, int(round(test_frac * n)))
    nva = max(1, int(round(val_frac * n)))
    te_s = set(songs[:nte]); va_s = set(songs[nte:nte + nva])
    te = np.isin(sid, list(te_s)); va = np.isin(sid, list(va_s))
    tr = ~(te | va)
    return tr, va, te


def ctx_feats(root_proba, sid, root_frame):
    """6 neighbour root-posteriors rotated into `root_frame` (target root at 0)."""
    feats = []
    for o in (-3, -2, -1, 1, 2, 3):
        nb = neighbor(root_proba, sid, o)
        nbr = np.empty_like(nb)
        for r in range(12):
            mm = root_frame == r
            if mm.any():
                nbr[mm] = np.roll(nb[mm], -r, axis=1)
        feats.append(nbr)
    return np.concatenate(feats, 1)


def run_seed(nn24, f48a, f48rr, roots, quals, sid, seed):
    tr, va, te = song_split(sid, seed)
    cnt = np.bincount(quals, minlength=KQ)
    cw = (cnt.sum() / (KQ * np.maximum(cnt, 1))).astype(np.float32)
    out = {}

    # ---------- ROOT HEAD (same MLP arch, both features) ----------
    def root_head(X):
        m = train_clf(X[tr], roots[tr], X[va], roots[va], X.shape[1], 12,
                      hid=(128, 64), epochs=50)
        proba = predict_proba(m, X)              # full-corpus posteriors
        acc = float((proba[te].argmax(1) == roots[te]).mean())
        return acc, proba

    nn_bass, nn_treb = nn24[:, :12], nn24[:, 12:]
    acc_nn, proba_nn = root_head(nn24)
    acc_bp, proba_bp = root_head(f48a)
    out['root_nnls'] = acc_nn
    out['root_bp48'] = acc_bp

    # ---------- QUALITY HEAD ----------
    def qual_bal(X, name):
        m = train_clf(X[tr], quals[tr], X[va], quals[va], X.shape[1], KQ,
                      hid=(128, 64), epochs=60, cw=cw)
        pr = predict_proba(m, X[te]).argmax(1)
        rec = balanced_recall(pr, quals[te], KQ)
        bal = float(np.nanmean(rec))
        dom = float(rec[2])
        return bal, dom, m

    # NNLS full recipe: oracle-root rotation (+ trigram context from NNLS root head)
    br = rotate_by_root(nn_bass, roots); tr_ = rotate_by_root(nn_treb, roots)
    Xrr = np.concatenate([br, tr_], 1)
    ctx_or = ctx_feats(proba_nn, sid, roots)
    Xrr_ctx = np.concatenate([Xrr, ctx_or], 1)
    out['qual_nnls_rot'], out['dom_nnls_rot'], _ = qual_bal(Xrr, 'nnls_rot')
    out['qual_nnls_full'], out['dom_nnls_full'], qm_full = qual_bal(Xrr_ctx, 'nnls_full')

    # NNLS cascade: rotate by PREDICTED root (deployable), same recipe
    pr_root = proba_nn.argmax(1)
    Xc = np.concatenate([rotate_by_root(nn_bass, pr_root), rotate_by_root(nn_treb, pr_root),
                         ctx_feats(proba_nn, sid, pr_root)], 1)
    mc = train_clf(Xc[tr], quals[tr], Xc[va], quals[va], Xc.shape[1], KQ,
                   hid=(128, 64), epochs=60, cw=cw)
    prc = predict_proba(mc, Xc[te]).argmax(1); recc = balanced_recall(prc, quals[te], KQ)
    out['qual_nnls_cascade'] = float(np.nanmean(recc))
    out['dom_nnls_cascade'] = float(recc[2])

    # BP48 baseline quality: feat48 root-relative (established recipe, oracle root frame)
    out['qual_bp48'], out['dom_bp48'], _ = qual_bal(f48rr, 'bp48')

    return out, int(tr.sum()), int(te.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
    bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    assert len(nn["root"]) == len(bp["root"]), "row count mismatch"
    # align check: same roots/song_ids row-for-row
    assert (nn["root"].astype(int) == bp["root"].astype(int)).all(), "root misalignment"
    assert (nn["song_id"] == bp["song_id"]).all(), "song_id misalignment"

    nn24 = nn["nnls24"].astype(np.float32)
    filled = np.abs(nn24).sum(1) > 0
    keep = filled                                 # only rows with extracted NNLS
    nn24 = nn24[keep]
    f48a = bp["feat48_abs"].astype(np.float32)[keep]
    f48rr = bp["feat48"].astype(np.float32)[keep]     # root-relative BP48
    roots = bp["root"].astype(np.int64)[keep] % 12
    quals = bp["quality_idx"].astype(np.int64)[keep]
    sid = bp["song_id"][keep]

    nsong = len(np.unique(sid))
    print(f"rows={keep.sum()}/{len(filled)} filled, songs={nsong}", flush=True)
    print(f"quality dist: {dict(zip(QUALITIES, np.bincount(quals, minlength=KQ).tolist()))}", flush=True)

    runs = []
    for s in range(a.seeds):
        o, ntr, nte = run_seed(nn24, f48a, f48rr, roots, quals, sid, s)
        runs.append(o)
        print(f"[seed {s}] root NNLS={o['root_nnls']:.3f} BP48={o['root_bp48']:.3f} | "
              f"qualbal NNLS_full={o['qual_nnls_full']:.3f} NNLS_rot={o['qual_nnls_rot']:.3f} "
              f"NNLS_casc={o['qual_nnls_cascade']:.3f} BP48={o['qual_bp48']:.3f} | "
              f"dom NNLS_full={o['dom_nnls_full']:.3f} BP48={o['dom_bp48']:.3f} "
              f"(tr {ntr}/te {nte})", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs]); return float(v.mean()), float(v.std())

    print("\n" + "=" * 66)
    print(f"RWC NNLS-24 full recipe vs BP48 baseline, {a.seeds}-seed song-grouped CV")
    print("=" * 66)
    keys = [
        ("root_nnls", "Root acc  NNLS-24 (MLP)"),
        ("root_bp48", "Root acc  BP48 (MLP, same arch)"),
        ("qual_nnls_full", "Qual bal  NNLS full (rot+trigram, oracle frame)"),
        ("qual_nnls_rot", "Qual bal  NNLS rotation-only (oracle frame)"),
        ("qual_nnls_cascade", "Qual bal  NNLS cascade (predicted root)"),
        ("qual_bp48", "Qual bal  BP48 root-relative (baseline)"),
        ("dom_nnls_full", "Dom rec   NNLS full"),
        ("dom_bp48", "Dom rec   BP48"),
    ]
    summ = {}
    for k, lbl in keys:
        m, sd = ms(k); summ[k] = [m, sd]
        print(f"  {lbl:48s}: {m:.3f} +/- {sd:.3f}")
    json.dump({"seeds": a.seeds, "n_rows": int(keep.sum()), "n_songs": int(nsong),
               "per_seed": runs, "summary": summ},
              open(REPO / "scratchpad/rwc_nnls_cv_result.json", "w"), indent=2)
    print("\nsaved scratchpad/rwc_nnls_cv_result.json")


if __name__ == "__main__":
    main()
