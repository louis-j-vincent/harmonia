"""STEP 1 — Trained bass-pc classifier on NNLS-24 features (RWC).

The highest-EV untried lever: every NNLS-bass number to date (0.797 all / 0.770 inv)
is the UNTRAINED bass-half argmax. This trains an MLP head (same arch as the verified
root/quality heads in multihead_training.py) on NNLS-24 to predict the SOUNDING BASS
pitch class, and compares it to the untrained argmax on the IDENTICAL pooled test rows.

Target: harmonia.data.corpus_schema.sounding_bass_pc(label, root)  (the project target).
Baseline: nn24[:,:12].argmax(1)  (untrained NNLS bass-half argmax).
Protocol: >=5-seed song-grouped 80/10/10 CV, predictions pooled over TEST rows.

Feature variants trained (all vs the same argmax baseline, same rows):
  - full24        : nn24 (bass[:12] + treble[12:]) absolute chroma
  - bass12        : nn24[:,:12] only
  - full24+argmax : full24 + one-hot(bass argmax) as an explicit prior
Every number printed comes from a completed run.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad")); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
    bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    assert (nn["root"].astype(int) == bp["root"].astype(int)).all()
    assert (nn["song_id"] == bp["song_id"]).all()

    nn24 = nn["nnls24"].astype(np.float32)
    keep = np.abs(nn24).sum(1) > 0
    nn24 = nn24[keep]
    roots = bp["root"].astype(np.int64)[keep] % 12
    sid = bp["song_id"][keep]
    labels = bp["labels"][keep]

    # GT sounding-bass pc via the project resolver; drop rows with no bass (N/X)
    gb = np.full(len(roots), -1, np.int64)
    for i in range(len(roots)):
        v = sounding_bass_pc(str(labels[i]), int(roots[i]))
        if v is not None:
            gb[i] = v % 12
    valid = gb >= 0
    inv_all = (gb != roots) & valid   # inversion = sounding bass != functional root

    print(f"rows={len(roots)} valid-bass={valid.sum()} songs={len(np.unique(sid))} "
          f"inversions={int(inv_all.sum())} ({inv_all.mean():.3%})", flush=True)

    bass_arg = nn24[:, :12].argmax(1)   # untrained baseline, per-row

    # feature builders
    def feats(kind):
        if kind == "full24":  return nn24
        if kind == "bass12":  return nn24[:, :12]
        if kind == "full24+argmax":
            oh = np.eye(12, dtype=np.float32)[bass_arg]
            return np.concatenate([nn24, oh], 1)
        raise ValueError(kind)

    variants = ["full24", "bass12", "full24+argmax"]
    pooled = {v: [] for v in variants}
    pooled_base, pooled_gt, pooled_inv = [], [], []

    for seed in range(a.seeds):
        tr, va, te = song_split(sid, seed)
        trv = tr & valid; vav = va & valid; tev = te & valid
        for v in variants:
            X = feats(v)
            m = train_clf(X[trv], gb[trv], X[vav], gb[vav], X.shape[1], 12,
                          hid=(128, 64), epochs=50)
            pred = predict_proba(m, X[tev]).argmax(1)
            pooled[v].append(pred)
        pooled_base.append(bass_arg[tev])
        pooled_gt.append(gb[tev])
        pooled_inv.append(inv_all[tev])
        # quick per-seed line (full24)
        gtt = gb[tev]; invt = inv_all[tev]
        p = pooled["full24"][-1]
        print(f"[seed {seed}] n={tev.sum()} full24 all={np.mean(p==gtt):.3f} "
              f"inv={np.mean(p[invt]==gtt[invt]) if invt.sum() else float('nan'):.3f} "
              f"| argmax all={np.mean(bass_arg[tev]==gtt):.3f} "
              f"inv={np.mean(bass_arg[tev][invt]==gtt[invt]) if invt.sum() else float('nan'):.3f}", flush=True)

    GT = np.concatenate(pooled_gt); INV = np.concatenate(pooled_inv)
    BASE = np.concatenate(pooled_base)
    n = len(GT)
    def acc(pred, mask=None):
        mask = np.ones(n, bool) if mask is None else mask
        return float((pred[mask] == GT[mask]).mean()) if mask.sum() else float('nan')

    print("\n" + "=" * 66)
    print(f"BASS-PC head on NNLS-24, {a.seeds}-seed song-grouped CV, pooled {n} test chords "
          f"({int(INV.sum())} inversions)")
    print("=" * 66)
    res = {}
    print(f"  {'UNTRAINED NNLS bass-argmax (baseline)':40s}  all={acc(BASE):.3f}  inv={acc(BASE, INV):.3f}  rootpos={acc(BASE, ~INV):.3f}")
    res["untrained_argmax"] = [acc(BASE), acc(BASE, INV), acc(BASE, ~INV)]
    for v in variants:
        P = np.concatenate(pooled[v])
        print(f"  {('TRAINED head [' + v + ']'):40s}  all={acc(P):.3f}  inv={acc(P, INV):.3f}  rootpos={acc(P, ~INV):.3f}")
        res["trained_" + v] = [acc(P), acc(P, INV), acc(P, ~INV)]

    json.dump({"seeds": a.seeds, "n_test": n, "n_inv": int(INV.sum()), "results": res},
              open(REPO / "scratchpad/bass_nnls_head_result.json", "w"), indent=2)
    print("\nsaved scratchpad/bass_nnls_head_result.json")


if __name__ == "__main__":
    main()
