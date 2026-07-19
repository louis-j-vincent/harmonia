"""Oracle ceiling for interval-aware root context.

Bounds the MAXIMUM accuracy an interval-aware neighbour model could reach if
the neighbour roots were known perfectly (GT).  Purely in absolute-PC / offset
probability space -- no chroma rotation, no shift-back involved.

For each chord we form a context-only prediction from the LEARNED absolute
transition matrix Tm and the neighbours' *GT* roots:
    ctx(cur) ∝ Tm[prev_gt, cur] * Tm_rev[next_gt, cur]
and compare argmax(ctx) to the true root.  This is the pure voice-leading
predictor with a perfect neighbour oracle -- the ceiling for context.

Also reports emission+context oracle (does perfect-neighbour context lift the
real emission posterior), on all + low-conf.  Uses the existing checkpoint's
posteriors and its own train/test split for Tm (no leakage: Tm from train).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match

CKPT = REPO / "data/models/_eval_only_rwc_bp48_fixed_root_2026_07_16.pt"
CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"


def make_mlp(in_dim, n):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(64, n))


def main():
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = np.load(CORPUS, allow_pickle=True)
    keep = filter_by_match(c["match"], minimum=MatchQuality.EXACT)
    X = c["feat48_abs"].astype(np.float32)[keep]
    root = c["root"].astype(int)[keep]
    sid = c["song_id"][keep]; t0 = c["t0"][keep]
    test_songs = set(d["test_songs"])
    te = np.isin(sid, list(test_songs)); tr = ~te

    model = make_mlp(48, 12); model.load_state_dict(d["root_model_state"]); model.eval()
    mean = d["root_mean"]; std = d["root_std"]
    with torch.no_grad():
        logits = model(torch.tensor(((X - mean) / std).astype(np.float32))).numpy()
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); post = e / e.sum(1, keepdims=True)
    conf = post.max(1)

    # Tm from TRAIN true roots
    Tm = np.ones((12, 12))
    for s in sorted(set(sid[tr].tolist())):
        idx = np.where(sid == s)[0]; idx = idx[np.argsort(t0[idx])]
        r = root[idx]
        for a, b in zip(r[:-1], r[1:]): Tm[a, b] += 1
    Tm /= Tm.sum(1, keepdims=True)

    idxs = np.where(te)[0]
    root_te = root[te]; conf_te = conf[te]
    thr = np.quantile(conf_te, 0.25); lc = conf_te <= thr

    ctx_only = np.zeros((te.sum(), 12))
    ctx_emit = np.zeros((te.sum(), 12))
    post_te = post[te]
    sid_te = sid[te]; t0_te = t0[te]
    pos = 0
    # map to per-song ordering
    lc_full = np.zeros(te.sum(), bool)
    lcount = 0
    order_local = np.empty(te.sum(), int)
    ptr = 0
    for s in sorted(set(sid_te.tolist())):
        loc = np.where(sid_te == s)[0]
        loc = loc[np.argsort(t0_te[loc])]
        r = root_te[loc]
        for k, li in enumerate(loc):
            cc = np.zeros(12)
            if k > 0:  cc += np.log(Tm[r[k-1]] + 1e-9)          # P(cur | prev_gt)
            if k < len(loc)-1: cc += np.log(Tm[:, r[k+1]] + 1e-9)  # P(cur | next_gt) reversed
            ctx_only[li] = cc
            ctx_emit[li] = np.log(post_te[li] + 1e-9) + cc

    def acc(pred, m=None):
        if m is None: return float((pred == root_te).mean())
        return float((pred[m] == root_te[m]).mean())

    p_ctx = ctx_only.argmax(1)
    p_emit = post_te.argmax(1)
    p_both = ctx_emit.argmax(1)
    print("=== ORACLE CEILING (perfect GT neighbour roots, learned Tm) ===")
    print(f"  n_test={te.sum()}  n_lowconf={lc.sum()}")
    print(f"  emission only (S0)      : all={acc(p_emit):.4f}  lowconf={acc(p_emit,lc):.4f}")
    print(f"  CONTEXT-ONLY (GT neigh) : all={acc(p_ctx):.4f}  lowconf={acc(p_ctx,lc):.4f}")
    print(f"  emission+GTcontext      : all={acc(p_both):.4f}  lowconf={acc(p_both,lc):.4f}")
    # how often is the true root even in the top-1 of context-only?
    print(f"\n  context-only recovers true root: all={acc(p_ctx):.3f} lowconf={acc(p_ctx,lc):.3f}")
    print(f"  (chance=0.083; corpus most-common single root would be ~0.10)")


if __name__ == "__main__":
    main()
