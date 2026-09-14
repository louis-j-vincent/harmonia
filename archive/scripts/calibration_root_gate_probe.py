"""calibration_* probe: is the RWC BP48 root model calibrated, and does a
calibration-gated neighbour-context fallback help ONLY on the low-confidence
subset (vs the already-dead universal context finding)?

Read-only w.r.t. corpus/checkpoint. Writes nothing but stdout.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

CKPT = REPO / "data/models/_eval_only_rwc_bp48_fixed_root_2026_07_16.pt"
CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"
rng = np.random.default_rng(0)


def make_mlp(in_dim, n):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(64, n))


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(conf)
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if i == 0:
            m |= conf == 0
        if m.sum() == 0:
            continue
        acc = correct[m].mean(); cf = conf[m].mean()
        e += (m.sum() / N) * abs(acc - cf)
        rows.append((bins[i], bins[i + 1], m.sum(), cf, acc))
    return e, rows


def main():
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = np.load(CORPUS, allow_pickle=True)
    model = make_mlp(48, 12)
    model.load_state_dict(d["root_model_state"]); model.eval()
    mean = d["root_mean"]; std = d["root_std"]
    X_abs = c["feat48_abs"].astype(np.float32)
    root = c["root"].astype(int)
    sid = c["song_id"]; t0 = c["t0"]; t1 = c["t1"]

    test_songs = list(d["test_songs"])
    # split held-out test songs into CALIB (fit T + pick threshold) vs EVAL
    ts = sorted(test_songs)
    rng.shuffle(ts)
    calib_songs = set(ts[:10]); eval_songs = set(ts[10:])
    train_mask = ~np.isin(sid, test_songs)

    def logits_for(mask):
        Xn = ((X_abs[mask] - mean) / std).astype(np.float32)
        with torch.no_grad():
            return model(torch.tensor(Xn)).numpy()

    def softmax(z, T=1.0):
        z = z / T; z = z - z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)

    cm = np.isin(sid, list(calib_songs)); em = np.isin(sid, list(eval_songs))
    zc, zc_y = logits_for(cm), root[cm]
    ze, ze_y = logits_for(em), root[em]

    # ---- Part 1: calibration (T=1) on EVAL ----
    p1 = softmax(ze, 1.0)
    conf1 = p1.max(1); pred1 = p1.argmax(1); corr1 = (pred1 == ze_y).astype(float)
    e1, rows1 = ece(conf1, corr1)
    print(f"=== Part 1: calibration on held-out EVAL ({em.sum()} chords, {len(eval_songs)} songs) ===")
    print(f"accuracy={corr1.mean():.4f}  mean_conf={conf1.mean():.4f}  ECE(T=1)={e1:.4f}")
    print("  bin_lo bin_hi   n   conf   acc")
    for lo, hi, n, cf, ac in rows1:
        print(f"  {lo:.2f}  {hi:.2f}  {n:4d}  {cf:.3f}  {ac:.3f}")

    # ---- fit temperature on CALIB (NLL) ----
    T = torch.tensor(1.0, requires_grad=True)
    zt = torch.tensor(zc); yt = torch.tensor(zc_y, dtype=torch.long)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    lf = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(); loss = lf(zt / T, yt); loss.backward(); return loss
    opt.step(closure)
    Topt = float(T.detach())
    p2 = softmax(ze, Topt)
    conf2 = p2.max(1); corr2 = (p2.argmax(1) == ze_y).astype(float)
    e2, _ = ece(conf2, corr2)
    print(f"\nfitted temperature T={Topt:.3f} (>1 => was overconfident)")
    print(f"ECE after temp-scaling on EVAL = {e2:.4f}  (was {e1:.4f})  "
          f"mean_conf={conf2.mean():.4f}  acc unchanged={corr2.mean():.4f}")

    # ---- Part 2: low-confidence subset characterization (on EVAL, calibrated) ----
    thr = np.quantile(conf2, 0.25)  # bottom 25% by calibrated confidence
    lo = conf2 <= thr; hi = conf2 > thr
    dur = (t1[em] - t0[em])
    note = X_abs[em][:, 12:24]; bass = X_abs[em][:, 24:36]

    def muddy(block):
        b = np.clip(block, 1e-9, None)
        pk = block.max(1) / (block.mean(1) + 1e-9)
        p = b / b.sum(1, keepdims=True)
        ent = -(p * np.log(p)).sum(1)
        return pk, ent
    pk_n, ent_n = muddy(note); pk_b, ent_b = muddy(bass)
    # P4/P5 error: wrong prediction whose interval to true root is 5 or 7 semitones
    err = p2.argmax(1) != ze_y
    interval = (p2.argmax(1) - ze_y) % 12
    p4p5 = err & np.isin(interval, [5, 7])

    print(f"\n=== Part 2: low-conf subset (bottom 25%, conf<= {thr:.3f}, n={lo.sum()}) vs high-conf (n={hi.sum()}) ===")
    print(f"  acc:        low={corr2[lo].mean():.3f}  high={corr2[hi].mean():.3f}")
    print(f"  clip_dur s: low={dur[lo].mean():.2f}  high={dur[hi].mean():.2f}")
    print(f"  note peak/mean: low={pk_n[lo].mean():.2f}  high={pk_n[hi].mean():.2f}  (lower=muddier)")
    print(f"  note entropy:   low={ent_n[lo].mean():.2f}  high={ent_n[hi].mean():.2f}  (higher=muddier)")
    print(f"  bass peak/mean: low={pk_b[lo].mean():.2f}  high={pk_b[hi].mean():.2f}")
    print(f"  P4/P5-error rate among errors: low={p4p5[lo].sum()/max(err[lo].sum(),1):.3f} "
          f"high={p4p5[hi].sum()/max(err[hi].sum(),1):.3f}")
    print(f"  frac of all P4/P5 errors captured by low-conf subset: {p4p5[lo].sum()/max(p4p5.sum(),1):.3f}")

    # ---- Part 2b: gated neighbour-context test ----
    # bigram transition from TRAINING songs true roots (no leakage)
    Tm = np.ones((12, 12))  # laplace
    for s in np.unique(sid[train_mask]):
        r = root[(sid == s)]
        for a, b in zip(r[:-1], r[1:]):
            Tm[a, b] += 1
    Tm /= Tm.sum(1, keepdims=True)

    # build per-chord neighbour context on EVAL using PREDICTED neighbour posteriors
    idx_e = np.where(em)[0]
    p_all = softmax(logits_for(em), Topt)  # eval posteriors, aligned to em order
    sid_e = sid[em]
    ctx = np.zeros((len(idx_e), 12))
    for i in range(len(idx_e)):
        s = sid_e[i]
        prev = p_all[i - 1] if i > 0 and sid_e[i - 1] == s else None
        nxt = p_all[i + 1] if i + 1 < len(idx_e) and sid_e[i + 1] == s else None
        cc = np.zeros(12)
        if prev is not None:
            cc += prev @ Tm            # P(cur | prev)
        if nxt is not None:
            cc += nxt @ Tm.T           # P(cur | next) reversed
        cc = cc / cc.sum() if cc.sum() > 0 else np.ones(12) / 12
        ctx[i] = cc

    # cheap premise: is true root recoverable from neighbours at all (low-conf)?
    prev_pred = np.array([p_all[i-1].argmax() if i>0 and sid_e[i-1]==sid_e[i] else -1 for i in range(len(idx_e))])
    next_pred = np.array([p_all[i+1].argmax() if i+1<len(idx_e) and sid_e[i+1]==sid_e[i] else -1 for i in range(len(idx_e))])
    neigh_has = (prev_pred == ze_y) | (next_pred == ze_y)
    print(f"\n=== Part 2b: gated context (transition from TRAIN true roots) ===")
    print(f"  premise: true root == a neighbour's predicted root:  low={neigh_has[lo].mean():.3f}  high={neigh_has[hi].mean():.3f}")

    logp = np.log(p2 + 1e-9); logc = np.log(ctx + 1e-9)
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        comb = (logp + lam * logc).argmax(1)
        acc_lo = (comb[lo] == ze_y[lo]).mean()
        acc_hi = (comb[hi] == ze_y[hi]).mean()
        acc_all = (comb == ze_y).mean()
        print(f"  lambda={lam:<4} acc_low={acc_lo:.3f}  acc_high={acc_hi:.3f}  acc_all={acc_all:.3f}")
    print(f"  (root-only baseline acc_low={corr2[lo].mean():.3f}  acc_high={corr2[hi].mean():.3f})")


if __name__ == "__main__":
    main()
