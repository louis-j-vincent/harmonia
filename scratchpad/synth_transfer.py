"""Synthetic-data transfer + augmentation harness.

Same small MLP for all conditions. Two single-segment targets:
  quality: feat48 (root-relative) -> 7-way
  root:    feat48_abs            -> 12-way
Conditions, all evaluated on held-out REAL RWC songs:
  A synth-only  -> real
  B real-only   -> real   (baseline)
  C real+synth  -> real   (augmentation)
Song-level splits on RWC (80/20), multiple seeds.
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import load_corpus

REAL = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"
SYNTH = REPO / "data/cache/synth/synth_bp48.npz"


def make_mlp(nin, nout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(nin, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(64, nout))


def train_eval(Xtr, ytr, Xte, yte, nout, *, epochs=60, lr=2e-3, batch=256, device="cpu"):
    import torch, torch.nn as nn
    mean = Xtr.mean(0).astype(np.float32); std = (Xtr.std(0) + 1e-9).astype(np.float32)
    Xtn = ((Xtr - mean) / std).astype(np.float32)
    Xen = ((Xte - mean) / std).astype(np.float32)
    Xt = torch.tensor(Xtn, device=device); yt = torch.tensor(ytr, dtype=torch.long, device=device)
    counts = np.bincount(ytr, minlength=nout).astype(float)
    w = 1.0 / (counts + 1.0); w /= w.sum(); w *= nout
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    model = make_mlp(Xtr.shape[1], nout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=wt)
    n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xen, device=device)).argmax(1).cpu().numpy()
    acc = (pred == yte).mean()
    # macro (balanced) accuracy across present classes
    accs = [ (pred[yte == c] == c).mean() for c in np.unique(yte) ]
    return float(acc), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["quality", "root"], default="quality")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--synth", default=str(SYNTH))
    args = ap.parse_args()

    real = load_corpus(REAL); synth = load_corpus(args.synth)
    if args.target == "quality":
        fkey, ykey, nout = "feat48", "quality_idx", 7
    else:
        fkey, ykey, nout = "feat48_abs", "root", 12
    Xr, yr, sid = real[fkey], real[ykey].astype(int), real["song_id"]
    Xs, ys = synth[fkey], synth[ykey].astype(int)

    songs = sorted(set(sid.tolist()))
    res = {"A_synth_only": [], "B_real_only": [], "C_augment": [],
           "A_bal": [], "B_bal": [], "C_bal": []}
    for s in range(args.seeds):
        rng = np.random.RandomState(s); sh = list(songs); rng.shuffle(sh)
        n_test = max(1, int(round(0.2 * len(sh)))); test = set(sh[:n_test])
        te = np.array([x in test for x in sid]); tr = ~te
        Xr_tr, yr_tr = Xr[tr], yr[tr]; Xr_te, yr_te = Xr[te], yr[te]

        a = train_eval(Xs, ys, Xr_te, yr_te, nout)
        b = train_eval(Xr_tr, yr_tr, Xr_te, yr_te, nout)
        Xc = np.vstack([Xr_tr, Xs]); yc = np.concatenate([yr_tr, ys])
        c = train_eval(Xc, yc, Xr_te, yr_te, nout)
        res["A_synth_only"].append(a[0]); res["A_bal"].append(a[1])
        res["B_real_only"].append(b[0]); res["B_bal"].append(b[1])
        res["C_augment"].append(c[0]); res["C_bal"].append(c[1])
        print(f"seed {s}: A(synth->real)={a[0]:.3f}/{a[1]:.3f}  "
              f"B(real->real)={b[0]:.3f}/{b[1]:.3f}  "
              f"C(aug->real)={c[0]:.3f}/{c[1]:.3f}  [acc/balanced]", flush=True)

    print(f"\n=== TARGET={args.target}  (mean +- std over {args.seeds} seeds) ===")
    for k in ["A_synth_only", "B_real_only", "C_augment"]:
        v = np.array(res[k]); bal = np.array(res[k.split('_')[0] + "_bal"])
        print(f"  {k:14s} acc={v.mean():.4f}+-{v.std():.4f}  balanced={bal.mean():.4f}+-{bal.std():.4f}")
    out = REPO / "scratchpad" / f"synth_transfer_{args.target}.json"
    out.write_text(json.dumps({k: list(map(float, v)) for k, v in res.items()}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
