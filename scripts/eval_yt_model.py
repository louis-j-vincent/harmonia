"""eval_yt_model.py — comprehensive ablation to determine yt corpus training bottleneck.

Experiments:
  1. Learning curve: accuracy vs corpus size (10 / 25 / 50 songs)
  2. Architecture: 128-64 vs 256-128 vs 64-32
  3. Features: 48-dim BP / 12-dim CQT / 60-dim BP+CQT / raw (un-normed) / with context
  4. Classes: 3-class (maj/min/dom) vs 5-class (+hdim,dim) vs 7-class

All evaluated with same song-level val split (seed=42, 15% val songs).
"""

from __future__ import annotations
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
from harmonia.data.yt_chord_corpus import load_corpus, QUALITIES

# ── helpers ──────────────────────────────────────────────────────────────────

def make_split(song_ids, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(song_ids.tolist()))
    n_val = max(1, int(len(unique) * val_frac))
    val_songs = set(rng.choice(unique, n_val, replace=False).tolist())
    val_mask = np.array([s in val_songs for s in song_ids.tolist()], dtype=bool)
    return val_mask, val_songs


def add_context(X, song_ids, window=1):
    N, D = X.shape; W = 2 * window + 1
    out = np.zeros((N, D * W), dtype=X.dtype)
    unique = list(dict.fromkeys(song_ids.tolist()))
    idx_map = {s: [] for s in unique}
    for i, s in enumerate(song_ids.tolist()): idx_map[s].append(i)
    for s, idxs in idx_map.items():
        idxs = np.array(idxs); n_s = len(idxs)
        for off in range(-window, window + 1):
            col = (off + window) * D
            src = np.clip(np.arange(n_s) + off, 0, n_s - 1)
            out[idxs, col:col + D] = X[idxs[src]]
    return out


def make_mlp(in_dim, n_classes, h1=128, h2=64):
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h1, h2),    nn.LayerNorm(h2),  nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h2, n_classes),
    )


def train_eval(X_tr, y_tr, X_val, y_val, n_cls, epochs=200, h1=128, h2=64):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mean = X_tr.mean(0).astype(np.float32); std = (X_tr.std(0) + 1e-9).astype(np.float32)
    Xtr = ((X_tr - mean) / std).astype(np.float32)
    Xvl = ((X_val - mean) / std).astype(np.float32)
    counts = np.bincount(y_tr, minlength=n_cls).astype(float)
    w = 1.0 / (counts + 1); w = w / w.sum() * n_cls
    model = make_mlp(X_tr.shape[1], n_cls, h1, h2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    n = len(Xt)
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xvl, device=device)).argmax(1).cpu().numpy()
    tr_pred = model(torch.tensor(Xtr, device=device)).argmax(1).cpu().numpy()
    return {
        "val_acc": float((pred == y_val).mean()),
        "train_acc": float((tr_pred == y_tr).mean()),
        "per_class": {QUALITIES[i]: float((pred[y_val == i] == i).mean())
                      for i in np.unique(y_val) if (y_val == i).sum() > 0},
    }


# ── load corpus ───────────────────────────────────────────────────────────────

print("Loading corpus_50.npz ...")
d = load_corpus(REPO / "data/cache/yt_corpus/corpus_50.npz")

keep = np.isin(d["match"], ["exact", "family"])
X48  = d["feat48"][keep].astype(np.float32)
X_cqt= d["feat12_cqt"][keep].astype(np.float32)
Xabs = d["feat48_abs"][keep].astype(np.float32)
y7   = d["quality_idx"][keep].astype(int)
sids = d["song_id"][keep]
unique_songs = list(dict.fromkeys(sids.tolist()))
print(f"Clean records: {keep.sum()} / {len(keep)}, songs: {len(unique_songs)}")
print()

val_mask, val_songs = make_split(sids)
tr = ~val_mask
print(f"Val songs ({len(val_songs)}): {val_songs}")
print(f"Train: {tr.sum()}, Val: {val_mask.sum()}")
print()

# ── feature sets ─────────────────────────────────────────────────────────────

# Raw un-normalised CQT (librosa returns it 0-1 already, but don't L2-norm)
# We don't have raw CQT stored, just normalized. Let's try different combos.

feats = {
    "BP 48d":          (X48,),
    "CQT 12d":         (X_cqt,),
    "BP+CQT 60d":      (np.concatenate([X48, X_cqt], 1),),
    "BP+CQT+ABS 108d": (np.concatenate([X48, X_cqt, Xabs], 1),),
}

# ── class schemes ─────────────────────────────────────────────────────────────
# 3-class: maj(0)/min(1)/dom(2) — filter out hdim/dim/aug/sus
# 5-class: maj/min/dom/hdim→3/dim→4 — remap hdim+aug+sus to hdim, dim+aug to dim
# 7-class: all

def remap5(y7):
    y5 = y7.copy()
    y5[y7 == 3] = 3   # hdim stays 3
    y5[y7 == 4] = 4   # dim stays 4
    y5[y7 == 5] = 3   # aug → hdim-ish (rare)
    y5[y7 == 6] = 2   # sus → dom-ish (rare)
    return y5, y7 <= 4  # filter aug/sus

classes = {
    "3-class": (lambda y: (y, y <= 2), ["maj", "min", "dom"]),
    "5-class": (lambda y: (remap5(y)[0], remap5(y)[1]), ["maj", "min", "dom", "hdim", "dim"]),
    "7-class": (lambda y: (y, np.ones(len(y), bool)), QUALITIES),
}

# ── experiment 1: LEARNING CURVE (3-class, BP+CQT 60d) ──────────────────────

print("=" * 70)
print("EXPERIMENT 1: Learning curve (3-class, BP+CQT 60d, fixed val)")
print("=" * 70)
X60 = np.concatenate([X48, X_cqt], 1)

# Subsample training songs
all_tr_songs = [s for s in unique_songs if s not in val_songs]
for n_songs in [10, 25, 50]:
    subset_songs = set(all_tr_songs[:n_songs - len(val_songs)])  # exclude val
    sub_tr = np.array([s in subset_songs for s in sids.tolist()]) & tr
    # filter to 3-class
    keep3_sub = sub_tr & (y7 <= 2)
    keep3_val = val_mask & (y7 <= 2)
    y3_tr  = y7[keep3_sub]; y3_val = y7[keep3_val]
    X3_tr  = X60[keep3_sub]; X3_val = X60[keep3_val]
    res = train_eval(X3_tr, y3_tr, X3_val, y3_val, n_cls=3, epochs=150)
    n_tr_songs = len(subset_songs)
    print(f"  n_train_songs={n_tr_songs:2d}  train={res['train_acc']:.3f}  val={res['val_acc']:.3f}  "
          f"maj={res['per_class'].get('maj',0):.2f}  min={res['per_class'].get('min',0):.2f}  "
          f"dom={res['per_class'].get('dom',0):.2f}")
print()

# ── experiment 2: ARCHITECTURE (3-class, BP+CQT 60d, all 50 songs) ──────────

print("=" * 70)
print("EXPERIMENT 2: Architecture sweep (3-class, BP+CQT 60d)")
print("=" * 70)
keep3 = y7 <= 2
X3_tr = X60[tr & keep3]; y3_tr = y7[tr & keep3]
X3_val= X60[val_mask & keep3]; y3_val = y7[val_mask & keep3]

for label, (h1, h2) in [("tiny 32-16", (32, 16)), ("small 64-32", (64, 32)),
                          ("std 128-64", (128, 64)), ("wide 256-128", (256, 128))]:
    res = train_eval(X3_tr, y3_tr, X3_val, y3_val, n_cls=3, h1=h1, h2=h2, epochs=150)
    print(f"  {label:15s}  train={res['train_acc']:.3f}  val={res['val_acc']:.3f}  "
          f"maj={res['per_class'].get('maj',0):.2f}  min={res['per_class'].get('min',0):.2f}  "
          f"dom={res['per_class'].get('dom',0):.2f}")
print()

# ── experiment 3: FEATURE SET (3-class, std 128-64) ─────────────────────────

print("=" * 70)
print("EXPERIMENT 3: Feature set (3-class, std 128-64)")
print("=" * 70)

for fname, (Xf,) in feats.items():
    X3_tr_f = Xf[tr & keep3]; X3_val_f = Xf[val_mask & keep3]
    res = train_eval(X3_tr_f, y3_tr, X3_val_f, y3_val, n_cls=3, epochs=150)
    print(f"  {fname:20s}  train={res['train_acc']:.3f}  val={res['val_acc']:.3f}  "
          f"maj={res['per_class'].get('maj',0):.2f}  min={res['per_class'].get('min',0):.2f}  "
          f"dom={res['per_class'].get('dom',0):.2f}")

# Context window
X60ctx = add_context(X60, sids, 1)
X3_tr_ctx = X60ctx[tr & keep3]; X3_val_ctx = X60ctx[val_mask & keep3]
res = train_eval(X3_tr_ctx, y3_tr, X3_val_ctx, y3_val, n_cls=3, epochs=150)
print(f"  {'BP+CQT ctx=1 180d':20s}  train={res['train_acc']:.3f}  val={res['val_acc']:.3f}  "
      f"maj={res['per_class'].get('maj',0):.2f}  min={res['per_class'].get('min',0):.2f}  "
      f"dom={res['per_class'].get('dom',0):.2f}")
print()

# ── experiment 4: CLASS SCHEME (BP+CQT 60d, std 128-64) ─────────────────────

print("=" * 70)
print("EXPERIMENT 4: Class scheme (BP+CQT 60d, std 128-64)")
print("=" * 70)

for cname, (remap_fn, cnames) in classes.items():
    y_r, cmask = remap_fn(y7)
    X_tr_c = X60[tr & cmask]; y_tr_c = y_r[tr & cmask]
    X_val_c= X60[val_mask & cmask]; y_val_c= y_r[val_mask & cmask]
    n_cls = len(cnames)
    res = train_eval(X_tr_c, y_tr_c, X_val_c, y_val_c, n_cls=n_cls, epochs=150)
    print(f"  {cname:10s}  train={res['train_acc']:.3f}  val={res['val_acc']:.3f}  ", end="")
    print("  ".join(f"{n}={res['per_class'].get(n, float('nan')):.2f}" for n in cnames))
print()

print("Done.")
