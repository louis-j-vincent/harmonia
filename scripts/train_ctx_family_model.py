"""Train and save the context-window entropy-gated family classifier.

Trains the 87.5%-CV MLP + entropy gate from experiment_ctx_model.py on the
FULL dataset (no CV fold held out) and saves weights to:
    harmonia/models/ctx_family_model.npz

The saved model is loaded by chord_pipeline_v1._CtxFamilyClassifier.

Usage:
    .venv/bin/python scripts/train_ctx_family_model.py
    .venv/bin/python scripts/train_ctx_family_model.py --n-songs 60 --epochs 80

Notes:
  - Requires: data/cache/ltas_family_dist.npz (run plot_family_likelihood --rebuild-cache)
  - Requires: data/accomp_db/ (the iReal/MMA corpus with hard-audio renders)
  - Runtime: ~20-40 min depending on n-songs (renders + Basic Pitch per song)
  - Output: harmonia/models/ctx_family_model.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# reuse collect() + helpers from experiment_ctx_model
from experiment_ctx_model import (
    CTX_K,
    FAMILIES,
    DIST_CACHE,
    _ctx_tensor,
    _cv_logreg_logits,
    _cv_torch_logits,
    _fit_entropy_gate,
    _apply_entropy_gate,
    _make_mlp,
    collect,
)


def train_and_save(n_songs: int, seed: int, epochs: int, out_path: Path) -> None:
    if not DIST_CACHE.exists():
        print(f"ERROR: {DIST_CACHE} missing — run: "
              ".venv/bin/python scripts/plot_family_likelihood.py --rebuild-cache")
        sys.exit(1)

    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(seed)
    print(f"Collecting data ({n_songs} songs, hard audio, oracle boundaries)...")
    records = collect(n_songs, dist, rng)
    if not records:
        print("ERROR: no records collected — check accomp_db and audio manifests.")
        sys.exit(1)
    print(f"  {len(records)} segments collected")

    y = np.array([r["y"] for r in records])

    # feature matrices
    X_chroma  = np.stack([r["chroma_mean"] for r in records])           # (N, 12)
    X_base_ll = np.stack([r["ll_mat"].max(axis=1) for r in records])    # (N, 5)
    X_logreg  = np.concatenate([X_chroma, X_base_ll], axis=1)           # (N, 17)

    ctx = _ctx_tensor(records, CTX_K)                                    # (N, 9, 5, 12)
    ctx_flat = ctx.reshape(len(records), -1)                             # (N, 540)
    X_ctx = np.concatenate([X_chroma, ctx_flat], axis=1)                # (N, 552)

    # ── Step 1: CV to get out-of-fold logits for entropy-gate fitting ─────────
    print("\nStep 1: 5-fold CV for base logits...")
    mu_base, _, _, oof_base = _cv_logreg_logits(X_logreg, y)
    print(f"  baseline family accuracy: {mu_base:.1%}")

    print("\nStep 2: 5-fold CV for MLP ctx logits...")
    mu_mlp, _, _, oof_mlp = _cv_torch_logits(_make_mlp, X_ctx, y, epochs=epochs)
    print(f"  MLP ctx accuracy: {mu_mlp:.1%}")

    # ── Step 2: Fit entropy gate on OOF logits ────────────────────────────────
    print("\nStep 3: Fitting entropy gate on OOF logits...")
    import torch
    w, b = _fit_entropy_gate(oof_base, oof_mlp, y)
    gate_preds = _apply_entropy_gate(oof_base, oof_mlp, w, b)
    gate_acc = float((gate_preds == y).mean())
    print(f"  gate accuracy: {gate_acc:.1%}  (w={w:.3f}, b={b:.3f})")

    # ── Step 3: Retrain on FULL data (no held-out fold) ───────────────────────
    print("\nStep 4: Retraining on full data...")

    # base LR on full data
    sc_base = StandardScaler().fit(X_logreg)
    clf_base = LogisticRegression(max_iter=2000, solver="lbfgs",
                                  class_weight="balanced", C=1.0)
    clf_base.fit(sc_base.transform(X_logreg), y)

    # MLP on full data
    sc_ctx = StandardScaler()
    X_ctx_sc = sc_ctx.fit_transform(X_ctx)
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    HIDDEN1, HIDDEN2 = 256, 128
    flat_dim = X_ctx.shape[1]

    mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_dim, HIDDEN1), nn.LayerNorm(HIDDEN1), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(HIDDEN1, HIDDEN2), nn.LayerNorm(HIDDEN2), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(HIDDEN2, 5),
    )
    counts = np.bincount(y, minlength=5).astype(float)
    wts = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32)
    wts = wts / wts.sum() * 5
    opt = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=wts)

    Xt = torch.tensor(X_ctx_sc, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    mlp.train()
    for ep in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(mlp(xb), yb).backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{epochs}")
    mlp.eval()

    # verify full-data MLP accuracy (train acc, informational only)
    with torch.no_grad():
        train_pred = mlp(Xt).argmax(1).numpy()
    print(f"  MLP train acc (overfit upper bound): {(train_pred == y).mean():.1%}")

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    print(f"\nSaving to {out_path}...")

    # serialize MLP state dict as object array (np.save compatible)
    mlp_state_np = {k: v.numpy() for k, v in mlp.state_dict().items()}

    save_dict: dict = {
        "gate_w": np.array(w),
        "gate_b": np.array(b),
        "flat_dim": np.array(flat_dim),
        "hidden1": np.array(HIDDEN1),
        "hidden2": np.array(HIDDEN2),
        "sc_mean": sc_ctx.mean_.astype(np.float32),
        "sc_std": sc_ctx.scale_.astype(np.float32),
        "mlp_state": np.array(mlp_state_np, dtype=object),
        "cv_gate_acc": np.array(gate_acc),
        "cv_base_acc": np.array(mu_base),
        "cv_mlp_acc": np.array(mu_mlp),
        "n_train": np.array(len(y)),
    }
    # embed the LTAS distribution for use at inference time
    for k in dist:
        save_dict[f"dist_{k}"] = dist[k]

    np.savez(out_path, **save_dict)
    print(f"Saved. CV gate acc: {gate_acc:.1%}")
    print("  To use: chord_pipeline_v1 automatically loads this model when present.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train and save the context-window entropy-gated family classifier."
    )
    ap.add_argument("--n-songs", type=int, default=40,
                    help="Number of songs for training (default: 40, more is better)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--epochs",  type=int, default=60,
                    help="MLP training epochs (default: 60)")
    ap.add_argument("--out",     type=Path,
                    default=REPO / "harmonia" / "models" / "ctx_family_model.npz",
                    help="Output path (default: harmonia/models/ctx_family_model.npz)")
    args = ap.parse_args()
    train_and_save(args.n_songs, args.seed, args.epochs, args.out)


if __name__ == "__main__":
    main()
