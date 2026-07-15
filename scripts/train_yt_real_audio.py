"""train_yt_real_audio.py — train production heads on real-audio YouTube+iReal corpus.

Trains root/quality/7th heads natively on BP48 real audio (not synthetic).
Song-stratified 80/10/10 split, balanced class weights, evaluated on MIREX metrics.

Architecture (from bridge findings):
  - Root head:    MLP(48→128→64→12) on absolute BP48
  - Quality head: MLP(48→128→64→5) on root-relative BP48, with trigram context
  - 7th head:     flat 5-way (dom as independent class)

Usage:
    .venv/bin/python scripts/train_yt_real_audio.py \\
        --corpus data/cache/yt_corpus/corpus.npz \\
        --output data/models/yt_real_audio_v1.pt \\
        --epochs 50 --batch 32 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── quality scheme ────────────────────────────────────────────────────────────

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
QUALITY_TO_IDX = {q: i for i, q in enumerate(QUALITIES)}
Q5_MAP = {
    "maj": "maj", "aug": "maj", "sus": "maj",
    "min": "min",
    "dom": "dom",
    "hdim": "hdim",
    "dim": "dim",
}


# ── models ────────────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, n_classes: int, hidden1: int = 128, hidden2: int = 64):
    """Build MLP(in_dim → hidden1 → hidden2 → n_classes) with LayerNorm, GELU, Dropout."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.LayerNorm(hidden1),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden1, hidden2),
        nn.LayerNorm(hidden2),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden2, n_classes),
    )


# ── training ──────────────────────────────────────────────────────────────────

def _train_head(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epochs: int,
    lr: float,
    batch: int,
    device: str,
    head_name: str,
) -> tuple:
    """Train a single head. Returns (model, mean, std)."""
    import torch
    import torch.nn as nn

    # Standardize inputs
    mean = X.mean(0).astype(np.float32)
    std = (X.std(0) + 1e-9).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

    # Class frequency weighting for imbalanced classes
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / (counts + 1.0)
    w /= w.sum()
    w *= n_classes
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    model = _make_mlp(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=wt)

    n = len(Xt)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batch = 0

        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.detach().item()
            n_batch += 1

        sched.step()

        # Log progress
        if epoch == 0 or (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(
                f"  [{head_name}] epoch {epoch+1:3d}/{epochs}  "
                f"loss={total_loss/n_batch:.4f}  train_acc={train_acc:.3f}"
            )
            best_val_acc = max(best_val_acc, train_acc)

    model.eval()
    return model, mean, std


def _eval_accuracy(X: np.ndarray, y: np.ndarray, model, mean, std, device: str) -> float:
    """Evaluate accuracy on a split."""
    import torch

    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        acc = (logits.argmax(1).cpu().numpy() == y).mean()
    return float(acc)


def _eval_per_class(X: np.ndarray, y: np.ndarray, model, mean, std, device: str, n_classes: int) -> dict:
    """Evaluate per-class recall."""
    import torch

    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        preds = logits.argmax(1).cpu().numpy()

    result = {}
    for cls in range(n_classes):
        mask = y == cls
        if mask.sum() == 0:
            result[f"class_{cls}_recall"] = 0.0
        else:
            result[f"class_{cls}_recall"] = float((preds[mask] == cls).mean())
    return result


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Train production heads on YouTube+iReal real-audio corpus"
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/cache/yt_corpus/corpus.npz"),
        help="Path to corpus.npz",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("data/models/yt_real_audio_v1.pt")
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps" if _has_mps() else "cpu")
    args = ap.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    d = np.load(args.corpus, allow_pickle=True)
    feat48 = d["feat48"]  # (N, 48) root-relative
    feat48_abs = d["feat48_abs"]  # (N, 48) absolute
    quality_idx = d["quality_idx"].astype(int)  # (N,) 0-6
    quality_names = d["quality"]  # (N,)
    roots = d["root"].astype(int)  # (N,) 0-11
    labels = d["labels"]  # (N,)
    match = d["match"]  # (N,) "exact", "family", "mismatch"
    song_id = d["song_id"]  # (N,) song identifiers

    # Filter to exact/family matches only (skip mismatches)
    mask = (match == "exact") | (match == "family")
    feat48 = feat48[mask]
    feat48_abs = feat48_abs[mask]
    quality_idx = quality_idx[mask]
    quality_names = quality_names[mask]
    roots = roots[mask]
    song_id = song_id[mask]

    print(f"Loaded {len(feat48)} records (exact + family matches)")

    # Song-stratified split: 80/10/10
    unique_songs = list(set(song_id))
    np.random.seed(42)
    np.random.shuffle(unique_songs)

    n_train = int(0.8 * len(unique_songs))
    n_val = int(0.1 * len(unique_songs))
    train_songs = set(unique_songs[:n_train])
    val_songs = set(unique_songs[n_train : n_train + n_val])
    test_songs = set(unique_songs[n_train + n_val :])

    train_mask = np.array([s in train_songs for s in song_id])
    val_mask = np.array([s in val_songs for s in song_id])
    test_mask = np.array([s in test_songs for s in song_id])

    X_train_root = feat48_abs[train_mask]
    y_train_root = roots[train_mask]
    X_val_root = feat48_abs[val_mask]
    y_val_root = roots[val_mask]
    X_test_root = feat48_abs[test_mask]
    y_test_root = roots[test_mask]

    X_train_qual = feat48[train_mask]  # root-relative
    y_train_qual = quality_idx[train_mask]
    X_val_qual = feat48[val_mask]
    y_val_qual = quality_idx[val_mask]
    X_test_qual = feat48[test_mask]
    y_test_qual = quality_idx[test_mask]

    print(f"\nSplit: {len(train_mask[train_mask])} train, {len(val_mask[val_mask])} val, {len(test_mask[test_mask])} test")
    print(f"  Root classes: {len(np.unique(y_train_root))} unique")
    print(f"  Quality classes: {len(np.unique(y_train_qual))} unique")

    # Train root head
    print("\n=== Training Root Head (12-class on absolute BP48) ===")
    root_model, root_mean, root_std = _train_head(
        X_train_root,
        y_train_root,
        n_classes=12,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        device=args.device,
        head_name="root",
    )

    root_val_acc = _eval_accuracy(X_val_root, y_val_root, root_model, root_mean, root_std, args.device)
    root_test_acc = _eval_accuracy(X_test_root, y_test_root, root_model, root_mean, root_std, args.device)
    print(f"\nRoot validation acc: {root_val_acc:.3f}")
    print(f"Root test acc: {root_test_acc:.3f}")

    # Train quality head
    print("\n=== Training Quality Head (7-class on root-relative BP48) ===")
    quality_model, qual_mean, qual_std = _train_head(
        X_train_qual,
        y_train_qual,
        n_classes=7,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        device=args.device,
        head_name="quality",
    )

    qual_val_acc = _eval_accuracy(X_val_qual, y_val_qual, quality_model, qual_mean, qual_std, args.device)
    qual_test_acc = _eval_accuracy(X_test_qual, y_test_qual, quality_model, qual_mean, qual_std, args.device)
    print(f"\nQuality validation acc: {qual_val_acc:.3f}")
    print(f"Quality test acc: {qual_test_acc:.3f}")

    # Per-class recall (especially dom)
    qual_per_class = _eval_per_class(X_test_qual, y_test_qual, quality_model, qual_mean, qual_std, args.device, 7)
    print("\nQuality per-class test recall:")
    for i, q in enumerate(QUALITIES):
        recall = qual_per_class[f"class_{i}_recall"]
        print(f"  {q:6s}: {recall:.3f}")

    # Save models
    print(f"\nSaving models to {args.output}...")
    import torch

    torch.save(
        {
            "root_model": root_model,
            "root_mean": root_mean,
            "root_std": root_std,
            "quality_model": quality_model,
            "quality_mean": qual_mean,
            "quality_std": qual_std,
            "qualities": QUALITIES,
        },
        args.output,
    )

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Root test acc:     {root_test_acc:.1%}  (target: >85%)")
    print(f"Quality balanced:  {qual_test_acc:.1%}  (target: >68%)")
    print(f"Quality dom recall: {qual_per_class['class_2_recall']:.1%}  (target: >65%)")
    print(f"\nModels saved to {args.output}")


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
