"""train_yt_exact_matches.py — retrain on exact iReal-to-inferred alignment matches only.

Filters corpus to only "exact" matches (highest alignment confidence), discards
"family" and "mismatch" records. Smaller dataset but cleaner labels.

Usage:
    .venv/bin/python scripts/train_yt_exact_matches.py \\
        --corpus data/cache/yt_corpus/corpus.npz \\
        --output data/models/yt_exact_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]


def _make_mlp(in_dim: int, n_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


def _train_head(X, y, n_classes, *, epochs, lr, batch, device, head_name):
    import torch
    import torch.nn as nn

    mean = X.mean(0).astype(np.float32)
    std = (X.std(0) + 1e-9).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

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

        if epoch == 0 or (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  [{head_name}] epoch {epoch+1:3d}/{epochs}  loss={total_loss/n_batch:.4f}  train_acc={train_acc:.3f}")

    model.eval()
    return model, mean, std


def _eval_accuracy(X, y, model, mean, std, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        acc = (logits.argmax(1).cpu().numpy() == y).mean()
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/cache/yt_corpus/corpus.npz"))
    ap.add_argument("--output", type=Path, default=Path("data/models/yt_exact_v1.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps" if _has_mps() else "cpu")
    args = ap.parse_args()

    print(f"Loading corpus from {args.corpus}...")
    d = np.load(args.corpus, allow_pickle=True)
    feat48 = d["feat48"]
    feat48_abs = d["feat48_abs"]
    quality_idx = d["quality_idx"].astype(int)
    roots = d["root"].astype(int)
    song_id = d["song_id"]
    match = d["match"]

    # Filter to EXACT matches only
    exact_mask = match == "exact"
    print(f"Filtering to exact matches: {exact_mask.sum()} / {len(match)} ({100*exact_mask.sum()/len(match):.1f}%)")

    feat48 = feat48[exact_mask]
    feat48_abs = feat48_abs[exact_mask]
    quality_idx = quality_idx[exact_mask]
    roots = roots[exact_mask]
    song_id = song_id[exact_mask]

    print(f"Exact-only corpus: {len(feat48)} records")

    # Split
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
    X_test_root = feat48_abs[test_mask]
    y_test_root = roots[test_mask]

    X_train_qual = feat48[train_mask]
    y_train_qual = quality_idx[train_mask]
    X_test_qual = feat48[test_mask]
    y_test_qual = quality_idx[test_mask]

    print(f"Split: {train_mask.sum()} train, {test_mask.sum()} test")

    print("\n=== Root Head ===")
    root_model, root_mean, root_std = _train_head(
        X_train_root, y_train_root, 12,
        epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device, head_name="root"
    )
    root_test = _eval_accuracy(X_test_root, y_test_root, root_model, root_mean, root_std, args.device)
    print(f"Root test acc: {root_test:.3f}  (target: >0.85)")

    print("\n=== Quality Head ===")
    quality_model, qual_mean, qual_std = _train_head(
        X_train_qual, y_train_qual, 7,
        epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device, head_name="quality"
    )
    qual_test = _eval_accuracy(X_test_qual, y_test_qual, quality_model, qual_mean, qual_std, args.device)
    print(f"Quality test acc: {qual_test:.3f}  (target: >0.68)")

    # Per-class
    import torch
    Xn = ((X_test_qual - qual_mean) / qual_std).astype(np.float32)
    with torch.no_grad():
        preds = quality_model(torch.tensor(Xn, device=args.device)).argmax(1).cpu().numpy()

    print("\nPer-class test recall:")
    for i, q in enumerate(QUALITIES):
        mask = y_test_qual == i
        if mask.sum() > 0:
            recall = (preds[mask] == i).mean()
            print(f"  {q:6s}: {recall:.3f}")

    # Save
    import torch
    torch.save({
        "root_model": root_model,
        "root_mean": root_mean,
        "root_std": root_std,
        "quality_model": quality_model,
        "quality_mean": qual_mean,
        "quality_std": qual_std,
        "qualities": QUALITIES,
    }, args.output)
    print(f"\nSaved to {args.output}")


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False


if __name__ == "__main__":
    main()
