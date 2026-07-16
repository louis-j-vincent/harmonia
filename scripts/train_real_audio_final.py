"""train_real_audio_final.py — production training on YouTube+iReal real-audio corpus.

Unified training script for either 10-song pilot or 50-song corpus.
Handles song-stratified split, balanced training, and MIREX metric reporting.

Usage:
    .venv/bin/python scripts/train_real_audio_final.py \\
        --corpus data/cache/yt_corpus/corpus_50.npz \\
        --output data/models/prod_real_audio_v1.pt \\
        --epochs 50 --batch 32 --lr 3e-4 \\
        --min-match exact  # or "exact+family"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
QUALITY_TO_IDX = {q: i for i, q in enumerate(QUALITIES)}


def _make_mlp(in_dim: int, n_classes: int):
    """MLP(in_dim → 128 → 64 → n_classes) with LayerNorm, GELU, Dropout."""
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


def _augment_root_by_roll(X_abs, y_root, n_shifts=12):
    """Pitch-shift augmentation for the ROOT head only.

    X_abs is feat48_abs: 4 concatenated 12-pc chroma-like blocks
    (onset⊕note⊕bass⊕treble) in absolute (non-root-relative) frame. Rolling
    each 12-wide block by k semitones and shifting the root label by +k mod 12
    is an exact label-preserving transform (validated 2026-07-15, Billboard
    BP48 campaign: +4.6pp root acc, 5x lower seed variance, from enforcing
    rotation-equivariance + balancing the root class marginal). Do NOT apply
    this to feat48 (root-relative quality features) — that space already
    encodes root-relative invariance and rolling it would break it.
    """
    n, d = X_abs.shape
    assert d % 12 == 0, f"expected feat48_abs width multiple of 12, got {d}"
    n_blocks = d // 12
    outs_X, outs_y = [X_abs], [y_root]
    for k in range(1, n_shifts):
        rolled = X_abs.reshape(n, n_blocks, 12)
        rolled = np.roll(rolled, shift=k, axis=2).reshape(n, d)
        outs_X.append(rolled)
        outs_y.append((y_root + k) % 12)
    return np.concatenate(outs_X, axis=0), np.concatenate(outs_y, axis=0)


def _standardize(X):
    """Return (X_normalized, mean, std)."""
    mean = X.mean(0).astype(np.float32)
    std = (X.std(0) + 1e-9).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)
    return Xn, mean, std


def _train_head(X, y, n_classes, *, epochs, lr, batch, device, head_name):
    """Train a single head, return (model, mean, std)."""
    import torch
    import torch.nn as nn

    Xn, mean, std = _standardize(X)
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

    # Balanced class weights
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
            print(
                f"  [{head_name:6s}] epoch {epoch+1:3d}/{epochs}  "
                f"loss={total_loss/n_batch:.4f}  train_acc={train_acc:.3f}"
            )

    model.eval()
    return model, mean, std


def _eval(X, y, model, mean, std, device):
    """Evaluate accuracy and per-class recall."""
    import torch

    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        preds = logits.argmax(1).cpu().numpy()

    overall_acc = (preds == y).mean()
    per_class = {}
    for cls in range(len(np.unique(y))):
        mask = y == cls
        if mask.sum() == 0:
            per_class[cls] = 0.0
        else:
            per_class[cls] = float((preds[mask] == cls).mean())

    return float(overall_acc), per_class, preds


def main():
    ap = argparse.ArgumentParser(
        description="Train production heads on real-audio YouTube+iReal corpus"
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/cache/yt_corpus/corpus_50.npz"),
        help="Corpus NPZ file",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/prod_real_audio_v1.pt"),
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--min-match",
        choices=["exact", "exact+family"],
        default="exact+family",
        help="Minimum alignment quality to accept",
    )
    ap.add_argument(
        "--device",
        default="mps" if _has_mps() else "cpu",
    )
    ap.add_argument(
        "--root-roll-augment",
        action="store_true",
        help="Pitch-shift augmentation for the root head (validated +4.6pp on Billboard BP48, 2026-07-15)",
    )
    args = ap.parse_args()

    # Load corpus
    print(f"Loading {args.corpus}...")
    d = load_corpus(args.corpus)
    feat48 = d["feat48"]
    feat48_abs = d["feat48_abs"]
    quality_idx = d["quality_idx"].astype(int)
    quality_names = d["quality"]
    roots = d["root"].astype(int)
    song_id = d["song_id"]
    match = d["match"]

    print(f"Total records: {len(feat48)}")
    print(f"Match distribution:")
    for match_type in ["exact", "family", "mismatch"]:
        count = (match == match_type).sum()
        pct = 100 * count / len(match)
        print(f"  {match_type:10s}: {count:6d} ({pct:5.1f}%)")

    # Filter
    if args.min_match == "exact":
        keep = filter_by_match(match, minimum=MatchQuality.EXACT)
    else:  # exact+family
        keep = filter_by_match(match, minimum=MatchQuality.FAMILY)

    feat48 = feat48[keep]
    feat48_abs = feat48_abs[keep]
    quality_idx = quality_idx[keep]
    quality_names = quality_names[keep]
    roots = roots[keep]
    song_id = song_id[keep]

    print(f"\nFiltered to {args.min_match}: {len(feat48)} records ({100*len(feat48)/len(match):.1f}%)")

    # Split (song-stratified 80/10/10)
    unique_songs = sorted(list(set(song_id)))
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

    # Extract splits
    X_train_root = feat48_abs[train_mask]
    y_train_root = roots[train_mask]
    X_val_root = feat48_abs[val_mask]
    y_val_root = roots[val_mask]
    X_test_root = feat48_abs[test_mask]
    y_test_root = roots[test_mask]

    X_train_qual = feat48[train_mask]
    y_train_qual = quality_idx[train_mask]
    X_val_qual = feat48[val_mask]
    y_val_qual = quality_idx[val_mask]
    X_test_qual = feat48[test_mask]
    y_test_qual = quality_idx[test_mask]

    print(
        f"\nSplit across {len(unique_songs)} songs:\n"
        f"  {len(train_mask[train_mask])} train / {len(val_mask[val_mask])} val / {len(test_mask[test_mask])} test\n"
    )

    # Train root head
    print("=== Root Head (12-class on absolute BP48) ===")
    if args.root_roll_augment:
        n_before = len(X_train_root)
        X_train_root, y_train_root = _augment_root_by_roll(X_train_root, y_train_root)
        print(f"Root-roll augmentation: {n_before} -> {len(X_train_root)} train records (x12)")
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

    root_val_acc, root_val_recall, _ = _eval(X_val_root, y_val_root, root_model, root_mean, root_std, args.device)
    root_test_acc, root_test_recall, root_test_preds = _eval(
        X_test_root, y_test_root, root_model, root_mean, root_std, args.device
    )

    print(f"Root val acc:  {root_val_acc:.1%}")
    print(f"Root test acc: {root_test_acc:.1%}  (target: >85%)")

    # Train quality head
    print("\n=== Quality Head (7-class on root-relative BP48) ===")
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

    qual_val_acc, qual_val_recall, _ = _eval(X_val_qual, y_val_qual, quality_model, qual_mean, qual_std, args.device)
    qual_test_acc, qual_test_recall, qual_test_preds = _eval(
        X_test_qual, y_test_qual, quality_model, qual_mean, qual_std, args.device
    )

    # Balanced accuracy (unweighted average of per-class recalls)
    qual_test_balanced = np.mean([qual_test_recall.get(i, 0.0) for i in range(7)])

    print(f"Quality val acc:     {qual_val_acc:.1%}")
    print(f"Quality test acc:    {qual_test_acc:.1%}")
    print(f"Quality test balanced: {qual_test_balanced:.1%}  (target: >68%)")

    print("\nQuality per-class test recall:")
    for i, q in enumerate(QUALITIES):
        recall = qual_test_recall.get(i, 0.0)
        print(f"  {q:6s}: {recall:.1%}")

    # Save models
    print(f"\nSaving to {args.output}...")
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

    # Summary and decision
    print("\n" + "="*70)
    print("FINAL METRICS (Real Audio, Song-Stratified Test Set)")
    print("="*70)
    print(f"Root accuracy:                {root_test_acc:.1%}  (target: >85%)")
    print(f"Quality balanced accuracy:    {qual_test_balanced:.1%}  (target: >68%)")
    print(f"Quality dom recall:           {qual_test_recall.get(2, 0.0):.1%}  (target: >65%)")
    print()

    all_pass = (
        root_test_acc >= 0.85
        and qual_test_balanced >= 0.68
        and qual_test_recall.get(2, 0.0) >= 0.65
    )

    if all_pass:
        print("✓ ALL TARGETS MET — SHIPPABLE")
    elif root_test_acc >= 0.75 and qual_test_balanced >= 0.60:
        print("⚠ PARTIAL (root/quality solid, dom recall below target)")
        print("  → Acceptable for research/beta, iterate before production")
    else:
        print("✗ UNDERPERFORMING (below research baseline)")
        print("  → Needs investigation (more data, better alignment, or domain-gap acceptance)")

    print()
    print(f"Models saved to {args.output}")


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False


if __name__ == "__main__":
    main()
