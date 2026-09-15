"""train_yt_chord_model.py — train a chord quality+root model on YouTube audio.

Trains two lightweight MLPs on real recordings aligned to iReal Pro GT charts:
  1. quality_head (7-class: maj/min/dom/hdim/dim/aug/sus) on root-shifted 48d chroma
  2. root_head    (12-class)                               on absolute 48d chroma

Both use a 2-hidden-layer architecture (128→64, GELU, LayerNorm, Dropout 0.3)
trained with AdamW + cosine LR on MPS (Apple M4) or CPU.

Input:  --corpus data/cache/yt_corpus/corpus.npz  (built by build_yt_corpus.py)
Output: harmonia/models/yt_chord_model.npz

Usage:
    .venv/bin/python scripts/train_yt_chord_model.py \\
        --corpus data/cache/yt_corpus/corpus.npz \\
        --epochs 100 --lr 3e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── model ────────────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, n_classes: int, hidden1: int = 128, hidden2: int = 64):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden1), nn.LayerNorm(hidden1), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(hidden1, hidden2), nn.LayerNorm(hidden2), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(hidden2, n_classes),
    )


# ── training loop ─────────────────────────────────────────────────────────────

def _train(X: np.ndarray, y: np.ndarray, n_classes: int,
           *,
           epochs: int, lr: float, batch: int, device: str,
           label: str, hidden1: int = 128, hidden2: int = 64) -> tuple:
    """Train a quality or root MLP.  Returns (model, scaler_mean, scaler_std)."""
    import torch
    import torch.nn as nn

    # standard-scale inputs
    mean = X.mean(0).astype(np.float32)
    std  = (X.std(0) + 1e-9).astype(np.float32)
    Xn   = ((X - mean) / std).astype(np.float32)

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y,  dtype=torch.long,    device=device)

    # simple class-frequency weighting for imbalanced qualities
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (1.0 / (counts + 1.0)); w /= w.sum(); w *= n_classes
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    model = _make_mlp(X.shape[1], n_classes, hidden1=hidden1, hidden2=hidden2).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=wt)

    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0; n_batch = 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward(); opt.step()
            total_loss += loss.detach().item(); n_batch += 1
        sched.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  [{label}] epoch {epoch+1:3d}/{epochs}  "
                  f"loss={total_loss/n_batch:.4f}  train_acc={acc:.3f}")

    model.eval()
    return model, mean, std


def _eval_split(X: np.ndarray, y: np.ndarray, model, mean, std, device: str) -> float:
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        acc = (logits.argmax(1).cpu().numpy() == y).mean()
    return float(acc)


# ── save / load ───────────────────────────────────────────────────────────────

def save_model(path: Path, qual_model, qual_mean, qual_std,
               root_model, root_mean, root_std,
               qualities: list[str], *, context_window: int = 0) -> None:
    """Save both heads to a single .npz file."""
    import torch

    def _state(m):
        return {k: v.cpu().numpy() for k, v in m.state_dict().items()}

    np.savez(
        path,
        qualities=np.array(qualities),
        qual_mean=qual_mean, qual_std=qual_std,
        root_mean=root_mean, root_std=root_std,
        qual_state=np.array(_state(qual_model), dtype=object),
        root_state=np.array(_state(root_model), dtype=object),
        context_window=np.int32(context_window),
    )
    print(f"Saved → {path}")


# ── eval report ───────────────────────────────────────────────────────────────

def _per_class_acc(X, y, model, mean, std, class_names, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, device=device)).argmax(1).cpu().numpy()
    n_cls = len(class_names)
    for ci, name in enumerate(class_names):
        mask = (y == ci)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == ci).mean()
        print(f"    {name:6s}  n={mask.sum():4d}  acc={acc:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def _add_context(X: np.ndarray, song_ids: np.ndarray, window: int = 1) -> np.ndarray:
    """Concatenate ±window segment neighbors within each song (zero-padding at edges).

    Input:  X of shape (N, D)
    Output: X_ctx of shape (N, D * (2*window + 1))
    Records from different songs are never mixed.
    """
    N, D = X.shape
    W = 2 * window + 1
    out = np.zeros((N, D * W), dtype=X.dtype)
    unique = list(dict.fromkeys(song_ids.tolist()))
    idx_map = {s: [] for s in unique}
    for i, s in enumerate(song_ids.tolist()):
        idx_map[s].append(i)
    for s, idxs in idx_map.items():
        idxs = np.array(idxs)
        n_s = len(idxs)
        for offset in range(-window, window + 1):
            col = (offset + window) * D
            src = np.clip(np.arange(n_s) + offset, 0, n_s - 1)
            out[idxs, col:col + D] = X[idxs[src]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  required=True, type=Path,
                    help="Path to corpus .npz (from build_yt_corpus.py)")
    ap.add_argument("--out",     default=REPO / "harmonia/models/yt_chord_model.npz",
                    type=Path, help="Output .npz path")
    ap.add_argument("--epochs",  default=80, type=int)
    ap.add_argument("--lr",      default=3e-4, type=float)
    ap.add_argument("--batch",   default=128, type=int)
    ap.add_argument("--val-frac",default=0.15, type=float,
                    help="Fraction of songs to hold out for validation")
    ap.add_argument("--skip-mismatches", action="store_true",
                    help="Exclude alignment-mismatch records from training")
    ap.add_argument("--context", default=1, type=int,
                    help="Segment context window ±k (0=none, 1=prev+curr+next, etc.)")
    ap.add_argument("--n-quality-classes", default=7, type=int, choices=[3, 7],
                    help="3=maj/min/dom only (skip hdim/dim/aug/sus), 7=all classes")
    ap.add_argument("--hidden1", default=128, type=int, help="MLP first hidden dim")
    ap.add_argument("--hidden2", default=64,  type=int, help="MLP second hidden dim")
    ap.add_argument("--use-abs-features", action="store_true",
                    help="Concatenate absolute (non-root-shifted) BP chroma to quality features")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # load corpus
    print(f"Loading corpus: {args.corpus}")
    from harmonia.data.yt_chord_corpus import QUALITIES, QUALITY_IDX, load_corpus
    d = load_corpus(args.corpus)

    # Build feature matrix
    has_cqt = "feat12_cqt" in d
    if has_cqt:
        base_q = np.concatenate([d["feat48"], d["feat12_cqt"]], axis=1).astype(np.float32)
        X_root = np.concatenate([d["feat48_abs"], d["feat12_cqt_abs"]], axis=1).astype(np.float32)
        base_label = "60-dim (48 BP + 12 CQT)"
    else:
        base_q = d["feat48"].astype(np.float32)
        X_root = d["feat48_abs"].astype(np.float32)
        base_label = "48-dim (BP only)"
    if args.use_abs_features and "feat48_abs" in d:
        X_qual = np.concatenate([base_q, d["feat48_abs"].astype(np.float32)], axis=1)
        print(f"Using {X_qual.shape[1]}-dim features ({base_label} + 48 abs BP)")
    else:
        X_qual = base_q
        print(f"Using {base_label}")

    y_qual   = d["quality_idx"].astype(np.int64)
    y_root   = d["root"].astype(np.int64)
    match    = d["match"]
    song_ids = d.get("song_id", np.array([""] * len(y_qual)))

    if args.skip_mismatches:
        keep = np.isin(match, ["exact", "family"])
        X_qual = X_qual[keep]; X_root = X_root[keep]
        y_qual = y_qual[keep]; y_root = y_root[keep]
        match  = match[keep]; song_ids = song_ids[keep]
        print(f"Keeping {keep.sum()} / {len(keep)} records (skip_mismatches=True)")

    # Optionally restrict to 3-class quality (maj/min/dom only)
    n_quality_classes = args.n_quality_classes
    if n_quality_classes == 3:
        keep3 = y_qual <= 2  # maj=0, min=1, dom=2; drop hdim/dim/aug/sus
        X_qual = X_qual[keep3]; X_root = X_root[keep3]
        y_qual = y_qual[keep3]; y_root = y_root[keep3]
        song_ids = song_ids[keep3]
        print(f"3-class mode: keeping {keep3.sum()} maj/min/dom records (dropped hdim/dim/aug/sus)")

    # Optional: add segment-neighbor context (±k segments within same song)
    if args.context > 0:
        print(f"Adding ±{args.context}-segment context → {X_qual.shape[1] * (2*args.context+1)}-dim")
        X_qual = _add_context(X_qual, song_ids, window=args.context)
        X_root = _add_context(X_root, song_ids, window=args.context)

    n = len(X_qual)
    print(f"Total records: {n}")
    quals = [QUALITIES[i] for i in np.unique(y_qual)]
    print(f"Quality classes present: {quals}")

    from collections import Counter
    print("Quality distribution:")
    for q, cnt in sorted(Counter(y_qual.tolist()).items()):
        print(f"  [{q}] {QUALITIES[q]:6s}  n={cnt:5d}  ({100*cnt/n:.1f}%)")

    # Song-level train/val split: hold out ~15% of songs entirely
    # This prevents the model from overfitting to a specific recording's acoustics.
    unique_songs = list(dict.fromkeys(song_ids.tolist()))  # preserve order
    rng = np.random.default_rng(42)
    n_val_songs = max(1, int(len(unique_songs) * args.val_frac))
    val_songs = set(rng.choice(unique_songs, n_val_songs, replace=False).tolist())
    print(f"Songs: {len(unique_songs)} total, {len(val_songs)} held out for val: {val_songs}")

    val_mask = np.array([s in val_songs for s in song_ids.tolist()], dtype=bool)
    tr_mask  = ~val_mask
    print(f"Train: {tr_mask.sum()}  Val: {val_mask.sum()}")

    # ── quality head ──────────────────────────────────────────────────────────
    print(f"\n=== Quality head ({n_quality_classes}-class, arch={args.hidden1}-{args.hidden2}) ===")
    qual_model, qual_mean, qual_std = _train(
        X_qual[tr_mask], y_qual[tr_mask], n_classes=n_quality_classes,
        epochs=args.epochs, lr=args.lr, batch=args.batch, device=device,
        label="quality", hidden1=args.hidden1, hidden2=args.hidden2,
    )
    val_acc_q = _eval_split(X_qual[val_mask], y_qual[val_mask],
                             qual_model, qual_mean, qual_std, device)
    print(f"  → val accuracy: {val_acc_q:.3f}")
    print("  Per-class val accuracy:")
    _per_class_acc(X_qual[val_mask], y_qual[val_mask],
                   qual_model, qual_mean, qual_std, QUALITIES[:n_quality_classes], device)

    # ── root head ─────────────────────────────────────────────────────────────
    print(f"\n=== Root head (12-class, arch={args.hidden1}-{args.hidden2}) ===")
    root_model, root_mean, root_std = _train(
        X_root[tr_mask], y_root[tr_mask], n_classes=12,
        epochs=args.epochs, lr=args.lr, batch=args.batch, device=device,
        label="root", hidden1=args.hidden1, hidden2=args.hidden2,
    )
    val_acc_r = _eval_split(X_root[val_mask], y_root[val_mask],
                             root_model, root_mean, root_std, device)
    print(f"  → val accuracy: {val_acc_r:.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to {args.out}")
    save_model(
        args.out,
        qual_model, qual_mean, qual_std,
        root_model, root_mean, root_std,
        QUALITIES[:n_quality_classes],
        context_window=args.context,
    )
    print(f"\nSummary: quality_val={val_acc_q:.3f}  root_val={val_acc_r:.3f}")


if __name__ == "__main__":
    main()
