"""Train the chord ProgressionEncoder (issue #21).

Masked-cloze / denoising objective: predict the centre chord's 5-class quality
from its ±6-chord neighbourhood. Each epoch, positions are stochastically masked
or quality-corrupted (with a matching low confidence gate) so the model must
learn harmonic grammar rather than copy the centre.

Split: project-standard modulo split (every 5th jazz1460 song → val, ~80/20).
The brief's literal '<70 train' would starve the model (70/1458 songs); see
progression_encoder.split_sequences docstring.

    .venv/bin/python scripts/train_progression_encoder.py --epochs 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.models.progression_encoder import (  # noqa: E402
    CTX, MASK_ID, WINDOW, ProgressionEncoder, load_jazz_sequences,
    split_sequences,
)

DB = REPO / "data" / "accomp_db" / "db.jsonl"
OUT = REPO / "harmonia" / "models" / "progression_encoder.pt"
N_QUAL = 5


class PositionDataset(Dataset):
    """One item per (song, centre index). Augmentation happens in collate."""

    def __init__(self, seqs, train: bool, seed: int = 0):
        self.items: list[tuple[np.ndarray, np.ndarray, int]] = []
        for seq in seqs:
            roots = np.array([r for r, _ in seq], dtype=np.int64)
            quals = np.array([q for _, q in seq], dtype=np.int64)
            for i in range(len(seq)):
                self.items.append((roots, quals, i))
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        roots, quals, i = self.items[idx]
        n = len(roots)
        r_c = roots[i]
        target = int(quals[i])

        root_rel = np.zeros(WINDOW, dtype=np.int64)
        qual = np.full(WINDOW, MASK_ID, dtype=np.int64)
        conf = np.zeros(WINDOW, dtype=np.float32)
        pad = np.ones(WINDOW, dtype=bool)

        for p in range(WINDOW):
            j = i + (p - CTX)
            if 0 <= j < n:
                pad[p] = False
                root_rel[p] = (roots[j] - r_c) % 12
                qual[p] = quals[j]
                conf[p] = 1.0

        if self.train:
            rng = self.rng
            # centre: 50% masked, else 25% corrupted (low conf), else clean high conf
            u = rng.random()
            if u < 0.50:
                qual[CTX] = MASK_ID
                conf[CTX] = 0.0
            elif u < 0.75:
                qual[CTX] = int(rng.integers(N_QUAL))
                conf[CTX] = float(rng.uniform(0.2, 0.6))
            else:
                conf[CTX] = float(rng.uniform(0.7, 1.0))
            # neighbours: 15% corrupted (low conf), else jittered high conf
            for p in range(WINDOW):
                if p == CTX or pad[p]:
                    continue
                if rng.random() < 0.15:
                    qual[p] = int(rng.integers(N_QUAL))
                    conf[p] = float(rng.uniform(0.2, 0.6))
                else:
                    conf[p] = float(rng.uniform(0.7, 1.0))
        else:
            # deterministic cloze: centre masked, neighbours clean
            qual[CTX] = MASK_ID
            conf[CTX] = 0.0

        return (torch.from_numpy(root_rel), torch.from_numpy(qual),
                torch.from_numpy(conf), torch.from_numpy(pad), target)


def collate(batch):
    rr, q, c, p, t = zip(*batch)
    return (torch.stack(rr), torch.stack(q), torch.stack(c),
            torch.stack(p), torch.tensor(t, dtype=torch.long))


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for rr, q, c, p, t in loader:
            rr, q, c, p = rr.to(device), q.to(device), c.to(device), p.to(device)
            logits = model(rr, q, c, p)
            pred = logits.argmax(-1).cpu()
            correct += (pred == t).sum().item()
            total += len(t)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    seqs = load_jazz_sequences(DB)
    train_seqs, val_seqs = split_sequences(seqs)
    print(f"jazz1460: {len(seqs)} songs -> {len(train_seqs)} train / "
          f"{len(val_seqs)} val")

    train_ds = PositionDataset(train_seqs, train=True, seed=args.seed)
    val_ds = PositionDataset(val_seqs, train=False)
    print(f"examples: {len(train_ds)} train / {len(val_ds)} val")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=512, collate_fn=collate)

    # val-majority baseline (predict the most common centre quality on val)
    val_targets = [t for *_, t in (val_ds[i] for i in range(len(val_ds)))]
    maj_cls = np.bincount(val_targets, minlength=N_QUAL).argmax()
    maj_acc = np.mean([t == maj_cls for t in val_targets])
    print(f"val majority baseline: class {maj_cls} acc={maj_acc:.1%}")

    model = ProgressionEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        nb = 0
        for rr, q, c, p, t in train_ld:
            rr, q, c, p, t = (rr.to(device), q.to(device), c.to(device),
                              p.to(device), t.to(device))
            opt.zero_grad()
            loss = crit(model(rr, q, c, p), t)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        va = evaluate(model, val_ld, device)
        hist.append((ep, tot_loss / nb, va))
        tag = ""
        if va > best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "hparams": {}, "val_cloze_acc": va}, OUT)
            tag = "  *best"
        print(f"epoch {ep:2d}  loss={tot_loss/nb:.4f}  val_cloze_acc={va:.1%}{tag}")

    print(f"\nbest val cloze acc = {best_val:.1%}  (majority {maj_acc:.1%})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
