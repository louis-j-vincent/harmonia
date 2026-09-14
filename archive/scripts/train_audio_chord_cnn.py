"""User's architecture: circular CNN over the 12-note chroma + an MLP branch for
the bass/register ordering. Compared fairly against logistic regression.

Rationale (docs/audio_chord_model_2026-07-05.md):
  - chroma (12 pitch classes) is the right representation for chord identity;
  - a CIRCULAR 1D convolution is transposition-aware — it learns a chord shape
    once and detects it at every root;
  - chroma discards who's on the bottom, so a separate MLP branch takes the
    bass/treble register chroma (the "which note is the bass / on top" signal).

Input per chord (root-relative, so directly comparable to the LR results):
  chroma channels: onset(12), note(12)         → (2, 12) for the circular CNN
  order features : bass-band(12), treble-band(12) → (24,) for the MLP branch

5-fold grouped-by-song CV. CPU, ~tens of seconds per level.

Usage: .venv/bin/python scripts/train_audio_chord_cnn.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

torch.manual_seed(0)


def l1(x):
    return x / (x.sum(axis=-1, keepdims=True) + 1e-9)


class ChordNet(nn.Module):
    """Circular CNN on chroma channels + MLP on bass/treble order features."""

    def __init__(self, n_classes: int, n_chroma_ch: int = 2, order_dim: int = 24):
        super().__init__()
        # circular convolution over the 12 pitch classes (transposition-aware)
        self.conv1 = nn.Conv1d(n_chroma_ch, 24, 5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(24, 24, 5, padding=2, padding_mode="circular")
        # two views of the conv map: aligned (keeps root at position 0) + a
        # transposition-invariant global pool (the chord shape regardless of root)
        self.order = nn.Sequential(nn.Linear(order_dim, 24), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(24 * 12 + 24 + 24, 96), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(96, n_classes),
        )

    def forward(self, chroma, order):
        x = F.relu(self.conv1(chroma))
        x = F.relu(self.conv2(x))                 # (B, 24, 12)
        aligned = x.flatten(1)                      # keeps which position is the root
        invariant = x.mean(dim=2)                   # transposition-invariant shape
        o = self.order(order)
        return self.head(torch.cat([aligned, invariant, o], dim=1))


def train_eval(Xc, Xo, y, groups, n_classes, epochs=60):
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(Xc, y, groups):
        net = ChordNet(n_classes)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        ct = torch.tensor(Xc[tr], dtype=torch.float32)
        ot = torch.tensor(Xo[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.long)
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = lossf(net(ct[idx], ot[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Xc[te], dtype=torch.float32),
                       torch.tensor(Xo[te], dtype=torch.float32)).argmax(1).numpy()
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    d = np.load(FEAT, allow_pickle=True)
    onset, note, bass, treble = (l1(d["onset"]), l1(d["note"]),
                                 l1(d["bass"]), l1(d["treble"]))
    groups = d["song"]
    Xc = np.stack([onset, note], axis=1)            # (N, 2, 12) chroma channels
    Xo = np.hstack([bass, treble])                   # (N, 24) order/register info
    print(f"{len(onset)} instances, {len(set(groups.tolist()))} songs, "
          f"5-fold grouped CV\n")
    print(f"{'level':<30}{'CNN+MLP (yours)':>18}{'logreg (prev)':>16}")
    print("-" * 64)
    lr_ref = {"family": 0.943, "base7": 0.879, "exact": 0.836}
    for name, key in [("FAMILY (5)", "family"), ("SEVENTH (14)", "base7"),
                      ("EXACT (18)", "exact")]:
        y = d[key].astype(int)
        m, s = train_eval(Xc, Xo, y, groups, int(y.max() + 1))
        print(f"{name:<30}{m:>16.1%}±{s*100:>3.0f}{lr_ref[key]:>15.1%}")


if __name__ == "__main__":
    main()
