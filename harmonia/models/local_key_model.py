"""harmonia/models/local_key_model.py — small learned section-key predictor (#23).

A light bidirectional-GRU classifier that maps a section's chord sequence
(list of (root_pc, qual5)) to one of 24 keys (12 major + 12 minor). Trained to
imitate the rules-based symbolic oracle (``local_key_data.oracle_section_key``)
so that the *mapping section-chords -> local key* is available as a learned
component for phase-2 (noisy MMA/YouTube audio), where the clean oracle rule
cannot be applied directly.

Key prediction is **transpose-equivariant**: transposing every chord up k
semitones transposes the key label up k. We bake that in with random-transpose
augmentation (shift all roots and the label tonic by the same k), which is the
correct symmetry and multiplies effective data 12x.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

N_QUAL = 5   # maj/min/dom/hdim/dim (progression_encoder.QUAL5)
N_KEYS = 24  # 12 major + 12 minor


def transpose_example(seq: list[tuple[int, int]], y: int, k: int
                      ) -> tuple[list[tuple[int, int]], int]:
    """Shift all chord roots and the key label by k semitones (equivariance)."""
    tonic, mode_off = y % 12, (y // 12) * 12
    return ([((r + k) % 12, q) for r, q in seq], (tonic + k) % 12 + mode_off)


class LocalKeyGRU(nn.Module):
    """Bi-GRU over (root, quality) chord embeddings -> 24-key logits."""

    def __init__(self, d_model: int = 48, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.root_emb = nn.Embedding(12, d_model)
        self.qual_emb = nn.Embedding(N_QUAL, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
            nn.Linear(2 * d_model, N_KEYS),
        )

    def forward(self, root, qual, lengths):
        # root, qual: (B, T) long; lengths: (B,) long
        x = self.root_emb(root) + self.qual_emb(qual)      # (B,T,d)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,T,2d)
        # mean-pool over real timesteps
        mask = (torch.arange(out.size(1), device=out.device)[None, :]
                < lengths[:, None]).unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)                            # (B,24)


def collate(batch: list[tuple[list[tuple[int, int]], int]], device: str = "cpu"):
    """batch of (seq, y) -> padded (root, qual, lengths, y) tensors."""
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lengths.max())
    root = torch.zeros(len(seqs), T, dtype=torch.long)
    qual = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        for t, (r, q) in enumerate(s):
            root[i, t], qual[i, t] = r, q
    y = torch.tensor(ys, dtype=torch.long)
    return (root.to(device), qual.to(device), lengths.to(device), y.to(device))


def load_model(path: Path, device: str = "cpu") -> LocalKeyGRU:
    ckpt = torch.load(path, map_location=device)
    model = LocalKeyGRU(**ckpt.get("hparams", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
