"""harmonia/models/local_key_seq_model.py — per-chord local-key sequence model (#20/#23).

A **many-to-many** bidirectional-GRU tagger: it emits one of 24 keys (12 major +
12 minor) at *every* chord position, distilled from the rule-based heuristic
(:func:`theory.local_key.continuity_scale_track_v2`) via per-position
cross-entropy — the teacher the user chose over the section oracle.

Why a whole-song bi-GRU rather than a fixed-window transformer (the
``ProgressionEncoder`` shape)? The gesture we want the model to smooth — a
descending-fifths chain of secondary dominants (Em7 A7 D7 G7#5) — has *no fixed
length*; recognising it as one move toward a single resolution needs context
that runs to the end of the chain, which a ±k window can truncate. A GRU over
the whole sequence sees unbounded left/right context and is the smallest thing
that can, in principle, out-smooth the heuristic's 2-chord lookahead. It also
reuses the existing ``LocalKeyGRU`` embedding/transpose machinery.

Key prediction is **transpose-equivariant by construction**: the dataset encodes
both chord roots and key targets *relative to the song's global tonic*
(:func:`local_key_seq_data.tokens_to_rel_example`), so transposing a whole song
leaves the (input, target) pair unchanged and the model learns each harmonic
motif once for all 12 keys — no random-transpose augmentation needed (unlike the
oracle-trained ``LocalKeyGRU``, which used absolute roots + augmentation).
Reconstruct an absolute key at inference by adding the global tonic back
(:func:`local_key_seq_data.rel_to_abs_key`).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

N_QUAL = 5   # maj/min/dom/hdim/dim (progression_encoder.QUAL5)
N_KEYS = 24  # 12 major + 12 minor
PAD_KEY = -100  # CrossEntropyLoss ignore_index for padded timesteps


class LocalKeySeqGRU(nn.Module):
    """Bi-GRU over (root, quality) chord embeddings → per-position 24-key logits."""

    def __init__(self, d_model: int = 64, layers: int = 2, dropout: float = 0.2):
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
        # root, qual: (B, T) long; lengths: (B,) long → logits (B, T, 24)
        x = self.root_emb(root) + self.qual_emb(qual)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,T,2d)
        return self.head(out)                                             # (B,T,24)


def collate(batch: list[tuple[list[tuple[int, int]], list[int]]], device: str = "cpu"):
    """batch of (seq, ys) → padded (root, qual, lengths, targets) tensors.

    Padded target timesteps are filled with ``PAD_KEY`` (= CrossEntropyLoss
    ignore_index) so padding contributes no loss and is excluded from accuracy.
    """
    seqs, yss = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lengths.max())
    root = torch.zeros(len(seqs), T, dtype=torch.long)
    qual = torch.zeros(len(seqs), T, dtype=torch.long)
    targets = torch.full((len(seqs), T), PAD_KEY, dtype=torch.long)
    for i, (s, ys) in enumerate(zip(seqs, yss)):
        for t, (r, q) in enumerate(s):
            root[i, t], qual[i, t] = r, q
        for t, y in enumerate(ys):
            targets[i, t] = y
    return (root.to(device), qual.to(device),
            lengths.to(device), targets.to(device))


@torch.no_grad()
def predict_sequence(
    model: LocalKeySeqGRU, seq: list[tuple[int, int]], device: str = "cpu"
) -> list[int]:
    """Greedy per-position argmax key idx for a single chord sequence."""
    model.eval()
    root, qual, lengths, _ = collate([(seq, [0] * len(seq))], device)
    logits = model(root, qual, lengths)          # (1,T,24)
    return logits[0].argmax(-1).cpu().tolist()


def load_seq_model(path: Path, device: str = "cpu") -> LocalKeySeqGRU:
    ckpt = torch.load(path, map_location=device)
    model = LocalKeySeqGRU(**ckpt.get("hparams", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
