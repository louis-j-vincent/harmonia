"""harmonia_min/chord_lm/model.py — a small chord transformer.

One architecture, two objectives, chosen by `causal`:

  causal=True   GPT-style. Predicts slot t+1 from slots <=t. Generative; this is
                the one that can *continue* a chart, and the one whose numbers
                are comparable to Chordonomicon's GPT-2 next-chord baseline.
  causal=False  BERT-style masked cloze. Predicts a masked slot from BOTH sides.
                This is the one that matters for the pipeline: when the acoustic
                model is unsure about half-bar 27, it has half-bars 1-26 AND
                28-end as context, and throwing away the future would be silly.

Three embeddings are summed:
  * token       — what chord (or REP / NC)
  * slot-in-bar — where in the bar this slot sits (downbeat vs backbeat at
                  half-bar grain). Metrical position is not decoration: a chord
                  change on the downbeat and one on beat 3 are different events,
                  and REP density differs sharply between the two.
  * position    — absolute slot index, learned.

Transposition equivariance is obtained by augmentation (random key shift per
sequence), not by construction — see `random_transpose` in data.py. That keeps
the model free to learn the few genuinely key-absolute effects (jazz keys
cluster on the flat side) while still seeing every progression in all 12 keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PAD, VOCAB_SIZE


@dataclass
class LMConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    slots_per_bar: int = 2
    causal: bool = True

    @property
    def n_params_estimate(self) -> int:
        emb = (VOCAB_SIZE + self.max_len + self.slots_per_bar) * self.d_model
        blk = self.n_layers * (4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff)
        return emb + blk


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.causal = cfg.causal

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        """x (B, T, D); key_padding (B, T) True where the slot is PAD."""
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        shape = (B, T, self.n_heads, self.d_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # (B, 1, 1, T) additive mask: padded keys are unattendable
        attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(key_padding[:, None, None, :], float("-inf"))
        if self.causal:
            causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device,
                                           dtype=x.dtype), diagonal=1)
            attn_mask = attn_mask + causal
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(h))
        return x + self.drop(self.ff(self.ln2(x)))


class ChordLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.slot = nn.Embedding(cfg.slots_per_bar, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight          # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, slot_ix: torch.Tensor | None = None
                ) -> torch.Tensor:
        """x (B, T) token ids -> logits (B, T, V)."""
        B, T = x.shape
        if slot_ix is None:
            slot_ix = (torch.arange(T, device=x.device) % self.cfg.slots_per_bar
                       ).expand(B, T)
        pos = torch.arange(T, device=x.device).expand(B, T)
        h = self.drop(self.tok(x) + self.pos(pos) + self.slot(slot_ix))
        key_padding = x.eq(PAD)
        for blk in self.blocks:
            h = blk(h, key_padding)
        return self.head(self.ln_f(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, prefix: list[int], n_slots: int, *, temperature: float = 1.0,
                 device: str = "cpu") -> list[int]:
        """Continue a chart (causal models only)."""
        assert self.cfg.causal, "generate() needs a causal model"
        self.eval()
        seq = list(prefix)
        for _ in range(n_slots):
            x = torch.tensor([seq[-self.cfg.max_len:]], device=device)
            slot_ix = (torch.arange(len(seq[-self.cfg.max_len:]), device=device)
                       % self.cfg.slots_per_bar).unsqueeze(0)
            logits = self(x, slot_ix)[0, -1] / max(temperature, 1e-6)
            logits[PAD] = -math.inf
            seq.append(int(torch.multinomial(F.softmax(logits, -1), 1)))
        return seq
