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
    rope: bool = True
    """Rotary (relative) positions instead of a learned absolute table.

    Measured, 2026-08-01: with absolute positions the model scores 72.1% on
    half-bars whose previous 4 bars already occurred verbatim earlier in the same
    song, while a dumb "copy what followed last time" rule scores 91.3%. Charts
    are built out of literal repeats (AABA, chorus 2 = chorus 1), so that gap is
    most of what a chord LM should find easy. Absolute positions make
    "match this pattern somewhere earlier" a different attention pattern for
    every offset; relative positions make it ONE pattern, which is the standard
    prerequisite for the induction heads that do the copying.
    """

    @property
    def n_params_estimate(self) -> int:
        emb = (VOCAB_SIZE + self.max_len + self.slots_per_bar) * self.d_model
        blk = self.n_layers * (4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff)
        return emb + blk


def build_rope(pos_ix: torch.Tensor, d_head: int, base: float = 10000.0
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T) positions -> (cos, sin), each (B, T, d_head//2)."""
    half = d_head // 2
    inv = base ** (-torch.arange(half, device=pos_ix.device,
                                 dtype=torch.float32) / half)
    ang = pos_ix.float().unsqueeze(-1) * inv
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
               ) -> torch.Tensor:
    """Rotate (B, H, T, D) queries/keys; cos/sin are (B, T, D//2)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos, sin = cos.unsqueeze(1).to(x.dtype), sin.unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


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

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor:
        """x (B, T, D); key_padding (B, T) True where the slot is PAD."""
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        shape = (B, T, self.n_heads, self.d_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
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

    def forward(self, x: torch.Tensor, slot_ix: torch.Tensor | None = None,
                pos_ix: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, T) token ids -> logits (B, T, V).

        `pos_ix` overrides the absolute slot positions. It exists for the
        context-length ablation: scoring a mid-song window must keep that
        window's TRUE positions, otherwise the model is told it is at the start
        of a song and the ablation measures a distribution shift rather than a
        loss of context.
        """
        B, T = x.shape
        if slot_ix is None:
            slot_ix = (torch.arange(T, device=x.device) % self.cfg.slots_per_bar
                       ).expand(B, T)
        if pos_ix is None:
            pos_ix = torch.arange(T, device=x.device).expand(B, T)
        h = self.tok(x) + self.slot(slot_ix)
        rope = None
        if self.cfg.rope:
            rope = build_rope(pos_ix, self.cfg.d_model // self.cfg.n_heads)
        else:
            h = h + self.pos(pos_ix)
        h = self.drop(h)
        key_padding = x.eq(PAD)
        for blk in self.blocks:
            h = blk(h, key_padding, rope)
        return self.head(self.ln_f(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def from_checkpoint(path, device: str = "cpu") -> "ChordLM":
        """Load a saved model, honouring the positional scheme it was TRAINED with.

        Checkpoints written before rotary positions existed have no `rope` key.
        Defaulting those to the current default (True) would hand a model trained
        with learned absolute positions a completely different position signal at
        inference and still load without error — a silent calibration bug of
        exactly the kind CLAUDE.md error-pattern #1 is about. Absent key => False.
        """
        blob = torch.load(path, map_location=device, weights_only=False)
        cfg_d = dict(blob["cfg"])
        cfg_d.setdefault("rope", False)
        known = set(LMConfig.__dataclass_fields__)
        net = ChordLM(LMConfig(**{k: v for k, v in cfg_d.items() if k in known}))
        net.load_state_dict(blob["state"])
        return net.to(device).eval()

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
