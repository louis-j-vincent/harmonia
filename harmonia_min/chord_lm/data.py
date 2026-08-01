"""harmonia_min/chord_lm/data.py — batching, augmentation, masking.

Sequences are per-SONG (never concatenated across songs: a chart's last bar has
nothing to say about the next tune's first). Songs longer than `max_len` are cut
into chunks aligned to bar boundaries, so slot parity — and therefore the
slot-in-bar embedding — stays meaningful in every chunk.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .grid import GriddedChart
from .vocab import BOS, EOS, MASK, N_CHORD, PAD, REP, VOCAB_SIZE, transpose


def random_transpose(tokens: list[int], semitones: int) -> list[int]:
    return [transpose(t, semitones) for t in tokens]


def chunk(chart: GriddedChart, max_len: int) -> list[list[int]]:
    """A chart's tokens as one or more sequences of at most `max_len` slots.

    BOS/EOS are NOT added: they would break slot parity (BOS is not a musical
    slot). Sequence start is instead marked implicitly — position 0 of the
    learned positional embedding — and the first slot is never REP by
    construction, so there is nothing dangling to resolve.
    """
    s = chart.slots_per_bar
    step = (max_len // s) * s
    if step == 0:
        return []
    out = []
    for i in range(0, len(chart.tokens), step):
        piece = chart.tokens[i:i + step]
        if len(piece) >= 2 * s:            # at least 2 bars to be worth training on
            out.append(piece)
    return out


@dataclass
class Batch:
    x: torch.Tensor           # (B, T) model input (may contain MASK)
    y: torch.Tensor           # (B, T) targets, PAD where no loss
    slot_ix: torch.Tensor     # (B, T) slot index within the bar
    orig: torch.Tensor        # (B, T) the un-masked, un-shifted tokens

    def to(self, device) -> "Batch":
        return Batch(*(t.to(device) for t in (self.x, self.y, self.slot_ix, self.orig)))


class ChordDataset:
    """Sequences + collation. `causal` picks the objective."""

    def __init__(self, charts: list[GriddedChart], *, max_len: int = 256,
                 slots_per_bar: int = 2, causal: bool = True,
                 augment: bool = True, mask_prob: float = 0.30,
                 min_mask_prob: float = 0.02, seed: int = 0):
        self.seqs = [c for ch in charts for c in chunk(ch, max_len)]
        self.max_len = max_len
        self.slots_per_bar = slots_per_bar
        self.causal = causal
        self.augment = augment
        self.mask_prob = mask_prob
        self.min_mask_prob = min_mask_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.seqs)

    def _collate(self, seqs: list[list[int]]) -> Batch:
        T = max(len(s) for s in seqs)
        B = len(seqs)
        orig = torch.full((B, T), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            orig[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        slot_ix = (torch.arange(T) % self.slots_per_bar).expand(B, T).clone()
        pad = orig.eq(PAD)

        if self.causal:
            x = orig.clone()
            y = torch.full((B, T), PAD, dtype=torch.long)
            y[:, :-1] = orig[:, 1:]          # predict the NEXT slot
            y[pad] = PAD
            y[:, -1] = PAD
            # never ask the model to predict a padded next-slot
            y[torch.roll(pad, -1, dims=1)] = PAD
        else:
            x = orig.clone()
            y = torch.full((B, T), PAD, dtype=torch.long)
            # Variable mask rate, sampled per batch. A fixed 15% would train the
            # model only on a regime the downstream task never sees: at
            # inference we mask ONE slot out of ~88 (~1%) and want the model to
            # exploit an almost-complete context. `mask_prob` is the upper end.
            rate = self.rng.uniform(self.min_mask_prob, self.mask_prob)
            r = torch.rand(B, T)
            sel = (r < rate) & ~pad
            if sel.sum() == 0:               # degenerate batch: force one target
                sel[0, 0] = ~pad[0, 0]
            y[sel] = orig[sel]
            # 80/10/10 BERT recipe: MASK / random chord / keep
            r2 = torch.rand(B, T)
            x[sel & (r2 < 0.8)] = MASK
            rand_sel = sel & (r2 >= 0.8) & (r2 < 0.9)
            if rand_sel.any():
                x[rand_sel] = torch.randint(0, N_CHORD, (int(rand_sel.sum()),))
        return Batch(x=x, y=y, slot_ix=slot_ix, orig=orig)

    def batches(self, batch_size: int, *, shuffle: bool = True):
        idx = list(range(len(self.seqs)))
        if shuffle:
            self.rng.shuffle(idx)
        # length-bucket so padding stays small
        idx.sort(key=lambda i: len(self.seqs[i]) // 32)
        groups = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
        if shuffle:
            self.rng.shuffle(groups)
        for g in groups:
            seqs = [self.seqs[i] for i in g]
            if self.augment:
                seqs = [random_transpose(s, self.rng.randrange(12)) for s in seqs]
            yield self._collate(seqs)
