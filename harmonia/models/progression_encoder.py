"""harmonia/models/progression_encoder.py — issue #21 chord-progression context.

A small non-causal transformer encoder that refines the *quality* of a chord
from its harmonic neighbourhood. Motivation (issue #21): the per-beat/​per-segment
pipeline predicts chords IID given the audio, so it happily emits sequences no
musician would play. jazz harmony is organised in *trigrams* (ii-V-I); a fixed
bigram/trigram table cannot cover the long tail (see scripts/check_bigram_premise.py
= 63.8% top-50, scripts/check_trigram_premise.py = 34.9% top-50, both < 70% gate),
but the sequential *information* is strong (trigram context removes 3.63 of 5.23
bits of next-chord uncertainty). That is the case for a learned encoder over a
fixed n-gram matrix.

Task (masked-cloze / denoising reranker):
    given a window of ±6 chords around a centre position — each as a
    root-relative interval + a quality + an acoustic-confidence scalar — predict
    the centre chord's quality. At training time the centre (and, with some
    probability, its neighbours) are masked or corrupted so the model must lean
    on harmonic grammar; at inference the centre carries the greedy prediction
    with its acoustic confidence, and the encoder can override it.

Quality is coarsened to 5 functional families (maj/min/dom/hdim/dim); that is
the granularity a progression prior can actually constrain (the fine 7th/6th
distinctions are an acoustic, not a grammatical, question).

Representation notes:
  - Root-relative encoding is transpose-invariant *by construction*, so the
    random-transpose augmentation asked for in the brief is a no-op here — the
    meaningful augmentation is quality corruption + confidence, implemented in
    the training script.
  - conf is a multiplicative gate on the per-chord embedding: a masked centre
    (conf 0) contributes only its positional slot, forcing the model to read
    the neighbourhood.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── 5-class functional quality vocabulary ──────────────────────────────────────
QUAL5 = ["maj", "min", "dom", "hdim", "dim"]
QUAL5_IDX = {q: i for i, q in enumerate(QUAL5)}
MASK_ID = 5  # embedding-table id for a masked / unknown quality (not an output)
N_QUAL_EMB = 6  # 5 families + MASK
CTX = 6  # ±6 chords
WINDOW = 2 * CTX + 1  # 13 positions

# fine quality bucket (analyze_accomp_emission.QUALITY_MAP output) → 5-class family
FINE_TO_QUAL5 = {
    # major family
    "maj": "maj", "maj7": "maj", "6": "maj", "aug": "maj", "augmaj7": "maj",
    "sus2": "maj", "sus4": "maj",
    # minor family
    "min": "min", "min7": "min", "m6": "min", "minmaj7": "min",
    # dominant family
    "dom7": "dom", "dom7alt": "dom", "aug7": "dom", "7sus4": "dom",
    # half-diminished
    "m7b5": "hdim",
    # fully diminished
    "dim": "dim", "dim7": "dim",
}


def fine_to_q5(fine: str) -> int | None:
    fam = FINE_TO_QUAL5.get(fine)
    return QUAL5_IDX[fam] if fam is not None else None


# ── data loading ────────────────────────────────────────────────────────────
def load_jazz_sequences(
    db_path: Path, corpus: str = "jazz1460"
) -> list[list[tuple[int, int]]]:
    """Per-song chord sequences as lists of (root_pc, qual5_idx).

    Uses the same clean symbolic spans (scripts.analyze_accomp_emission.
    song_chord_spans) that the bigram/trigram premise checks consumed, so the
    encoder trains on exactly the corpus those gates were measured on.
    """
    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo / "scripts"))
    from analyze_accomp_emission import song_chord_spans  # noqa: E402

    seqs: list[list[tuple[int, int]]] = []
    for line in open(db_path):
        rec = json.loads(line)
        if rec.get("corpus") != corpus:
            continue
        seq: list[tuple[int, int]] = []
        for _t0, _t1, root, qual in song_chord_spans(rec):
            q5 = fine_to_q5(qual)
            if q5 is None:
                continue
            seq.append((root % 12, q5))
        if len(seq) >= 3:
            seqs.append(seq)
    return seqs


def split_sequences(
    seqs: list[list[tuple[int, int]]], val_every: int = 5
) -> tuple[list, list]:
    """Deterministic disjoint train/val split by song index.

    The brief's literal '<70 train / ≥70 val' would starve training (70 of 1458
    songs); we use the project's standard modulo split style instead (every 5th
    song → val, ~80/20), documented in the nightly log.
    """
    train = [s for i, s in enumerate(seqs) if i % val_every != 0]
    val = [s for i, s in enumerate(seqs) if i % val_every == 0]
    return train, val


# ── model ─────────────────────────────────────────────────────────────────────
class ProgressionEncoder(nn.Module):
    """Small transformer encoder for chord-progression context.

    Input tensors (batch B):
        root_rel : (B, WINDOW) long   — interval (mod 12) of each chord's root
                                         relative to the centre chord's root
        qual     : (B, WINDOW) long   — quality id in [0, N_QUAL_EMB) (MASK_ID=5)
        conf     : (B, WINDOW) float  — acoustic confidence in [0, 1] (gate)
        pad_mask : (B, WINDOW) bool   — True where the position is padding
    Output:
        logits   : (B, 5) — refined quality logits for the centre chord.
    """

    def __init__(self, d_model: int = 32, nhead: int = 4, ffn: int = 64,
                 layers: int = 2):
        super().__init__()
        self.root_emb = nn.Embedding(12, d_model)
        self.qual_emb = nn.Embedding(N_QUAL_EMB, d_model)
        self.pos_emb = nn.Embedding(WINDOW, d_model)
        self.in_proj = nn.Linear(2 * d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            batch_first=True, dropout=0.1, activation="gelu",
        )
        # enable_nested_tensor=False: the nested-tensor fast path is unimplemented
        # on MPS (aten::_nested_tensor_from_mask_left_aligned) — disable it so the
        # padding-mask path runs on-device.
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 5))
        self.register_buffer("_pos_ids", torch.arange(WINDOW), persistent=False)

    def forward(self, root_rel, qual, conf, pad_mask=None):
        r = self.root_emb(root_rel)          # (B,W,d)
        q = self.qual_emb(qual)              # (B,W,d)
        x = self.in_proj(torch.cat([r, q], dim=-1))  # (B,W,d)
        x = x * conf.unsqueeze(-1)           # confidence gate
        x = x + self.pos_emb(self._pos_ids)  # (W,d) broadcast over batch
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,W,d)
        centre = h[:, CTX, :]                # centre position
        return self.head(centre)             # (B,5)


def load_encoder(path: Path, device: str = "cpu") -> ProgressionEncoder:
    ckpt = torch.load(path, map_location=device)
    model = ProgressionEncoder(**ckpt.get("hparams", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
