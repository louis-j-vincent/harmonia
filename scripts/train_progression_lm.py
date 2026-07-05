"""Chord-progression language model over scale-relative degrees (user's idea).

Represent every chord as its degree within the (ground-truth) key + its quality —
a transposition-invariant, functional token (e.g. "II:min7", "V:dom7", "I:maj7").
Then ask: given the last few chords, how well can we predict the next one? Compare

  unigram     — always guess the most common chord (floor)
  bigram      — P(next | previous) table (what the old 'progression' test used)
  MLP(last-4) — feedforward net on the last 4 chords
  LSTM        — recurrent net over the whole history (the real sequence model)

on the full jazz corpus (symbolic, ~1450 songs), split by song. Reports next-chord
top-1 / top-3, plus degree-only and quality-only accuracy (predicting the next
root vs the next quality). This measures whether progression is a good *indicator*
and whether a sequence model beats a bigram.

Usage: .venv/bin/python scripts/train_progression_lm.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import merged_events, parse_key  # noqa: E402

torch.manual_seed(0)
DB = REPO / "data" / "accomp_db" / "db.jsonl"
BUCKET_BASE7 = {
    "maj": "majT", "6": "majT", "maj7": "maj7", "dom7": "dom7", "dom7alt": "dom7",
    "min": "minT", "m6": "minT", "min7": "min7", "minmaj7": "minmaj7",
    "dim": "dimT", "dim7": "dim7", "m7b5": "m7b5",
    "aug": "augT", "aug7": "aug7", "augmaj7": "augmaj7",
    "sus2": "susT", "sus4": "susT", "7sus4": "7sus4",
}
DEG_NAMES = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]


def build_sequences():
    seqs = []
    for rec in map(json.loads, open(DB)):
        if rec["corpus"] != "jazz1460":
            continue
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, _ = k
        seq = []
        for chord, _, _ in merged_events(rec):
            parsed = parse_chord(chord)
            if parsed is None or parsed[1] not in BUCKET_BASE7:
                continue
            deg = (parsed[0] - tonic) % 12
            seq.append((deg, BUCKET_BASE7[parsed[1]]))
        if len(seq) >= 4:
            seqs.append((rec["song_id"], seq))
    return seqs


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb=48, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x):
        return self.out(self.lstm(self.emb(x))[0])


class MLPLM(nn.Module):
    def __init__(self, vocab, k=4, emb=48, hid=256):
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(vocab + 1, emb)  # +1 = pad
        self.net = nn.Sequential(nn.Linear(emb * k, hid), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hid, vocab))

    def forward(self, ctx):
        return self.net(self.emb(ctx).flatten(1))


def main():
    seqs = build_sequences()
    vocab = {}
    for _, s in seqs:
        for t in s:
            vocab.setdefault(t, len(vocab))
    inv = {v: k for k, v in vocab.items()}
    V = len(vocab)
    n_trans = sum(len(s) - 1 for _, s in seqs)
    print(f"{len(seqs)} jazz songs, {n_trans} chord transitions, vocab {V} "
          f"scale-relative chord types\n")

    rng = np.random.default_rng(0)
    train = [(sid, s) for sid, s in seqs if rng.random() < 0.8]
    test = [(sid, s) for sid, s in seqs if (sid, s) not in train]
    test = [(sid, s) for sid, s in seqs if sid not in {t[0] for t in train}]

    PAD = V

    def top_acc(logits, target, k=1):
        topk = logits.topk(k, dim=-1).indices
        return (topk == target.unsqueeze(-1)).any(-1).float().mean().item()

    # ── bigram + unigram baselines ────────────────────────────────────────────
    bg = defaultdict(Counter)
    ug = Counter()
    for _, s in train:
        for a, b in zip(s, s[1:]):
            bg[vocab[a]][vocab[b]] += 1
        for t in s:
            ug[vocab[t]] += 1
    ug_top = ug.most_common(1)[0][0]
    bg_best = {a: c.most_common(1)[0][0] for a, c in bg.items()}
    n = ok_u = ok_b = 0
    for _, s in test:
        for a, b in zip(s, s[1:]):
            n += 1
            ok_u += ug_top == vocab[b]
            ok_b += bg_best.get(vocab[a], ug_top) == vocab[b]
    print(f"{'model':<16}{'next top-1':>12}{'next top-3':>12}{'degree top-1':>14}{'quality top-1':>15}")
    print("-" * 69)
    print(f"{'unigram':<16}{ok_u/n:>12.1%}{'-':>12}{'-':>14}{'-':>15}")
    print(f"{'bigram':<16}{ok_b/n:>12.1%}{'-':>12}{'-':>14}{'-':>15}")

    # ── neural models ─────────────────────────────────────────────────────────
    def make_lstm_batch(data):
        xs, ys = [], []
        for _, s in data:
            ids = [vocab[t] for t in s]
            xs.append(ids[:-1]); ys.append(ids[1:])
        return xs, ys

    def eval_seq_model(model, is_lstm):
        model.eval()
        tot = 0; t1 = t3 = dok = qok = 0
        with torch.no_grad():
            for _, s in test:
                ids = [vocab[t] for t in s]
                if is_lstm:
                    x = torch.tensor([ids[:-1]])
                    logits = model(x)[0]           # (L-1, V)
                    tgt = torch.tensor(ids[1:])
                else:
                    logits, tgt = [], []
                    for i in range(1, len(ids)):
                        ctx = ids[max(0, i - 4):i]
                        ctx = [PAD] * (4 - len(ctx)) + ctx
                        logits.append(model(torch.tensor([ctx]))[0])
                        tgt.append(ids[i])
                    logits = torch.stack(logits); tgt = torch.tensor(tgt)
                tot += len(tgt)
                t1 += top_acc(logits, tgt, 1) * len(tgt)
                t3 += top_acc(logits, tgt, 3) * len(tgt)
                pred = logits.argmax(-1)
                for p, g in zip(pred.tolist(), tgt.tolist()):
                    dok += inv[p][0] == inv[g][0]
                    qok += inv[p][1] == inv[g][1]
        return t1 / tot, t3 / tot, dok / tot, qok / tot

    # LSTM
    lstm = LSTMLM(V)
    opt = torch.optim.Adam(lstm.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    xs, ys = make_lstm_batch(train)
    for _ in range(15):
        order = np.random.permutation(len(xs))
        for i in order:
            opt.zero_grad()
            x = torch.tensor([xs[i]]); y = torch.tensor([ys[i]])
            loss = lossf(lstm(x)[0], y[0])
            loss.backward(); opt.step()
    a = eval_seq_model(lstm, True)
    print(f"{'LSTM':<16}{a[0]:>12.1%}{a[1]:>12.1%}{a[2]:>14.1%}{a[3]:>15.1%}")

    # MLP(last-4)
    mlp = MLPLM(V, k=4)
    opt = torch.optim.Adam(mlp.parameters(), lr=2e-3)
    ctxs, tgts = [], []
    for _, s in train:
        ids = [vocab[t] for t in s]
        for i in range(1, len(ids)):
            c = ids[max(0, i - 4):i]
            ctxs.append([PAD] * (4 - len(c)) + c); tgts.append(ids[i])
    ctxs = torch.tensor(ctxs); tgts = torch.tensor(tgts)
    for _ in range(25):
        perm = torch.randperm(len(ctxs))
        for j in range(0, len(ctxs), 512):
            idx = perm[j:j + 512]
            opt.zero_grad()
            loss = lossf(mlp(ctxs[idx]), tgts[idx])
            loss.backward(); opt.step()
    a = eval_seq_model(mlp, False)
    print(f"{'MLP(last-4)':<16}{a[0]:>12.1%}{a[1]:>12.1%}{a[2]:>14.1%}{a[3]:>15.1%}")

    print("\nDegree top-1 = predicting the next chord's ROOT (scale degree); "
          "quality top-1 = its quality.\nIf a sequence model clearly beats the "
          "bigram, longer progression context carries real signal.")


if __name__ == "__main__":
    main()
