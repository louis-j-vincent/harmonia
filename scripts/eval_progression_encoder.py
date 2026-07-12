"""Standalone evaluation of the chord ProgressionEncoder (issue #21).

Masked-cloze quality task on the jazz1460 val split (project modulo split, every
5th song): mask the centre chord's quality, predict it from the ±6 neighbourhood.
This is the symbolic analogue of the issue-#18 "vacuum bake-off" — it isolates
how much of a chord's *quality* is recoverable from harmonic grammar alone,
before any acoustics.

Baselines (extending the premise-check story):
  - majority    : always predict the most common quality (unigram prior)
  - bigram      : argmax P(q_centre | interval, q_prev) from train counts
  - trigram     : argmax P(q_centre | q_prev2, iv1, q_prev, iv2) w/ bigram backoff
  - encoder     : ProgressionEncoder centre-cloze prediction

All on the same held-out songs, 5-class (maj/min/dom/hdim/dim).

    .venv/bin/python scripts/eval_progression_encoder.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.models.progression_encoder import (  # noqa: E402
    CTX, MASK_ID, QUAL5, WINDOW, load_encoder, load_jazz_sequences,
    split_sequences,
)

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CKPT = REPO / "harmonia" / "models" / "progression_encoder.pt"
N_QUAL = 5


def cloze_positions(seqs):
    """Yield (seq, i) for every centre position with a defined target."""
    for seq in seqs:
        for i in range(len(seq)):
            yield seq, i


# ── n-gram baselines fit on the train split ─────────────────────────────────────
def fit_ngram(train_seqs):
    uni = Counter()
    bi = defaultdict(Counter)     # key (iv1, q_prev) -> q_centre
    tri = defaultdict(Counter)    # key (q_prev2, iv1, q_prev, iv2)... predict q_c
    for seq in train_seqs:
        for i in range(len(seq)):
            rc, qc = seq[i]
            uni[qc] += 1
            if i >= 1:
                rp, qp = seq[i - 1]
                bi[((rc - rp) % 12, qp)][qc] += 1
            if i >= 2:
                rp2, qp2 = seq[i - 2]
                rp, qp = seq[i - 1]
                tri[(qp2, (rp - rp2) % 12, qp, (rc - rp) % 12)][qc] += 1
    uni_pred = uni.most_common(1)[0][0]
    return uni_pred, bi, tri


def predict_ngram(seq, i, uni_pred, bi, tri):
    rc, _ = seq[i]
    # trigram with backoff
    if i >= 2:
        rp2, qp2 = seq[i - 2]
        rp, qp = seq[i - 1]
        key = (qp2, (rp - rp2) % 12, qp, (rc - rp) % 12)
        if tri.get(key):
            tri_p = tri[key].most_common(1)[0][0]
        else:
            tri_p = None
    else:
        tri_p = None
    if i >= 1:
        rp, qp = seq[i - 1]
        bkey = ((rc - rp) % 12, qp)
        bi_p = bi[bkey].most_common(1)[0][0] if bi.get(bkey) else uni_pred
    else:
        bi_p = uni_pred
    return bi_p, (tri_p if tri_p is not None else bi_p)


def encoder_batch(seq, i):
    n = len(seq)
    rc = seq[i][0]
    root_rel = np.zeros(WINDOW, dtype=np.int64)
    qual = np.full(WINDOW, MASK_ID, dtype=np.int64)
    conf = np.zeros(WINDOW, dtype=np.float32)
    pad = np.ones(WINDOW, dtype=bool)
    for p in range(WINDOW):
        j = i + (p - CTX)
        if 0 <= j < n:
            pad[p] = False
            root_rel[p] = (seq[j][0] - rc) % 12
            qual[p] = seq[j][1]
            conf[p] = 1.0
    qual[CTX] = MASK_ID
    conf[CTX] = 0.0
    return root_rel, qual, conf, pad


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seqs = load_jazz_sequences(DB)
    train_seqs, val_seqs = split_sequences(seqs)
    print(f"jazz1460: {len(train_seqs)} train / {len(val_seqs)} val songs")

    positions = list(cloze_positions(val_seqs))
    targets = np.array([seq[i][1] for seq, i in positions])
    print(f"val cloze positions: {len(positions)}")

    uni_pred, bi, tri = fit_ngram(train_seqs)

    # n-gram baselines
    bi_correct = tri_correct = uni_correct = 0
    for seq, i in positions:
        bp, tp = predict_ngram(seq, i, uni_pred, bi, tri)
        uni_correct += (uni_pred == seq[i][1])
        bi_correct += (bp == seq[i][1])
        tri_correct += (tp == seq[i][1])
    n = len(positions)

    # encoder
    model = load_encoder(CKPT, device)
    enc_correct = 0
    B = 1024
    rr_b, q_b, c_b, p_b = [], [], [], []
    idx = 0
    preds = np.zeros(n, dtype=np.int64)
    for k, (seq, i) in enumerate(positions):
        rr, q, c, p = encoder_batch(seq, i)
        rr_b.append(rr); q_b.append(q); c_b.append(c); p_b.append(p)
        if len(rr_b) == B or k == n - 1:
            with torch.no_grad():
                logits = model(
                    torch.tensor(np.stack(rr_b)).to(device),
                    torch.tensor(np.stack(q_b)).to(device),
                    torch.tensor(np.stack(c_b)).to(device),
                    torch.tensor(np.stack(p_b)).to(device),
                )
            pr = logits.argmax(-1).cpu().numpy()
            preds[idx:idx + len(pr)] = pr
            idx += len(pr)
            rr_b, q_b, c_b, p_b = [], [], [], []
    enc_correct = int((preds == targets).sum())

    print(f"\n{'model':<12} {'cloze quality acc':>18}")
    print("-" * 32)
    print(f"{'majority':<12} {uni_correct / n:>17.1%}")
    print(f"{'bigram':<12} {bi_correct / n:>17.1%}")
    print(f"{'trigram':<12} {tri_correct / n:>17.1%}")
    print(f"{'ENCODER':<12} {enc_correct / n:>17.1%}")

    # per-class encoder recall + confusion
    print(f"\nEncoder per-class recall (5-class):")
    for qi, qn in enumerate(QUAL5):
        m = targets == qi
        if m.sum():
            print(f"  {qn:<5} n={m.sum():>5}  recall={np.mean(preds[m] == qi):.1%}")


if __name__ == "__main__":
    main()
