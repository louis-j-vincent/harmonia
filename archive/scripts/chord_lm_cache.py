"""How much is the within-song repetition cache worth on top of the LM?

    .venv/bin/python scripts/chord_lm_cache.py --model chord_lm_causal

Fits one mixing weight per match length on VAL, reports on TEST, and splits the
result into repeated-context and novel-context slots so it is visible that the
cache only ever helps where it has something to copy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, data, evaluate, model as M, repetition, vocab

WIDTHS = (16, 12, 8, 6, 4, 2)


@torch.no_grad()
def model_logprobs(net: M.ChordLM, seq: list[int], device: str) -> np.ndarray:
    """(T, V) log P(seq[i] | seq[:i]); row 0 is the model's unconditional prior."""
    T = len(seq)
    x = torch.tensor([seq], device=device)
    slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar).unsqueeze(0)
    lp = F.log_softmax(net(x, slot_ix), -1)[0].cpu().numpy()
    out = np.zeros((T, vocab.VOCAB_SIZE), dtype=np.float32)
    out[0] = -np.log(vocab.VOCAB_SIZE)
    out[1:] = lp[:-1]
    return out


def gather(net, seqs, device):
    cache = repetition.RepetitionCache(widths=WIDTHS)
    lps, cps, mts, ys = [], [], [], []
    for s in seqs:
        if len(s) < 4:
            continue
        lp = model_logprobs(net, s, device)
        cp, mt = cache.distributions(s)
        lps.append(lp[1:]); cps.append(cp[1:]); mts.append(mt[1:])
        ys.append(np.array(s[1:], dtype=np.int64))
    return lps, cps, mts, ys


def score(lps, cps, mts, ys, lam) -> evaluate.Metrics:
    acc = evaluate.MetricAccumulator()
    for lp, cp, mt, y in zip(lps, cps, mts, ys):
        acc.add(repetition.mix(lp, cp, mt, lam) if lam else lp, y)
    return acc.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chord_lm_causal")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = M.ChordLM.from_checkpoint(Path("data/models") / f"{args.model}.pt", device)
    print(f"model {args.model}  (rope={net.cfg.rope})")

    splits = corpus.load_splits(slots_per_bar=2)
    va = [c for ch in splits.val for c in data.chunk(ch, 256)]
    te = [c for ch in splits.test for c in data.chunk(ch, 256)]

    print("scoring val...")
    v = gather(net, va, device)
    lam = repetition.fit_lambdas(*v, widths=WIDTHS)
    print("  fitted mixing weight per match length (0 = cache ignored):")
    for w in WIDTHS:
        print(f"    {w:>2} slots = {w//2:>2} bars matched -> lambda {lam[w]:.2f}")

    print("\nscoring test...")
    t = gather(net, te, device)
    print(evaluate.HEADER)
    print("-" * 100)
    base = score(*t, None)
    print(base.table_row("LM alone"))
    mixed = score(*t, lam)
    print(mixed.table_row("LM + repetition cache"))
    print(mixed.table_row("  delta").replace("delta", "delta") if False else "")

    d_tok = 100 * (mixed.token_acc - base.token_acc)
    d_id = 100 * (mixed.chord_acc_given_change - base.chord_acc_given_change)
    print(f"\n  delta: token {d_tok:+.2f}pp   identity|change {d_id:+.2f}pp   "
          f"ppl {base.ppl:.3f} -> {mixed.ppl:.3f}")

    # where does it act?
    n_match = sum(int((mt > 0).sum()) for mt in t[2])
    n_tot = sum(len(mt) for mt in t[2])
    print(f"  cache had something to copy on {n_match}/{n_tot} slots "
          f"({100*n_match/n_tot:.1f}%)")

    acc_m = evaluate.MetricAccumulator()
    acc_n = evaluate.MetricAccumulator()
    for lp, cp, mt, y in zip(*t):
        mixed_lp = repetition.mix(lp, cp, mt, lam)
        sel = mt > 0
        if sel.any():
            acc_m.add(mixed_lp[sel], y[sel])
        if (~sel).any():
            acc_n.add(mixed_lp[~sel], y[~sel])
    print(f"  on matched slots  : {100*acc_m.result().token_acc:.2f}% token acc")
    print(f"  on unmatched slots: {100*acc_n.result().token_acc:.2f}% token acc")


if __name__ == "__main__":
    main()
