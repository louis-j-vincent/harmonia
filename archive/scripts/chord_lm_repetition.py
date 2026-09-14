"""Does whole-song context actually buy anything? (Louis, 2026-08-01)

The expectation under test: "un LLM qui a comme contexte toute la chanson
devrait facilement refaire les patterns" — a model that can see the whole song
should nail the repeats, because bar 33 of an AABA tune is bar 1 again.

Two measurements, both on the held-out test songs:

  A. ACCURACY vs AVAILABLE CONTEXT. Re-score every slot with only the last
     k slots visible, k = 2, 4, 8, 16, 32, 64, all. If long context matters, the
     curve keeps climbing past one section (16 slots = 8 bars). If it flattens
     early, the model is really a local-grammar model and whole-song context is
     decoration.

  B. ACCURACY on REPEATED vs NOVEL context. A slot is "repeated" when the 8
     slots (4 bars) before it have already occurred verbatim earlier in the SAME
     song — i.e. the answer is literally copyable from earlier in the chart.
     Louis's claim predicts near-ceiling accuracy on that subset.

The split matters for the pipeline: if the model's strength is copying repeats,
it helps most exactly where our acoustic model is weakest (later choruses, solos)
— and it means the honest headline number is the NOVEL-context one, because the
repeated-context number is partly bookkeeping rather than harmony.

    .venv/bin/python scripts/chord_lm_repetition.py --model chord_lm_run1_causal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, data, evaluate, model as M, vocab

CTX_SIG = 8   # 4 bars — the window that defines "we have seen this before"


load = M.ChordLM.from_checkpoint


@torch.no_grad()
def score_with_context(net: M.ChordLM, seqs: list[list[int]], device: str,
                       k: int | None) -> tuple[evaluate.Metrics, np.ndarray]:
    """Predict each slot from at most the previous `k` slots (None = all).

    Each window keeps its TRUE absolute positions (`pos_ix`) and its TRUE
    slot-in-bar indices. Left-padding a mid-song window to position 0 would tell
    the model it is looking at the start of a tune, and the ablation would then
    measure that distribution shift instead of the loss of context.
    """
    acc = evaluate.MetricAccumulator()
    correct_flags: list[np.ndarray] = []
    for s in seqs:
        T = len(s)
        if T < 2:
            continue
        if k is None:
            x = torch.tensor([s], device=device)
            slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                       ).unsqueeze(0)
            lp = F.log_softmax(net(x, slot_ix), -1)[0, :-1].cpu().numpy()
            tgt = np.array(s[1:], dtype=np.int64)
        else:
            # one window per predicted position, each holding the last k slots
            wins, slots, tgts = [], [], []
            for i in range(1, T):
                lo = max(0, i - k)
                w = s[lo:i]
                wins.append(w)
                slots.append(list(range(lo, i)))
                tgts.append(s[i])
            W = max(len(w) for w in wins)
            X = torch.full((len(wins), W), vocab.PAD, dtype=torch.long)
            S = torch.zeros((len(wins), W), dtype=torch.long)
            P = torch.zeros((len(wins), W), dtype=torch.long)
            last = []
            for j, (w, sl) in enumerate(zip(wins, slots)):
                # left-pad so the predicted slot is always the final position,
                # but carry the window's real positions in S (slot-in-bar) and
                # P (absolute), so only the CONTEXT is ablated
                X[j, W - len(w):] = torch.tensor(w)
                S[j, W - len(w):] = torch.tensor([p % net.cfg.slots_per_bar
                                                  for p in sl])
                P[j, W - len(w):] = torch.tensor(sl)
                last.append(W - 1)
            X, S, P = X.to(device), S.to(device), P.to(device)
            out = []
            for b0 in range(0, len(wins), 256):
                sl = slice(b0, b0 + 256)
                o = F.log_softmax(net(X[sl], S[sl], P[sl]), -1)
                rows = torch.arange(o.shape[0], device=device)
                out.append(o[rows, torch.tensor(last[sl], device=device)].cpu().numpy())
            lp = np.concatenate(out, 0)
            tgt = np.array(tgts, dtype=np.int64)
        acc.add(lp, tgt)
        correct_flags.append(lp.argmax(1) == tgt)
    return acc.result(), correct_flags


def repeated_mask(seq: list[int], w: int = CTX_SIG) -> np.ndarray:
    """For each predicted position i (1..T-1): has seq[i-w:i] occurred earlier?"""
    T = len(seq)
    out = np.zeros(T - 1, dtype=bool)
    seen: dict[tuple, int] = {}
    for i in range(1, T):
        lo = max(0, i - w)
        sig = tuple(seq[lo:i])
        if len(sig) == w and sig in seen:
            out[i - 1] = True
        seen.setdefault(sig, i)
    return out


def copy_prediction(seq: list[int], w: int = CTX_SIG) -> list[int | None]:
    """Baseline: for position i, what followed the FIRST earlier occurrence of
    the preceding `w` slots? None when there is no earlier occurrence."""
    T = len(seq)
    out: list[int | None] = [None] * (T - 1)
    seen: dict[tuple, int] = {}
    for i in range(1, T):
        lo = max(0, i - w)
        sig = tuple(seq[lo:i])
        if len(sig) == w:
            j = seen.get(sig)
            if j is not None and j < T:
                out[i - 1] = seq[j]
            seen.setdefault(sig, i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chord_lm_run1_causal")
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load(Path("data/models") / f"{args.model}.pt", device)

    splits = corpus.load_splits(slots_per_bar=2)
    te = [c for ch in splits.test for c in data.chunk(ch, args.max_len)]
    print(f"model {args.model}  |  {len(te)} test sequences\n")

    print("A. accuracy vs available context")
    print(f"   {'context':<22} {'ppl':>7} {'tok%':>7} {'chg%':>7} {'id|chg%':>8}")
    full_flags = None
    for k in (2, 4, 8, 16, 32, 64, None):
        m, flags = score_with_context(net, te, device, k)
        label = "whole song" if k is None else f"{k} slots = {k//2} bars"
        print(f"   {label:<22} {m.ppl:7.3f} {100*m.token_acc:7.2f} "
              f"{100*m.change_acc:7.2f} {100*m.chord_acc_given_change:8.2f}")
        if k is None:
            full_flags = flags

    print("\nB. repeated vs novel context (whole-song context, 4-bar signature)")
    rep_ok = rep_n = nov_ok = nov_n = 0
    copy_ok = copy_n = 0
    for seq, flags in zip([s for s in te if len(s) >= 2], full_flags):
        rm = repeated_mask(seq)
        cp = copy_prediction(seq)
        n = min(len(rm), len(flags))
        rm, fl = rm[:n], flags[:n]
        rep_ok += int(fl[rm].sum());  rep_n += int(rm.sum())
        nov_ok += int(fl[~rm].sum()); nov_n += int((~rm).sum())
        for i in range(n):
            if rm[i] and cp[i] is not None:
                copy_n += 1
                copy_ok += int(cp[i] == seq[i + 1])
    tot = rep_n + nov_n
    print(f"   repeated context : {100*rep_ok/max(rep_n,1):6.2f}%  "
          f"({rep_n} slots, {100*rep_n/tot:.1f}% of all)")
    print(f"   novel context    : {100*nov_ok/max(nov_n,1):6.2f}%  "
          f"({nov_n} slots, {100*nov_n/tot:.1f}% of all)")
    print(f"   overall          : {100*(rep_ok+nov_ok)/tot:6.2f}%")
    print(f"\n   CONTROL — pure copy rule on the repeated subset: "
          f"{100*copy_ok/max(copy_n,1):6.2f}%  ({copy_n} slots)")
    print("   (find the earliest earlier occurrence of the same 4 bars, copy what")
    print("    followed it. This is the signal the model should already be getting")
    print("    for free; the gap between it and the model's repeated-context score")
    print("    is what a repetition mechanism would still be worth.)")


if __name__ == "__main__":
    main()
