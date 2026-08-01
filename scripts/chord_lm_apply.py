"""Look at what the trained chord LM actually does — two inspectable artifacts.

    .venv/bin/python scripts/chord_lm_apply.py

1. GENERATE — the causal model writes a 16-bar chart from a 2-bar prompt. If it
   learned grammar, these read as music; if it learned bigram statistics, they
   wander. This is the cheapest honest check that is not a number.

2. SECOND-OPINION — the masked model fills in each half-bar of our own charts
   (harmonia_min/state/charts/*.json) from both-sided context, without seeing
   that slot. Where it is confident AND disagrees with the pipeline, it is
   proposing a fix. Those rows are printed for the ear to arbitrate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import model as M, vocab
from harmonia_min.chord_lm.from_app import load_app_charts
from harmonia_min.chord_lm.grid import GriddedChart


def load(path: Path, device: str) -> M.ChordLM:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = M.LMConfig(**blob["cfg"])
    net = M.ChordLM(cfg)
    net.load_state_dict(blob["state"])
    return net.to(device).eval()


@torch.no_grad()
def cloze_all(net: M.ChordLM, tokens: list[int], device: str,
              batch: int = 128) -> np.ndarray:
    """(T, V) distribution for every slot, each computed with that slot masked."""
    T = len(tokens)
    base = torch.tensor(tokens, dtype=torch.long)
    out = np.zeros((T, vocab.VOCAB_SIZE), dtype=np.float32)
    for start in range(0, T, batch):
        pos = list(range(start, min(start + batch, T)))
        x = base.unsqueeze(0).repeat(len(pos), 1).clone()
        for r, p in enumerate(pos):
            x[r, p] = vocab.MASK
        x = x.to(device)
        slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                   ).expand(len(pos), T)
        lp = F.log_softmax(net(x, slot_ix), -1)
        rows = torch.arange(len(pos), device=device)
        out[start:start + len(pos)] = lp[rows, torch.tensor(pos, device=device)].cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="chord_lm")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="LM probability above which a disagreement is reported")
    args = ap.parse_args()

    device = M.torch.backends.mps.is_available() and "mps" or "cpu"
    mdir = Path("data/models")
    causal = load(mdir / f"{args.models}_causal.pt", device)
    masked = load(mdir / f"{args.models}_masked.pt", device)

    print("=" * 78)
    print("1. GENERATION — causal model, 16 bars from a 2-bar ii-V prompt")
    print("=" * 78)
    prompts = {
        "ii-V in C":   [vocab.chord_id(2, "min"), vocab.REP,
                        vocab.chord_id(7, "dom"), vocab.REP],
        "minor ii-V":  [vocab.chord_id(2, "hdim"), vocab.REP,
                        vocab.chord_id(7, "dom"), vocab.REP],
        "I-vi in F":   [vocab.chord_id(5, "maj"), vocab.REP,
                        vocab.chord_id(2, "min"), vocab.REP],
    }
    for name, prompt in prompts.items():
        for temp in (0.7, 1.0):
            seq = causal.generate(prompt, 28, temperature=temp, device=device)
            g = GriddedChart(title="", tokens=seq, slots_per_bar=2, beats_per_bar=4,
                             sections=[])
            print(f"\n  {name}  (temperature {temp})")
            for line in g.render(bars_per_line=4).splitlines():
                print("    " + line)

    print("\n" + "=" * 78)
    print("2. SECOND OPINION — masked model vs our own charts")
    print("=" * 78)
    for chart in load_app_charts():
        lp = cloze_all(masked, chart.tokens, device)
        prob = np.exp(lp)
        pred = prob.argmax(1)
        agree = int((pred == np.array(chart.tokens)).sum())
        conf = prob.max(1)
        flags = [i for i in range(len(chart.tokens))
                 if pred[i] != chart.tokens[i] and conf[i] >= args.threshold]
        print(f"\n  {chart.title}  —  LM agrees on {agree}/{len(chart.tokens)} slots "
              f"({100*agree/len(chart.tokens):.0f}%), "
              f"{len(flags)} confident disagreements (p>={args.threshold})")
        for i in flags[:12]:
            bar, slot = divmod(i, chart.slots_per_bar)
            print(f"    bar {bar+1:>3} slot {slot+1}: chart says "
                  f"{vocab.token_name(chart.tokens[i]):<8} "
                  f"LM says {vocab.token_name(int(pred[i])):<8} (p={conf[i]:.2f})")
        if len(flags) > 12:
            print(f"    ... and {len(flags)-12} more")


if __name__ == "__main__":
    main()
