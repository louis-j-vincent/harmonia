"""Train + evaluate the symbolic section-key model (#23).

Baseline = "predict the song global key for every section" (accuracy = 1 - the
oracle modulation rate, by construction). The learned GRU is trained to imitate
the rules-based oracle from the section's chords alone, with random-transpose
augmentation. We report overall accuracy vs baseline AND accuracy on the
*modulated* subset specifically (the transferable capability for phase-2 audio,
since the global baseline is 0% there by definition).

Usage:
    python scripts/train_local_key_model.py [--epochs 40]
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from harmonia.models.local_key_data import (
    DEFAULT_DB, build_examples, split_examples,
)
from harmonia.models.local_key_model import (
    LocalKeyGRU, collate, transpose_example,
)


def evaluate(model, examples, device):
    model.eval()
    correct = tot = 0
    mod_c = mod_t = 0
    nonmod_c = nonmod_t = 0
    with torch.no_grad():
        for i in range(0, len(examples), 256):
            chunk = examples[i:i + 256]
            root, qual, lengths, _ = collate(
                [(e["seq"], e["y"]) for e in chunk], device)
            pred = model(root, qual, lengths).argmax(1).cpu().numpy()
            for e, p in zip(chunk, pred):
                ok = int(p) == e["y"]
                correct += ok
                tot += 1
                if e["modulated"]:
                    mod_c += ok
                    mod_t += 1
                else:
                    nonmod_c += ok
                    nonmod_t += 1
    return (correct / tot, mod_c / max(mod_t, 1), nonmod_c / max(nonmod_t, 1),
            mod_t, nonmod_t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--margin", type=float, default=6.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    print("Building dataset...")
    ex = build_examples(DEFAULT_DB, margin=args.margin)
    train, val = split_examples(ex)
    print(f"  train={len(train)}  val={len(val)}  "
          f"(songs: {len({e['song_idx'] for e in train})} / "
          f"{len({e['song_idx'] for e in val})})")

    # --- baseline: always predict global key ---
    def base_acc(subset):
        c = sum(e["y"] == e["y_global"] for e in subset)
        return c / len(subset)
    val_mod = [e for e in val if e["modulated"]]
    print(f"\nBASELINE (always global key): val acc = {base_acc(val):.1%}  "
          f"(= 1 - val modulation rate {1 - base_acc(val):.1%})")
    print(f"  modulated-subset baseline acc = 0.0% by construction "
          f"(n={len(val_mod)})")

    # --- train GRU ---
    device = args.device
    model = LocalKeyGRU().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()

    best = 0.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        tot_loss = 0.0
        for i in range(0, len(train), 128):
            chunk = train[i:i + 128]
            # random-transpose augmentation (equivariance)
            batch = []
            for e in chunk:
                k = random.randint(0, 11)
                s, y = transpose_example(e["seq"], e["y"], k)
                batch.append((s, y))
            root, qual, lengths, y = collate(batch, device)
            logits = model(root, qual, lengths)
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(chunk)
        acc, macc, nacc, mt, nt = evaluate(model, val, device)
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}  loss {tot_loss/len(train):.3f}  "
                  f"val acc {acc:.1%}  mod {macc:.1%} (n={mt})  "
                  f"nonmod {nacc:.1%} (n={nt})")

    model.load_state_dict(best_state)
    acc, macc, nacc, mt, nt = evaluate(model, val, device)
    print(f"\nBEST MODEL: val acc {acc:.1%}  |  modulated-subset {macc:.1%} "
          f"(n={mt})  |  non-modulated {nacc:.1%} (n={nt})")
    print(f"GAIN over global baseline: {acc - base_acc(val):+.1%}")

    out = DEFAULT_DB.parent.parent / "cache" / "local_key_gru.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": {}}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
