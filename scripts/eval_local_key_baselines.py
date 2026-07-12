"""Compare the three local-key predictors on the *same* val split (#23).

  1. always-global-key baseline (0% modulated recall by construction)
  2. ported client-side continuity heuristic (zero parameters, new)
  3. LocalKeyGRU (learned, imitates the oracle)

All three are scored against the rules-based section-key oracle on the every-5th
song validation split shared with train_local_key_model.py.

Usage:  python scripts/eval_local_key_baselines.py
"""
from __future__ import annotations

import numpy as np
import torch

from harmonia.models.local_key_data import (
    DEFAULT_DB, build_examples, split_examples,
)
from harmonia.models.local_key_heuristic import (
    build_heuristic_examples, evaluate_heuristic,
)
from harmonia.models.local_key_model import LocalKeyGRU, collate, load_model


def main() -> None:
    margin = 6.0

    # oracle examples (for baseline + GRU) and heuristic examples share song_idx,
    # so split_examples gives the identical val split for both.
    ex = build_examples(DEFAULT_DB, margin=margin)
    _, val = split_examples(ex)
    hex_ = build_heuristic_examples(DEFAULT_DB, margin=margin)
    _, hval = split_examples(hex_)
    assert len(val) == len(hval), (len(val), len(hval))

    val_mod = [e for e in val if e["modulated"]]
    n_mod = len(val_mod)

    # 1. baseline: always global key
    base_c = sum(e["y"] == e["y_global"] for e in val)
    base_acc = base_c / len(val)

    # 2. heuristic
    h = evaluate_heuristic(hval)

    # 3. GRU (if a checkpoint is present)
    gru_line = "  GRU checkpoint not found (run train_local_key_model.py)"
    ckpt = DEFAULT_DB.parent.parent / "cache" / "local_key_gru.pt"
    if ckpt.exists():
        model = load_model(ckpt, device="cpu")
        model.eval()
        correct = mod_c = 0
        with torch.no_grad():
            for i in range(0, len(val), 256):
                chunk = val[i:i + 256]
                root, qual, lengths, _ = collate(
                    [(e["seq"], e["y"]) for e in chunk], "cpu")
                pred = model(root, qual, lengths).argmax(1).cpu().numpy()
                for e, p in zip(chunk, pred):
                    ok = int(p) == e["y"]
                    correct += ok
                    if e["modulated"]:
                        mod_c += ok
        gru_line = (f"  LocalKeyGRU            {correct/len(val):6.1%}"
                    f"      {mod_c/max(n_mod,1):6.1%}")

    print(f"val sections = {len(val)}   modulated = {n_mod} "
          f"({n_mod/len(val):.1%})\n")
    print("  baseline               accuracy   mod-recall")
    print(f"  always-global-key      {base_acc:6.1%}       0.0%")
    print(f"  continuity heuristic   {h['acc']:6.1%}      {h['mod_acc']:6.1%}"
          f"   (non-mod {h['nonmod_acc']:.1%})")
    print(gru_line)


if __name__ == "__main__":
    main()
