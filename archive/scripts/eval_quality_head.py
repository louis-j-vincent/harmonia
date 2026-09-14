"""eval_quality_head.py — score a trained 5-way quality head on its hold-out songs.

Loads a checkpoint from train_quality_head.py, reconstructs the SAME song-level
hold-out split (the checkpoint stores the held-out song list + seed + feature
spec), and reports quality metrics against iReal GT:

  - quality accuracy   : strict 5-way maj/min/dom/hdim/dim
  - majmin accuracy    : third-class correct (maj-third {maj,dom} vs min-third
                         {min,hdim,dim}) — the "majmin" MIREX-style number
  - 7ths / exact        : strict quality == GT quality (identical to 5-way here,
                         since the 5 classes already encode the seventh family)
  - partial-credit      : family-or-better — prediction shares the GT triad
                         family (maj-ish {maj,dom} / min-ish {min} / dim-ish
                         {hdim,dim}); rewards maj7-vs-maj style near-misses
  - per-class precision/recall + confusion matrix

SCOPE / how this compares to Mission 1's benchmark
  This head predicts QUALITY ONLY on a root-shifted feature; the root is oracle
  (iReal GT) here, so these are quality-conditioned-on-correct-root numbers — an
  upper bound on the end-to-end majmin/7ths that Mission 1 measures with the
  model's own (imperfect) root. Read the delta vs Mission 1 as "how much of the
  real-audio quality gap this retrain closes, holding root fixed".

USAGE
  .venv/bin/python scripts/eval_quality_head.py --ckpt data/models/quality_head_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train_quality_head import (  # noqa: E402  (sibling script, same dir on path)
    QUALITY5, THIRD_OF_Q5, load_real, make_mlp,
)

# triad-family per 5-way quality for partial credit: 0=maj-ish, 1=min-ish, 2=dim-ish
FAMILY_OF_Q5 = np.array([0, 1, 0, 2, 2])

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=REPO / "data" / "models" / "quality_head_v1.pt")
    args = ap.parse_args()

    import torch
    if not args.ckpt.exists():
        sys.exit(f"ERROR: checkpoint {args.ckpt} not found — run train_quality_head.py first.")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class_names = list(ck["class_names"])
    assert class_names == QUALITY5, f"checkpoint class mismatch: {class_names}"

    # rebuild model
    model = make_mlp(ck["in_dim"], len(class_names), ck["h1"], ck["h2"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # reload corpus with the SAME feature spec and reconstruct the hold-out split
    X, y, songs = load_real(ck["features"])
    hold = set(ck["holdout_songs"])
    hold_mask = np.array([s in hold for s in songs.tolist()], dtype=bool)
    Xh, yh = X[hold_mask], y[hold_mask]
    print(f"checkpoint: {args.ckpt.name}  features={ck['features']}  seed={ck['seed']}")
    print(f"hold-out: {len(hold)} songs, {hold_mask.sum()} segments\n")

    mean = ck["scaler_mean"]; std = ck["scaler_std"]
    Xn = ((Xh - mean) / std).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, device=device)).argmax(1).cpu().numpy()

    # ── metrics ───────────────────────────────────────────────────────────────
    quality_acc = float((pred == yh).mean())
    majmin_acc = float((THIRD_OF_Q5[pred] == THIRD_OF_Q5[yh]).mean())
    family_acc = float((FAMILY_OF_Q5[pred] == FAMILY_OF_Q5[yh]).mean())

    print("METRICS (root = iReal GT / oracle; quality head only)")
    print(f"  quality acc (strict 5-way / 7ths / exact): {quality_acc:.3f}")
    print(f"  majmin acc  (third-class):                 {majmin_acc:.3f}")
    print(f"  partial-credit (family-or-better):         {family_acc:.3f}")
    base = float(np.bincount(yh, minlength=len(class_names)).max() / len(yh))
    print(f"  majority base rate:                        {base:.3f}\n")

    print("PER-CLASS")
    print(f"  {'class':6s} {'n':>5s} {'prec':>6s} {'rec':>6s}")
    for ci, name in enumerate(class_names):
        tp = int(((pred == ci) & (yh == ci)).sum())
        fp = int(((pred == ci) & (yh != ci)).sum())
        fn = int(((pred != ci) & (yh == ci)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {name:6s} {int((yh == ci).sum()):5d} {prec:6.3f} {rec:6.3f}")

    print("\nCONFUSION (row=GT, col=pred)")
    C = np.zeros((len(class_names), len(class_names)), int)
    for t, p in zip(yh, pred):
        C[t, p] += 1
    print("         " + " ".join(f"{n:>5s}" for n in class_names))
    for i, n in enumerate(class_names):
        print(f"  {n:6s} " + " ".join(f"{C[i, j]:>5d}" for j in range(len(class_names))))


if __name__ == "__main__":
    main()
