"""train_quality_head.py — retrain the 5-way chord-quality head on REAL audio.

Mission 2. Quality head = maj / min / dom / hdim / dim on root-shifted chroma
(the root is fixed upstream by the beat-seq root model; this head only decides
the third/seventh family). The existing production quality head was trained on
synthetic MMA renders and drops ~40pp on real recordings (docs/known_issues.md
#19: real-audio q5 acc ~44%). This script retrains it on the real-audio corpus.

DATA
  data/cache/yt_corpus/corpus_50.npz  — 7195 iReal-labelled segments, 50 songs,
  built by harmonia/data/yt_chord_corpus.py from YouTube audio + iReal Pro GT.
  Keys used: feat48 (root-shifted 48d BP chroma), feat12_cqt (root-shifted 12d
  CQT), quality_idx (7-class), song_id, match (exact/family/mismatch GT-agree).

  We keep only match in {exact, family} (root is correct → the quality label is
  trustworthy) and the 5 target classes maj/min/dom/hdim/dim. aug/sus are ~2.7%
  of clean segments and outside the 5-way scheme, so they are dropped.

SPLIT
  Held out BY SONG (never by segment) so no song appears in both train and
  hold-out — segment-level splitting would leak a song's timbre/mix across the
  boundary. Deterministic: `rng = default_rng(--seed)`.

SYNTH SMOKE (--synth-smoke)
  Trains the same MLP on the synthetic MMA oracle table
  (data/cache/audio_chord_features.npz, 5-way *family* labels
  major/minor/dim/aug/sus) purely to verify the architecture + training loop can
  learn from 48d chroma before touching real audio. NB: that table's label space
  is triad-family, not the maj/min/dom/hdim/dim quality space, so it is a
  plumbing check only — its accuracy is not comparable to the real-audio head.

USAGE
  # real-audio training (the actual Mission-2 run):
  .venv/bin/python scripts/train_quality_head.py --epochs 200
  # verify the loop learns on synthetic data first:
  .venv/bin/python scripts/train_quality_head.py --synth-smoke --epochs 60
  # fast plumbing check (few epochs, scratch checkpoint):
  .venv/bin/python scripts/train_quality_head.py --epochs 5 --out /tmp/smoke.pt

OUTPUT
  data/models/quality_head_v1.pt  (default) — torch checkpoint with state_dict,
  StandardScaler stats, class names, feature spec, and the held-out song list so
  eval_quality_head.py can reproduce the exact split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# 5-way quality scheme for this head (subset of the corpus 7-class scheme).
QUALITY5 = ["maj", "min", "dom", "hdim", "dim"]
# third-class per 5-way quality for the majmin metric: 0=major-third, 1=minor-third
# maj & dom have a major third; min, hdim, dim have a minor third.
THIRD_OF_Q5 = np.array([0, 1, 0, 1, 1])

CORPUS = REPO / "data" / "cache" / "yt_corpus" / "corpus_50.npz"
SYNTH = REPO / "data" / "cache" / "audio_chord_features.npz"
DEFAULT_OUT = REPO / "data" / "models" / "quality_head_v1.pt"


# ── model ─────────────────────────────────────────────────────────────────────

def make_mlp(in_dim: int, n_classes: int, h1: int = 128, h2: int = 64):
    import torch.nn as nn
    # Matches the yt_chord_model / eval_yt_model architecture (LayerNorm + GELU +
    # Dropout 0.3) so numbers are comparable to the existing quality head.
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h2, n_classes),
    )


# ── data ──────────────────────────────────────────────────────────────────────

def load_real(features: str):
    """Return X, y5 (0..4), song_id for clean 5-way real-audio segments."""
    d = np.load(CORPUS, allow_pickle=True)
    quals = [str(q) for q in d["qualities"]]           # 7-class corpus scheme
    q7 = d["quality_idx"].astype(int)
    match = d["match"]
    song = d["song_id"]

    # remap corpus 7-class idx -> our 5-way idx (or -1 to drop)
    remap = np.full(len(quals), -1, dtype=int)
    for ci, name in enumerate(quals):
        if name in QUALITY5:
            remap[ci] = QUALITY5.index(name)
    y5_all = remap[q7]

    clean = np.isin(match, ["exact", "family"])         # trustworthy GT root+family
    keep = clean & (y5_all >= 0)

    feat = _feature_matrix(d, features)[keep]
    return feat.astype(np.float32), y5_all[keep].astype(int), song[keep]


def _feature_matrix(d, features: str) -> np.ndarray:
    if features == "bp48":
        return d["feat48"]
    if features == "cqt12":
        return d["feat12_cqt"]
    if features == "bp48+cqt12":
        return np.concatenate([d["feat48"], d["feat12_cqt"]], axis=1)
    raise ValueError(f"unknown --features {features!r}")


def load_synth():
    """Synthetic MMA oracle table → (X48, family_labels, song). Smoke test only."""
    d = np.load(SYNTH, allow_pickle=True)
    # 4 channels × 12d = 48d, same block layout as corpus feat48 (already root-relative).
    X = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]]).astype(np.float32)
    y = d["family"].astype(int)
    labels = [str(x) for x in d["family_labels"]]
    return X, y, d["song"], labels


def song_split(song_ids: np.ndarray, holdout: int, seed: int):
    """Hold out `holdout` whole songs (deterministic). Returns bool train mask + set."""
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(song_ids.tolist()))
    holdout = min(holdout, max(1, len(unique) - 1))
    hold = set(rng.choice(unique, size=holdout, replace=False).tolist())
    train_mask = np.array([s not in hold for s in song_ids.tolist()], dtype=bool)
    return train_mask, hold


# ── training ──────────────────────────────────────────────────────────────────

def train(X_tr, y_tr, n_cls, *, epochs, lr, batch, h1, h2, device, seed, label="quality"):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    mean = X_tr.mean(0).astype(np.float32)
    std = (X_tr.std(0) + 1e-9).astype(np.float32)
    Xn = ((X_tr - mean) / std).astype(np.float32)

    counts = np.bincount(y_tr, minlength=n_cls).astype(float)
    w = 1.0 / (counts + 1.0)
    w = w / w.sum() * n_cls                              # class-balanced CE (rare hdim/dim)

    model = make_mlp(X_tr.shape[1], n_cls, h1, h2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    Xt = torch.tensor(Xn, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    n = len(Xt)
    g = torch.Generator(device=device).manual_seed(seed)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  [{label}] epoch {ep + 1:3d}/{epochs}  loss={tot / max(1, n // batch):.4f}  "
                  f"train_acc={acc:.3f}")

    model.eval()
    return model, mean, std


def report(model, mean, std, X, y, class_names, device, *, third_of=None):
    """Print per-class precision/recall + accuracy on a held-out set."""
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, device=device)).argmax(1).cpu().numpy()
    acc = float((pred == y).mean())
    print(f"  hold-out accuracy: {acc:.3f}  (n={len(y)})")
    print(f"  {'class':6s} {'n':>5s} {'prec':>6s} {'rec':>6s}")
    for ci, name in enumerate(class_names):
        tp = int(((pred == ci) & (y == ci)).sum())
        fp = int(((pred == ci) & (y != ci)).sum())
        fn = int(((pred != ci) & (y == ci)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {name:6s} {int((y == ci).sum()):5d} {prec:6.3f} {rec:6.3f}")
    if third_of is not None:
        majmin = float((third_of[pred] == third_of[y]).mean())
        print(f"  majmin (third-class) accuracy: {majmin:.3f}")
    return acc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--h1", type=int, default=128)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-songs", type=int, default=5,
                    help="number of whole songs held out (default 5; task spec = 5)")
    ap.add_argument("--features", choices=["bp48", "cqt12", "bp48+cqt12"], default="bp48")
    ap.add_argument("--synth-smoke", action="store_true",
                    help="first train the same MLP on synthetic MMA family labels (plumbing check)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  seed={args.seed}  features={args.features}\n")

    if args.synth_smoke:
        print("=" * 70)
        print("SYNTH SMOKE — MMA oracle family labels (plumbing check, not comparable)")
        print("=" * 70)
        Xs, ys, songs, labels = load_synth()
        tr_mask, hold = song_split(songs, holdout=args.holdout_songs, seed=args.seed)
        print(f"  {len(labels)}-way family {labels}; train={tr_mask.sum()} hold={(~tr_mask).sum()} "
              f"(hold songs: {sorted(hold)})")
        m, mu, sd = train(Xs[tr_mask], ys[tr_mask], len(labels), epochs=args.epochs, lr=args.lr,
                          batch=args.batch, h1=args.h1, h2=args.h2, device=device,
                          seed=args.seed, label="synth")
        report(m, mu, sd, Xs[~tr_mask], ys[~tr_mask], labels, device)
        print()

    print("=" * 70)
    print("REAL AUDIO — corpus_50 quality head (maj/min/dom/hdim/dim)")
    print("=" * 70)
    if not CORPUS.exists():
        sys.exit(f"ERROR: {CORPUS} not found — gated on Mission 1 corpus build.")

    X, y, songs = load_real(args.features)
    tr_mask, hold = song_split(songs, holdout=args.holdout_songs, seed=args.seed)
    n_songs = len(set(songs.tolist()))
    print(f"  {len(y)} clean 5-way segments, {n_songs} songs "
          f"(train {n_songs - len(hold)} / hold-out {len(hold)})")
    print(f"  class mix: " + "  ".join(f"{q}={int((y == i).sum())}" for i, q in enumerate(QUALITY5)))
    print(f"  hold-out songs ({len(hold)}): {sorted(hold)}")
    print(f"  train={tr_mask.sum()}  hold-out={(~tr_mask).sum()}\n")

    model, mean, std = train(X[tr_mask], y[tr_mask], len(QUALITY5), epochs=args.epochs, lr=args.lr,
                             batch=args.batch, h1=args.h1, h2=args.h2, device=device,
                             seed=args.seed, label="quality")

    print("\nHOLD-OUT REPORT")
    report(model, mean, std, X[~tr_mask], y[~tr_mask], QUALITY5, device, third_of=THIRD_OF_Q5)

    # ── save checkpoint ───────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "scaler_mean": mean,
        "scaler_std": std,
        "class_names": QUALITY5,
        "third_of_class": THIRD_OF_Q5,
        "in_dim": X.shape[1],
        "h1": args.h1,
        "h2": args.h2,
        "features": args.features,
        "holdout_songs": sorted(hold),
        "seed": args.seed,
        "corpus": str(CORPUS),
    }
    torch.save(ckpt, args.out)
    print(f"\nSaved checkpoint → {args.out}")


if __name__ == "__main__":
    main()
