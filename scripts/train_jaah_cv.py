"""Multi-seed song-stratified cross-validation for the JAAH real-audio corpus.

JAAH is small (~113 songs, fewer after the verification gate), so a single
80/10/10 split (as scripts/train_real_audio_final.py does) is high-variance.
This wrapper reuses that script's model/train/eval functions verbatim and runs
N song-stratified splits with different seeds, reporting mean +/- std for the
three headline metrics — root acc, quality balanced acc, dom recall — directly
comparable to Billboard's numbers. Does NOT modify the shared trainer.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import (
    _train_head, _eval, _augment_root_by_roll, QUALITIES,
)


def one_split(d, seed, *, min_match, roll, epochs, lr, batch, device, test_frac=0.2):
    keep = filter_by_match(d["match"],
                           minimum=MatchQuality.EXACT if min_match == "exact"
                           else MatchQuality.FAMILY)
    feat48 = d["feat48"][keep]; feat48_abs = d["feat48_abs"][keep]
    quality_idx = d["quality_idx"].astype(int)[keep]
    roots = d["root"].astype(int)[keep]; song_id = d["song_id"][keep]

    songs = sorted(set(song_id.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in song_id])
    te = ~tr

    # root head (absolute feats, optional roll augment)
    Xtr, ytr = feat48_abs[tr], roots[tr]
    if roll:
        Xtr, ytr = _augment_root_by_roll(Xtr, ytr)
    rm, rmean, rstd = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                                  device=device, head_name="root")
    root_acc, _, _ = _eval(feat48_abs[te], roots[te], rm, rmean, rstd, device)

    # quality head (root-relative feats)
    qm, qmean, qstd = _train_head(feat48[tr], quality_idx[tr], 7, epochs=epochs, lr=lr,
                                  batch=batch, device=device, head_name="qual")
    q_acc, q_recall, _ = _eval(feat48[te], quality_idx[te], qm, qmean, qstd, device)
    q_bal = float(np.mean([q_recall.get(i, 0.0) for i in range(7)]))
    dom = q_recall.get(QUALITIES.index("dom"), 0.0)
    return {"root": root_acc, "qual_bal": q_bal, "qual_acc": q_acc,
            "dom": dom, "n_test_songs": len(test_songs),
            "n_train": int(tr.sum()), "n_test": int(te.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/jaah/jaah_bp48.npz")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--min-match", choices=["exact", "exact+family"], default="exact")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--roll", action="store_true", help="root-roll augmentation")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    d = load_corpus(a.corpus)
    songs = sorted(set(d["song_id"].tolist()))
    print(f"Corpus {a.corpus.name}: {len(d['root'])} records, {len(songs)} songs, device={dev}")
    qi = d["quality_idx"].astype(int)
    print("Quality marginal:", {q: int((qi == i).sum()) for i, q in enumerate(QUALITIES)})
    print(f"Root-roll augment: {a.roll}   min-match: {a.min_match}\n")

    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(d, s, min_match=a.min_match, roll=a.roll, epochs=a.epochs,
                      lr=a.lr, batch=a.batch, device=dev)
        runs.append(r)
        print(f"  root={r['root']:.3f}  qual_bal={r['qual_bal']:.3f}  "
              f"dom={r['dom']:.3f}  (train {r['n_train']} / test {r['n_test']} "
              f"over {r['n_test_songs']} songs)\n", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs]); return v.mean(), v.std()
    print("=" * 60)
    print(f"JAAH CV over {a.seeds} song-stratified splits (roll={a.roll}):")
    for k, lbl in [("root", "Root acc"), ("qual_bal", "Quality balanced acc"),
                   ("qual_acc", "Quality raw acc"), ("dom", "Dom recall")]:
        m, sd = ms(k)
        print(f"  {lbl:24s}: {m:.1%} +/- {sd:.1%}")
    print("\nBillboard best (reference): root 54-56%, quality balanced ~20%")


if __name__ == "__main__":
    main()
