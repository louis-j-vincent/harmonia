"""ROOT-HEAD normalization/dimensionality ablation grid on RWC BP48 — multi-seed CV.

SCOPE CUT 2026-07-16: per explicit user direction ("Ok until the root isn't
at a higher quality, theres no need to predict the rest by renormalizing the
root") — quality/7th heads and the top-k cascade marginalization are DROPPED
from this run entirely. Root accuracy is the only thing being tested here;
downstream quality work is gated on root actually improving first. (The
dropped work is preserved, unrun, in scratchpad/rwc_structured.py if revived
later.)

Grid: {no-norm(absolute), key-relative} x {12-dim folded, 48-dim full-register}
  - "no-norm"     : feat48_abs as-is (what's been used all session).
  - "key-relative": block-roll chroma by -key_tonic, tonic estimated per song
                    from that song's own summed chroma via
                    harmonia.theory.key_profiles.infer_key (Krumhansl-Schmuckler).
                    Non-circular: tonic does NOT depend on the per-chord root.
                    Head predicts scale-degree=(root-tonic); absolute root =
                    (degree+tonic)%12 for eval.
  - "12-dim"      : fold the 4 register/type blocks (onset,note,bass,treble)
                    into one 12-vec (mean, re-L2-normed).
  - "48-dim"      : keep the 4 blocks (register-preserving).
Root-relative normalization for the ROOT head itself is INVALID (circular —
cannot rotate by the root you're predicting) -> not tested.
Roll augmentation applies to every variant (roll feature + shift target in the
feature's own frame); valid because it augments the target class, not because
the feature is absolute.

Comparison target: RWC flat-MLP baseline (train_jaah_cv.py --roll, 6 seeds,
scratchpad/rwc_cv.log): root acc 64.0% +/- 2.0%. That baseline IS the abs_48
variant here (same feature, same roll-aug) — included as an in-run sanity
check that this script reproduces it before trusting the other 3 cells.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from harmonia.theory.key_profiles import infer_key
from train_real_audio_final import _train_head


def _softmax(model, Xn, device):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xn.astype(np.float32), device=device)), 1).cpu().numpy()


def _fold12(f48):
    """Fold 4 register/type blocks -> one 12-dim chroma, re-L2-normed."""
    v = f48.reshape(-1, 4, 12).mean(1)
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _roll_aug_generic(X, target, block_width, n_shifts=12):
    """Roll aug in the feature's own frame: roll each block by k, shift target
    by +k. Valid for absolute (target=root) and key-relative (target=degree)."""
    n, d = X.shape; nb = d // block_width
    Xs, ys = [X], [target]
    for k in range(1, n_shifts):
        Xs.append(np.roll(X.reshape(n, nb, block_width), k, 2).reshape(n, d))
        ys.append((target + k) % 12)
    return np.concatenate(Xs), np.concatenate(ys)


def _train_eval_root(Xtr_frame, tgt_tr, Xte_frame, tonic_te, true_root_te, bw,
                     *, epochs, lr, batch, device, name):
    Xa, ya = _roll_aug_generic(Xtr_frame, tgt_tr, bw)
    m, mean, std = _train_head(Xa, ya, 12, epochs=epochs, lr=lr, batch=batch,
                               device=device, head_name=name)
    pred_frame = _softmax(m, (Xte_frame - mean) / std, device).argmax(1)
    abs_pred = (pred_frame + tonic_te) % 12
    return float((abs_pred == true_root_te).mean())


def song_tonics(feat48_abs, song_id):
    """One key tonic (0-11) per song, estimated from that song's summed
    folded chroma. Non-circular (no per-chord root used)."""
    fold = _fold12(feat48_abs)
    tonic = {}
    for s in sorted(set(song_id.tolist())):
        m = song_id == s
        tonic[s] = infer_key(fold[m].sum(0)).tonic
    return np.array([tonic[s] for s in song_id], dtype=int)


def one_split(d, tonics_all, seed, *, epochs=60, lr=3e-4, batch=64, device, test_frac=0.2):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    fa = d["feat48_abs"][keep]
    roots = d["root"].astype(int)[keep]; sid = d["song_id"][keep]
    tonic = tonics_all[keep]

    songs = sorted(set(sid.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test = set(songs[:n_test])
    tr = np.array([s not in test for s in sid]); te = ~tr

    fa12 = _fold12(fa)
    kr48 = np.stack([np.roll(fa[i].reshape(4, 12), -tonic[i], 1).reshape(48) for i in range(len(fa))])
    kr12 = np.stack([np.roll(fa12[i], -tonic[i]) for i in range(len(fa12))])
    deg = (roots - tonic) % 12
    zeros_te = np.zeros(int(te.sum()), int)

    grid = {}
    grid["abs_48"] = _train_eval_root(fa[tr], roots[tr], fa[te], zeros_te, roots[te], 12,
                                      epochs=epochs, lr=lr, batch=batch, device=device, name="abs48")
    grid["abs_12"] = _train_eval_root(fa12[tr], roots[tr], fa12[te], zeros_te, roots[te], 12,
                                      epochs=epochs, lr=lr, batch=batch, device=device, name="abs12")
    grid["keyrel_48"] = _train_eval_root(kr48[tr], deg[tr], kr48[te], tonic[te], roots[te], 12,
                                         epochs=epochs, lr=lr, batch=batch, device=device, name="kr48")
    grid["keyrel_12"] = _train_eval_root(kr12[tr], deg[tr], kr12[te], tonic[te], roots[te], 12,
                                         epochs=epochs, lr=lr, batch=batch, device=device, name="kr12")
    return dict(grid=grid, n_train=int(tr.sum()), n_test=int(te.sum()), n_test_songs=n_test)


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = dict(load_corpus(REPO / "data/cache/rwc/rwc_bp48.npz"))
    tonics = song_tonics(d["feat48_abs"], d["song_id"])
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"RWC root-grid: device={dev} seeds={seeds}\n", flush=True)

    runs = []
    for s in range(seeds):
        print(f"=== seed {s} ===", flush=True)
        r = one_split(d, tonics, s, device=dev)
        runs.append(r)
        g = r["grid"]
        print(f"  abs_48={g['abs_48']:.3f}  abs_12={g['abs_12']:.3f}  "
              f"keyrel_48={g['keyrel_48']:.3f}  keyrel_12={g['keyrel_12']:.3f}  "
              f"(train {r['n_train']} / test {r['n_test']} over {r['n_test_songs']} songs)\n", flush=True)

    def ms(v):
        arr = np.array([r["grid"][v] for r in runs]); return arr.mean(), arr.std()
    print("=" * 60)
    print(f"RWC root-normalization ablation, {seeds}-seed CV (mean +/- std):")
    for v in ("abs_48", "abs_12", "keyrel_48", "keyrel_12"):
        m, sd = ms(v)
        print(f"  {v:10s}: {m:.3f} +/- {sd:.3f}")
    print("\nBaseline (train_jaah_cv.py --roll, scratchpad/rwc_cv.log): root 0.640 +/- 0.020")
    print("abs_48 above should reproduce that baseline (same feature+aug) as an in-run sanity check.")


if __name__ == "__main__":
    main()
