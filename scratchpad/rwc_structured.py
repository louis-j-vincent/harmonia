"""Structured multi-head + root-normalization ablation on RWC BP48 — multi-seed CV.

Two questions, both answered on clean real production-BP48 audio (RWC), with
identical song-stratified splits per seed and 6-seed mean+/-std rigor:

A) ROOT-HEAD normalization/dimensionality ablation grid (the new priority):
     {no-norm(absolute), key-relative} x {12-dim folded, 48-dim full-register}
   - "no-norm"     : feat48_abs as-is (what's been used all session).
   - "key-relative": block-roll chroma by -key_tonic, where tonic is estimated
                     per song from that song's own summed chroma via
                     harmonia.theory.key_profiles.infer_key (Krumhansl-Schmuckler).
                     Non-circular: tonic does NOT depend on the per-chord root.
                     The head then predicts scale-degree=(root-tonic); absolute
                     root = (degree+tonic)%12 for eval.
   - "12-dim"      : fold the 4 register/type blocks (onset,note,bass,treble)
                     into one 12-vec (mean, re-L2-normed).
   - "48-dim"      : keep the 4 blocks (register-preserving).
   Root-relative normalization for the ROOT head itself is INVALID (circular:
   you cannot rotate by the root you are trying to predict) -> NOT tested,
   flagged here explicitly.
   Roll augmentation applies to every variant (roll feature + shift target in
   the feature's own frame); it is valid because it augments the target class,
   not because the feature is absolute.

B) FAMILY(quality) + 7th heads: input chroma ALWAYS oracle-root-relative
   (feat48 == block-roll of feat48_abs by -true_root, verified exact). Plus the
   deployable cascade: rotate by PREDICTED root, marginalize over top-k root
   posteriors (k=1 hard / 3 / 5). This is the structured design minus the
   trigram context (deliberately not ported).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from harmonia.theory.key_profiles import infer_key
from train_real_audio_final import _train_head, QUALITIES

DOM = QUALITIES.index("dom")


# ----------------------------- helpers --------------------------------------
def _softmax(model, Xn, device):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xn.astype(np.float32), device=device)), 1).cpu().numpy()


def _block_roll(f48, shift):
    """Roll each 12-block of an (N,48) feature by `shift` (scalar or (N,))."""
    N = f48.shape[0]; r = f48.reshape(N, 4, 12)
    if np.isscalar(shift):
        out = np.roll(r, shift, 2)
    else:
        out = np.stack([np.roll(r[i], int(shift[i]), 2) for i in range(N)])
    return out.reshape(N, 48)


def _fold12(f48):
    """Fold 4 register/type blocks -> one 12-dim chroma, re-L2-normed."""
    v = f48.reshape(-1, 4, 12).mean(1)
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _roll_aug_generic(X, target, block_width, n_shifts=12):
    """Roll aug in the feature's own frame: roll each `block_width` block by k,
    shift target by +k. Valid for both absolute (target=root) and key-relative
    (target=degree) frames — it augments the target class, not the origin."""
    n, d = X.shape; nb = d // block_width
    Xs, ys = [X], [target]
    for k in range(1, n_shifts):
        Xs.append(np.roll(X.reshape(n, nb, block_width), k, 2).reshape(n, d))
        ys.append((target + k) % 12)
    return np.concatenate(Xs), np.concatenate(ys)


def _bal_dom(preds, y, n=7):
    rec = {c: (float((preds[y == c] == c).mean()) if (y == c).sum() else 0.0) for c in range(n)}
    dom = rec[DOM] if n == 7 else None
    return float(np.mean([rec[c] for c in range(n)])), dom


def _train_eval_root(Xtr_frame, tgt_tr, Xte_frame, tonic_te, true_root_te, bw,
                     *, epochs, lr, batch, device):
    """Generic root head. Trains on target in feature-frame, maps back to
    absolute root for scoring. bw = block width (12 or 48-dim -> 12)."""
    Xa, ya = _roll_aug_generic(Xtr_frame, tgt_tr, bw)
    m, mean, std = _train_head(Xa, ya, 12, epochs=epochs, lr=lr, batch=batch,
                               device=device, head_name="root")
    pred_frame = _softmax(m, (Xte_frame - mean) / std, device).argmax(1)
    abs_pred = (pred_frame + tonic_te) % 12          # tonic_te is 0 for no-norm
    return float((abs_pred == true_root_te).mean()), m, mean, std


# ---------------------- per-song key tonic (non-circular) -------------------
def song_tonics(d):
    """Estimate one key tonic (0-11) per song from that song's summed chroma."""
    sid = d["song_id"]; fold = _fold12(d["feat48_abs"])
    tonic = {}
    for s in sorted(set(sid.tolist())):
        m = sid == s
        chroma = fold[m].sum(0)               # raw magnitude carries evidence
        tonic[s] = infer_key(chroma).tonic
    return np.array([tonic[s] for s in sid], dtype=int)


# ------------------------------ one split -----------------------------------
def one_split(d, tonics_all, seed, *, epochs=60, lr=3e-4, batch=64, device, test_frac=0.2):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    f48 = d["feat48"][keep]; fa = d["feat48_abs"][keep]
    qidx = d["quality_idx"].astype(int)[keep]
    has7 = d["has7"][keep].astype(int)
    roots = d["root"].astype(int)[keep]; sid = d["song_id"][keep]
    tonic = tonics_all[keep]

    songs = sorted(set(sid.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test = set(songs[:n_test])
    tr = np.array([s not in test for s in sid]); te = ~tr

    out = {"n_test": int(te.sum()), "n_test_songs": n_test}

    # ---- ROOT ablation grid: {no-norm, key-rel} x {12, 48} ----
    fa12 = _fold12(fa)
    # key-relative frames: per-row block-roll by -tonic
    kr48 = np.stack([np.roll(fa[i].reshape(4, 12), -tonic[i], 1).reshape(48) for i in range(len(fa))])
    kr12 = np.stack([np.roll(fa12[i], -tonic[i]) for i in range(len(fa12))])
    deg = (roots - tonic) % 12

    grid = {}
    grid["abs_48"], *_ = (_train_eval_root(fa[tr], roots[tr], fa[te], np.zeros(te.sum(), int),
                                           roots[te], 12, epochs=epochs, lr=lr, batch=batch, device=device),)
    grid["abs_12"], *_ = (_train_eval_root(fa12[tr], roots[tr], fa12[te], np.zeros(te.sum(), int),
                                           roots[te], 12, epochs=epochs, lr=lr, batch=batch, device=device),)
    grid["keyrel_48"], *_ = (_train_eval_root(kr48[tr], deg[tr], kr48[te], tonic[te],
                                              roots[te], 12, epochs=epochs, lr=lr, batch=batch, device=device),)
    grid["keyrel_12"], *_ = (_train_eval_root(kr12[tr], deg[tr], kr12[te], tonic[te],
                                              roots[te], 12, epochs=epochs, lr=lr, batch=batch, device=device),)
    out["root_grid"] = grid

    # root posteriors from the DEFAULT (abs_48 + roll) head, for cascade
    Xa, ya = _roll_aug_generic(fa[tr], roots[tr], 12)
    rm, rmean, rstd = _train_head(Xa, ya, 12, epochs=epochs, lr=lr, batch=batch,
                                  device=device, head_name="root_casc")
    root_probs = _softmax(rm, (fa[te] - rmean) / rstd, device)

    # ---- FAMILY(quality) head: oracle root-relative ----
    qm, qmean, qstd = _train_head(f48[tr], qidx[tr], 7, epochs=epochs, lr=lr, batch=batch,
                                  device=device, head_name="qual")
    q_or = _softmax(qm, (f48[te] - qmean) / qstd, device).argmax(1)
    out["q_oracle"] = _bal_dom(q_or, qidx[te])

    # pure-flat quality baseline (absolute, no root structure)
    qmf, qfm, qfs = _train_head(fa[tr], qidx[tr], 7, epochs=epochs, lr=lr, batch=batch,
                                device=device, head_name="qual_flat")
    q_flat = _softmax(qmf, (fa[te] - qfm) / qfs, device).argmax(1)
    out["q_flat"] = _bal_dom(q_flat, qidx[te])

    # cascade: rotate feat48_abs[te] by -r, marginalize over top-k root posteriors
    per_root = np.stack([_softmax(qm, (_block_roll(fa[te], -r) - qmean) / qstd, device)
                         for r in range(12)])                       # (12,Nte,7)
    out["cascade"] = {}
    for k in (1, 3, 5):
        topk = np.argsort(-root_probs, 1)[:, :k]
        w = np.take_along_axis(root_probs, topk, 1); w /= w.sum(1, keepdims=True)
        qmix = np.zeros((per_root.shape[1], 7), np.float32)
        for j in range(k):
            qmix += w[:, j:j+1] * per_root[topk[:, j], np.arange(per_root.shape[1])]
        out["cascade"][k] = _bal_dom(qmix.argmax(1), qidx[te])

    # ---- 7th head: oracle root-relative, binary has-7th ----
    sm, s_m, s_s = _train_head(f48[tr], has7[tr], 2, epochs=epochs, lr=lr, batch=batch,
                               device=device, head_name="seventh")
    s_pred = _softmax(sm, (f48[te] - s_m) / s_s, device).argmax(1)
    out["seventh"] = _bal_dom(s_pred, has7[te], n=2)[0], float((s_pred[has7[te] == 1] == 1).mean() if (has7[te] == 1).sum() else 0)
    return out


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = dict(load_corpus(REPO / "data/cache/rwc/rwc_bp48.npz"))
    # derive has-7th from full Harte labels
    labs = d["labels"]
    d["has7"] = np.array([("7" in l.split(":", 1)[1] if ":" in l else False) for l in labs], bool)
    tonics = song_tonics(d)
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"RWC structured+ablation: device={dev} seeds={seeds}  has7 rate={d['has7'].mean():.3f}\n")

    runs = []
    for s in range(seeds):
        print(f"=== seed {s} ===", flush=True)
        r = one_split(d, tonics, s, device=dev)
        runs.append(r)
        g = r["root_grid"]
        print(f"  ROOT  abs_48={g['abs_48']:.3f} abs_12={g['abs_12']:.3f} "
              f"keyrel_48={g['keyrel_48']:.3f} keyrel_12={g['keyrel_12']:.3f}")
        c = r["cascade"]
        print(f"  FAM   flat_abs={r['q_flat'][0]:.3f}/{r['q_flat'][1]:.3f} "
              f"oracle={r['q_oracle'][0]:.3f}/{r['q_oracle'][1]:.3f} "
              f"casc k1={c[1][0]:.3f}/{c[1][1]:.3f} k3={c[3][0]:.3f}/{c[3][1]:.3f} k5={c[5][0]:.3f}/{c[5][1]:.3f}")
        print(f"  7TH   bal={r['seventh'][0]:.3f} has7_recall={r['seventh'][1]:.3f}"
              f"   (test {r['n_test']}/{r['n_test_songs']} songs)\n", flush=True)

    def ms(f):
        v = np.array([f(x) for x in runs]); return v.mean(), v.std()
    print("=" * 70)
    print(f"RWC {seeds}-seed CV (mean +/- std)")
    print("-- ROOT-HEAD ablation grid (root accuracy) --")
    for v in ("abs_48", "abs_12", "keyrel_48", "keyrel_12"):
        m, sd = ms(lambda r: r["root_grid"][v]); print(f"   {v:10s}: {m:.3f} +/- {sd:.3f}")
    print("-- FAMILY(quality) balanced acc / dom recall --")
    print(f"   flat_abs  : {ms(lambda r:r['q_flat'][0])[0]:.3f} +/- {ms(lambda r:r['q_flat'][0])[1]:.3f}   dom {ms(lambda r:r['q_flat'][1])[0]:.3f}")
    print(f"   oracle    : {ms(lambda r:r['q_oracle'][0])[0]:.3f} +/- {ms(lambda r:r['q_oracle'][0])[1]:.3f}   dom {ms(lambda r:r['q_oracle'][1])[0]:.3f}")
    for k in (1, 3, 5):
        print(f"   cascade k{k}: {ms(lambda r:r['cascade'][k][0])[0]:.3f} +/- {ms(lambda r:r['cascade'][k][0])[1]:.3f}   dom {ms(lambda r:r['cascade'][k][1])[0]:.3f} +/- {ms(lambda r:r['cascade'][k][1])[1]:.3f}")
    print("-- 7TH head (oracle root-rel) --")
    print(f"   balanced  : {ms(lambda r:r['seventh'][0])[0]:.3f} +/- {ms(lambda r:r['seventh'][0])[1]:.3f}   has7_recall {ms(lambda r:r['seventh'][1])[0]:.3f}")


if __name__ == "__main__":
    main()
