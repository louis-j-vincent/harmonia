"""Feature-domain bridge experiments in BP48 production space.

Q1: 12 vs 24 vs 48 dim (which representation for chord quality).
Q2: normalization A/B/C (raw / relative-key / relative-root).
Q3: cross-domain transfer NNLS(Billboard) -> BP48(accomp), the bridge gap.

All song-stratified, class-weighted MLP, report balanced acc + per-class recall.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(42)
torch.manual_seed(42)
Q5 = ["maj", "min", "dom", "hdim", "dim"]


def roll12(block, shift):
    """Roll a (...,12) block along last axis by -shift (put pc `shift` at index 0)."""
    return np.roll(block, -shift, axis=-1)


def load_bp48():
    d = np.load(REPO / "data/cache/bp48_absolute.npz", allow_pickle=True)
    return d


def norm_rows(X):
    """L2-normalize each 12-dim block independently within the row."""
    X = X.reshape(X.shape[0], -1, 12)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    X = X / np.clip(n, 1e-8, None)
    return X.reshape(X.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, din, nout, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, nout))

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xte, yte, nout=5, epochs=60):
    cw = np.bincount(ytr, minlength=nout).astype(float)
    cw = cw.sum() / (nout * np.clip(cw, 1, None))
    w = torch.tensor(cw, dtype=torch.float32)
    m = MLP(Xtr.shape[1], nout)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr)
    m.train()
    n = len(Xt); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); out = m(Xt[idx]); loss = lossf(out, yt[idx])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    # balanced acc + per-class recall
    recs = []
    for c in range(nout):
        mask = yte == c
        recs.append((pred[mask] == c).mean() if mask.sum() else np.nan)
    bal = np.nanmean(recs)
    return bal, recs


def split_songs(songs):
    uniq = np.array(sorted(set(songs.tolist())))
    RNG.shuffle(uniq)
    n_te = max(1, int(0.25 * len(uniq)))
    te = set(uniq[:n_te].tolist())
    return np.array([s not in te for s in songs]), np.array([s in te for s in songs])


def build_features(d, blocks, frame):
    """blocks: subset of ['onset','note','bass','treble']; frame: raw/key/root."""
    root = d["root"]; tonic = d["tonic"]
    parts = []
    for b in blocks:
        X = d[b].astype(np.float32)  # (N,12) absolute
        if frame == "root":
            X = np.stack([roll12(X[i], root[i]) for i in range(len(X))])
        elif frame == "key":
            X = np.stack([roll12(X[i], tonic[i]) for i in range(len(X))])
        parts.append(X)
    return np.concatenate(parts, axis=1)


def run_q1_q2():
    d = load_bp48()
    y = d["q5"].astype(int)
    keep = y >= 0
    songs = d["song"][keep]
    tr, te = split_songs(songs)

    def prep(blocks, frame):
        X = build_features(d, blocks, frame)[keep]
        X = norm_rows(X)
        return X

    print("\n===== Q1: DIMENSIONALITY (root-relative frame, q5 quality) =====")
    print(f"{'feature':<28}{'dim':>5}{'bal':>8}{'maj':>7}{'min':>7}{'dom':>7}{'hdim':>7}{'dim':>7}")
    q1 = [
        ("onset (12, single chroma)", ["onset"]),
        ("onset+note (24, no register)", ["onset", "note"]),
        ("bass+treble (24, register)", ["bass", "treble"]),
        ("all 4 blocks (48)", ["onset", "note", "bass", "treble"]),
    ]
    yk = y[keep]
    for name, blocks in q1:
        X = prep(blocks, "root")
        bal, recs = train_eval(X[tr], yk[tr], X[te], yk[te])
        print(f"{name:<28}{X.shape[1]:>5}{bal:>8.3f}" + "".join(f"{r:>7.2f}" for r in recs))

    print("\n===== Q2: NORMALIZATION (48-dim, q5 quality) =====")
    print(f"{'scheme':<34}{'bal':>8}{'maj':>7}{'min':>7}{'dom':>7}{'hdim':>7}{'dim':>7}")
    for name, frame in [("A-none / raw absolute", "raw"),
                        ("C-relative to KEY (tonic->C)", "key"),
                        ("A-relative to ROOT (root->C)", "root")]:
        X = prep(["onset", "note", "bass", "treble"], frame)
        bal, recs = train_eval(X[tr], yk[tr], X[te], yk[te])
        print(f"{name:<34}{bal:>8.3f}" + "".join(f"{r:>7.2f}" for r in recs))


def run_q3_bridge():
    """Cross-domain: NNLS Billboard bass+treble -> BP48 bass+treble, root-relative."""
    print("\n===== Q3: CROSS-DOMAIN BRIDGE (NNLS Billboard -> BP48 accomp) =====")
    # NNLS billboard: 24-dim = bass[0:12]+treble[12:24], A-referenced -> roll +9 to C, then root-relative.
    nb = np.load(REPO / "data/cache/bass_root_features.npz", allow_pickle=True)
    Xn = nb["feats"].astype(np.float32)  # (N,24) A-ref
    rn = nb["roots"].astype(int); qn = nb["quals"].astype(int)
    # A-ref -> C-ref: roll each 12-block by +9 (i.e. shift -(-9)=... ). extract_bass_root: roll +9 aligns A->C
    bassA, trebA = Xn[:, :12], Xn[:, 12:]
    def to_root_rel(block12, roots, aoff=9):
        # A-ref index -> C-ref: np.roll(x, aoff). then root-relative: roll by -root
        out = np.empty_like(block12)
        for i in range(len(block12)):
            c = np.roll(block12[i], aoff)          # A->C
            out[i] = np.roll(c, -roots[i])         # root->0
        return out
    Xn_rr = np.concatenate([to_root_rel(bassA, rn), to_root_rel(trebA, rn)], axis=1)
    Xn_rr = norm_rows(Xn_rr)

    # BP48 accomp bass+treble root-relative
    d = load_bp48(); y = d["q5"].astype(int); keep = y >= 0
    Xb = build_features(d, ["bass", "treble"], "root")[keep]
    Xb = norm_rows(Xb); yb = y[keep]
    songs = d["song"][keep]; tr, te = split_songs(songs)

    # in-domain BP48 (upper bound)
    bal_in, _ = train_eval(Xb[tr], yb[tr], Xb[te], yb[te])
    # in-domain NNLS (sanity)
    nsong = nb["song_id"]
    ntr, nte = split_songs(nsong)
    bal_nn, _ = train_eval(Xn_rr[ntr], qn[ntr], Xn_rr[nte], qn[nte])
    # cross: train NNLS(all), test BP48(all)
    bal_x, recs_x = train_eval(Xn_rr, qn, Xb, yb)
    # cross + per-dim standardization bridge: standardize both to zero-mean/unit-var per dim
    mu, sd = Xn_rr.mean(0), Xn_rr.std(0) + 1e-6
    Xn_z = (Xn_rr - mu) / sd
    mub, sdb = Xb.mean(0), Xb.std(0) + 1e-6
    Xb_z = (Xb - mub) / sdb
    bal_xz, recs_xz = train_eval(Xn_z, qn, Xb_z, yb)

    print(f"in-domain BP48 (train BP48/test BP48):   bal={bal_in:.3f}")
    print(f"in-domain NNLS (train NNLS/test NNLS):   bal={bal_nn:.3f}")
    print(f"CROSS raw   (train NNLS -> test BP48):   bal={bal_x:.3f}  dom={recs_x[2]:.2f}")
    print(f"CROSS z-norm bridge (per-dim standardize): bal={bal_xz:.3f}  dom={recs_xz[2]:.2f}")


if __name__ == "__main__":
    run_q1_q2()
    run_q3_bridge()
