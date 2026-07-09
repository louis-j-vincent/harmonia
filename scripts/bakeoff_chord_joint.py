"""Does JOINT (root × quality) canonical chord prediction beat the SPLIT approach?

Same leak-free vacuum as bakeoff_root_models.py (irealb, oracle chord segments, mean
chroma ±4 ctx, disjoint 35/35 split).  Compares three ways to get the full chord:

  split : canonical root head → root; separate LR classifies quality in the
          root-aligned (canonical) frame given the PREDICTED root.  (≈ what the
          pipeline does today: v4 root + root-conditioned family clf.)
  dft   : canonical root head → root; quality from |DFT| magnitudes (v3's head —
          shift-invariant but throws away the phase that separates {0,4,7} vs {0,3,7}).
  joint : one weight-tied MLP emits a K-way quality distribution PER candidate root;
          argmax over the 12×K grid = (root, quality) jointly.

Metrics (per-segment, disjoint eval): root / majmin / sevenths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import bakeoff_root_models as bo
from analyze_accomp_emission import song_chord_spans
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _pool_beats

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CACHE = REPO / "data" / "cache" / "bakeoff_chord_joint.pkl"
NCTX = bo.NCTX

# 6-class quality (finer than the 5-class v3 scheme: keeps min vs min7 for 7ths)
QUAL = ["maj", "min", "maj7", "min7", "7", "dim"]
QIDX = {q: i for i, q in enumerate(QUAL)}
QMAP = {
    "maj": "maj", "6": "maj", "maj6": "maj", "aug": "maj",
    "maj7": "maj7", "majmaj7": "maj7",
    "7": "7", "9": "7", "dom7": "7", "dom7alt": "7", "aug7": "7",
    "min": "min", "m": "min", "m6": "min", "min6": "min",
    "min7": "min7", "minmaj7": "min7",
    "dim": "dim", "dim7": "dim", "m7b5": "dim", "hdim7": "dim",
}
MAJMIN = {"maj": "M", "maj7": "M", "7": "M", "min": "m", "min7": "m", "dim": None}


def qual6(tok):
    q = QMAP.get(tok)
    return QIDX[q] if q is not None else None


# ── collect per-chord (mean chroma, root, quality) ────────────────────────────

def collect(rec, renderer, sf2, ex):
    spb = 60.0 / rec["tempo"]; n_beats = rec["n_bars"] * rec["beats_per_bar"]
    spans = []
    for t0, t1, r, q in song_chord_spans(rec):
        qi = qual6(q)
        if t1 > t0 and qi is not None:
            spans.append((t0, t1, r % 12, qi))
    if not spans:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)
    bt = np.arange(n_beats + 1) * spb
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
    F = bo.beat_feats(onset_b, note_b)
    chords = []
    for t0, t1, root, qi in spans:
        b0, b1 = int(round(t0 / spb)), int(round(t1 / spb))
        b0 = max(0, b0); b1 = min(n_beats, max(b1, b0 + 1))
        if b1 <= b0:
            continue
        chords.append((bo._l2norm_blocks(F[b0:b1].mean(0)).astype(np.float32), int(root), int(qi)))
    return chords


def build_cache(n):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()][:n]
    print(f"Rendering {len(songs)} songs...")
    data = {}
    for i, rec in enumerate(songs):
        print(f"  [{i+1}/{len(songs)}]", end="\r", flush=True)
        try:
            c = collect(rec, renderer, sf2, ex)
        except Exception as e:
            print(f"\n SKIP {rec['song_id']}: {e}"); continue
        if c:
            data[rec["song_id"]] = c
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"\nCached {len(data)} songs")
    return data


def windowize(chords):
    feats = np.stack([c[0] for c in chords])
    roots = np.array([c[1] for c in chords]); quals = np.array([c[2] for c in chords])
    m, d = len(chords), 48
    X = np.zeros((m, d * (2 * NCTX + 1)), np.float32)
    for i in range(m):
        X[i] = np.concatenate([feats[i + o] if 0 <= i + o < m else np.zeros(d, np.float32)
                               for o in range(-NCTX, NCTX + 1)])
    return X, roots, quals


def assemble(data, sids):
    Xs, rs, qs = [], [], []
    for sid in sids:
        X, r, q = windowize(data[sid]); Xs.append(X); rs.append(r); qs.append(q)
    return np.vstack(Xs), np.concatenate(rs), np.concatenate(qs)


# ── models ────────────────────────────────────────────────────────────────────

def canon_root_proba(Xtr, ytr, Xev, epochs=25):
    return bo.train_torch(Xtr, ytr, "canon", epochs=epochs)(Xev)


def split_quality(Xtr, rtr, qtr, Xev, root_pred):
    """LR quality in canonical frame: train on TRUE-root-rolled feats, test on PRED-root."""
    d = Xtr.shape[1]
    Xc = np.stack([Xtr[i, bo.roll_idx(d, rtr[i])] for i in range(len(Xtr))])
    sc = StandardScaler().fit(Xc)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xc), qtr)
    Xe = np.stack([Xev[i, bo.roll_idx(d, root_pred[i])] for i in range(len(Xev))])
    return clf.predict(sc.transform(Xe))


def dft_quality(Xtr, qtr, Xev):
    from train_beat_seq_model_v3 import dft_features
    Dtr = dft_features(Xtr); Dev = dft_features(Xev)
    sc = StandardScaler().fit(Dtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Dtr), qtr)
    return clf.predict(sc.transform(Dev))


def joint_canon(Xtr, rtr, qtr, Xev, K=6, hidden=96, epochs=25, seed=0):
    """Weight-tied MLP → K quality logits per candidate root; argmax over 12×K."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    ytr = (rtr * K + qtr).astype(np.int64)
    canonX = np.stack([Xtr[i, bo.roll_idx(d, rtr[i])] for i in range(len(Xtr))])
    mu, sd = canonX.mean(0).astype(np.float32), (canonX.std(0) + 1e-6).astype(np.float32)
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, K))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    mt, st = torch.tensor(mu), torch.tensor(sd)

    def logits(X):
        out = torch.zeros(len(X), 12, K)
        for r in range(12):
            cf = torch.tensor((X[:, bo.roll_idx(d, r)] - mu) / sd)
            out[:, r, :] = net(cf)
        return out.reshape(len(X), 12 * K)

    N = len(Xtr); bs = 256
    for ep in range(epochs):
        perm = np.random.permutation(N)
        for s in range(0, N, bs):
            bi = perm[s:s + bs]
            loss = lossf(logits(Xtr[bi]), torch.tensor(ytr[bi]))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cell = logits(Xev).numpy().argmax(1)
    return cell // K, cell % K  # root, qual


# ── metrics ───────────────────────────────────────────────────────────────────

def report(name, rp, qp, rgt, qgt, out):
    root = (rp == rgt).mean()
    mm_mask = np.array([MAJMIN[QUAL[q]] is not None for q in qgt])
    mm = ((rp == rgt) & np.array([MAJMIN[QUAL[a]] == MAJMIN[QUAL[b]]
                                  for a, b in zip(qp, qgt)]))[mm_mask].mean()
    sev = ((rp == rgt) & (qp == qgt)).mean()
    out.append((name, root, mm, sev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild or not CACHE.exists():
        data = build_cache(args.n)
    else:
        data = pickle.load(open(CACHE, "rb")); print(f"Loaded cache: {len(data)} songs")

    sids = sorted(data.keys())
    Xtr, rtr, qtr = assemble(data, sids[0::2])
    Xev, rev, qev = assemble(data, sids[1::2])
    print(f"train {len(rtr)} / eval {len(rev)} chords\n")

    root_p = canon_root_proba(Xtr, rtr, Xev).argmax(1)
    out = []
    report("split (canon root + canon-LR qual)", root_p,
           split_quality(Xtr, rtr, qtr, Xev, root_p), rev, qev, out)
    report("dft   (canon root + |DFT| qual)", root_p,
           dft_quality(Xtr, qtr, Xev), rev, qev, out)
    jr, jq = joint_canon(Xtr, rtr, qtr, Xev)
    report("joint (canon root×qual)", jr, jq, rev, qev, out)

    print(f"{'model':<38} {'root':>7} {'majmin':>7} {'7ths':>7}")
    print("-" * 62)
    for name, r, m, s in out:
        print(f"{name:<38} {r:>7.1%} {m:>7.1%} {s:>7.1%}")


if __name__ == "__main__":
    main()
