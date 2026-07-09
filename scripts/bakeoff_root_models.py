"""Vacuum bake-off of root-prediction models on irealb/jazz1460.

Clean, leak-free comparison — the ONLY thing that varies is the model:
  * song-level disjoint split (train = even-indexed songs, eval = odd)
  * EVERY learnable model trained on train split, evaluated on the SAME eval segments
  * grid = ORACLE chord segments (GT boundaries) → isolates root-ID from segmentation
  * unit feature = mean chroma over the chord's beats (48d, 4 L2-normed blocks),
    with ±4 neighbouring chords as context (9 units × 48d = 432d)
  * single headline metric: per-segment root accuracy (unweighted + duration-weighted)
    + the +5/+7 (root↔4th/5th) share of the remaining errors

Models:
  abs_query   LR on the query chord's 48d only         (acoustics, no context, key-biased)
  abs_ctx     LR on the 432d ±4 window                  (acoustics + context, key-biased)
  canon       weight-tied canonical scorer on 432d      (Path A: key-AGNOSTIC progression)
  ltas_canon  canonical scorer on 5-family log-lik feats (Path A via family-LL — the LTAS idea)
  twopath     canon + abs, jointly learned & merged      (USER's proposal)
  viterbi     abs_ctx emissions + relative root-bigram   (MY proposal: progression prior)
  twopath+vit twopath emissions + relative root-bigram   (both ideas combined)

Cache: renders+extracts each song once → data/cache/bakeoff_jazz_feats.pkl
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
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _chroma88, _pool_beats

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CACHE = REPO / "data" / "cache" / "bakeoff_jazz_feats.pkl"
NCTX = 4  # chords of context each side


# ── feature extraction (per song → per-chord mean chroma) ─────────────────────

def _l2norm_blocks(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for s in range(0, len(v), 12):
        n = np.linalg.norm(out[s:s+12])
        if n > 1e-9:
            out[s:s+12] /= n
    return out


def beat_feats(onset_b, note_b):
    n = len(onset_b)
    F = np.zeros((n, 48), np.float32)
    for b in range(n):
        F[b] = np.concatenate([
            _chroma88(onset_b[b]), _chroma88(note_b[b]),
            _chroma88(onset_b[b], 0, 52), _chroma88(onset_b[b], 60, 200),
        ])
    return F


def collect_song(rec, renderer, sf2, ex):
    """Return list of per-chord (feat48, root, n_beats) in order."""
    spb = 60.0 / rec["tempo"]
    n_beats = rec["n_bars"] * rec["beats_per_bar"]
    spans = [(t0, t1, r % 12) for t0, t1, r, q in song_chord_spans(rec)
             if t1 > t0 and q in BUCKET_FAMILY]
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
    F = beat_feats(onset_b, note_b)
    chords = []
    for t0, t1, root in spans:
        b0, b1 = int(round(t0 / spb)), int(round(t1 / spb))
        b0 = max(0, b0); b1 = min(n_beats, max(b1, b0 + 1))
        if b1 <= b0:
            continue
        feat = _l2norm_blocks(F[b0:b1].mean(0))
        chords.append((feat.astype(np.float32), int(root), int(b1 - b0)))
    return chords


def build_cache(n_songs):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    songs = songs[:n_songs]
    print(f"Rendering {len(songs)} jazz songs (one-time)...")
    data = {}
    for i, rec in enumerate(songs):
        print(f"  [{i+1}/{len(songs)}] {rec['song_id']}", end="\r", flush=True)
        try:
            ch = collect_song(rec, renderer, sf2, ex)
        except Exception as e:
            print(f"\n  SKIP {rec['song_id']}: {e}"); continue
        if ch:
            data[rec["song_id"]] = ch
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"\nCached {len(data)} songs → {CACHE}")
    return data


# ── assemble windowed matrices ────────────────────────────────────────────────

def windowize(chords):
    """chords: list of (feat48, root, nbeats). → X (m, 9*48), y (m,), w (m,)."""
    feats = np.stack([c[0] for c in chords])
    roots = np.array([c[1] for c in chords])
    wts = np.array([c[2] for c in chords], float)
    m = len(chords); d = 48; W = 2 * NCTX + 1
    X = np.zeros((m, d * W), np.float32)
    for i in range(m):
        row = []
        for off in range(-NCTX, NCTX + 1):
            j = i + off
            row.append(feats[j] if 0 <= j < m else np.zeros(d, np.float32))
        X[i] = np.concatenate(row)
    return X, roots, wts


def assemble(data, sids):
    Xs, ys, ws, songs = [], [], [], []
    for sid in sids:
        X, y, w = windowize(data[sid])
        Xs.append(X); ys.append(y); ws.append(w); songs += [sid] * len(y)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.array(songs)


# ── models ────────────────────────────────────────────────────────────────────

def roll_idx(d, r):
    idx = np.arange(d)
    for s in range(0, d, 12):
        idx[s:s+12] = s + (np.arange(12) + r) % 12
    return idx


FAM_TMPL = {  # pitch-class sets → L1-normed 12d template (for LTAS-style feature)
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6],
    "aug": [0, 4, 8], "dom7": [0, 4, 7, 10],
}


def _fam_templates():
    T = np.full((5, 12), 1e-3, np.float32)
    for k, (name, pcs) in enumerate(FAM_TMPL.items()):
        for p in pcs:
            T[k, p] = 1.0
    T /= T.sum(1, keepdims=True)
    return np.log(T)  # (5,12) log-template


def ltas_feats(X, r):
    """Per-candidate family log-lik of each window block's onset-chroma, rolled by -r.
    X (m, 9*48) → (m, 9*5). Uses block 0 (onset chroma) of each of the 9 units."""
    logT = _fam_templates()
    m = X.shape[0]
    Xr = X[:, roll_idx(X.shape[1], r)]
    out = np.zeros((m, 9 * 5), np.float32)
    for u in range(9):
        onset = Xr[:, u*48:u*48+12]          # onset chroma block of unit u
        out[:, u*5:u*5+5] = onset @ logT.T     # (m,5) family LL
    return out


# --- sklearn absolute LRs ---

def fit_abs(Xtr, ytr, cols=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xt = Xtr[:, cols] if cols is not None else Xtr
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xt), ytr)
    return sc, clf, cols


def proba_abs(model, X):
    sc, clf, cols = model
    Xt = X[:, cols] if cols is not None else X
    p = clf.predict_proba(sc.transform(Xt))
    out = np.zeros((len(X), 12), np.float32)
    for i, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, i]
    return out


# --- torch canonical / two-path ---

def _std(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def train_torch(Xtr, ytr, mode, feat="chroma", epochs=30, hidden=64, seed=0):
    """mode in {'canon','abs','twopath'}. feat in {'chroma','ltas'} for canonical path.
    Returns a predictor closure proba(X)->(n,12)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]

    # candidate feature builder
    if feat == "ltas":
        cand = lambda X, r: ltas_feats(X, r)
        cdim = 9 * 5
    else:
        cand = lambda X, r: X[:, roll_idx(d, r)]
        cdim = d

    # standardizers (from true-root canonical view for canon path; raw for abs)
    canonX = np.stack([cand(Xtr[i:i+1], ytr[i])[0] for i in range(len(Xtr))])
    cmu, csd = _std(canonX)
    amu, asd = _std(Xtr)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if mode in ("canon", "twopath"):
                self.canon = nn.Sequential(nn.Linear(cdim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if mode in ("abs", "twopath"):
                self.absn = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 12))

        def canon_logits(self, X):  # X (B,d) numpy
            import torch
            sc = torch.zeros(len(X), 12)
            for r in range(12):
                cf = torch.tensor((cand(X, r) - cmu) / csd)
                sc[:, r] = self.canon(cf).squeeze(-1)
            return sc

        def abs_logits(self, X):
            import torch
            return self.absn(torch.tensor((X - amu) / asd))

        def logits(self, X):
            if mode == "canon":
                return self.canon_logits(X)
            if mode == "abs":
                return self.abs_logits(X)
            return self.canon_logits(X) + self.abs_logits(X)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    N = len(Xtr); bs = 256
    yt = ytr.astype(np.int64)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        for s in range(0, N, bs):
            bi = perm[s:s+bs]
            lg = net.logits(Xtr[bi])
            loss = lossf(lg, torch.tensor(yt[bi]))
            opt.zero_grad(); loss.backward(); opt.step()

    def proba(X):
        import torch
        with torch.no_grad():
            lg = net.logits(X).numpy()
        lg -= lg.max(1, keepdims=True)
        e = np.exp(lg); return e / e.sum(1, keepdims=True)
    return proba


# --- bass-anchored models (rotation fixed by an OBSERVABLE, not the oracle root) ---

def bass_anchor(X):
    """Per row, dominant bass pitch class of the QUERY chord = argmax of its mean
    bass-register chroma block (block 2, cols +24:+36 of the center unit)."""
    c = NCTX * 48
    return X[:, c+24:c+36].argmax(1).astype(int)  # (m,)


def roll_rows(X, anchors):
    """Roll every 12d block of each row by -anchor[i] (bass PC → pc 0)."""
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = X[i, roll_idx(X.shape[1], int(anchors[i]))]
    return out


def ltas_from_window(Xw):
    """Rolled window (m, 9*48) → (m, 9*5) family log-lik of each unit's onset chroma."""
    logT = _fam_templates()
    m = Xw.shape[0]
    out = np.zeros((m, 9 * 5), np.float32)
    for u in range(9):
        out[:, u*5:u*5+5] = Xw[:, u*48:u*48+12] @ logT.T
    return out


def offset_to_abs(P_off, anchors):
    """P_off (m,12) over (root-bass)%12  →  absolute-root posterior (m,12)."""
    out = np.zeros_like(P_off)
    for i in range(len(P_off)):
        out[i] = np.roll(P_off[i], int(anchors[i]))  # P_abs[(a+o)%12] = P_off[o]
    return out


def fit_ba(Xtr, ytr, atr, feat="chroma"):
    """Bass-anchored LR predicting offset=(root-bass)%12. Returns predictor over ABS root."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xw = roll_rows(Xtr, atr)
    F = ltas_from_window(Xw) if feat == "ltas" else Xw
    yoff = (ytr - atr) % 12
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(F), yoff)

    def proba_abs_root(X):
        a = bass_anchor(X)
        Xw = roll_rows(X, a)
        Ff = ltas_from_window(Xw) if feat == "ltas" else Xw
        p = clf.predict_proba(sc.transform(Ff))
        P_off = np.zeros((len(X), 12), np.float32)
        for i, c in enumerate(clf.classes_):
            P_off[:, int(c)] = p[:, i]
        return offset_to_abs(P_off, a)
    return proba_abs_root


# --- viterbi progression re-ranker ---

def fit_rel_transition(data, sids, smooth=1.0):
    """Learn P(delta) where delta=(root_next-root_prev)%12 from train chord bigrams."""
    counts = np.full(12, smooth)
    for sid in sids:
        roots = [c[1] for c in data[sid]]
        for a, b in zip(roots[:-1], roots[1:]):
            counts[(b - a) % 12] += 1
    return np.log(counts / counts.sum())  # (12,) log P(delta)


def viterbi_rerank(emit_proba, songs, logP_delta):
    """Per song: argmax path over roots with emission=log proba, transition=logP(delta)."""
    out = np.zeros(len(emit_proba), int)
    logE = np.log(emit_proba + 1e-9)
    T = np.zeros((12, 12), np.float32)  # T[i,j] = logP(j from i)
    for i in range(12):
        for j in range(12):
            T[i, j] = logP_delta[(j - i) % 12]
    for sid in np.unique(songs):
        idx = np.where(songs == sid)[0]
        n = len(idx)
        dp = np.zeros((n, 12)); bp = np.zeros((n, 12), int)
        dp[0] = logE[idx[0]]
        for t in range(1, n):
            for j in range(12):
                sc = dp[t-1] + T[:, j]
                bp[t, j] = int(sc.argmax()); dp[t, j] = sc.max() + logE[idx[t], j]
        j = int(dp[-1].argmax())
        path = [j]
        for t in range(n-1, 0, -1):
            j = bp[t, j]; path.append(j)
        out[idx] = path[::-1]
    return out


# ── eval ──────────────────────────────────────────────────────────────────────

def score(name, pred, y, w, results):
    ok = pred == y
    acc = ok.mean(); wacc = (ok * w).sum() / w.sum()
    err = ~ok
    iv = (pred[err] - y[err]) % 12
    p57 = ((iv == 5).sum() + (iv == 7).sum()) / max(err.sum(), 1)
    results.append((name, acc, wacc, p57))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=70)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild or not CACHE.exists():
        data = build_cache(args.n_songs)
    else:
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cache: {len(data)} songs")

    sids = sorted(data.keys())
    tr_sids = sids[0::2]; ev_sids = sids[1::2]  # disjoint even/odd
    Xtr, ytr, wtr, str_ = assemble(data, tr_sids)
    Xev, yev, wev, sev = assemble(data, ev_sids)
    print(f"train segs {len(ytr)} ({len(tr_sids)} songs) | eval segs {len(yev)} ({len(ev_sids)} songs)\n")

    results = []

    # sklearn absolute baselines
    center = slice(NCTX * 48, (NCTX + 1) * 48)  # query chord's own 48d
    m = fit_abs(Xtr, ytr, cols=center)   # query only
    score("abs_query (query chroma, no ctx)", proba_abs(m, Xev).argmax(1), yev, wev, results)
    abs_ctx_model = fit_abs(Xtr, ytr)  # full 432d
    abs_ctx_p = proba_abs(abs_ctx_model, Xev)
    score("abs_ctx (LR, ±4 ctx)", abs_ctx_p.argmax(1), yev, wev, results)

    # torch canonical / two-path
    canon_p = train_torch(Xtr, ytr, "canon")(Xev)
    score("canon (Path A: key-agnostic)", canon_p.argmax(1), yev, wev, results)
    ltas_p = train_torch(Xtr, ytr, "canon", feat="ltas")(Xev)
    score("ltas_canon (Path A via family-LL)", ltas_p.argmax(1), yev, wev, results)
    two_p = train_torch(Xtr, ytr, "twopath")(Xev)
    score("twopath (USER: canon+abs merged)", two_p.argmax(1), yev, wev, results)

    # bass-anchored models (rotation fixed by observed bass, not oracle root)
    atr = bass_anchor(Xtr); aev = bass_anchor(Xev)
    score("bass=root (naive anchor)", aev, yev, wev, results)
    ba_abs_p = fit_ba(Xtr, ytr, atr, feat="chroma")(Xev)
    score("bass_anchored (full chroma)", ba_abs_p.argmax(1), yev, wev, results)
    ba_ltas_p = fit_ba(Xtr, ytr, atr, feat="ltas")(Xev)
    score("bass_anchored (family-LL)", ba_ltas_p.argmax(1), yev, wev, results)

    # SUPER-MODEL: ensemble full-chroma canonical view + bass-anchored view
    score("canon ⊕ bass_anchored (avg)", (canon_p + ba_abs_p).argmax(1), yev, wev, results)
    score("canon ⊕ bass_anchored (prod)",
          (np.log(canon_p + 1e-9) + np.log(ba_abs_p + 1e-9)).argmax(1), yev, wev, results)

    # viterbi progression re-rank (MY proposal) over abs_ctx + over twopath
    logPd = fit_rel_transition(data, tr_sids)
    score("viterbi (MINE: abs_ctx + root-bigram)",
          viterbi_rerank(abs_ctx_p, sev, logPd), yev, wev, results)
    score("twopath + viterbi (combined)",
          viterbi_rerank(two_p, sev, logPd), yev, wev, results)

    # report
    print(f"{'model':<40} {'segAcc':>7} {'wAcc':>7} {'+5/7err':>8}")
    print("-" * 66)
    for name, acc, wacc, p57 in sorted(results, key=lambda r: -r[2]):
        print(f"{name:<40} {acc:>7.1%} {wacc:>7.1%} {p57:>8.1%}")


if __name__ == "__main__":
    main()
