"""Context-window family classifier: full key×family LL matrix as input.

Per segment, the feature is:
  - 12d  : LTAS chroma mean (point estimate, root-shifted)
  - (9, 5, 12): per-context-position × 5-families × 12-keys LL matrix
                position 4 is the current segment, 0-3 are the 4 previous,
                5-8 are the 4 next.

Models compared (all 5-fold CV, balanced, hard audio, oracle boundaries):
  logreg_base       — 12d chroma + 5d mean-LL  (the existing ~80% baseline)
  logreg_ctx_flat   — 12d chroma + flatten(9×5×12) ctx  (MLP-style)
  mlp_ctx           — PyTorch 2-layer MLP on same flat input
  cnn_ctx           — 1D CNN over 9 time steps, (5×12)=60 channels
  lstm_ctx          — LSTM over 9 time steps, 60d input

Accuracy metric: standard + hierarchical (predict child → parent counted correct).

Usage:
    .venv/bin/python scripts/experiment_ctx_model.py
    .venv/bin/python scripts/experiment_ctx_model.py --n-songs 50
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp as sp_logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import (
    SCENARIOS, SOUNDFONTS, LEAD_PROGRAMS,
    make_melody, render_to_array, stem_midi, time_varying_degrade,
)
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
DIST_CACHE = REPO / "data" / "cache" / "ltas_family_dist.npz"
PLOT_OUT   = REPO / "docs" / "plots" / "ctx_model_comparison.png"

FAMILIES   = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = ["#58d4ff", "#a65fd4", "#e34948", "#e0a03b", "#1baf7a"]
NOTE       = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
HOP        = 512
CTX_K      = 4   # ±4 segments

# Parent mapping for hierarchical accuracy
# child family → same family (family level is flat, no hierarchy there)
# but at exact level: predict dom7 when GT=maj → family correct → hierarchical credit
# Here we only classify family, so hierarchical means: if pred is in the same
# family subtree as GT. At the family level this IS the standard accuracy.
# We implement it at exact level when we have that info, but here just track
# "correct family AND correct family" = standard accuracy for now.
# The hierarchical bonus applies when we extend to exact — see note at bottom.


# ── audio / chroma ────────────────────────────────────────────────────────────

def _render_hard(midi_path, rng):
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm    = pretty_midi.PrettyMIDI(str(midi_path))
    scen  = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf    = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    if gains.get("melody", 0) > 0.01:
        mel_pm = pretty_midi.PrettyMIDI()
        m = make_melody(pm, int(rng.choice(LEAD_PROGRAMS)), rng)
        if m: mel_pm.instruments.append(m); stems["melody"] = mel_pm
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr2 = render_to_array(renderer, s, sf, reverb=False)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


def _ltas_frames(audio, sr):
    raw  = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=HOP)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    ct   = librosa.frames_to_time(np.arange(raw.shape[1]), sr=sr, hop_length=HOP)
    return raw / ltas, ct


# ── LL computation ────────────────────────────────────────────────────────────

def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def compute_key_family_ll(chroma_mean: np.ndarray, dist: dict) -> np.ndarray:
    """
    Given a 12d absolute-pitch LTAS chroma mean, compute LL for each
    (family, key) pair.

    Returns (5, 12) matrix where [fi, r] = LL of rolling x by -r
    against family fi's root-shifted distribution.

    This is the full information matrix — the 5d max-LL vector is just
    argmax over axis 1 of this.
    """
    n = np.linalg.norm(chroma_mean)
    if n < 1e-9:
        return np.zeros((5, 12), dtype=np.float32)
    x = chroma_mean / n
    ll = np.zeros((5, 12), dtype=np.float32)
    for fi, fam in enumerate(FAMILIES):
        mu  = dist[f"{fam}_mu"]
        std = dist[f"{fam}_std"]
        for r in range(12):
            ll[fi, r] = _diag_ll(np.roll(x, -r), mu, std)
    return ll


def compute_softmax_att_ll(frames_shifted: np.ndarray, dist: dict) -> np.ndarray:
    """
    Per-frame softmax-attention aggregate over (5,12) LL matrix.
    Returns (5, 12) — the attention-weighted sum of per-frame (5,12) matrices.
    """
    T = frames_shifted.shape[1]
    if T == 0:
        return np.zeros((5, 12), dtype=np.float32)

    frame_mats = np.zeros((T, 5, 12), dtype=np.float32)
    for t in range(T):
        x = frames_shifted[:, t]
        nn = np.linalg.norm(x)
        if nn < 1e-9: continue
        frame_mats[t] = compute_key_family_ll(x, dist)

    # confidence of each frame = max LL across all (family, key)
    confidence = frame_mats.max(axis=(1, 2))      # (T,)
    w = np.exp(confidence - confidence.max())
    w = w / (w.sum() + 1e-12)                     # (T,) softmax weights
    return (w[:, None, None] * frame_mats).sum(axis=0)  # (5, 12)


# ── data collection ───────────────────────────────────────────────────────────

def collect(n_songs, dist, rng):
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail  = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    records = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{n_songs}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], rng)
        except Exception:
            continue
        chroma, ct = _ltas_frames(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}

        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY: continue
            fam = BUCKET_FAMILY[p[1]]
            if fam not in FAMILIES: continue
            root = int(root_gt % 12)

            i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            frames_abs     = chroma[:, i0:i1]
            frames_shifted = np.roll(frames_abs, -root, axis=0)

            # chroma mean (root-shifted, L2-normed)
            mean_s = frames_shifted.mean(axis=1)
            nn = np.linalg.norm(mean_s)
            chroma_mean = (mean_s / nn).astype(np.float32) if nn > 1e-9 else np.zeros(12, np.float32)

            # (5, 12) LL matrix via softmax-att aggregation over frames
            ll_mat = compute_softmax_att_ll(frames_shifted, dist)  # (5, 12)

            records.append({
                "y":           FAMILIES.index(fam),
                "chroma_mean": chroma_mean,       # (12,) root-shifted: root at index 0
                "ll_mat":      ll_mat,            # (5, 12)  ← full key×family info
                "root_pc":     root,              # (int) GT root 0–11, needed for key unification
            })
    print()
    return records


# ── context window assembly ───────────────────────────────────────────────────

def _ctx_tensor(records, k=CTX_K):
    """
    For each segment i, build a (2k+1, 5, 12) context tensor of ll_mat values,
    zero-padded at boundaries.

    Key unification: each neighbor j's ll_mat is rolled so that column 0
    corresponds to segment i's root (not j's own root).  This makes the
    tensor fully key-agnostic — chord i's root is always "C" from the model's
    perspective, and context positions encode *relative* key relationships.

    Roll amount: delta = (root_j - root_i) % 12
    np.roll(ll_mat_j, -delta, axis=1) shifts j's axis so that key=root_i
    lands at column 0.

    Returns (N, 2k+1, 5, 12) float32.
    """
    N = len(records)
    W = 2 * k + 1
    out = np.zeros((N, W, 5, 12), dtype=np.float32)
    for i in range(N):
        root_i = records[i]["root_pc"]
        for j, offset in enumerate(range(-k, k + 1)):
            ni = i + offset
            if 0 <= ni < N:
                root_j = records[ni]["root_pc"]
                delta  = (root_j - root_i) % 12
                # roll j's key axis so key=root_i lands at column 0
                out[i, j] = np.roll(records[ni]["ll_mat"], -delta, axis=1)
    return out


# ── sklearn baseline ──────────────────────────────────────────────────────────

def _cv_logreg(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5, 5), int)
    sc   = StandardScaler()
    for tr, va in skf.split(X, y):
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr]); pred = clf.predict(Xva)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_tot


# ── PyTorch models ────────────────────────────────────────────────────────────

def _has_torch():
    try:
        import torch; return True
    except ImportError:
        return False


def _cv_torch(model_fn, X_tensor, y, n_splits=5, epochs=40, lr=3e-3, batch=64):
    """
    X_tensor : (N, ...) float32 numpy array
    model_fn : callable(input_dim) → nn.Module
    """
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5, 5), int)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    for fold, (tr, va) in enumerate(skf.split(X_tensor, y)):
        Xtr = torch.tensor(X_tensor[tr], dtype=torch.float32).to(device)
        Xva = torch.tensor(X_tensor[va], dtype=torch.float32).to(device)
        ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
        yva = torch.tensor(y[va], dtype=torch.long).to(device)

        # class weights
        counts = np.bincount(y[tr], minlength=5).astype(float)
        wts = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32).to(device)
        wts = wts / wts.sum() * 5

        model = model_fn(X_tensor.shape[1:]).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(weight=wts)

        ds  = TensorDataset(Xtr, ytr)
        dl  = DataLoader(ds, batch_size=batch, shuffle=True)

        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).cpu().numpy()
        yva_np = yva.cpu().numpy()
        accs.append((pred == yva_np).mean())
        cm_tot += confusion_matrix(yva_np, pred, labels=list(range(5)))
        print(f"    fold {fold+1}/{n_splits}  val={accs[-1]:.1%}", flush=True)

    return float(np.mean(accs)), float(np.std(accs)), cm_tot


def _make_mlp(flat_dim):
    """2-layer MLP: flat_dim → 256 → 128 → 5."""
    import torch.nn as nn
    d = int(np.prod(flat_dim))
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, 5),
    )


class _CNN(object):
    """1D CNN: (batch, W, 5*12) → 5.  W = time steps (9)."""
    def __new__(cls, shape):
        import torch.nn as nn
        W, F5, K12 = shape  # (9, 5, 12)
        C = F5 * K12        # 60 channels
        return nn.Sequential(
            _Reshape(-1, C, W),         # (B, 60, 9) — channels-first
            nn.Conv1d(C, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),    # (B, 64, 1)
            nn.Flatten(),
            nn.Linear(64, 5),
        )


class _LSTM(object):
    """LSTM: sequence of W steps × (5*12)d → 5."""
    def __new__(cls, shape):
        import torch.nn as nn
        W, F5, K12 = shape
        return _LSTMModel(input_dim=F5 * K12, hidden=128, n_layers=2, n_class=5)


class _Reshape(object):
    """Helper nn.Module for reshape inside Sequential."""
    def __new__(cls, *shape):
        import torch.nn as nn
        class R(nn.Module):
            def __init__(self): super().__init__(); self.shape = shape
            def forward(self, x): return x.reshape(self.shape[0], *self.shape[1:])
        return R()


class _LSTMModel(object):
    def __new__(cls, input_dim, hidden, n_layers, n_class):
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.flat = nn.Flatten(start_dim=2)   # (B, W, 5*12)
                self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                    batch_first=True, dropout=0.2)
                self.head = nn.Linear(hidden, n_class)
            def forward(self, x):
                x = self.flat(x)            # (B, W, 60)
                _, (h, _) = self.lstm(x)   # h: (n_layers, B, hidden)
                return self.head(h[-1])     # (B, n_class)
        return M()


# ── entropy gate ─────────────────────────────────────────────────────────────

def _entropy(logits: np.ndarray) -> np.ndarray:
    """Shannon entropy of softmax(logits), shape (N,)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs  = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)   # (N,)


def _fit_entropy_gate(logits_base, logits_ctx, y_tr):
    """
    Fit 2 scalars (w, b) s.t.
        α = sigmoid(w · H(p_base) + b)
        p_final = α · softmax(logits_base) + (1-α) · softmax(logits_ctx)
    by minimising cross-entropy on (logits_base_tr, logits_ctx_tr, y_tr).

    Returns (w, b) as floats.
    """
    import torch, torch.nn as nn

    def _softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x); return e / e.sum(axis=1, keepdims=True)

    pb = torch.tensor(_softmax(logits_base), dtype=torch.float32)
    pc = torch.tensor(_softmax(logits_ctx),  dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    H  = -(pb * (pb + 1e-12).log()).sum(1, keepdim=True)   # (N,1) entropy

    w = torch.nn.Parameter(torch.tensor([1.0]))
    b = torch.nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        alpha  = torch.sigmoid(w * H + b)          # (N,1)
        p_mix  = alpha * pb + (1 - alpha) * pc     # (N,5)
        loss   = nn.NLLLoss()(p_mix.log(), yt)
        loss.backward(); return loss
    opt.step(closure)

    return float(w.item()), float(b.item())


def _apply_entropy_gate(logits_base, logits_ctx, w, b):
    """Apply fitted (w, b) gate. Returns hard predictions (N,)."""
    import torch
    def _softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x); return e / e.sum(axis=1, keepdims=True)
    pb = torch.tensor(_softmax(logits_base), dtype=torch.float32)
    pc = torch.tensor(_softmax(logits_ctx),  dtype=torch.float32)
    H  = -(pb * (pb + 1e-12).log()).sum(1, keepdim=True)
    alpha = torch.sigmoid(torch.tensor(w) * H + torch.tensor(b))
    p_mix = alpha * pb + (1 - alpha) * pc
    return p_mix.argmax(1).numpy()


def _cv_entropy_gate(logits_base_all, logits_ctx_all, y, n_splits=5):
    """
    CV over entropy-gated blend of base and ctx logits.
    Both inputs: (N, 5) raw logits from their respective classifiers.
    """
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs    = []; cm_tot = np.zeros((5, 5), int)
    ws, bs  = [], []

    for tr, va in skf.split(logits_base_all, y):
        w, b = _fit_entropy_gate(logits_base_all[tr], logits_ctx_all[tr], y[tr])
        ws.append(w); bs.append(b)
        pred = _apply_entropy_gate(logits_base_all[va], logits_ctx_all[va], w, b)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))

    print(f"    gate params: w={np.mean(ws):.3f}±{np.std(ws):.3f}  "
          f"b={np.mean(bs):.3f}±{np.std(bs):.3f}")
    print(f"    α interpretation: H_max={np.log(5):.2f} → "
          f"α={1/(1+np.exp(-(np.mean(ws)*np.log(5)+np.mean(bs)))):.2f} (use ctx), "
          f"H=0 → α={1/(1+np.exp(-np.mean(bs))):.2f} (use base)")
    return float(np.mean(accs)), float(np.std(accs)), cm_tot


# ── collect logits from CV folds ──────────────────────────────────────────────

def _cv_logreg_logits(X, y, n_splits=5):
    """Like _cv_logreg but returns out-of-fold logits (N,5) alongside metrics."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5, 5), int)
    sc   = StandardScaler()
    oof_logits = np.zeros((len(y), 5), dtype=np.float32)
    for tr, va in skf.split(X, y):
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        oof_logits[va] = clf.decision_function(Xva)
        pred = clf.predict(Xva)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_tot, oof_logits


def _cv_torch_logits(model_fn, X_tensor, y, n_splits=5, epochs=60, lr=3e-3, batch=64):
    """Like _cv_torch but also returns out-of-fold logits (N,5)."""
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs   = []; cm_tot = np.zeros((5, 5), int)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    oof_logits = np.zeros((len(y), 5), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X_tensor, y)):
        Xtr = torch.tensor(X_tensor[tr], dtype=torch.float32).to(device)
        Xva = torch.tensor(X_tensor[va], dtype=torch.float32).to(device)
        ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
        yva = torch.tensor(y[va], dtype=torch.long).to(device)
        counts = np.bincount(y[tr], minlength=5).astype(float)
        wts = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32).to(device)
        wts = wts / wts.sum() * 5
        model   = model_fn(X_tensor.shape[1:]).to(device)
        opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(weight=wts)
        dl      = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva).cpu().numpy()
        oof_logits[va] = logits
        pred = logits.argmax(1)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
        print(f"    fold {fold+1}/{n_splits}  val={accs[-1]:.1%}", flush=True)

    return float(np.mean(accs)), float(np.std(accs)), cm_tot, oof_logits


# ── hierarchical accuracy ─────────────────────────────────────────────────────

def hierarchical_acc(cm: np.ndarray) -> float:
    """At family level, hierarchical == standard (no parent above family)."""
    return cm.diagonal().sum() / cm.sum()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=40)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--epochs",  type=int, default=60)
    args = ap.parse_args()

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); sys.exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting ({args.n_songs} songs, hard audio, oracle bounds)...")
    records = collect(args.n_songs, dist, rng)
    print(f"  {len(records)} segments")

    y = np.array([r["y"] for r in records])

    # ── build all feature matrices ────────────────────────────────────────────
    # (1) logreg baseline: 12d chroma + 5d max-LL
    X_base_ll = np.stack([r["ll_mat"].max(axis=1) for r in records])   # (N, 5)
    X_chroma  = np.stack([r["chroma_mean"] for r in records])           # (N, 12)
    X_logreg  = np.concatenate([X_chroma, X_base_ll], axis=1)          # (N, 17)

    # (2) ctx tensor: (N, 9, 5, 12)
    ctx = _ctx_tensor(records, CTX_K)

    # (3) logreg+ctx flat: 12d chroma + flatten(9×5×12)
    ctx_flat   = ctx.reshape(len(records), -1)                          # (N, 540)
    X_logreg_ctx = np.concatenate([X_chroma, ctx_flat], axis=1)        # (N, 552)

    results        = {}
    oof_base_logits = None
    oof_mlp_logits  = None

    print("\n── LogReg baseline (17d) ──")
    mu, sd, cm, oof_base_logits = _cv_logreg_logits(X_logreg, y)
    results["logreg_base"] = (mu, sd, cm)
    print(f"  {mu:.1%} ± {sd:.1%}")

    print("\n── LogReg + ctx flat (552d) ──")
    mu, sd, cm, _ = _cv_logreg_logits(X_logreg_ctx, y)
    results["logreg_ctx"] = (mu, sd, cm)
    print(f"  {mu:.1%} ± {sd:.1%}")

    if _has_torch():
        import torch
        print("\n── MLP (flat ctx) ──")
        mu, sd, cm, oof_mlp_logits = _cv_torch_logits(_make_mlp, X_logreg_ctx, y,
                                                        epochs=args.epochs)
        results["mlp"] = (mu, sd, cm)
        print(f"  {mu:.1%} ± {sd:.1%}")

        print("\n── CNN (9 steps × 60d) ──")
        mu, sd, cm, _ = _cv_torch_logits(_CNN, ctx, y, epochs=args.epochs)
        results["cnn"] = (mu, sd, cm)
        print(f"  {mu:.1%} ± {sd:.1%}")

        print("\n── LSTM (9 steps × 60d) ──")
        mu, sd, cm, _ = _cv_torch_logits(_LSTM, ctx, y, epochs=args.epochs)
        results["lstm"] = (mu, sd, cm)
        print(f"  {mu:.1%} ± {sd:.1%}")

        # ── Entropy gate: blend base (point estimate) with MLP (context) ─────────
        # α = sigmoid(w·H(p_base) + b)  — when base is confident (low H), trust it;
        # when ambiguous (high H), let context dominate.
        if oof_base_logits is not None and oof_mlp_logits is not None:
            print("\n── Entropy gate (base ↔ MLP) ──")
            mu, sd, cm = _cv_entropy_gate(oof_base_logits, oof_mlp_logits, y)
            results["gate_base_mlp"] = (mu, sd, cm)
            print(f"  {mu:.1%} ± {sd:.1%}")
    else:
        print("\n(PyTorch not installed — skipping MLP/CNN/LSTM/gate. pip install torch)")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'Model':18s}  {'Acc':>7s}  {'±':>5s}  {'Δ base':>8s}")
    print("  " + "-"*44)
    base = results["logreg_base"][0]
    for name, (mu, sd, cm) in results.items():
        delta = mu - base
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:18s}  {mu:7.1%}  {sd:5.1%}  {sign}{delta*100:.1f}pp")

    print("\nPer-family recall:")
    fam_labels = ["major","minor","diminished","augmented","suspended"]
    header = f"  {'family':12s}" + "".join(f"  {n:>12s}" for n in results)
    print(header)
    for i, fam in enumerate(fam_labels):
        row = f"  {fam:12s}"
        for name, (mu, sd, cm) in results.items():
            r = cm[i,i] / max(cm[i].sum(), 1)
            row += f"  {r:12.1%}"
        print(row)

    # ── plot ──────────────────────────────────────────────────────────────────
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 4, 4.5), facecolor="#0d1520")
    if n_models == 1: axes = [axes]
    fig.suptitle("Context-window family classifier — model comparison\n"
                 "(hard audio, oracle bounds, GT root, 5-fold CV)",
                 color="#e2e8f0", fontsize=11, y=1.02)
    labels_short = ["maj","min","dim","aug","sus"]
    base_acc = results["logreg_base"][0]
    for ax, (name, (mu, sd, cm)) in zip(axes, results.items()):
        cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_facecolor("#0d1520"); ax.spines[:].set_color("#253447")
        ax.set_xticks(range(5)); ax.set_xticklabels(labels_short, fontsize=8, color="#88aacc")
        ax.set_yticks(range(5)); ax.set_yticklabels(labels_short, fontsize=8, color="#88aacc")
        ax.tick_params(colors="#5a6a7e")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#fff" if cm_n[i,j] > 0.5 else "#5a6a7e")
        delta = mu - base_acc
        sign  = "+" if delta >= 0 else ""
        ax.set_title(f"{name}\n{mu:.1%} ± {sd:.1%}  ({sign}{delta*100:.1f}pp)",
                     color="#e2e8f0", fontsize=8.5, pad=5)
    plt.tight_layout()
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight", facecolor="#0d1520")
    plt.close(fig)
    print(f"\n→ {PLOT_OUT}")
