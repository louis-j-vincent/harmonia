"""Train beat_seq_model_v3: architecturally key-invariant root + quality heads.

Two new heads, both invariant/equivariant *by construction* rather than by
rotation augmentation (which v2 leans on):

  ROOT head — canonical-form scorer (equivariant by weight-tying).
    For each of 12 root candidates r, roll all chroma blocks in the 240d
    windowed feature by -r (candidate root → pc 0), then score with a SHARED
    MLP → scalar.  argmax over candidates = predicted root.  Because the same
    function scores every candidate, rolling the input by s and the label by s
    merely cyclically permutes the 12 scores → identical loss.  Rotation
    augmentation is therefore a *no-op* for this head (verified in --ablate).
    Optional extra per-candidate feature: Gaussian log-likelihood of the rolled
    centre-beat chroma under chroma_root_template.npz (mu, sigma).

  QUALITY head — DFT-magnitude features (invariant by construction).
    Each 12d chroma block → |rfft|[:7] (7 mags).  |DFT| is exactly invariant to
    circular pitch-class shift, so the head literally cannot encode a key-biased
    quality prior.  4 blocks × 7 × 5 beats = 140d → LR over 5 quality classes
    {major, minor, dom7, maj7, dim}.

Data pipeline mirrors v2 (iReal jazz + POP909 piano, songs 001-005 held out).
Saved model (beat_seq_model_v3.npz) is pure-numpy loadable via the V3Model
class below — no torch needed at inference.

Usage:
    .venv/bin/python scripts/train_beat_seq_model_v3.py            # train + save + eval
    .venv/bin/python scripts/train_beat_seq_model_v3.py --eval-only # eval saved model
    .venv/bin/python scripts/train_beat_seq_model_v3.py --ablate    # augmentation / template ablations
"""
from __future__ import annotations

import argparse
import json
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_accomp_emission import song_chord_spans
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _chroma88, _pool_beats, _reg_raw

DB   = REPO / "data" / "accomp_db" / "db.jsonl"
POP  = REPO / "data" / "pop909" / "POP909"
OUT  = REPO / "harmonia" / "models" / "beat_seq_model_v3.npz"
TEMPLATE = REPO / "harmonia" / "models" / "chroma_root_template.npz"
NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

HARTE_TO_PC = {n: i for i, n in enumerate(NOTE)}
HARTE_TO_PC.update({"Db":1,"Eb":3,"Gb":6,"Ab":8,"Bb":10})

QUALITY5 = ["major", "minor", "dom7", "maj7", "dim"]
Q5_IDX = {q: i for i, q in enumerate(QUALITY5)}

# raw quality token (MMA bucket or Harte suffix) → 5-class quality (or None = drop).
# sus/half-nothing tokens map to None (no unambiguous third → excluded from the
# quality head; they are also excluded from the majmin metric, mir_eval style).
Q5_MAP = {
    "maj": "major", "6": "major", "maj6": "major", "aug": "major",
    "maj7": "maj7", "majmaj7": "maj7", "augmaj7": "maj7",
    "dom7": "dom7", "dom7alt": "dom7", "7": "dom7", "aug7": "dom7", "9": "dom7",
    "min": "minor", "min7": "minor", "m6": "minor", "min6": "minor",
    "minmaj7": "minor", "m": "minor",
    "dim": "dim", "dim7": "dim", "m7b5": "dim", "hdim7": "dim",
}


def quality5(tok: str) -> int | None:
    q = Q5_MAP.get(tok)
    return Q5_IDX[q] if q is not None else None


# ── feature helpers ───────────────────────────────────────────────────────────

def beat_features(onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
    """48d per-beat feature (identical to v1/v2): 4 L2-normed 12d chroma blocks."""
    n = len(onset_b)
    F = np.zeros((n, 48), np.float32)
    for b in range(n):
        F[b] = np.concatenate([
            _chroma88(onset_b[b]),
            _chroma88(note_b[b]),
            _chroma88(onset_b[b], 0, 52),
            _chroma88(onset_b[b], 60, 200),
        ])
    return F


def windowed_features(F: np.ndarray, window: int) -> np.ndarray:
    """Concatenate ±window neighbours → (n_beats, 48*(2w+1))."""
    n, d = F.shape
    out = np.zeros((n, d * (2 * window + 1)), np.float32)
    for b in range(n):
        row = []
        for delta in range(-window, window + 1):
            nb = b + delta
            row.append(F[nb] if 0 <= nb < n else np.zeros(d, np.float32))
        out[b] = np.concatenate(row)
    return out


def dft_features(X: np.ndarray) -> np.ndarray:
    """windowed chroma (n, 12*nblocks) → (n, 7*nblocks) DFT magnitudes.

    Each 12d block → |rfft(block, n=12)[:7]| (DC + |F1..F5| + Nyquist).
    Exactly invariant to circular shift of the block.
    """
    n, d = X.shape
    nb = d // 12
    blocks = X.reshape(n, nb, 12)
    mags = np.abs(np.fft.rfft(blocks, n=12, axis=2))[:, :, :7]  # (n, nb, 7)
    return mags.reshape(n, nb * 7).astype(np.float32)


def _roll_index(d: int, r: int) -> np.ndarray:
    """Gather index that rolls every 12d block of a d-vector by -r (root r → pc 0)."""
    idx = np.arange(d)
    for start in range(0, d, 12):
        idx[start:start+12] = start + (np.arange(12) + r) % 12
    return idx


# ── canonical-form root scorer (numpy inference + torch training) ─────────────

class V3Model:
    """Pure-numpy loader/predictor for beat_seq_model_v3.npz.

    predict_proba(onset_b, note_b) -> (root_proba (n,12), qual_proba (n,5))
    mirroring _BeatSeqModel's onset/note interface so the eval is a drop-in.
    """

    def __init__(self, path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        self.window = int(d["window"][0])
        # root head (canonical MLP)
        self.rW1 = d["root_W1"]; self.rb1 = d["root_b1"]
        self.rW2 = d["root_W2"]; self.rb2 = d["root_b2"]
        self.root_mean = d["root_mean"]; self.root_scale = d["root_scale"]
        self.use_template = bool(d["root_use_template"][0])
        self.mu = d["template_mu"]; self.sigma = d["template_sigma"]
        # quality head (LR over DFT mags)
        self.qmean = d["qual_mean"]; self.qscale = d["qual_scale"]
        self.qcoef = d["qual_coef"]; self.qint = d["qual_intercept"]
        self.qclasses = d["qual_classes"]

    # -- feature construction from pooled beats --
    def _windowed(self, onset_b, note_b):
        F = beat_features(onset_b, note_b)
        return windowed_features(F, self.window)  # (n, 48*(2w+1))

    # -- root scoring --
    def _cand_features(self, X: np.ndarray, r: int) -> np.ndarray:
        """Roll all blocks by -r; optionally append centre-beat template log-lik."""
        Xr = X[:, _roll_index(X.shape[1], r)]
        if not self.use_template:
            return Xr
        cb = self.window * 4  # onset-full block of the centre beat
        x = Xr[:, cb*12:(cb+1)*12]
        ll = (-0.5 * (((x - self.mu) / self.sigma) ** 2).sum(1)).astype(np.float32)
        return np.concatenate([Xr, ll[:, None]], axis=1)

    def root_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.zeros((n, 12), np.float32)
        for r in range(12):
            feat = self._cand_features(X, r)
            z = (feat - self.root_mean) / self.root_scale
            h = np.maximum(z @ self.rW1 + self.rb1, 0.0)
            scores[:, r] = (h @ self.rW2 + self.rb2)[:, 0]
        scores -= scores.max(1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(1, keepdims=True)

    def qual_proba(self, X: np.ndarray) -> np.ndarray:
        D = dft_features(X)
        z = (D - self.qmean) / self.qscale
        logits = z @ self.qcoef.T + self.qint
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(1, keepdims=True)
        # remap to QUALITY5 order via classes
        out = np.zeros((len(X), 5), np.float32)
        for i, c in enumerate(self.qclasses):
            out[:, int(c)] = p[:, i]
        return out

    def predict_proba(self, onset_b, note_b):
        X = self._windowed(onset_b, note_b)
        return self.root_proba(X), self.qual_proba(X)


# ── data collection ───────────────────────────────────────────────────────────

def collect_ireal_song(rec, renderer, sf2, ex, window):
    from build_audio_chord_features import BUCKET_FAMILY  # noqa
    spb  = 60.0 / rec["tempo"]
    bpb  = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb
    spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec) if t1 > t0]

    def gt(t):
        for t0, t1, root, q in spans:
            if t0 <= t < t1:
                return root, quality5(q)
        return None, None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    bt = np.arange(n_beats + 1) * spb
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)
    F = beat_features(onset_b, note_b)
    W = windowed_features(F, window)

    roots = np.full(n_beats, -1, int); quals = np.full(n_beats, -1, int)
    for b in range(n_beats):
        r, q = gt((b + 0.5) * spb)
        if r is not None:
            roots[b] = r
        if q is not None:
            quals[b] = q
    return W, roots, quals


def collect_pop909_song(sid, renderer, sf2, ex, window):
    song_dir  = POP / sid
    midi_path = song_dir / f"{sid}.mid"
    chord_txt = song_dir / "chord_midi.txt"
    beat_txt  = song_dir / "beat_midi.txt"
    if not midi_path.exists() or not chord_txt.exists():
        return None

    spans = []
    for line in chord_txt.read_text().splitlines():
        p = line.split()
        if len(p) < 3 or p[2] == "N":
            continue
        root_str = p[2].split(":")[0]
        tok = p[2].split(":")[1].split("/")[0] if ":" in p[2] else ""
        if root_str in HARTE_TO_PC:
            spans.append((float(p[0]), float(p[1]), HARTE_TO_PC[root_str], quality5(tok)))
    if not spans:
        return None

    beat_times = []
    for line in beat_txt.read_text().splitlines():
        p = line.split()
        if p:
            try:
                beat_times.append(float(p[0]))
            except ValueError:
                pass
    if len(beat_times) < 2:
        return None
    bt = np.array(beat_times + [beat_times[-1] + (beat_times[-1] - beat_times[-2])])

    def gt(t):
        for t0, t1, root, q in spans:
            if t0 <= t < t1:
                return root, q
        return None, None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(midi_path, tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    n_beats = len(bt) - 1
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)
    F = beat_features(onset_b, note_b)
    W = windowed_features(F, window)

    roots = np.full(n_beats, -1, int); quals = np.full(n_beats, -1, int)
    for b in range(n_beats):
        r, q = gt(0.5 * (bt[b] + bt[b+1]))
        if r is not None:
            roots[b] = r
        if q is not None:
            quals[b] = q
    return W, roots, quals


def build_training_set(n_jazz, n_pop, window):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex  = PitchExtractor(cache_dir=None)

    W_all, root_all, qual_all = [], [], []

    recs  = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // n_jazz, 1)][: n_jazz]
    print(f"iReal: {len(songs)} songs")
    for i, rec in enumerate(songs):
        print(f"  [{i+1}/{len(songs)}] {rec['song_id']}", end="\r", flush=True)
        try:
            W, roots, quals = collect_ireal_song(rec, renderer, sf2, ex, window)
        except Exception as e:
            print(f"\n  SKIP {rec['song_id']}: {e}"); continue
        W_all.append(W); root_all.append(roots); qual_all.append(quals)

    HOLD = {"001", "002", "003", "004", "005"}
    pop_sids = sorted(d.name for d in POP.iterdir()
                      if d.is_dir() and d.name not in HOLD and (d / f"{d.name}.mid").exists())
    pop_sids = pop_sids[: n_pop]
    print(f"\nPOP909: {len(pop_sids)} songs")
    for i, sid in enumerate(pop_sids):
        print(f"  [{i+1}/{len(pop_sids)}] {sid}", end="\r", flush=True)
        try:
            res = collect_pop909_song(sid, renderer, sf2, ex, window)
        except Exception as e:
            print(f"\n  SKIP {sid}: {e}"); continue
        if res is None:
            continue
        W, roots, quals = res
        W_all.append(W); root_all.append(roots); qual_all.append(quals)

    W = np.vstack(W_all)
    roots = np.concatenate(root_all)
    quals = np.concatenate(qual_all)
    print(f"\nTotal beats: {len(W)}  (root-labeled {np.sum(roots>=0)}, qual-labeled {np.sum(quals>=0)})")
    return W, roots, quals


# ── training ──────────────────────────────────────────────────────────────────

def train_root_head(W, roots, window, use_template, augment, hidden=96,
                    epochs=40, batch=512, lr=1e-3, seed=0, verbose=True):
    """Train the canonical-form root scorer. Returns weight dict + standardizer."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed)
    mask = roots >= 0
    X = W[mask].astype(np.float32); y = roots[mask].astype(np.int64)
    d = X.shape[1]
    in_dim = d + (1 if use_template else 0)

    # precompute per-candidate roll indices
    roll_idx = np.stack([_roll_index(d, r) for r in range(12)])  # (12, d)
    tmpl = np.load(TEMPLATE)
    mu, sigma = tmpl["mu"].astype(np.float32), tmpl["sigma"].astype(np.float32)
    cb = window * 4

    def cand_feats(Xb):
        """Xb (B,d) -> (B,12,in_dim) rolled candidate features."""
        Xc = Xb[:, roll_idx]  # (B,12,d)
        if not use_template:
            return Xc
        x = Xc[:, :, cb*12:(cb+1)*12]                        # (B,12,12)
        ll = -0.5 * (((x - mu) / sigma) ** 2).sum(2)         # (B,12)
        return np.concatenate([Xc, ll[:, :, None]], axis=2)  # (B,12,in_dim)

    # standardizer from canonical (true-root-aligned) features — a fixed transform
    # applied identically to every candidate, so equivariance is preserved.
    canon = cand_feats(X)[np.arange(len(X)), y]  # (N, in_dim) true-root view
    mean = canon.mean(0); scale = canon.std(0) + 1e-6

    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()

    mean_t = torch.tensor(mean); scale_t = torch.tensor(scale)
    N = len(X)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        tot = 0.0
        for s in range(0, N, batch):
            bi = perm[s:s+batch]
            Xb, yb = X[bi], y[bi]
            if augment:
                # _roll_index(d, sh) rolls blocks by -sh (transpose DOWN by sh),
                # so the true root also moves down: (yb - sh) % 12.  For an
                # exactly-equivariant scorer this is a *no-op* in expectation.
                sh = np.random.randint(0, 12, size=len(bi))
                Xb = np.stack([Xb[i][_roll_index(d, sh[i])] for i in range(len(bi))])
                yb = (yb - sh) % 12
            cf = torch.tensor(cand_feats(Xb))            # (B,12,in_dim)
            cf = (cf - mean_t) / scale_t
            logits = net(cf).squeeze(-1)                 # (B,12)
            loss = lossf(logits, torch.tensor(yb))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            with torch.no_grad():
                cf = torch.tensor(cand_feats(X)); cf = (cf - mean_t) / scale_t
                acc = (net(cf).squeeze(-1).argmax(1).numpy() == y).mean()
            print(f"    root ep {ep:2d}  loss {tot/N:.3f}  train_acc {acc:.1%}")

    lin1, lin2 = net[0], net[2]
    return dict(
        root_W1=lin1.weight.detach().numpy().T.astype(np.float32),
        root_b1=lin1.bias.detach().numpy().astype(np.float32),
        root_W2=lin2.weight.detach().numpy().T.astype(np.float32),
        root_b2=lin2.bias.detach().numpy().astype(np.float32),
        root_mean=mean.astype(np.float32), root_scale=scale.astype(np.float32),
        root_use_template=np.array([use_template], bool),
        template_mu=mu, template_sigma=sigma,
    )


def train_quality_head(W, quals):
    mask = quals >= 0
    D = dft_features(W[mask]); y = quals[mask].astype(int)
    sc = StandardScaler().fit(D)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(D), y)
    acc = (clf.predict(sc.transform(D)) == y).mean()
    print(f"    quality train acc {acc:.1%}  (n={len(y)}, classes={list(clf.classes_)})")
    return dict(
        qual_mean=sc.mean_.astype(np.float32), qual_scale=sc.scale_.astype(np.float32),
        qual_coef=clf.coef_.astype(np.float32), qual_intercept=clf.intercept_.astype(np.float32),
        qual_classes=clf.classes_.astype(int),
    )


# ── evaluation on rendered POP909 001-005 ─────────────────────────────────────

def _tempo_grid_beats(y, sr):
    import librosa
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    period = 60.0 / max(tempo_bpm, 1.0)
    _, raw_frames = librosa.beat.beat_track(y=y, sr=sr)
    raw_times = librosa.frames_to_time(raw_frames, sr=sr)
    if len(raw_times) > 0:
        ph = (raw_times % period) / period
        phase0 = float(np.arctan2(np.sin(2*np.pi*ph).mean(),
                                  np.cos(2*np.pi*ph).mean()) / (2*np.pi) * period) % period
    else:
        phase0 = 0.0
    dur = librosa.get_duration(y=y, sr=sr)
    bt = np.arange(0, dur + period, period) + phase0
    bt = bt[bt < dur + 0.5 * period]
    return bt, tempo_bpm


def _gt_majmin(tok):
    q = Q5_MAP.get(tok)
    if q in ("major", "maj7", "dom7"):
        return "maj"
    if q == "minor":
        return "min"
    return None  # dim / sus / aug / unknown → excluded (mir_eval majmin style)


def _pred_majmin(q5_idx):
    return "maj" if QUALITY5[q5_idx] in ("major", "maj7", "dom7") else "min"


def evaluate(model: V3Model, songs=("001","002","003","004","005"), v2_root=True, tag=""):
    from harmonia.data.pop909_parser import POP909Parser
    parser = POP909Parser(POP)
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache")

    bsm2 = None
    if v2_root:
        from harmonia.models.chord_pipeline_v1 import _BeatSeqModel, MODELS
        p = MODELS / "beat_seq_model_v2.npz"
        if p.exists():
            bsm2 = _BeatSeqModel(p)

    rows = []
    agg = {"gt_r": [], "p_r": [], "gt_mm": [], "p_mm": [], "v2_r": [], "song": []}
    for sid in songs:
        wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            print(f"  skip {sid}: no render"); continue
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        bt, tempo = _tempo_grid_beats(y, sr)
        acts = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)
        root_p, qual_p = model.predict_proba(onset_b, note_b)
        p_root = root_p.argmax(1); p_q = qual_p.argmax(1)
        v2_root_pred = bsm2.predict_proba(onset_b, note_b).argmax(1) if bsm2 else None

        song = parser.parse_song(sid)
        n = min(len(bt) - 1, len(p_root))
        r_ok = mm_ok = mm_tot = r_tot = 0
        for b in range(n):
            ev = song.chord_at_time(0.5 * (bt[b] + bt[b+1]))
            if ev is None:
                continue
            lbl = ev.label
            root_str = lbl.split(":")[0].split("/")[0]
            if root_str not in HARTE_TO_PC:
                continue
            gt_r = HARTE_TO_PC[root_str]
            tok = lbl.split(":")[1].split("/")[0] if ":" in lbl else ""
            gt_mm = _gt_majmin(tok)
            agg["gt_r"].append(gt_r); agg["p_r"].append(int(p_root[b]))
            agg["song"].append(sid)
            if v2_root_pred is not None:
                agg["v2_r"].append(int(v2_root_pred[b]))
            r_tot += 1; r_ok += int(p_root[b] == gt_r)
            if gt_mm is not None:
                mm_tot += 1
                pmm = _pred_majmin(int(p_q[b]))
                good = (p_root[b] == gt_r) and (pmm == gt_mm)
                mm_ok += int(good)
                agg["gt_mm"].append((gt_r, gt_mm)); agg["p_mm"].append((int(p_root[b]), pmm))
        rows.append((sid, r_tot, r_ok / max(r_tot,1), mm_tot, mm_ok / max(mm_tot,1)))

    print(f"\n=== v3 eval {tag} (per-beat, POP909 001-005) ===")
    print(f"  {'song':>5} {'nR':>5} {'root':>7} {'nMM':>5} {'majmin':>7}")
    for sid, rt, ra, mt, ma in rows:
        flag = "  ⚠ 2× tempo" if sid == "002" else ""
        print(f"  {sid:>5} {rt:>5} {ra:>7.1%} {mt:>5} {ma:>7.1%}{flag}")

    gt_r = np.array(agg["gt_r"]); p_r = np.array(agg["p_r"])
    root_acc = (gt_r == p_r).mean()
    mm_pairs = list(zip(agg["gt_mm"], agg["p_mm"]))
    mm_acc = np.mean([g == p for g, p in mm_pairs]) if mm_pairs else float("nan")
    print(f"  {'CORPUS':>5} {len(gt_r):>5} {root_acc:>7.1%} {len(mm_pairs):>5} {mm_acc:>7.1%}")

    # exclude song 002 (2× tempo) as an alternate read
    song_arr = np.array(agg["song"])
    keep = song_arr != "002"
    if keep.any():
        print(f"  root acc excl. 002: {(gt_r[keep]==p_r[keep]).mean():.1%}  (n={keep.sum()})")
    if agg["v2_r"]:
        v2_r = np.array(agg["v2_r"])
        print(f"  [calibration] v2 root acc via this harness: {(v2_r==gt_r).mean():.1%}")
    return root_acc, mm_acc


# ── main ──────────────────────────────────────────────────────────────────────

def save_model(out, window, root_d, qual_d):
    np.savez(out, window=np.array([window], int), **root_d, **qual_d)
    print(f"Saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jazz", type=int, default=50)
    ap.add_argument("--n-pop",  type=int, default=60)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-template", action="store_true", help="disable chroma-template prior on root head")
    ap.add_argument("--augment", action="store_true", help="add rotation augmentation to root head (should be a no-op)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="run augmentation / template ablations")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cache", default=None, help="cache built training set to this .npz")
    args = ap.parse_args()

    if args.eval_only:
        evaluate(V3Model(Path(args.out)), tag="(saved)")
        return

    # build (or load cached) training set
    if args.cache and Path(args.cache).exists():
        z = np.load(args.cache)
        W, roots, quals = z["W"], z["roots"], z["quals"]
        print(f"Loaded cached training set: {W.shape}")
    else:
        W, roots, quals = build_training_set(args.n_jazz, args.n_pop, args.window)
        if args.cache:
            np.savez(args.cache, W=W, roots=roots, quals=quals)
            print(f"Cached training set → {args.cache}")

    if args.ablate:
        print("\n### ABLATION ###")
        for use_tmpl in (True, False):
            for aug in (False, True):
                tag = f"template={use_tmpl} augment={aug}"
                print(f"\n-- {tag} --")
                rd = train_root_head(W, roots, args.window, use_tmpl, aug,
                                     hidden=args.hidden, epochs=args.epochs, verbose=False)
                qd = train_quality_head(W, quals)
                tmp = Path(tempfile.gettempdir()) / "v3_ablate_scratch.npz"
                save_model(tmp, args.window, rd, qd)
                evaluate(V3Model(tmp), v2_root=False, tag=tag)
                tmp.unlink(missing_ok=True)
        return

    print("\n### TRAIN ROOT HEAD (canonical scorer) ###")
    rd = train_root_head(W, roots, args.window, not args.no_template, args.augment,
                         hidden=args.hidden, epochs=args.epochs)
    print("### TRAIN QUALITY HEAD (DFT magnitudes) ###")
    qd = train_quality_head(W, quals)
    save_model(Path(args.out), args.window, rd, qd)
    evaluate(V3Model(Path(args.out)), tag="(final)")


if __name__ == "__main__":
    main()
