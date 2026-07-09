"""Train beat_seq_model_v4: per-beat root model = key-agnostic canonical scorer
ENSEMBLED with a bass-anchored scorer (rotation fixed by the observed bass PC).

Winner of the 2026-07-09 per-beat bake-off (scripts/bakeoff_root_perbeat.py):
clean disjoint jazz split, per-beat root — abs_ctx/v2-style 86.7% → canon 92.9%
→ canon⊕bass-anchored 93.3%.  See docs/known_issues.md #18.

Two heads, both key-invariant, producing a per-beat (n,12) root posterior:
  canon : for each candidate root r, roll the ±4-beat 432d window by -r, shared
          MLP → scalar; softmax over 12 = key-agnostic root posterior.
  bass  : anchor rotation on the observed bass PC (argmax of the centre beat's bass
          chroma), roll by -anchor, LR predicts offset=(root-bass)%12, mapped back
          to absolute root.  Removes the "we already assumed root" caveat.
final root posterior = softmax_mix(canon) + softmax(bass), argmax.

Trained on jazz1460 (per-beat cache) + POP909 piano (001-005 held out).  Pure-numpy
loadable via V4Model (mirrors _BeatSeqModel.predict_proba(onset_b, note_b)).

Usage:
    .venv/bin/python scripts/train_beat_seq_model_v4.py [--n-pop 60] [--eval-only]
"""
from __future__ import annotations

import argparse
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
from bakeoff_root_perbeat import cache_path, windowize_song, assemble as jazz_assemble
from train_beat_seq_model_v3 import train_root_head, HARTE_TO_PC
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _pool_beats

POP = REPO / "data" / "pop909" / "POP909"
OUT = REPO / "harmonia" / "models" / "beat_seq_model_v4.npz"
WINDOW = 4  # ±4 beats


# ── per-beat feature builders ─────────────────────────────────────────────────

def collect_pop_perbeat(sid, renderer, sf2, ex):
    song_dir = POP / sid
    midi = song_dir / f"{sid}.mid"; chord_txt = song_dir / "chord_midi.txt"
    beat_txt = song_dir / "beat_midi.txt"
    if not midi.exists() or not chord_txt.exists():
        return None
    spans = []
    for line in chord_txt.read_text().splitlines():
        p = line.split()
        if len(p) < 3 or p[2] == "N":
            continue
        rs = p[2].split(":")[0].split("/")[0]
        if rs in HARTE_TO_PC:
            spans.append((float(p[0]), float(p[1]), HARTE_TO_PC[rs]))
    if not spans:
        return None
    bts = []
    for line in beat_txt.read_text().splitlines():
        p = line.split()
        if p:
            try:
                bts.append(float(p[0]))
            except ValueError:
                pass
    if len(bts) < 2:
        return None
    bt = np.array(bts + [bts[-1] + (bts[-1] - bts[-2])])

    def gt(t):
        for t0, t1, r in spans:
            if t0 <= t < t1:
                return r
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(midi, tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
    F = bo.beat_feats(onset_b, note_b)
    n = len(bt) - 1
    roots = np.array([gt(0.5 * (bt[b] + bt[b+1])) for b in range(n)], dtype=object)
    keep = np.array([r is not None for r in roots])
    return F, roots, keep


DSCACHE = REPO / "data" / "cache" / "v4_trainset.npz"


def build_dataset(n_pop):
    if DSCACHE.exists():
        z = np.load(DSCACHE)
        print(f"Loaded v4 trainset cache: {z['W'].shape}")
        return z["W"], z["y"]
    with open(cache_path(1), "rb") as f:
        jazz = pickle.load(f)
    print(f"jazz per-beat cache: {len(jazz)} songs")
    data = dict(jazz)

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    HOLD = {"001", "002", "003", "004", "005"}
    pop_sids = sorted(d.name for d in POP.iterdir()
                      if d.is_dir() and d.name not in HOLD and (d / f"{d.name}.mid").exists())[:n_pop]
    print(f"POP909: rendering {len(pop_sids)} songs...")
    for i, sid in enumerate(pop_sids):
        print(f"  [{i+1}/{len(pop_sids)}] {sid}", end="\r", flush=True)
        try:
            res = collect_pop_perbeat(sid, renderer, sf2, ex)
        except Exception as e:
            print(f"\n  SKIP {sid}: {e}"); continue
        if res is not None:
            data[f"pop_{sid}"] = res
    print()
    # assemble windowed ±4
    W, y, _, _ = jazz_assemble_all(data)
    print(f"Total: {len(y)} labeled beats  ({W.shape[1]}d windows)")
    np.savez(DSCACHE, W=W, y=y)
    return W, y


def jazz_assemble_all(data):
    """Windowize every song in data, concat. (reuses the per-beat windowizer.)"""
    Xs, ys = [], []
    for sid, (F, roots, keep) in data.items():
        X = windowize_song(F)
        for b in range(len(F)):
            if keep[b]:
                Xs.append(X[b]); ys.append(int(roots[b]))
    return np.vstack(Xs), np.array(ys), None, None


# ── bass-anchored head (export weights) ───────────────────────────────────────

def train_ba_head(W, y):
    a = bo.bass_anchor(W)
    Xw = bo.roll_rows(W, a)
    yoff = (y - a) % 12
    sc = StandardScaler().fit(Xw)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xw), yoff)
    acc = (clf.predict(sc.transform(Xw)) == yoff).mean()
    print(f"  bass-anchored train offset-acc {acc:.1%}")
    return dict(
        ba_mean=sc.mean_.astype(np.float32), ba_scale=sc.scale_.astype(np.float32),
        ba_coef=clf.coef_.astype(np.float32), ba_intercept=clf.intercept_.astype(np.float32),
        ba_classes=clf.classes_.astype(int),
    )


# ── pure-numpy loader ─────────────────────────────────────────────────────────

class V4Model:
    """Per-beat root model: canon ⊕ bass-anchored. predict_proba(onset_b,note_b)->(n,12)."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.window = int(d["window"][0])
        self.rW1, self.rb1 = d["root_W1"], d["root_b1"]
        self.rW2, self.rb2 = d["root_W2"], d["root_b2"]
        self.rmean, self.rscale = d["root_mean"], d["root_scale"]
        self.use_template = bool(d["root_use_template"][0])
        self.mu, self.sigma = d["template_mu"], d["template_sigma"]
        self.bmean, self.bscale = d["ba_mean"], d["ba_scale"]
        self.bcoef, self.bint = d["ba_coef"], d["ba_intercept"]
        self.bclasses = d["ba_classes"]

    def _windowed(self, onset_b, note_b):
        F = bo.beat_feats(onset_b, note_b)
        n, d = F.shape; w = self.window
        out = np.zeros((n, d * (2*w+1)), np.float32)
        for b in range(n):
            row = [F[b+o] if 0 <= b+o < n else np.zeros(d, np.float32)
                   for o in range(-w, w+1)]
            out[b] = np.concatenate(row)
        return out

    def _canon_proba(self, X):
        n, d = X.shape
        sc = np.zeros((n, 12), np.float32)
        cb = self.window * 4
        for r in range(12):
            Xr = X[:, bo.roll_idx(d, r)]
            feat = Xr
            if self.use_template:
                x = Xr[:, cb*12:(cb+1)*12]
                ll = (-0.5 * (((x - self.mu) / self.sigma) ** 2).sum(1)).astype(np.float32)
                feat = np.concatenate([Xr, ll[:, None]], axis=1)
            z = (feat - self.rmean) / self.rscale
            h = np.maximum(z @ self.rW1 + self.rb1, 0.0)
            sc[:, r] = (h @ self.rW2 + self.rb2)[:, 0]
        sc -= sc.max(1, keepdims=True)
        e = np.exp(sc); return e / e.sum(1, keepdims=True)

    def _ba_proba(self, X):
        a = bo.bass_anchor(X)
        Xw = bo.roll_rows(X, a)
        z = (Xw - self.bmean) / self.bscale
        lg = z @ self.bcoef.T + self.bint
        lg -= lg.max(1, keepdims=True)
        e = np.exp(lg); p = e / e.sum(1, keepdims=True)
        P_off = np.zeros((len(X), 12), np.float32)
        for i, c in enumerate(self.bclasses):
            P_off[:, int(c)] = p[:, i]
        return bo.offset_to_abs(P_off, a)

    def predict_proba(self, onset_b, note_b):
        X = self._windowed(onset_b, note_b)
        p = self._canon_proba(X) + self._ba_proba(X)
        return p / p.sum(1, keepdims=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pop", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.eval_only:
        print("loaded V4Model:", args.out)
        return

    W, y = build_dataset(args.n_pop)
    print("### canon head (±4) ###")
    rd = train_root_head(W, y, WINDOW, use_template=False, augment=False,
                         hidden=96, epochs=args.epochs)
    print("### bass-anchored head ###")
    bd = train_ba_head(W, y)
    np.savez(args.out, window=np.array([WINDOW], int), **rd, **bd)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
