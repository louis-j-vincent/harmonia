"""Per-beat bake-off — the deployment-relevant regime where emissions are NOISY and
the progression prior / bass-anchoring should finally earn their keep.

Same clean rules as the oracle-segment bake-off (bakeoff_root_models.py):
  irealb/jazz1460, exact metronomic tempo grid, disjoint even/odd song split,
  every learnable model trained on train split only, evaluated on the SAME eval beats.
Grid = per BEAT (or half-beat via --subdiv 2).  Feature per beat = 48d (4 L2-normed
chroma blocks), context ±4 beats (9×48=432d).

Reports overall / interior / boundary root accuracy + +5/7 error share, so we see
exactly where each model wins (the earlier diagnostic showed boundary beats are the
hard ones).  Reuses the model code from bakeoff_root_models to guarantee identical defs.
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

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _pool_beats

import bakeoff_root_models as bo  # reuse: beat_feats, roll_idx, canonical/BA/viterbi, etc.

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NCTX = bo.NCTX  # 4


def cache_path(subdiv):
    return REPO / "data" / "cache" / f"bakeoff_jazz_perbeat_s{subdiv}.pkl"


def collect_song(rec, renderer, sf2, ex, subdiv):
    spb = 60.0 / rec["tempo"]
    n_beats = rec["n_bars"] * rec["beats_per_bar"]
    spans = [(t0, t1, r % 12) for t0, t1, r, q in song_chord_spans(rec)
             if t1 > t0 and q in BUCKET_FAMILY]
    if not spans:
        return None

    def gtroot(t):
        for t0, t1, r in spans:
            if t0 <= t < t1:
                return r
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    step = spb / subdiv
    n_units = n_beats * subdiv
    bt = np.arange(n_units + 1) * step
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
    F = bo.beat_feats(onset_b, note_b)  # (n_units, 48)
    roots = np.array([gtroot((u + 0.5) * step) for u in range(n_units)], dtype=object)
    keep = np.array([r is not None for r in roots])
    return F, roots, keep


def build_cache(n_songs, subdiv):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()][:n_songs]
    print(f"Rendering {len(songs)} songs (subdiv={subdiv})...")
    data = {}
    for i, rec in enumerate(songs):
        print(f"  [{i+1}/{len(songs)}] {rec['song_id']}", end="\r", flush=True)
        try:
            res = collect_song(rec, renderer, sf2, ex, subdiv)
        except Exception as e:
            print(f"\n  SKIP {rec['song_id']}: {e}"); continue
        if res is not None:
            data[rec["song_id"]] = res
    p = cache_path(subdiv)
    with open(p, "wb") as f:
        pickle.dump(data, f)
    print(f"\nCached {len(data)} songs → {p}")
    return data


def windowize_song(F):
    """(n,48) → (n, 9*48) per-beat ±4 window (zero-padded edges)."""
    n, d = F.shape
    W = 2 * NCTX + 1
    X = np.zeros((n, d * W), np.float32)
    for b in range(n):
        row = []
        for off in range(-NCTX, NCTX + 1):
            j = b + off
            row.append(F[j] if 0 <= j < n else np.zeros(d, np.float32))
        X[b] = np.concatenate(row)
    return X


def assemble(data, sids):
    Xs, ys, songs, bnd = [], [], [], []
    for sid in sids:
        F, roots, keep = data[sid]
        X = windowize_song(F)
        prev = None
        for b in range(len(F)):
            if not keep[b]:
                prev = None
                continue
            r = int(roots[b])
            Xs.append(X[b]); ys.append(r); songs.append(sid)
            bnd.append(prev is not None and r != prev)
            prev = r
    return np.vstack(Xs), np.array(ys), np.array(songs), np.array(bnd)


def score(name, pred, y, bnd, results):
    ok = pred == y
    err = ~ok
    iv = (pred[err] - y[err]) % 12
    p57 = ((iv == 5).sum() + (iv == 7).sum()) / max(err.sum(), 1)
    results.append((name, ok.mean(), ok[~bnd].mean(), ok[bnd].mean(), p57))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=70)
    ap.add_argument("--subdiv", type=int, default=1, help="1=per beat, 2=per half-beat")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    p = cache_path(args.subdiv)
    if args.rebuild or not p.exists():
        data = build_cache(args.n_songs, args.subdiv)
    else:
        with open(p, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cache: {len(data)} songs (subdiv={args.subdiv})")

    sids = sorted(data.keys())
    tr, ev = sids[0::2], sids[1::2]
    Xtr, ytr, _, _ = assemble(data, tr)
    Xev, yev, sev, bev = assemble(data, ev)
    print(f"train {len(ytr)} beats ({len(tr)} songs) | eval {len(yev)} beats "
          f"({len(ev)} songs, {bev.mean():.0%} boundary)\n")

    results = []

    # baselines
    abs_ctx = bo.fit_abs(Xtr, ytr)
    abs_ctx_p = bo.proba_abs(abs_ctx, Xev)
    score("abs_ctx (LR ±4, key-biased) [~v2]", abs_ctx_p.argmax(1), yev, bev, results)

    # canonical + bass-anchored + super-model
    canon_p = bo.train_torch(Xtr, ytr, "canon")(Xev)
    score("canon (key-agnostic)", canon_p.argmax(1), yev, bev, results)
    ba_p = bo.fit_ba(Xtr, ytr, bo.bass_anchor(Xtr), feat="chroma")(Xev)
    score("bass_anchored (full chroma)", ba_p.argmax(1), yev, bev, results)
    super_p = canon_p + ba_p
    score("canon ⊕ bass_anchored (super)", super_p.argmax(1), yev, bev, results)

    # progression prior (Viterbi) — expected to help most here (noisy per-beat emissions)
    logPd = bo.fit_rel_transition_frames(data, tr) if hasattr(bo, "fit_rel_transition_frames") \
        else _fit_trans_perbeat(data, tr)
    score("canon + viterbi", bo.viterbi_rerank(canon_p, sev, logPd), yev, bev, results)
    score("super + viterbi", bo.viterbi_rerank(super_p / super_p.sum(1, keepdims=True),
          sev, logPd), yev, bev, results)

    print(f"{'model':<40} {'overall':>8} {'interior':>9} {'boundary':>9} {'+5/7err':>8}")
    print("-" * 78)
    for name, o, i, b, p in sorted(results, key=lambda r: -r[1]):
        print(f"{name:<40} {o:>8.1%} {i:>9.1%} {b:>9.1%} {p:>8.1%}")


def _fit_trans_perbeat(data, sids, smooth=1.0):
    """Relative root-bigram over consecutive BEATS (dominated by delta=0 = persistence)."""
    counts = np.full(12, smooth)
    for sid in sids:
        _, roots, keep = data[sid]
        seq = [int(r) for r, k in zip(roots, keep) if k]
        for a, b in zip(seq[:-1], seq[1:]):
            counts[(b - a) % 12] += 1
    return np.log(counts / counts.sum())


if __name__ == "__main__":
    main()
