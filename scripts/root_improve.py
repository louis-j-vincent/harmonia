"""Push root accuracy past the linear-model baseline (~91%). Root caps everything
downstream, so this is the highest-value labeling lever.

Renders + extracts ABSOLUTE per-oracle-segment chroma ONCE and caches it, then sweeps
models and feature sets offline (fast iteration). Compares vs the current LR-on-48d.

Feature blocks (all absolute, not root-relative):
  base48      onset+note+bass+treble chroma (12 each), L2-normalized
  templates   for each candidate root r, max over chord-families of chroma·template
              (12 scores) — explicit music-theory root hypotheses
  bass3       bass split into 3 sub-registers (catch the root even under walking bass)

Models: LogisticRegression, HistGradientBoosting, MLP. 5-fold by song.

Usage: .venv/bin/python scripts/root_improve.py --n-songs 60 [--rebuild]
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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from root_model_experiment import TEMPLATES, chroma88  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CACHE = REPO / "data" / "cache" / "root_abs_feats.npz"


def bass_sub(v88, edges=(0, 40, 46, 52)):
    """Bass register split into sub-bands (root often the lowest sustained band)."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        out.append(chroma88(v88, lo, hi))
    return np.concatenate(out)


def build_cache(n_songs, augment):
    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // n_songs, 1)][: n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)
    conds = [False, True] if augment else [False]
    base, tmpl, b3, roots, grp = [], [], [], [], []
    for rec in songs:
      for degrade in conds:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            t = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], t, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(t); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(t, y, sr)
            acts = ex.extract(t, use_cache=False)
        finally:
            t.unlink(missing_ok=True)
        ft = acts.frame_times; spb = 60.0 / rec["tempo"]
        for t0, t1, root, _q in song_chord_spans(rec):
            mma = None
            for ev in rec["chord_timeline"]:
                if int(round(((ev["bar"] - 1) * rec["beats_per_bar"] + ev["beat"]))) == int(round(t0 / spb)):
                    mma = ev["mma"]; break
            pp = parse_chord(mma) if mma else None
            if pp is None or pp[1] not in BUCKET_FAMILY:
                continue
            m = (ft >= t0) & (ft < t1)
            if m.sum() < 1:
                continue
            on = acts.onset_probs[m].sum(0); nt = acts.note_probs[m].sum(0)
            oc = chroma88(on)
            base.append(np.concatenate([oc, chroma88(nt), chroma88(on, 0, 52), chroma88(on, 60, 200)]))
            tmpl.append(np.array([max(oc @ t for r2, t in TEMPLATES if r2 == r) for r in range(12)]))
            b3.append(bass_sub(on))
            roots.append(root % 12); grp.append(rec["song_id"])
    np.savez(CACHE, base=np.array(base), tmpl=np.array(tmpl), b3=np.array(b3),
             roots=np.array(roots), grp=np.array(grp))
    print(f"cached {len(roots)} segments → {CACHE}")


def cv(X, y, grp, model):
    n = len(y); pred = np.zeros(n, int)
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = model(); m.fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return (pred == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=60)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild or not CACHE.exists():
        build_cache(args.n_songs, args.augment)
    d = np.load(CACHE, allow_pickle=True)
    base, tmpl, b3, roots, grp = d["base"], d["tmpl"], d["b3"], d["roots"], d["grp"]
    print(f"\n=== root accuracy sweep, {len(roots)} segments, {len(set(grp))} songs ===\n")

    feats = {
        "base48": base,
        "base48+templates": np.hstack([base, tmpl]),
        "base48+templates+bass3": np.hstack([base, tmpl, b3]),
    }
    models = {
        "LogReg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, alpha=1e-3),
    }
    print(f"{'features':<26} " + " ".join(f"{m:>8}" for m in models))
    for fname, X in feats.items():
        row = [f"{cv(X, roots, grp, mk):.1%}" for mk in models.values()]
        print(f"{fname:<26} " + " ".join(f"{v:>8}" for v in row))
    print("\n  (baseline is base48 + LogReg. Looking for a clear, held-out lift.)")


if __name__ == "__main__":
    main()
