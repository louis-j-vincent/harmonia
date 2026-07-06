"""Train a chord-change detector: per-beat audio features → P(chord changes here).

The missing brick of the end-to-end pipeline. Renders songs, extracts Basic Pitch,
computes per-beat change features (change_features.py), labels each beat with the
ground-truth chord change (from the chart), and fits a logistic regression. Reports
detection precision/recall/F (5-fold by song) and saves the model coefficients.
Disk-safe: one WAV at a time, deleted after.

Usage: .venv/bin/python scripts/train_change_detector.py --n-songs 40 [--degrade]
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from change_features import FEATURE_NAMES, beat_change_features  # noqa: E402
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
OUT = REPO / "harmonia" / "models" / "change_detector.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=40)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(DB)]
    songs = [r for r in records if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(7)
    X, y, grp = [], [], []
    for si, rec in enumerate(songs):
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"] * bpb
        # GT per-beat chord (family), for change labels
        chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"] for e in rec["chord_timeline"]}
        fam = [None] * nb
        cur = None
        for b in range(nb):
            if b in chord_at:
                p = parse_chord(chord_at[b])
                cur = BUCKET_FAMILY.get(p[1]) if (p and p[1] in BUCKET_FAMILY) else cur
            fam[b] = cur
        change = np.array([1 if b > 0 and fam[b] != fam[b - 1] and fam[b] is not None else 0
                           for b in range(nb)])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp,
                            RenderConfig(soundfont_path=renderer._find_soundfont("MuseScore_General.sf2")))
            if args.degrade:
                a, sr = sf.read(tmp); a = a.mean(1) if a.ndim > 1 else a
                sf.write(tmp, time_varying_degrade(a.astype("float32"), sr, rng), sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        onset_b = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        feats = beat_change_features(onset_b)
        for b in range(1, nb):                # skip beat 0 (always a boundary)
            X.append(feats[b]); y.append(change[b]); grp.append(rec["song_id"])
        if (si + 1) % 10 == 0:
            print(f"  … {si+1}/{len(songs)} songs")

    X = np.array(X); y = np.array(y); grp = np.array(grp)
    print(f"\n{len(y)} beats, {y.mean():.1%} are chord changes\n")

    # CV detection F-score
    precs, recs, fs = [], [], []
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        tp = ((pred == 1) & (y[te] == 1)).sum(); fp = ((pred == 1) & (y[te] == 0)).sum()
        fn = ((pred == 0) & (y[te] == 1)).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        precs.append(p); recs.append(r); fs.append(2 * p * r / (p + r + 1e-9))
    print(f"Change-detection (per-beat, 5-fold by song):")
    print(f"    precision {np.mean(precs):.1%}   recall {np.mean(recs):.1%}   F {np.mean(fs):.1%}")
    # feature importances (final model on all data)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    imp = sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("    top signals:", ", ".join(f"{n}({w:+.2f})" for n, w in imp[:5]))

    if args.save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "features": FEATURE_NAMES}))
        print(f"\nSaved detector → {OUT}")


if __name__ == "__main__":
    main()
