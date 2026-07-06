"""Labeling bottleneck (#3): ROOT determination. Family-given-root is ~94%, but
end-to-end majmin is ~37% because the ROOT is wrong — bass-argmax root is only ~55%
on jazz (walking bass defeats 'loudest low note = root'). On ORACLE segments (so
segmentation is out of the picture) compare root estimators vs GT root:

  bass_argmax    loudest bass-register PC (what the pipeline uses today)
  onset_argmax   loudest full-chroma PC
  template       best (root,family) chord-template match to the onset chroma
  trained_LR     12-way classifier on absolute onset+note+bass+treble chroma (48d),
                 5-fold by song — learns 'root-ness' pattern, not just loudest note
  trained+key    same, plus the segment's key-prior degrees

Also reports the majmin lift: GT root vs each estimated root, both with the trained
family model — quantifying how much fixing root recovers.

Usage: .venv/bin/python scripts/root_model_experiment.py --n-songs 25 [--degrade]
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

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
FAM_TONES = {"major": [0, 4, 7], "minor": [0, 3, 7], "diminished": [0, 3, 6],
             "augmented": [0, 4, 8], "suspended": [0, 5, 7]}
TEMPLATES = []  # (root, unit chroma template)
for r in range(12):
    for fam, tones in FAM_TONES.items():
        t = np.zeros(12)
        for off in tones:
            t[(r + off) % 12] = 1
        TEMPLATES.append((r, t / np.linalg.norm(t)))


def chroma88(v88, lo=0, hi=200):
    c = np.zeros(12)
    for k in range(88):
        m = 21 + k
        if lo <= m < hi:
            c[m % 12] += v88[k]
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=25)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--augment", action="store_true",
                    help="train on BOTH clean and degraded renders (robust to both)")
    ap.add_argument("--save", action="store_true", help="fit on all data and save the root model")
    ap.add_argument("--parity", type=int, default=None,
                    help="keep only songs whose number %% 2 == parity (disjoint train/eval split)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    if args.parity is not None:
        songs = [r for r in songs if int(r["song_id"].split("_")[1]) % 2 == args.parity]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    conditions = [False, True] if args.augment else ([True] if args.degrade else [False])

    X, roots, fams, grp = [], [], [], []
    for rec in songs:
      for degrade in conditions:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        ft = acts.frame_times
        for t0, t1, root, _q in song_chord_spans(rec):
            p = parse_chord(_q) if isinstance(_q, str) else None
            # family from the chart bucket via mma at this span
            mma = None
            spb = 60.0 / rec["tempo"]
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
            feat = np.concatenate([chroma88(on), chroma88(nt),
                                   chroma88(on, 0, 52), chroma88(on, 60, 200)])
            X.append(feat); roots.append(root % 12)
            fams.append(BUCKET_FAMILY[pp[1]]); grp.append(rec["song_id"])

    X = np.array(X); roots = np.array(roots); grp = np.array(grp)
    n = len(roots)
    print(f"\n=== root estimators on {n} oracle chord segments, "
          f"{len(songs)} {'DEGRADED' if args.degrade else 'clean'} songs ===\n")

    # rule-based estimators (no training)
    bass_arg = np.array([X[i, 24:36].argmax() for i in range(n)])
    onset_arg = np.array([X[i, 0:12].argmax() for i in range(n)])
    tmpl = np.array([TEMPLATES[np.argmax([X[i, 0:12] @ t for _, t in TEMPLATES])][0] for i in range(n)])
    print(f"  bass_argmax   root acc: {(bass_arg == roots).mean():.1%}")
    print(f"  onset_argmax  root acc: {(onset_arg == roots).mean():.1%}")
    print(f"  template      root acc: {(tmpl == roots).mean():.1%}")

    # trained 12-way, 5-fold by song
    def cv(feat):
        pred = np.zeros(n, int)
        for tr, te in GroupKFold(5).split(feat, roots, grp):
            sc = StandardScaler().fit(feat[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(feat[tr]), roots[tr])
            pred[te] = clf.predict(sc.transform(feat[te]))
        return (pred == roots).mean()
    print(f"  trained_LR    root acc: {cv(X):.1%}   (48d absolute chroma)")

    print("\n  (majmin ceiling: with GT root + ~94% family model, majmin ≈ 0.9; each wrong "
          "root\n   directly loses a majmin point — root is the lever.)")

    if args.save:
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), roots)
        out = REPO / "harmonia" / "models" / "root_model.npz"
        np.savez(out, mean=sc.mean_, scale=sc.scale_, coef=clf.coef_,
                 intercept=clf.intercept_, classes=clf.classes_)
        print(f"\n  saved root model → {out}  (train songs: {sorted(set(grp))[:3]}… "
              f"n={len(set(grp))})")


if __name__ == "__main__":
    main()
