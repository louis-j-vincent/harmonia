"""Retrain the audio chord model ON hard audio (augmentation) and measure the lift.

Currently the model is trained on clean audio and deployed on hard audio (86.6%
family / 75.2% seventh). This tests whether training WITH hard audio recovers
accuracy: 5-fold grouped-by-song CV on the hard renders, comparing
  clean-only training      (baseline — never saw hard audio)
  clean + hard training     (augmentation)
on the held-out hard fold. Family / seventh / exact. CSV → docs/results/.

BP activations for the hard renders are already cached (data/cache/accomp_hard).

Usage: .venv/bin/python scripts/retrain_on_hard.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import (BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,  # noqa: E402
                                        EXACT_IDX, FAM_IDX, full_chroma, reg_chroma)
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
HARD_MANIFEST = REPO / "data" / "accomp_db" / "audio_hard" / "manifest_hard.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
RESULTS = REPO / "docs" / "results"


def extract_hard():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_hard")
    feats, fam, b7, exact, song = [], [], [], [], []
    for m in map(json.loads, open(HARD_MANIFEST)):
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
        chord_at = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"] for ev in rec["chord_timeline"]}
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), nb)
            mma = chord_at.get(b0)
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
                continue
            rt = root % 12
            rr = lambda c: np.roll(c, -rt)
            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            if on_c.sum() < 1e-9:
                continue
            feats.append(np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                                    rr(reg_chroma(onset[b0:b1], 0, 52)),
                                    rr(reg_chroma(onset[b0:b1], 60, 200))]))
            fam.append(FAM_IDX[BUCKET_FAMILY[p[1]]])
            b7.append(BASE7_IDX[BUCKET_BASE7[p[1]]])
            exact.append(EXACT_IDX[p[1]])
            song.append(m["song_id"])
    return (np.array(feats), np.array(fam), np.array(b7), np.array(exact), np.array(song))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xclean = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    yclean = {"family": d["family"].astype(int), "seventh": d["base7"].astype(int),
              "exact": d["exact"].astype(int)}

    Xh, fam, b7, exact, song = extract_hard()
    yh = {"family": fam, "seventh": b7, "exact": exact}
    print(f"{len(Xh)} hard chords, {len(set(song.tolist()))} songs; "
          f"{len(Xclean)} clean chords\n")

    def fit_predict(Xtr, ytr, Xte, nc):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
        return pred

    rows = []
    print(f"{'level':<10}{'clean-only (deploy)':>20}{'clean+hard (augmented)':>24}")
    for level in ("family", "seventh", "exact"):
        nc = int(max(yclean[level].max(), yh[level].max()) + 1)
        gkf = GroupKFold(5)
        acc_clean, acc_aug = [], []
        for tr, te in gkf.split(Xh, yh[level], song):
            # clean-only: train on clean, test on hard-test
            p0 = fit_predict(Xclean, yclean[level], Xh[te], nc)
            acc_clean.append((p0 == yh[level][te]).mean())
            # augmented: train on clean + hard-train, test on hard-test
            Xtr = np.vstack([Xclean, Xh[tr]])
            ytr = np.concatenate([yclean[level], yh[level][tr]])
            p1 = fit_predict(Xtr, ytr, Xh[te], nc)
            acc_aug.append((p1 == yh[level][te]).mean())
        a0, a1 = float(np.mean(acc_clean)), float(np.mean(acc_aug))
        print(f"{level:<10}{a0:>19.1%}{a1:>23.1%}   (Δ {a1-a0:+.1%})")
        rows.append({"level": level, "clean_only": round(a0, 3),
                     "clean_plus_hard": round(a1, 3), "gain": round(a1 - a0, 3)})

    with open(RESULTS / "retrain_hard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV → {RESULTS/'retrain_hard.csv'}")


if __name__ == "__main__":
    main()
