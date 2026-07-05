"""Train every noise-impacted aspect on degraded audio (user's ask), disk-safe.

The clean-trained model is out-of-distribution on grubby/phone audio → low accuracy
AND miscalibrated certainty → structure folding fails. This retrains on the degraded
distribution and measures whether it recovers all three:
  1. accuracy (family/seventh/exact),
  2. certainty calibration (ECE),
  3. structure-folding gain (single vs certainty-weighted fold),
comparing the CLEAN-trained model vs a model trained WITH degraded audio, both
evaluated on held-out degraded audio (5-fold by song).

Disk-safe: renders each song's degraded audio to a temp WAV, extracts Basic Pitch,
then deletes the WAV immediately (peak = one WAV).

Usage: .venv/bin/python scripts/train_on_degraded.py --n-songs 30 --variants 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import warnings
from collections import defaultdict
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
from analyze_accomp_priors import parse_key  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import (BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,  # noqa: E402
                                        EXACT_IDX, FAM_IDX, full_chroma, reg_chroma)
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
RESULTS = REPO / "docs" / "results"


def ece(conf, correct, nb=10):
    e, edges = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < nb - 1 else conf <= 1.0)
        if m.sum():
            e += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return e


def section_pos(labels):
    sec, start, i = {}, {}, 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        for b in range(i, j):
            sec[b], start[b] = labels[i], i
        i = j
    return sec, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=30)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--corpus", default="jazz1460")
    ap.add_argument("--clean", action="store_true", help="no degradation (clean audio)")
    ap.add_argument("--out", default="train_on_degraded")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(DB)]
    songs = [r for r in records if r["corpus"] == args.corpus and r["beats_per_bar"] == 4]
    songs = [r for r in songs if (REPO / r["midi_path"]).exists()][:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)
    rows = []
    for si, rec in enumerate(songs):
        k = parse_key(rec["key"])
        if k is None:
            continue
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"] * bpb
        sec, start = section_pos(rec["section_per_bar"])
        chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"] for e in rec["chord_timeline"]}
        for v in range(args.variants):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=renderer._find_soundfont("MuseScore_General.sf2")))
                if not args.clean:
                    a, sr = sf.read(tmp)
                    a = a.mean(1) if a.ndim > 1 else a
                    sf.write(tmp, time_varying_degrade(a.astype("float32"), sr, rng), sr)
                acts = ex.extract(tmp, use_cache=False)
            finally:
                tmp.unlink(missing_ok=True)          # delete WAV immediately (disk-safe)
            onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
            note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
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
                feat = np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                                  rr(reg_chroma(onset[b0:b1], 0, 52)), rr(reg_chroma(onset[b0:b1], 60, 200))])
                bar = b0 // bpb
                rows.append({"feat": feat, "fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                             "b7": BASE7_IDX[BUCKET_BASE7[p[1]]], "ex": EXACT_IDX[p[1]],
                             "song": rec["song_id"], "v": v,
                             "slot": (sec.get(bar, ("?", bar))[0], b0 - start.get(bar, bar) * bpb, root)})
        if (si + 1) % 10 == 0:
            print(f"  … {si+1}/{len(songs)} songs, {len(rows)} chords")

    Xd = np.stack([r["feat"] for r in rows])
    song = np.array([r["song"] for r in rows])
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    kind = "clean" if args.clean else "degraded"
    m0 = "jazz-only" if args.corpus != "jazz1460" else f"clean-trained"
    m1 = f"+{args.corpus[:4]}-train" if args.corpus != "jazz1460" else "+degraded-train"
    print(f"\n{len(rows)} {args.corpus} {kind} chords, {len(set(song.tolist()))} songs "
          f"(test); base training = jazz clean\n")

    out = []
    print(f"{'level':<9}{'':<17}{'accuracy':>10}{'ECE':>8}{'fold gain':>11}")
    for lvl, key in [("family", "fam"), ("seventh", "b7"), ("exact", "ex")]:
        yd = np.array([r[key] for r in rows]); yc = d[{"fam": "family", "b7": "base7", "ex": "exact"}[key]].astype(int)
        nc = int(max(yd.max(), yc.max()) + 1)
        for name, use_deg in [(m0, False), (m1, True)]:
            accs, eces, single_acc, fold_acc = [], [], [], []
            for tr, te in GroupKFold(5).split(Xd, yd, song):
                Xtr = np.vstack([Xc, Xd[tr]]) if use_deg else Xc
                ytr = np.concatenate([yc, yd[tr]]) if use_deg else yc
                scaler = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
                P = np.full((len(te), nc), 1e-9); P[:, clf.classes_] = clf.predict_proba(scaler.transform(Xd[te]))
                P /= P.sum(1, keepdims=True)
                pred = P.argmax(1); conf = P.max(1)
                accs.append((pred == yd[te]).mean())
                eces.append(ece(conf, pred == yd[te]))
                # certainty-weighted fold on the test fold
                Pf = P.copy(); grp = defaultdict(list)
                for i, gi in enumerate(te):
                    grp[rows[gi]["slot"]].append(i)
                for g in grp.values():
                    if len(g) >= 2:
                        g = np.array(g); w = conf[g] / (conf[g].sum() + 1e-9)
                        Pf[g] = (P[g] * w[:, None]).sum(0)
                single_acc.append((P.argmax(1) == yd[te]).mean())
                fold_acc.append((Pf.argmax(1) == yd[te]).mean())
            a, e = np.mean(accs), np.mean(eces)
            fg = np.mean(fold_acc) - np.mean(single_acc)
            print(f"{lvl if name==m0 else '':<9}{name:<17}{a:>9.1%}{e:>8.3f}{fg:>+11.1%}")
            out.append({"level": lvl, "model": name, "accuracy": round(float(a), 3),
                        "ece": round(float(e), 3), "fold_gain": round(float(fg), 3)})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"\nCSV → {RESULTS/'train_on_degraded.csv'}")


if __name__ == "__main__":
    main()
