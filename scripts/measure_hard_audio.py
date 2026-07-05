"""Measure the audio chord model + key prior on HARD audio, sliced by difficulty.

Uses build_accomp_audio_hard.py's renders (multi-instrument, mix-imbalanced,
noisy). Reports family/seventh accuracy overall and per difficulty axis
(scenario, SNR, groove), for audio-alone vs audio+key-prior — showing where the
clean-pilot conclusions break down and the priors start paying off.

The audio model is trained on the CLEAN pilot (data/cache/audio_chord_features.npz)
and evaluated on the HARD renders — i.e. "trained on easy, deployed on hard",
the realistic deployment gap.

All result tables are written as CSVs under docs/results/ for the blog.

Usage: .venv/bin/python scripts/measure_hard_audio.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import (BUCKET_BASE7, BUCKET_FAMILY, full_chroma,  # noqa: E402
                                        reg_chroma)
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
HARD_MANIFEST = REPO / "data" / "accomp_db" / "audio_hard" / "manifest_hard.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
RESULTS = REPO / "docs" / "results"
FAM = ["major", "minor", "diminished", "augmented", "suspended"]
FAMI = {f: i for i, f in enumerate(FAM)}
B7 = sorted(set(BUCKET_BASE7.values()))
B7I = {b: i for i, b in enumerate(B7)}
B7_FAM = np.array([FAMI[BUCKET_FAMILY[b]] for b in BUCKET_FAMILY
                   for _ in [0] if False])
_b7fam = {BUCKET_BASE7[b]: FAMI[BUCKET_FAMILY[b]] for b in BUCKET_FAMILY}
B7_FAM = np.array([_b7fam[b] for b in B7])


def extract_hard():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_hard")
    rows = []
    for m in map(json.loads, open(HARD_MANIFEST)):
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic = k[0]
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
            root_t = root % 12
            rr = lambda c: np.roll(c, -root_t)
            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            if on_c.sum() < 1e-9:
                continue
            feat = np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                              rr(reg_chroma(onset[b0:b1], 0, 52)),
                              rr(reg_chroma(onset[b0:b1], 60, 200))])
            rows.append({
                "feat": feat, "deg": (root - tonic) % 12,
                "mode": 0 if k[1] == "major" else 1,
                "fam": FAMI[BUCKET_FAMILY[p[1]]], "b7": B7I[BUCKET_BASE7[p[1]]],
                "scenario": m["scenario"], "snr": m["snr_db"], "groove": m["groove"],
            })
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    # audio model trained on CLEAN pilot
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc = StandardScaler().fit(Xc)
    clf_fam = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xc), d["family"].astype(int))
    clf_b7 = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xc), d["base7"].astype(int))
    # key prior tables P(fam|deg,mode), P(b7|deg,mode) from clean labels
    keyfam = defaultdict(lambda: np.ones(5) * 0.5)
    keyb7 = defaultdict(lambda: np.ones(len(B7)) * 0.5)
    for deg, mode, fam, b7 in zip(d["degree"], d["mode"], d["family"], d["base7"]):
        keyfam[(int(mode), int(deg))][int(fam)] += 1
        keyb7[(int(mode), int(deg))][int(b7)] += 1

    rows = extract_hard()
    if not rows:
        sys.exit("No hard renders yet — run build_accomp_audio_hard.py first.")
    X = sc.transform(np.stack([r["feat"] for r in rows]))
    pf = np.full((len(rows), 5), 1e-9); pf[:, clf_fam.classes_] = clf_fam.predict_proba(X)
    pb = np.full((len(rows), len(B7)), 1e-9); pb[:, clf_b7.classes_] = clf_b7.predict_proba(X)
    yf = np.array([r["fam"] for r in rows]); yb = np.array([r["b7"] for r in rows])

    def key_logs(level):
        out = np.zeros((len(rows), 5 if level == "fam" else len(B7)))
        tbl = keyfam if level == "fam" else keyb7
        for i, r in enumerate(rows):
            c = tbl[(r["mode"], r["deg"])]
            out[i] = np.log(c / c.sum())
        return out

    lkf, lkb = key_logs("fam"), key_logs("b7")

    def acc(logaudio, logkey, y, w):
        return ((logaudio + w * logkey).argmax(1) == y).mean()

    def best_w(logaudio, logkey, y):
        base = acc(logaudio, logkey, y, 0)
        b = (base, 0.0)
        for w in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
            a = acc(logaudio, logkey, y, w)
            if a > b[0]:
                b = (a, w)
        return base, b[0], b[1]

    la_f, la_b = np.log(pf), np.log(pb)

    # ── overall + per-slice CSV ───────────────────────────────────────────────
    def slice_rows(keyfn, label):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[keyfn(r)].append(i)
        res = []
        for gval, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
            idx = np.array(idx)
            fb, ff, fw = best_w(la_f[idx], lkf[idx], yf[idx])
            bb, bf, bw = best_w(la_b[idx], lkb[idx], yb[idx])
            res.append({"slice": label, "value": gval, "n": len(idx),
                        "family_audio": round(fb, 3), "family_audio_key": round(ff, 3),
                        "family_key_recovery": round(ff - fb, 3), "family_best_w": fw,
                        "seventh_audio": round(bb, 3), "seventh_audio_key": round(bf, 3),
                        "seventh_key_recovery": round(bf - bb, 3), "seventh_best_w": bw})
        return res

    allres = []
    allres += slice_rows(lambda r: "ALL", "overall")
    allres += slice_rows(lambda r: r["scenario"], "scenario")
    allres += slice_rows(lambda r: ("clean" if r["snr"] is None else f"snr{int(r['snr'])}"), "snr")
    allres += slice_rows(lambda r: r["groove"], "groove")

    csv_path = RESULTS / "hard_audio_priors.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allres[0].keys()))
        w.writeheader(); w.writerows(allres)

    print(f"{len(rows)} hard chords. Results → {csv_path}\n")
    print(f"{'slice':<10}{'value':<18}{'n':>5}{'famA':>7}{'famA+key':>10}{'+rec':>7}"
          f"{'7thA':>7}{'7thA+key':>10}{'+rec':>7}")
    for r in allres:
        print(f"{r['slice']:<10}{str(r['value']):<18}{r['n']:>5}{r['family_audio']:>7.1%}"
              f"{r['family_audio_key']:>10.1%}{r['family_key_recovery']:>+7.1%}"
              f"{r['seventh_audio']:>7.1%}{r['seventh_audio_key']:>10.1%}"
              f"{r['seventh_key_recovery']:>+7.1%}")


if __name__ == "__main__":
    main()
