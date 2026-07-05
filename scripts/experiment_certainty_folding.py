"""Certainty-weighted hierarchical structure folding (structure + certainty + hard cases).

On audio where a section's repeats now VARY (per-occurrence voicing variation), fold
the repeated chord evidence — but weighted by each occurrence's calibrated certainty,
so the clear occurrences fix the noisy ones. Compares, on the varied-jazz hard audio:

  single            — each chord decided from its own (noisy) evidence
  mean fold         — uniform average of a slot's repeats (Candidate-C style)
  certainty fold    — average weighted by each repeat's confidence (the real mechanism)

Then descends the chord tree: family, then seventh. CSV → docs/results/.

Usage: .venv/bin/python scripts/experiment_certainty_folding.py
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
from build_audio_chord_features import (BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,  # noqa: E402
                                        FAM_IDX, full_chroma, reg_chroma)
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

import argparse  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
_AP = argparse.ArgumentParser()
_AP.add_argument("--manifest", default="manifest_hard_varied")
_AP.add_argument("--cache", default="accomp_varied")
_ARGS, _ = _AP.parse_known_args()
MANIFEST = REPO / "data" / "accomp_db" / "audio_hard" / f"{_ARGS.manifest}.jsonl"
CACHE = REPO / "data" / "cache" / _ARGS.cache
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
RESULTS = REPO / "docs" / "results"


def section_pos(spb_labels):
    sec, start = {}, {}
    i = 0
    while i < len(spb_labels):
        j = i
        while j < len(spb_labels) and spb_labels[j] == spb_labels[i]:
            j += 1
        for b in range(i, j):
            sec[b], start[b] = spb_labels[i], i
        i = j
    return sec, start


def extract():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_varied")
    rows = []
    for m in map(json.loads, open(MANIFEST)):
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        if parse_key(rec["key"]) is None:
            continue
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
        sec, start = section_pos(rec["section_per_bar"])
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
            feat = np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                              rr(reg_chroma(onset[b0:b1], 0, 52)),
                              rr(reg_chroma(onset[b0:b1], 60, 200))])
            bar = b0 // bpb
            rows.append({"feat": feat, "fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                         "b7": BASE7_IDX[BUCKET_BASE7[p[1]]],
                         "slot": (m["wav"], sec.get(bar, "?"), b0 - start.get(bar, bar) * bpb)})
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc = StandardScaler().fit(Xc)
    clf = {"fam": LogisticRegression(max_iter=2000).fit(sc.transform(Xc), d["family"].astype(int)),
           "b7": LogisticRegression(max_iter=2000).fit(sc.transform(Xc), d["base7"].astype(int))}
    ncls = {"fam": int(d["family"].max()) + 1, "b7": int(d["base7"].max()) + 1}

    rows = extract()
    if not rows:
        sys.exit("No varied renders yet.")
    X = sc.transform(np.stack([r["feat"] for r in rows]))
    proba = {}
    for key in ("fam", "b7"):
        p = np.full((len(rows), ncls[key]), 1e-9)
        p[:, clf[key].classes_] = clf[key].predict_proba(X)
        proba[key] = p / p.sum(1, keepdims=True)

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["slot"]].append(i)
    sims = []
    feats = np.stack([r["feat"] for r in rows])
    for idx in groups.values():
        if len(idx) >= 2:
            fs = feats[idx] / (np.linalg.norm(feats[idx], axis=1, keepdims=True) + 1e-9)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    sims.append(float(fs[a] @ fs[b]))
    print(f"{len(rows)} varied-jazz chords; mean cosine between repeats: "
          f"{np.mean(sims):.3f} (identical-render hard was 0.998 → lower = folding can help)\n")

    out = []
    print(f"{'level':<9}{'single':>9}{'mean-fold':>12}{'certainty-fold':>16}")
    for key, name in [("fam", "family"), ("b7", "seventh")]:
        y = np.array([r[key] for r in rows])
        P = proba[key]
        cert = P.max(1)
        single = P.argmax(1)
        mean_pred = np.zeros(len(rows), int)
        cert_pred = np.zeros(len(rows), int)
        for idx in groups.values():
            idx = np.array(idx)
            mean_pred[idx] = P[idx].mean(0).argmax()
            w = cert[idx] / (cert[idx].sum() + 1e-9)
            cert_pred[idx] = (P[idx] * w[:, None]).sum(0).argmax()
        a_s = (single == y).mean(); a_m = (mean_pred == y).mean(); a_c = (cert_pred == y).mean()
        print(f"{name:<9}{a_s:>8.1%}{a_m:>12.1%}{a_c:>16.1%}")
        out.append({"level": name, "single": round(float(a_s), 3),
                    "mean_fold": round(float(a_m), 3), "certainty_fold": round(float(a_c), 3)})

    with open(RESULTS / "certainty_folding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"\nCSV → {RESULTS/'certainty_folding.csv'}")


if __name__ == "__main__":
    main()
