"""Structure battle, step 1: does folding repeated sections clean up chords on HARD audio?

On clean audio, folding repeated sections did nothing (MMA renders repeats
identically → no noise to average out). On HARD audio the same sections are buried
under independently-varying noise/mix, so averaging the audio evidence across a
section's repeats should DENOISE and give a cleaner chord. This is the user's
hypothesis that structure helps most on the hard cases — tested directly.

For each chord: classify from its own (noisy) audio vs from the audio AVERAGED over
all repeats of the same (section-label, position-in-section) within the render,
using the ground-truth structure. Clean-trained model. Family + seventh.
CSV → docs/results/structure_on_hard.csv.

Usage: .venv/bin/python scripts/experiment_structure_on_hard.py
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

DB = REPO / "data" / "accomp_db" / "db.jsonl"
HARD_MANIFEST = REPO / "data" / "accomp_db" / "audio_hard" / "manifest_hard.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
RESULTS = REPO / "docs" / "results"
B7_FAM = None


def section_pos(section_per_bar):
    sec, start = {}, {}
    i = 0
    while i < len(section_per_bar):
        j = i
        while j < len(section_per_bar) and section_per_bar[j] == section_per_bar[i]:
            j += 1
        for b in range(i, j):
            sec[b], start[b] = section_per_bar[i], i
        i = j
    return sec, start


def extract():
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

    rows = extract()
    print(f"{len(rows)} hard chords\n")

    # fold: average feature over repeats of the same slot within a render
    groups = defaultdict(list)
    for r in rows:
        groups[r["slot"]].append(r)
    folded = {}
    sims = []
    for g in groups.values():
        avg = np.mean([r["feat"] for r in g], axis=0)
        for r in g:
            folded[id(r)] = avg
        if len(g) >= 2:
            fs = [r["feat"] / (np.linalg.norm(r["feat"]) + 1e-9) for r in g]
            for a in range(len(fs)):
                for b in range(a + 1, len(fs)):
                    sims.append(float(fs[a] @ fs[b]))
    rep = sum(1 for g in groups.values() if len(g) >= 2)
    print(f"repeated slots: {rep}; mean cosine between hard-audio repeats: "
          f"{np.mean(sims) if sims else float('nan'):.3f} (clean was 0.93 → lower = more to gain)\n")

    Xsingle = sc.transform(np.stack([r["feat"] for r in rows]))
    Xfold = sc.transform(np.stack([folded[id(r)] for r in rows]))

    print(f"{'level':<10}{'single (noisy)':>16}{'structure-folded':>18}{'gain':>8}")
    out = []
    for level, key in [("family", "fam"), ("seventh", "b7")]:
        y = np.array([r[key] for r in rows])
        a_single = (clf[key].predict(Xsingle) == y).mean()
        a_fold = (clf[key].predict(Xfold) == y).mean()
        print(f"{level:<10}{a_single:>15.1%}{a_fold:>18.1%}{a_fold-a_single:>+8.1%}")
        out.append({"level": level, "single": round(float(a_single), 3),
                    "structure_folded": round(float(a_fold), 3),
                    "gain": round(float(a_fold - a_single), 3)})
    with open(RESULTS / "structure_on_hard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"\nCSV → {RESULTS/'structure_on_hard.csv'}")
    print("If folding helps here (unlike on clean audio), structure genuinely cleans "
          "up chords\non realistic noisy audio — the user's hypothesis, validated.")


if __name__ == "__main__":
    main()
