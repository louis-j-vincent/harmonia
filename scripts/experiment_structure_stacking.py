"""Structure stacking (AABA): combine the repeats of a chord into one guess.

User's idea: if the song is AABA, the chord at a given spot in section A is the
SAME chord every time A comes around — so gather those repeats and average the
(noisy) audio evidence before naming the chord. More looks → a more robust guess.

Tested three ways, single-observation vs stacked, with fixed nearest-template
classification at each tree level:
  (1) STRUCTURAL — average repeats of (section-label, position-in-section) within
      a render, using the ground-truth song structure.
  (2) CROSS-RENDER — average the two independent renders of the same chord slot
      (different soundfont / noise / transpose). This is the "each repeat is
      played a little differently" case that real performances have but a single
      deterministic MMA render does not — the condition stacking actually needs.

Also reports how DIFFERENT the repeats' audio really is (mean cosine), which
predicts whether stacking can help at all.

Usage: .venv/bin/python scripts/experiment_structure_stacking.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from learn_stage1_mapping import pool_beats, to_chroma  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"

BUCKET_FAMILY = {
    "maj": "major", "maj7": "major", "6": "major", "dom7": "major", "dom7alt": "major",
    "min": "minor", "min7": "minor", "m6": "minor", "minmaj7": "minor",
    "dim": "diminished", "dim7": "diminished", "m7b5": "diminished",
    "aug": "augmented", "aug7": "augmented", "augmaj7": "augmented",
    "sus2": "suspended", "sus4": "suspended", "7sus4": "suspended",
}
BUCKET_BASE7 = {
    "maj": "majT", "6": "majT", "maj7": "maj7", "dom7": "dom7", "dom7alt": "dom7",
    "min": "minT", "m6": "minT", "min7": "min7", "minmaj7": "minmaj7",
    "dim": "dimT", "dim7": "dim7", "m7b5": "m7b5",
    "aug": "augT", "aug7": "aug7", "augmaj7": "augmaj7",
    "sus2": "susT", "sus4": "susT", "7sus4": "7sus4",
}


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


def collect():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")
    items = []
    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        spb = 60.0 / m["tempo"]
        bpb = m["beats_per_bar"]
        n_beats = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        chroma = to_chroma(onset)
        sec, start = section_pos(rec["section_per_bar"])
        chord_at = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"] for ev in rec["chord_timeline"]}
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), n_beats)
            mma = chord_at.get(b0)
            parsed = parse_chord(mma) if mma else None
            if parsed is None or parsed[1] not in BUCKET_FAMILY or b1 <= b0:
                continue
            root_t = (root + m["transpose"]) % 12
            v = np.roll(chroma[b0:b1].sum(axis=0), -root_t)
            if v.sum() < 1e-9:
                continue
            bar = b0 // bpb
            items.append({
                "song": m["song_id"], "wav": m["wav"], "chroma": v / v.sum(),
                "fam": BUCKET_FAMILY[parsed[1]], "b7": BUCKET_BASE7[parsed[1]],
                "exact": parsed[1],
                "sec": sec.get(bar, "?"), "pos": b0 - start.get(bar, bar) * bpb,
                "slot": (m["song_id"], sec.get(bar, "?"), b0 - start.get(bar, bar) * bpb),
            })
    return items


def centroids(items, label):
    acc, n = defaultdict(lambda: np.zeros(12)), Counter()
    for it in items:
        acc[it[label]] += it["chroma"] / (np.linalg.norm(it["chroma"]) + 1e-9)
        n[it[label]] += 1
    labs = sorted(acc)
    M = np.stack([acc[c] / n[c] for c in labs])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return labs, M


def classify(chroma, labs, M):
    v = chroma / (np.linalg.norm(chroma) + 1e-9)
    return labs[int(np.argmax(M @ v))]


def acc_for(items, get_chroma, label, labs, M):
    ok = sum(classify(get_chroma(it), labs, M) == it[label] for it in items)
    return ok / len(items)


def main():
    items = collect()
    print(f"{len(items)} chord instances\n")

    # how similar are repeats, really?
    for keyname, keyfn in [("structural (section,pos in one render)", lambda it: (it["wav"], it["slot"])),
                           ("cross-render (same slot, 2 renders)", lambda it: it["slot"])]:
        groups = defaultdict(list)
        for it in items:
            groups[keyfn(it)].append(it)
        sims = []
        for g in groups.values():
            if len(g) >= 2:
                cs = [x["chroma"] / (np.linalg.norm(x["chroma"]) + 1e-9) for x in g]
                for a in range(len(cs)):
                    for b in range(a + 1, len(cs)):
                        sims.append(float(cs[a] @ cs[b]))
        rep = sum(1 for g in groups.values() if len(g) >= 2)
        print(f"repeat similarity — {keyname}: mean cosine "
              f"{np.mean(sims) if sims else float('nan'):.3f} over {rep} repeated groups")
    print()

    # stacked chroma per grouping
    def stacked_map(keyfn):
        groups = defaultdict(list)
        for it in items:
            groups[keyfn(it)].append(it)
        avg = {}
        for k, g in groups.items():
            m = np.mean([x["chroma"] for x in g], axis=0)
            for x in g:
                avg[id(x)] = m
        return avg

    struct_avg = stacked_map(lambda it: (it["wav"], it["slot"]))
    cross_avg = stacked_map(lambda it: it["slot"])

    print(f"{'level':<12}{'single':>10}{'structural-stack':>18}{'cross-render-stack':>20}")
    print("-" * 60)
    for label, name in [("fam", "family"), ("b7", "seventh"), ("exact", "exact")]:
        labs, M = centroids(items, label)
        a_single = acc_for(items, lambda it: it["chroma"], label, labs, M)
        a_struct = acc_for(items, lambda it: struct_avg[id(it)], label, labs, M)
        a_cross = acc_for(items, lambda it: cross_avg[id(it)], label, labs, M)
        print(f"{name:<12}{a_single:>10.1%}{a_struct:>18.1%}{a_cross:>20.1%}")

    print("\nReading: if repeats are near-identical (high cosine), structural stacking")
    print("can't denoise and won't help — real performances vary more, which is what")
    print("cross-render approximates here (independent soundfont/noise per repeat).")


if __name__ == "__main__":
    main()
