"""Measure accuracy gain from motif-based voting vs GT-fold vs raw audio.

Three conditions (global model, no LOO — measures folding delta):
  1. audio     — raw model probabilities, no folding
  2. GT fold   — certainty-weighted vote grouped by (GT section, pos, root)   [oracle]
  3. motif fold— certainty-weighted vote grouped by bar-aligned motif clusters [inferred]

Runs in ~2 min on 150 songs by using pre-computed chord features from the npz.

Usage:
    .venv/bin/python scripts/eval_motif_fold_accuracy.py
    .venv/bin/python scripts/eval_motif_fold_accuracy.py --limit 20   # quick test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harmonia.models.motif import Chord as MChord, find_motifs
from analyze_accomp_emission import parse_chord, song_chord_spans
from analyze_accomp_priors import parse_key
from build_audio_chord_features import (
    BASE7, BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,
    EXACT, EXACT_IDX, FAM_IDX,
)

DB   = REPO / "data" / "accomp_db" / "db.jsonl"
FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"


def _build_chord_meta(rec: dict, manifest_entry: dict) -> list[dict]:
    """Build per-chord metadata from DB record only (no audio). Returns list aligned to npz rows."""
    spb = 60.0 / manifest_entry["tempo"]
    bpb = manifest_entry["beats_per_bar"]
    nb  = manifest_entry["n_bars"] * bpb
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"] for e in rec["chord_timeline"]}
    sec: dict[int, tuple] = {}
    i, lab = 0, rec["section_per_bar"]
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        for b in range(i, j):
            sec[b] = (lab[i], i)
        i = j

    meta = []
    for t0, t1, root, _q in song_chord_spans(rec):
        b0 = int(round(t0 / spb))
        b1 = min(int(round(t1 / spb)), nb)
        mma = chord_at.get(b0)
        p = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
            continue
        bar = b0 // bpb
        seclab, secstart = sec.get(bar, ("?", bar))
        pos_in_sec = b0 - secstart * bpb
        meta.append({
            "root": root,
            "b0": b0, "bar": bar,
            "gt_qual": p[1],
            "gt_fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
            "gt_b7":  BASE7_IDX[BUCKET_BASE7[p[1]]],
            "gt_ex":  EXACT_IDX[p[1]],
            "slot":   (seclab, pos_in_sec, root),
        })
    return meta


def _motif_groups(chords: list[dict]) -> dict[str, list[int]]:
    """Group chord indices by motif slot (shape-transposing, bar-aligned)."""
    mc = [MChord(root=c["root"] % 12,
                 qual=BUCKET_FAMILY.get(c["gt_qual"], "dom7"),
                 label=str(c["root"]),
                 bar=c["bar"]) for c in chords]
    if not mc:
        return {}
    n_bars = max(c.bar for c in mc) + 1
    avg_cpb = max(1, round(len(mc) / n_bars))
    min_len = max(1, avg_cpb * 2)
    max_len = min(min_len * 4, 32)
    try:
        motifs = find_motifs(mc, shape=True, min_len=min_len, max_len=max_len, min_count=2)
    except Exception:
        return {}
    chord_motif: dict[int, str] = {}
    for m in motifs:
        if m.length < 2:
            continue
        for occ_start in m.occurrences:
            for k in range(m.length):
                idx = occ_start + k
                if idx < len(mc):
                    chord_motif[idx] = f"{m.key}:{k}"
    groups: dict[str, list[int]] = defaultdict(list)
    for i, mk in chord_motif.items():
        groups[mk].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _vote(prob: dict, groups: dict[str, list[int]]) -> None:
    for lv in ("fam", "b7", "ex"):
        cert = prob[lv].max(1)
        for g in groups.values():
            g_arr = np.array(g)
            w = cert[g_arr] / (cert[g_arr].sum() + 1e-9)
            prob[lv][g_arr] = (prob[lv][g_arr] * w[:, None]).sum(0)


def _acc(prob: dict, chords: list[dict]) -> dict[str, float]:
    return {
        "fam": float(np.mean([prob["fam"][i].argmax() == chords[i]["gt_fam"] for i in range(len(chords))])),
        "b7":  float(np.mean([prob["b7"][i].argmax()  == chords[i]["gt_b7"]  for i in range(len(chords))])),
        "ex":  float(np.mean([prob["ex"][i].argmax()  == chords[i]["gt_ex"]  for i in range(len(chords))])),
    }


def eval_song(rec: dict, man_entry: dict, song_indices: np.ndarray,
              d: dict, sc: StandardScaler, clf: dict, ncl: dict) -> dict | None:
    chords = _build_chord_meta(rec, man_entry)
    # song_indices: npz row indices for transpose=0 of this song
    song_feats = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])[song_indices]

    # The npz builder skips chords with zero onset energy; reconcile by GT family match
    if len(song_feats) != len(chords):
        npz_gt = d["family"][song_indices].astype(int)
        keep, npz_i = [], 0
        for ci, c in enumerate(chords):
            if npz_i < len(npz_gt) and npz_gt[npz_i] == c["gt_fam"]:
                keep.append(ci)
                npz_i += 1
        chords = [chords[i] for i in keep]
        if len(chords) != len(song_feats):
            return None

    if not chords:
        return None

    X = sc.transform(song_feats)
    prob_base: dict[str, np.ndarray] = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob_base[lv] = p / p.sum(1, keepdims=True)

    r_audio = _acc(copy.deepcopy(prob_base), chords)

    prob_gt = copy.deepcopy(prob_base)
    gt_groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chords):
        gt_groups[str(c["slot"])].append(i)
    gt_groups = {k: v for k, v in gt_groups.items() if len(v) >= 2}
    _vote(prob_gt, gt_groups)
    r_gt = _acc(prob_gt, chords)

    prob_motif = copy.deepcopy(prob_base)
    motif_groups = _motif_groups(chords)
    _vote(prob_motif, motif_groups)
    r_motif = _acc(prob_motif, chords)

    return {
        "title": rec["title"], "n": len(chords),
        "audio": r_audio, "gt": r_gt, "motif": r_motif,
        "n_gt_groups":    sum(1 for g in gt_groups.values()),
        "n_motif_groups": sum(1 for g in motif_groups.values()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl")):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm
    avail = sorted([sid for sid in recs if sid in man], key=lambda s: recs[s]["title"])
    if args.limit:
        avail = avail[:args.limit]

    d = np.load(FEAT, allow_pickle=True)
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}

    print("Fitting global classifier...", flush=True)
    Xall = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc = StandardScaler().fit(Xall)
    clf = {lv: LogisticRegression(max_iter=500, solver="lbfgs").fit(
               sc.transform(Xall), d[k].astype(int))
           for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}
    print(f"  done. Evaluating {len(avail)} songs...", flush=True)

    results = []
    for i, sid in enumerate(avail):
        rec = recs[sid]
        print(f"\r[{i+1:3d}/{len(avail)}] {rec['title'][:32]:32s}", end="", flush=True)
        all_idx = np.where(d["song"] == sid)[0]
        # first half = transpose-0 entries (npz stores t=0, then t=transposed)
        song_indices = all_idx[:len(all_idx) // 2] if len(all_idx) > 0 else all_idx
        try:
            r = eval_song(rec, man[sid], song_indices, d, sc, clf, ncl)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n  SKIP {rec['title']}: {e}")

    if not results:
        print("\nNo results.")
        sys.exit(1)

    print(f"\n\nN = {len(results)} songs\n")
    W = 68
    print("=" * W)
    print(f"{'Condition':18s}  {'Family':>8s}  {'Seventh':>8s}  {'Exact':>8s}")
    print("-" * W)
    for cond, lbl in [("audio", "Audio only"), ("motif", "Motif fold"), ("gt", "GT fold [oracle]")]:
        fam = np.mean([r[cond]["fam"] for r in results])
        b7  = np.mean([r[cond]["b7"]  for r in results])
        ex  = np.mean([r[cond]["ex"]  for r in results])
        print(f"{lbl:18s}  {fam:8.1%}  {b7:8.1%}  {ex:8.1%}")
    print("=" * W)

    d_fam_m = np.mean([r["motif"]["fam"] - r["audio"]["fam"] for r in results])
    d_b7_m  = np.mean([r["motif"]["b7"]  - r["audio"]["b7"]  for r in results])
    d_ex_m  = np.mean([r["motif"]["ex"]  - r["audio"]["ex"]  for r in results])
    d_fam_g = np.mean([r["gt"]["fam"]    - r["audio"]["fam"] for r in results])
    d_b7_g  = np.mean([r["gt"]["b7"]     - r["audio"]["b7"]  for r in results])
    d_ex_g  = np.mean([r["gt"]["ex"]     - r["audio"]["ex"]  for r in results])

    print(f"\nDelta motif vs audio:   family {d_fam_m:+.1%}  seventh {d_b7_m:+.1%}  exact {d_ex_m:+.1%}")
    print(f"Delta GT vs audio:      family {d_fam_g:+.1%}  seventh {d_b7_g:+.1%}  exact {d_ex_g:+.1%}")
    eps = 1e-9
    def rat(a, b): return f"{a/(b+eps):.0%}" if abs(b) > 0.001 else "N/A"
    print(f"Motif/GT delta ratio:   family {rat(d_fam_m,d_fam_g)}  seventh {rat(d_b7_m,d_b7_g)}  exact {rat(d_ex_m,d_ex_g)}")
    print(f"\nAvg groups — GT: {np.mean([r['n_gt_groups'] for r in results]):.1f}  "
          f"motif: {np.mean([r['n_motif_groups'] for r in results]):.1f}")
