"""Offline p_self sweep using cached per-beat posteriors (scratchpad/beat_posteriors/*.npz).
No audio download / Basic Pitch re-run — just re-decode + re-score boundary
matching against Billboard GT for each candidate p_self.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mirdata

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.chord_hmm import viterbi as hmm_viterbi

bb = mirdata.initialize("billboard")

SONGS = {"1111": "1111", "887": "887", "1027": "1027", "362": "362"}
POST_DIR = REPO / "scratchpad" / "beat_posteriors"


def gt_changes(tid):
    t = bb.track(tid)
    cd = t.chords_full
    ivs, labs = np.asarray(cd.intervals), np.asarray(cd.labels)
    order = np.argsort(ivs[:, 0])
    ivs, labs = ivs[order], labs[order]
    starts, ends, lbls = [], [], []
    for (s, e), l in zip(ivs, labs):
        if starts and lbls[-1] == l and abs(s - ends[-1]) < 1e-6:
            ends[-1] = e
            continue
        starts.append(s); ends.append(e); lbls.append(l)
    starts = np.array(starts)
    return starts[1:]  # interior boundaries


def match_boundaries(gt_b, pred_b, tol):
    used_pred = set()
    tp_gt = []
    for i, g in enumerate(gt_b):
        cands = [(abs(g - p), j) for j, p in enumerate(pred_b) if j not in used_pred and abs(g - p) <= tol]
        if cands:
            cands.sort()
            j = cands[0][1]
            used_pred.add(j)
            tp_gt.append(i)
    fn = len(gt_b) - len(tp_gt)
    fp = len(pred_b) - len(used_pred)
    return len(tp_gt), fn, fp


def decode(root_p, qual_p, bt, tempo_bpm, p_self):
    n_beats = root_p.shape[0]
    n_root = root_p.shape[1]
    n_qual = qual_p.shape[1]
    n_states = n_root * n_qual
    log_emission = np.empty((n_beats, n_states), dtype=np.float64)
    for i in range(n_beats):
        log_root = np.log(np.clip(root_p[i], 1e-9, None))
        log_qual = np.log(np.clip(qual_p[i], 1e-9, None))
        log_emission[i] = (log_root[:, None] + log_qual[None, :]).ravel()
    off_diag = (1.0 - p_self) / max(n_states - 1, 1)
    log_transition = np.full((n_states, n_states), np.log(off_diag), dtype=np.float64)
    np.fill_diagonal(log_transition, np.log(p_self))
    log_init = np.full(n_states, -np.log(n_states), dtype=np.float64)
    path, _ = hmm_viterbi(log_emission, log_transition, log_init)
    smoothed = [(int(s) // n_qual, int(s) % n_qual) for s in path]
    # coalesce
    starts = []
    j = 0
    while j < n_beats:
        r, q = smoothed[j]
        k = j
        while k < n_beats and smoothed[k] == (r, q):
            k += 1
        starts.append(float(bt[j]))
        j = k
    return np.array(starts[1:]), len(starts)


results = {}
for tid, key in SONGS.items():
    d = np.load(POST_DIR / f"bb_{key}.npz")
    root_p, qual_p, bt, tempo_bpm = d["root_p"], d["qual_p"], d["bt"], float(d["tempo_bpm"])
    gt_b = gt_changes(tid)
    beat_dur = 60.0 / max(tempo_bpm, 1.0)
    results[tid] = {}
    for p_self in [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pred_b, n_spans = decode(root_p, qual_p, bt, tempo_bpm, p_self) if p_self > 0 else (None, None)
        if p_self == 0.0:
            continue
        tp, fn, fp = match_boundaries(gt_b, pred_b, 0.5)
        n_gt = len(gt_b)
        n_pred = len(pred_b)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[tid][p_self] = dict(n_spans=n_spans, n_gt=n_gt, n_pred_b=n_pred, P=round(prec, 3), R=round(rec, 3), F1=round(f1, 3))
        print(f"bb_{tid} p_self={p_self:.2f}: n_spans={n_spans} n_gt_b={n_gt} n_pred_b={n_pred} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")
    print()

import json
(REPO / "scratchpad" / "p_self_sweep_results.json").write_text(json.dumps(results, indent=2))
print("saved sweep results")
