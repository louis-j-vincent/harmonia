"""Check whether Viterbi temporal smoothing (chord_pipeline_v1.py fix,
2026-07-15) regresses root labeling accuracy, using cached per-beat
posteriors (no re-download). Compares GT root (sampled at each Billboard
chords_full interval's midpoint) against:
  (a) raw per-beat argmax root at the beat covering that midpoint (the OLD
      behavior's root label, pre-smoothing)
  (b) Viterbi-smoothed root at that same beat, for several p_self values
Root accuracy should be ~unchanged; quality/family flips due to smoothing
persistence are expected and separately noted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mirdata

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.chord_hmm import viterbi as hmm_viterbi

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

bb = mirdata.initialize("billboard")


def parse_root(label: str):
    if label in ("N", "X"):
        return None
    tok = label.split(":")[0].split("/")[0]
    if not tok or tok[0] not in PC:
        return None
    pc = PC[tok[0]]
    for c in tok[1:]:
        if c == "#":
            pc += 1
        elif c == "b":
            pc -= 1
        else:
            break
    return pc % 12


def decode(root_p, qual_p, bt, p_self):
    n_beats, n_root = root_p.shape
    n_qual = qual_p.shape[1]
    n_states = n_root * n_qual
    log_emission = np.empty((n_beats, n_states))
    for i in range(n_beats):
        lr = np.log(np.clip(root_p[i], 1e-9, None))
        lq = np.log(np.clip(qual_p[i], 1e-9, None))
        log_emission[i] = (lr[:, None] + lq[None, :]).ravel()
    off = (1 - p_self) / (n_states - 1)
    lt = np.full((n_states, n_states), np.log(off))
    np.fill_diagonal(lt, np.log(p_self))
    li = np.full(n_states, -np.log(n_states))
    path, _ = hmm_viterbi(log_emission, lt, li)
    return [(int(s) // n_qual, int(s) % n_qual) for s in path]


for tid in ["1111", "887", "1027", "362"]:
    d = np.load(REPO / f"scratchpad/beat_posteriors/bb_{tid}.npz")
    root_p, qual_p, bt = d["root_p"], d["qual_p"], d["bt"]
    n_beats = root_p.shape[0]
    raw_root = root_p.argmax(1)

    t = bb.track(tid)
    cd = t.chords_full
    ivs, labs = np.asarray(cd.intervals), np.asarray(cd.labels)

    def beat_idx(tm):
        i = np.searchsorted(bt, tm, side="right") - 1
        return int(np.clip(i, 0, n_beats - 1))

    gt_roots, raw_preds = [], []
    for (s, e), l in zip(ivs, labs):
        r = parse_root(l)
        if r is None:
            continue
        mid = (s + e) / 2
        if mid > bt[-1]:
            continue
        bi = beat_idx(mid)
        gt_roots.append(r)
        raw_preds.append(int(raw_root[bi]))
    gt_roots = np.array(gt_roots)
    raw_preds = np.array(raw_preds)
    raw_acc = (gt_roots == raw_preds).mean()

    print(f"bb_{tid}: n_gt_intervals(with root)={len(gt_roots)}  raw_argmax_root_acc={raw_acc:.3f}")
    for p_self in [0.0, 0.1, 0.15, 0.2, 0.3]:
        if p_self == 0.0:
            continue
        sm = decode(root_p, qual_p, bt, p_self)
        sm_root = np.array([sm[beat_idx((s + e) / 2)][0] for (s, e), l in zip(ivs, labs) if parse_root(l) is not None])
        acc = (gt_roots == sm_root).mean()
        print(f"  p_self={p_self:.2f}: smoothed_root_acc={acc:.3f}  delta={acc - raw_acc:+.3f}")
    print()
