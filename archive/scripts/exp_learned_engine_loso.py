"""exp_learned_engine_loso.py — in-domain honesty control for the learned-engine
screen (companion to exp_learned_engine_transition.py).

The frozen-set result answers "does the learned engine beat musx on REAL audio?"
but conflates two things if it loses: (a) the head is a weak engine, or (b) it is
a fine engine that does not TRANSFER (POP909 MIDI-synth -> real audio domain gap).
This script isolates them: POP909 song-disjoint k-fold, segment-level root +
quality accuracy IN-DOMAIN, delta ablated.  If in-domain root >> frozen root, the
loss is transfer, not a broken head (the honest verdict qualifier).

Cache-only (reuses the 41 POP909 nnls bothchroma caches).  Modifies nothing.

Usage: .venv/bin/python scripts/exp_learned_engine_loso.py --folds 5 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import exp_learned_engine_transition as E  # noqa: E402
from harmonia.data.pop909_parser import POP909Parser  # noqa: E402


def grouped_records(n_songs, win):
    """POP909 Segs tagged with song id (for song-disjoint folds)."""
    parser = POP909Parser(E.POP909_DIR)
    stems = sorted(p.stem for p in E.NN_CACHE.glob("*.npz")
                   if p.stem.isdigit() and len(p.stem) == 3)
    groups: dict[str, list] = {}
    used = 0
    for st in stems:
        if used >= n_songs:
            break
        sg = parser.parse_song(st)
        if sg is None or len(sg.chord_events) < 5:
            continue
        try:
            frames, times = E.load_c_frames(st)
        except FileNotFoundError:
            continue
        med = float(np.median(frames.sum(1)) + 1e-9)
        evs = [e for e in sg.chord_events if e.root >= 0]
        prev_q = E.NQ
        recs = []
        for e in evs:
            t0, t1 = float(e.start_beat), float(e.end_beat)
            if t1 <= t0:
                continue
            x = E.clip_pool_abs(frames, times, t0, t1)
            d = E.onset_delta_abs(frames, times, t0, t1, med, win)
            q = E.QIDX[E.reduce_quality(e.quality)]
            recs.append(E.Seg(x, d, int(e.root) % 12, q, prev_q, t0, t1))
            prev_q = q
        if recs:
            groups[st] = recs
            used += 1
    return groups, used


def evaluate(groups, use_delta, folds, seeds):
    sids = sorted(groups)
    rng = np.random.default_rng(0)
    order = list(sids)
    rng.shuffle(order)
    fold_of = {s: i % folds for i, s in enumerate(order)}
    root_acc, qual_acc, joint_acc = [], [], []
    for seed in range(seeds):
        rc = qc = jc = tot = 0
        for f in range(folds):
            tr = [r for s in sids if fold_of[s] != f for r in groups[s]]
            te = [r for s in sids if fold_of[s] == f for r in groups[s]]
            if not te or not tr:
                continue
            head = E.HeadA(tr, seed, use_delta=use_delta, use_prevqual=False)
            X = np.array([r.x_abs for r in te], np.float32)
            D = np.array([r.d_abs for r in te], np.float32)
            roots = head.root_proba(X, D).argmax(1)
            prevq = np.array([r.prev_qual for r in te], np.int64)
            quals = head.quality(X, D, roots, prevq)
            gt_r = np.array([r.root for r in te])
            gt_q = np.array([r.qual for r in te])
            rc += int((roots == gt_r).sum())
            qc += int((quals == gt_q).sum())
            jc += int(((roots == gt_r) & (quals == gt_q)).sum())
            tot += len(te)
        root_acc.append(rc / tot); qual_acc.append(qc / tot); joint_acc.append(jc / tot)
    return (np.mean(root_acc), np.std(root_acc),
            np.mean(qual_acc), np.mean(joint_acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=41)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--win", type=float, default=0.20)
    args = ap.parse_args()

    groups, used = grouped_records(args.songs, args.win)
    n = sum(len(v) for v in groups.values())
    print(f"POP909 in-domain song-disjoint {args.folds}-fold "
          f"({used} songs, {n} segs, {args.seeds} seeds)")
    for name, ud in (("no-delta", False), ("with-delta", True)):
        r_mu, r_sd, q_mu, j_mu = evaluate(groups, ud, args.folds, args.seeds)
        print(f"  {name:11} root={r_mu:.4f}+/-{r_sd:.4f}  "
              f"quality={q_mu:.4f}  joint={j_mu:.4f}")


if __name__ == "__main__":
    main()
