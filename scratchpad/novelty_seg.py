"""novelty_seg.py — Task 1b: phase-FREE segmentation via a boundary-novelty
curve computed from the learned key-norm embedding, instead of a fixed-phase
block grid (nuclear_spans). Both unsupervised phase-selection heuristics in
phase_fix.py/phase_fix2.py failed to recover the oracle gap (see
docs/known_issues.md "Call 2 follow-up" entry, 2026-07-18) — this sidesteps
the phase-selection problem entirely rather than trying to solve it: instead
of tiling the song into fixed-phase blocks and picking one global phase, score
EVERY bar position as a candidate boundary using local embedding contrast
(classic SSM novelty-curve idea, MSAF/Foote-style, but the similarity function
is the trained learned key-norm encoder instead of raw chroma), then peak-pick
boundaries. No global phase parameter -- boundaries are decided locally per
song, so misalignment of a single rigid grid can't happen structurally.

Algorithm per song (key-normalized):
  1. For each interior bar i (w <= i <= n-w), embed window [i-w,i) and [i,i+w)
     with the trained variable-span encoder; novelty(i) = 1 - cosine(before,after).
  2. Boundaries = local maxima of novelty(i) with novelty(i) >= thresh and
     min-distance `w` apart (simple greedy peak-picking).
  3. Always include bar 0 and bar n as boundaries -> segments.
  4. Label segments via the SAME union-find over segment embeddings as
     symstruct_adaptive.py's label_segments (cosine >= tau_label).
Reuses keynorm_varspan.pt (trained on stride-2 windows at ALL phases, so it is
already phase-agnostic as a similarity function -- the point of this script is
to also make the SEGMENTATION step phase-agnostic, not just the encoder).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import key_pc, MAXSPAN
from symstruct_adaptive import load_encoder, SongEmbedder, label_segments
from symstruct import load_corpus, vmeasure, predict_blockmatch


def novelty_curve(feat, se, w):
    n = len(feat)
    idxs = list(range(w, n - w + 1))
    if not idxs:
        return [], np.array([])
    before = [(max(0, i - w), i) for i in idxs]
    after = [(i, min(n, i + w)) for i in idxs]
    Eb = se.emb(before)
    Ea = se.emb(after)
    nov = 1.0 - (Eb * Ea).sum(dim=1).numpy()
    return idxs, nov


def peak_pick(idxs, nov, thresh, min_dist):
    """Greedy: repeatedly take the highest remaining peak >= thresh, suppress
    a window of +-min_dist around it, until none remain."""
    if len(idxs) == 0:
        return []
    order = np.argsort(-nov)
    taken = []
    blocked = np.zeros(len(idxs), dtype=bool)
    for k in order:
        if nov[k] < thresh:
            break
        if blocked[k]:
            continue
        taken.append(idxs[k])
        for j, ix in enumerate(idxs):
            if abs(ix - idxs[k]) < min_dist:
                blocked[j] = True
    return sorted(taken)


def segments_from_boundaries(n, bounds):
    pts = sorted(set([0] + bounds + [n]))
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1) if pts[i + 1] > pts[i]]


def predict_novelty(feat, model, keystr, w, thresh, tau_label, keynorm=True,
                     min_seg=None):
    n = len(feat)
    shift = (-key_pc(keystr) % 12) if keynorm else 0
    se = SongEmbedder(feat, model, shift, MAXSPAN)
    if n <= 2 * w:
        return ["A"] * n
    idxs, nov = novelty_curve(feat, se, w)
    min_dist = min_seg if min_seg else w
    bounds = peak_pick(idxs, nov, thresh, min_dist)
    segs = segments_from_boundaries(n, bounds)
    return label_segments(feat, model, shift, segs, tau_label, MAXSPAN, se=se)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="scratchpad/keynorm_varspan.pt")
    args = ap.parse_args()
    model, ck = load_encoder(args.enc)
    keynorm = ck["args"].get("keynorm", True)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    val = [corpus[i] for i in ck["val_ids"]][:150]
    test = [corpus[i] for i in ck["test_ids"]]

    ws = [2, 4, 8]
    threshs = [0.15, 0.25, 0.35, 0.45]
    labels = [0.65, 0.75, 0.85]

    best = None
    for w in ws:
        for th in threshs:
            for tl in labels:
                vfs = [vmeasure(c["labels"], predict_novelty(
                    c["feat"], model, c.get("key"), w, th, tl, keynorm))[0]
                       for c in val]
                m = float(np.mean(vfs))
                if best is None or m > best[0]:
                    best = (m, w, th, tl)
    print("VAL best: V_F=%.3f  w=%d thresh=%.2f tau_label=%.2f" % best)
    _, w, th, tl = best

    vfs_test = [vmeasure(c["labels"], predict_novelty(
        c["feat"], model, c.get("key"), w, th, tl, keynorm))[0] for c in test]
    b8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0]
          for c in test]
    print("=== novelty-curve segmentation (phase-free), TEST n=%d ===" % len(test))
    print("  novelty-seg V_F   = %.3f" % np.mean(vfs_test))
    print("  flat block8 (ref) = %.3f" % np.mean(b8))
    print("  [learned-union scale=8 (deployed) ref = 0.689]")


if __name__ == "__main__":
    main()
