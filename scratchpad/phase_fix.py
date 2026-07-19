"""phase_fix.py — Task 1a: unsupervised PHASE selection for the deployable
learned-union structure predictor (scale=8, keynorm_varspan.pt encoder).

Per docs/known_issues.md "GRID PHASE MISALIGNMENT" entry: fixed 8-bar blocks
starting at bar 0 are the DOMINANT remaining per-bar V_F loss source (oracle
phase correction 0.679->0.738 on the OLD hard-match block8). This script
tests whether the same idea helps the CURRENT deployable winner (learned
key-norm union at scale=8, V_F~0.69), using the repeat_clarity() unsupervised
heuristic from symstruct_adaptive_scale.py (already validated for scale
selection) applied to PHASE selection instead.

phase in [0, size): the grid's first block is [0,phase) (partial, kept only if
phase>0), then regular `size`-bar blocks from `phase` onward -- same convention
as premise_check_phase.py's blockmatch_phased, extended to the learned union
method.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import nuclear_spans, key_pc, MAXSPAN
from symstruct_adaptive import load_encoder, SongEmbedder
from symstruct_adaptive_scale import repeat_clarity
from symstruct import load_corpus, vmeasure, predict_blockmatch


def nuclear_spans_phase(n, size, phase):
    phase = phase % size
    sp = []
    if phase > 0:
        sp.append((0, min(phase, n)))
    i = phase
    while i < n:
        j = min(i + size, n)
        sp.append((i, j))
        i = j
    sp = [(s, e) for (s, e) in sp if e > s]
    return sp


def learned_union_labels_phased(feat, model, shift, size, tau, phase, se=None):
    n = len(feat)
    if n < size:
        return ["A"] * n
    spans = nuclear_spans_phase(n, size, phase)
    if se is None:
        se = SongEmbedder(feat, model, shift, MAXSPAN)
    E = se.emb(spans)
    S = (E @ E.t()).numpy()
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="scratchpad/keynorm_varspan.pt")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--size", type=int, default=8)
    args = ap.parse_args()

    model, ck = load_encoder(args.enc)
    keynorm = ck["args"].get("keynorm", True)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    val = [corpus[i] for i in ck["val_ids"]]
    test = [corpus[i] for i in ck["test_ids"]]
    size = args.size

    def eval_song_phases(c):
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        se = SongEmbedder(c["feat"], model, shift, MAXSPAN)
        return {p: learned_union_labels_phased(c["feat"], model, shift, size,
                                                args.tau, p, se=se)
                for p in range(size)}

    def run(split, name):
        phase0, oracle, unsup, b8 = [], [], [], []
        best_phase_hist = collections.Counter()
        for c in split:
            gt = c["labels"]
            labs = eval_song_phases(c)
            vfs = {p: vmeasure(gt, labs[p])[0] for p in range(size)}
            phase0.append(vfs[0])
            best_p = max(vfs, key=vfs.get)
            oracle.append(vfs[best_p])
            best_phase_hist[best_p] += 1
            up = max(range(size), key=lambda p: repeat_clarity(labs[p]))
            unsup.append(vfs[up])
            b8.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=8))[0])
        n_nonzero = sum(c for p, c in best_phase_hist.items() if p != 0)
        print("=== %s (n=%d) ===" % (name, len(split)))
        print("  phase=0 (current)             V_F=%.3f" % np.mean(phase0))
        print("  UNSUPERVISED clarity-phase     V_F=%.3f" % np.mean(unsup))
        print("  GT-ORACLE best-phase           V_F=%.3f  <- ceiling" % np.mean(oracle))
        print("  flat block8 (ref)              V_F=%.3f" % np.mean(b8))
        print("  fraction with nonzero optimal phase: %.1f%%"
              % (100.0 * n_nonzero / len(split)))
        return dict(phase0=np.mean(phase0), unsup=np.mean(unsup),
                    oracle=np.mean(oracle), b8=np.mean(b8))

    run(val, "VAL")
    run(test, "TEST")


if __name__ == "__main__":
    main()
