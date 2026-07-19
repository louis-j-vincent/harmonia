"""symstruct_adaptive_scale.py — ADAPTIVE per-song SCALE selection using the
learned key-norm (variable-span) encoder.

The fixed-scale oracle's +0.05 over block8 came entirely from picking the right
section scale PER SONG (docs/known_issues.md: level8 optimal on 56%, but 16/32
win on the rest). This tests whether the learned encoder's own (unsupervised)
recurrence signal can pick that scale — a lighter, more robust form of "adaptive
hierarchy" than free-form merging, targeting exactly where the headroom is.

For each song: learned-union labels at scales {4,8,16,32}; then
  (a) each fixed scale mean V_F,
  (b) GT-ORACLE best scale (ceiling),
  (c) UNSUPERVISED selector: pick the scale maximizing repeat-clarity.
Loads the variable-span encoder. No audio/commits.
"""
from __future__ import annotations
import sys, argparse, collections
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import (nuclear_spans, bar_tokens, key_pc, NC_ROOT,
                               NC_QUAL, MAXSPAN)
from symstruct_adaptive import load_encoder, SongEmbedder
from symstruct import load_corpus, vmeasure, predict_blockmatch


def learned_union_labels(feat, model, shift, size, tau):
    n = len(feat)
    if n < size:
        return ["A"] * n
    spans = nuclear_spans(n, size)
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


def repeat_clarity(labels):
    """Unsupervised: fraction of bars whose label recurs (>=2 runs), lightly
    penalizing over-fragmentation. Higher = cleaner song-form repetition."""
    runs = []
    for l in labels:
        if not runs or runs[-1][0] != l:
            runs.append([l, 1])
        else:
            runs[-1][1] += 1
    if len(runs) <= 1:
        return 0.0
    cnt = collections.Counter(l for l, _ in runs)
    rep_labels = {l for l, c in cnt.items() if c >= 2}
    rep_bars = sum(n for l, n in runs if l in rep_labels)
    frac_rep = rep_bars / len(labels)
    frac_distinct_runs = len(cnt) / len(runs)
    return frac_rep * (1 - 0.5 * (1 - min(1.0, 2 * (1 - frac_distinct_runs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="scratchpad/keynorm_varspan.pt")
    ap.add_argument("--tau", type=float, default=0.75)
    args = ap.parse_args()
    model, ck = load_encoder(args.enc)
    keynorm = ck["args"].get("keynorm", True)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    val = [corpus[i] for i in ck["val_ids"]]
    test = [corpus[i] for i in ck["test_ids"]]
    scales = (4, 8, 16, 32)

    # pick tau on val (maximize best-fixed-scale? use scale=8 as anchor)
    def eval_song_scales(c, tau):
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        out = {}
        for sz in scales:
            lab = learned_union_labels(c["feat"], model, shift, sz, tau)
            out[sz] = lab
        return out

    for tau in [args.tau]:
        per_scale = {sz: [] for sz in scales}
        oracle, unsup, b8 = [], [], []
        for c in test:
            gt = c["labels"]
            labs = eval_song_scales(c, tau)
            vfs = {sz: vmeasure(gt, labs[sz])[0] for sz in scales}
            for sz in scales:
                per_scale[sz].append(vfs[sz])
            oracle.append(max(vfs.values()))
            # unsupervised pick
            best_sz = max(scales, key=lambda sz: repeat_clarity(labs[sz]))
            unsup.append(vfs[best_sz])
            b8.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=8))[0])
        # constrained selector: default scale 8, upgrade to 16 only when its
        # clarity beats 8 by a margin (avoids the fine-scale over-pick).
        for margin in (0.05, 0.10, 0.15):
            con = []
            for c in test:
                gt = c["labels"]
                labs = eval_song_scales(c, tau)
                c8, c16 = repeat_clarity(labs[8]), repeat_clarity(labs[16])
                pick = 16 if (c16 > c8 + margin) else 8
                con.append(vmeasure(gt, labs[pick])[0])
            print("  CONSTRAINED sel (8->16 margin=%.2f) V_F=%.3f" % (margin, np.mean(con)))
        print("=== learned encoder, per-song SCALE selection (tau=%.2f), TEST ===" % tau)
        for sz in scales:
            print("  fixed learned-union scale=%-2d  V_F=%.3f" % (sz, np.mean(per_scale[sz])))
        print("  UNSUPERVISED best-scale/song  V_F=%.3f  <- clarity selector" % np.mean(unsup))
        print("  GT-ORACLE  best-scale/song    V_F=%.3f  <- ceiling" % np.mean(oracle))
        print("  flat block8 (ref)             V_F=%.3f" % np.mean(b8))
        print("  [full-corpus refs: block8=0.681, fixed-scale hard-match oracle=0.732]")


if __name__ == "__main__":
    main()
