"""Premise check (research-loop Phase 0): is the reported '0.732 clean-GT oracle'
comparable to the '0.68-0.70 flat-block8' number it's benchmarked against?

symstruct_grammar.sanity_allpairs() computes V-measure over BLOCK-level units
(one frame per 8-bar block, GT label = majority vote within the block). Every
other V_F number in this project (flat block8, hierarchy, learned encoder) is
computed over PER-BAR units (one frame per bar) via symstruct.vmeasure().

These are not the same metric. Block-level V-measure has ~8x fewer, coarser
units per song and discards intra-block boundary information via majority
vote. This script re-derives the clean-GT sanity number BOTH ways on the same
clustering, to quantify how much of the '0.732 ceiling' vs '0.68 block8' gap
is a metric-granularity artifact rather than a real matching-quality gap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from symstruct import load_corpus, vmeasure, _bar_sig
from symstruct_grammar import _nuclear_spans, _cluster_types

corpus = load_corpus()
multi = [c for c in corpus if len(set(c["labels"])) >= 2]
print("corpus: %d multi-section tunes" % len(multi))

for size in (4, 8, 16):
    block_vfs = []
    perbar_vfs = []
    for c in multi:
        feat, gt = c["feat"], c["labels"]
        n = len(feat)
        if n < size:
            continue
        sigs = [_bar_sig(feat[i]) for i in range(n)]
        spans = _nuclear_spans(n, size)
        bsigs = [sigs[s:e] for s, e in spans]
        cid = _cluster_types(bsigs, sim_threshold=0.75, method="union")
        m = len(spans)
        if m < 2:
            continue

        # (a) BLOCK-level V_F as originally reported: majority-vote GT label per
        # block, one frame per block (this is what produced "0.732").
        import collections
        glab_block = []
        for (s, e) in spans:
            c_ = collections.Counter(gt[s:e])
            glab_block.append(c_.most_common(1)[0][0])
        block_vf = vmeasure([str(x) for x in glab_block], [str(x) for x in cid])[0]
        block_vfs.append(block_vf)

        # (b) PER-BAR V_F: expand the SAME clustering back to per-bar predicted
        # labels, compare against the REAL per-bar GT (no majority-vote loss).
        # This is apples-to-apples with flat block8's 0.68-0.70.
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        per_bar_pred = []
        for (s, e), cidx in zip(spans, cid):
            lab = letters[cidx % 26] if cidx is not None else "?"
            per_bar_pred += [lab] * (e - s)
        per_bar_pred = per_bar_pred[:n]
        if len(per_bar_pred) < n:
            per_bar_pred += [per_bar_pred[-1] if per_bar_pred else "A"] * (n - len(per_bar_pred))
        perbar_vf = vmeasure(gt, per_bar_pred)[0]
        perbar_vfs.append(perbar_vf)

    print("size=%2d  BLOCK-level V_F=%.3f (n=%d)   PER-BAR V_F=%.3f (n=%d)   gap=%.3f" % (
        size, np.mean(block_vfs), len(block_vfs),
        np.mean(perbar_vfs), len(perbar_vfs),
        np.mean(block_vfs) - np.mean(perbar_vfs)))
