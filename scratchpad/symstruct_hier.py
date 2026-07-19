"""symstruct_hier.py — HIERARCHICAL multi-scale symbolic structure model.

Extends flat block8 (V_F 0.68) into a parent-child pipeline over block sizes
4,8,16,32 bars. Each level's repeat clustering combines (a) its OWN direct
transposition-invariant chord match with (b) child-level agreement (do the
constituent sub-blocks already belong to matching repeat groups). Evaluated on
the same 1992-tune iReal corpus with the same V-measure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratchpad.symstruct import (load_corpus, vmeasure, _bar_sig, _block_sim,
                                   predict_blockmatch, predict_fixed8, NQ)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _spans(n, size):
    starts = list(range(0, n, size))
    sp = [(s, min(s + size, n)) for s in starts]
    if len(sp) >= 2 and (sp[-1][1] - sp[-1][0]) < size / 2:
        s, e = sp.pop()
        sp[-1] = (sp[-1][0], e)
    return sp


def level_labels(feat, size, sigs, *, child_bar_labels=None, alpha=1.0,
                 sim_threshold=0.75):
    """Per-bar repeat labels at one block size, optionally child-informed.

    sim(P,Q) = alpha * direct_chord_sim  +  (1-alpha) * child_label_agreement
    child_label_agreement = fraction of aligned bars whose child-level (size/2)
    labels match — i.e. 'both halves independently matched a repeat'.
    """
    n = len(feat)
    if n < size:
        # whole song one block
        return ["A"] * n, [("A", (0, n))]
    spans = _spans(n, size)
    labels = []
    reps = []  # (letter, (s,e))
    for (s, e) in spans:
        assigned = None
        for let, (rs, re) in reps:
            L = min(e - s, re - rs)
            if L <= 0:
                continue
            direct = _block_sim(sigs[s:s + L], sigs[rs:rs + L])
            if child_bar_labels is not None and alpha < 1.0:
                agree = np.mean([child_bar_labels[s + k] == child_bar_labels[rs + k]
                                 for k in range(L)])
                sim = alpha * direct + (1 - alpha) * agree
            else:
                sim = direct
            if sim >= sim_threshold:
                assigned = let
                break
        if assigned is None:
            assigned = LETTERS[len(reps) % 26]
            reps.append((assigned, (s, e)))
        labels.append(assigned)
    bar_labels = []
    for (s, e), lab in zip(spans, labels):
        bar_labels += [lab] * (e - s)
    return bar_labels[:n], list(zip([l for l in labels], spans))


def build_hierarchy(feat, *, alpha=0.5, sim_threshold=0.75,
                    sizes=(4, 8, 16, 32)):
    """Bottom-up: each level informed by the one below. Returns dict size->per-bar labels."""
    n = len(feat)
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    hier = {}
    prev = None
    for sz in sizes:
        if sz > n:
            break
        bl, _ = level_labels(feat, sz, sigs, child_bar_labels=prev,
                             alpha=(alpha if prev is not None else 1.0),
                             sim_threshold=sim_threshold)
        hier[sz] = bl
        prev = bl
    if not hier:
        hier[4] = ["A"] * n
    return hier


def _repeat_clarity(bar_labels):
    """Unsupervised 'is there real repeat structure at this grain' score:
    fraction of bars whose section label occurs in >1 distinct run (repeats),
    times (1 - unique-run penalty). Higher = clearer song-form repetition."""
    # runs
    runs = []
    for i, l in enumerate(bar_labels):
        if not runs or runs[-1][0] != l:
            runs.append([l, 1])
        else:
            runs[-1][1] += 1
    from collections import Counter
    cnt = Counter(l for l, _ in runs)
    n_runs = len(runs)
    if n_runs <= 1:
        return 0.0
    # bars covered by a label that appears in >=2 runs
    rep_labels = {l for l, c in cnt.items() if c >= 2}
    rep_bars = sum(n for l, n in runs if l in rep_labels)
    frac_rep = rep_bars / len(bar_labels)
    # penalize all-unique (n_runs == n_distinct) and over-fragmentation
    frac_distinct_runs = len(cnt) / n_runs
    return frac_rep * (1.0 - 0.5 * (1 - min(1.0, 2 * (1 - frac_distinct_runs))))


def predict_hier_adaptive(feat, *, alpha=0.5, sim_threshold=0.75):
    """Pick, per song, the hierarchy level with the strongest repeat clarity."""
    hier = build_hierarchy(feat, alpha=alpha, sim_threshold=sim_threshold)
    best_sz, best_score, best = None, -1, None
    for sz, bl in hier.items():
        sc = _repeat_clarity(bl)
        # mild preference for the section-scale grain (8/16) over very fine (4)
        if sc > best_score:
            best_score, best_sz, best = sc, sz, bl
    return best if best is not None else ["A"] * len(feat)


if __name__ == "__main__":
    import random
    random.seed(0)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    print("corpus: %d multi-section tunes" % len(corpus), file=sys.stderr)

    # per-fixed-level V_F, hierarchical child-informed levels, adaptive, flat block8
    def evalfn(fn, split):
        return np.mean([vmeasure(c["labels"], fn(c["feat"]))[0] for c in split])

    # alpha sweep for the child-informed combination weight
    print("\n== child-informed alpha sweep (level=16, informed by level-8) ==")
    for alpha in (1.0, 0.75, 0.5, 0.25):
        vs = []
        for c in corpus:
            h = build_hierarchy(c["feat"], alpha=alpha)
            sz = 16 if 16 in h else max(h)
            vs.append(vmeasure(c["labels"], h[sz])[0])
        print("  alpha=%.2f  level16 V_F=%.3f" % (alpha, np.mean(vs)))

    print("\n== per-fixed-level V_F (child-informed, alpha=0.5) ==")
    for sz in (4, 8, 16, 32):
        vs = []
        for c in corpus:
            h = build_hierarchy(c["feat"], alpha=0.5)
            if sz in h:
                vs.append(vmeasure(c["labels"], h[sz])[0])
        print("  level%-2d  V_F=%.3f  (n=%d)" % (sz, np.mean(vs), len(vs)))

    print("\n== headline comparison (1992 tunes) ==")
    print("  flat block8         V_F=%.3f" % evalfn(lambda f: predict_blockmatch(f, base_bars=8), corpus))
    print("  hier adaptive a=0.5 V_F=%.3f" % evalfn(lambda f: predict_hier_adaptive(f, alpha=0.5), corpus))
    print("  hier adaptive a=1.0 V_F=%.3f" % evalfn(lambda f: predict_hier_adaptive(f, alpha=1.0), corpus))
    # oracle upper bound: best level per song (uses GT — ceiling only)
    vs = []
    for c in corpus:
        h = build_hierarchy(c["feat"], alpha=0.5)
        vs.append(max(vmeasure(c["labels"], bl)[0] for bl in h.values()))
    print("  ORACLE best-level    V_F=%.3f  (GT-picked ceiling)" % np.mean(vs))
