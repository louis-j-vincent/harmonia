"""phase_fix2.py — Task 1a retry after phase_fix.py's negative result.

DIAGNOSIS of why repeat_clarity() failed as a phase selector (phase_fix.py:
val 0.670->0.594, test 0.689->0.627, WORSE than phase=0): repeat_clarity
rewards "looks repetitive" (high frac of bars in a run that recurs, low
fragmentation) but block matching at a bad phase can accidentally produce
SPURIOUS repeats (misaligned blocks straddling two different sections can
still cosine-match by chance), which reads as high clarity but is wrong
structure. Clarity measures the OUTPUT of the union-find, so it can't tell a
true merge from a lucky misaligned one -- it's confounded with the very
process it's supposed to police, unlike in the (validated) scale-selection
use, where different scales don't create this same failure mode as sharply.

New hypothesis: pick the phase using a signal that depends only on WHERE
block boundaries fall relative to the chord sequence itself, not on the
clustering's output. Real section boundaries usually coincide with a chord
CHANGE (root or quality differs from the previous bar) -- score each phase by
how many of its interior block boundaries land on a chord-change bar,
normalized by boundary count. This is unsupervised (no GT) and independent of
the merge/clarity confound above.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import nuclear_spans, key_pc, bar_tokens, MAXSPAN
from symstruct_adaptive import load_encoder, SongEmbedder
from phase_fix import nuclear_spans_phase, learned_union_labels_phased
from symstruct import load_corpus, vmeasure, predict_blockmatch


def chord_change_mask(feat):
    n = len(feat)
    toks = [bar_tokens(feat[i]) for i in range(n)]
    return [1 if i > 0 and toks[i] != toks[i - 1] else 0 for i in range(n)]


def boundary_alignment_score(feat, size, phase):
    """Fraction of INTERIOR block boundaries (not bar 0, not song end) that
    land on a chord-change bar."""
    n = len(feat)
    cc = chord_change_mask(feat)
    spans = nuclear_spans_phase(n, size, phase)
    interior_starts = [s for (s, e) in spans[1:]]  # skip first span's start
    if not interior_starts:
        return 0.0
    hits = sum(cc[s] for s in interior_starts if s < n)
    return hits / len(interior_starts)


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

    def run(split, name):
        phase0, oracle, unsup_cc, b8 = [], [], [], []
        for c in split:
            gt = c["labels"]
            shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
            se = SongEmbedder(c["feat"], model, shift, MAXSPAN)
            labs = {p: learned_union_labels_phased(c["feat"], model, shift, size,
                                                    args.tau, p, se=se)
                    for p in range(size)}
            vfs = {p: vmeasure(gt, labs[p])[0] for p in range(size)}
            phase0.append(vfs[0])
            oracle.append(max(vfs.values()))
            cc_scores = {p: boundary_alignment_score(c["feat"], size, p)
                         for p in range(size)}
            best_cc_p = max(cc_scores, key=cc_scores.get)
            unsup_cc.append(vfs[best_cc_p])
            b8.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=8))[0])
        print("=== %s (n=%d) ===" % (name, len(split)))
        print("  phase=0 (current)              V_F=%.3f" % np.mean(phase0))
        print("  UNSUPERVISED chord-change-phase V_F=%.3f" % np.mean(unsup_cc))
        print("  GT-ORACLE best-phase            V_F=%.3f  <- ceiling" % np.mean(oracle))
        print("  flat block8 (ref)               V_F=%.3f" % np.mean(b8))

    run(val, "VAL")
    run(test, "TEST")
    run_constrained(test, "TEST", model, ck, keynorm, size, args.tau)


def run_constrained(split, name, model, ck, keynorm, size, tau, margins=(0.05,0.1,0.15,0.2)):
    for margin in margins:
        con, phase0 = [], []
        for c in split:
            gt = c["labels"]
            shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
            se = SongEmbedder(c["feat"], model, shift, MAXSPAN)
            cc0 = boundary_alignment_score(c["feat"], size, 0)
            best_p, best_score = 0, cc0
            for p in range(1, size):
                sc = boundary_alignment_score(c["feat"], size, p)
                if sc > best_score:
                    best_score, best_p = sc, p
            pick = best_p if (best_score > cc0 + margin) else 0
            lab = learned_union_labels_phased(c["feat"], model, shift, size, tau, pick, se=se)
            con.append(vmeasure(gt, lab)[0])
            lab0 = learned_union_labels_phased(c["feat"], model, shift, size, tau, 0, se=se)
            phase0.append(vmeasure(gt, lab0)[0])
        print("  CONSTRAINED cc-phase margin=%.2f  V_F=%.3f  (phase0=%.3f)" %
              (margin, np.mean(con), np.mean(phase0)))


if __name__ == "__main__":
    main()
