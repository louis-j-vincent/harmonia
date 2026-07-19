"""symstruct_grammar.py — GRAMMAR-INDUCTION (RePair/Sequitur) symbolic structure.

Redesign of the failed fixed-scale hierarchy (symstruct_hier.py, oracle 0.732).
Instead of forcing ONE global block size per song, structure emerges locally and
per-section via iterative pair-merging:

  Step 1  Nuclear blocks: split song into fixed 2- or 4-bar units.
  Step 2  Types: ALL-PAIRS transposition-invariant chord-sig similarity between
          nuclear blocks -> union-find connected components -> type labels
          (A B A B C D A B ...).  (greedy left-to-right also available.)
  Step 3  RePair: repeatedly replace the most-frequent adjacent type-pair (count
          >= merge_threshold) with a new composite symbol, recording the rule.
          Recurse until no pair repeats -> a per-song grammar tree/forest.
  Step 4  Default cut (no GT): coarsest merge-level whose per-bar labelling has
          2..8 distinct symbols.
  Step 5  Evaluation: (a) LEVEL oracle = best global merge-level per song; (b)
          FRONTIER oracle = best MIXED-depth cut through the parse forest (the
          real test: does the right answer EXIST in the tree, allowing different
          sections to be read off at different depths). Both vs iReal *A/*B/*C GT
          with the same V-measure (mir_eval.segment.nce).

Reuses symstruct.py: load_corpus, vmeasure, _bar_sig, _block_sim, predict_blockmatch.
No audio. No commits. Server untouched.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratchpad.symstruct import (load_corpus, vmeasure, _bar_sig, _block_sim,
                                   predict_blockmatch)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── Step 1: nuclear blocks ────────────────────────────────────────────────────
def _nuclear_spans(n, size):
    """Fixed `size`-bar spans; fold a too-short tail (<size/2) into the previous."""
    sp = [(s, min(s + size, n)) for s in range(0, n, size)]
    if len(sp) >= 2 and (sp[-1][1] - sp[-1][0]) < size / 2:
        s, e = sp.pop()
        sp[-1] = (sp[-1][0], e)
    return sp


# ── Step 2: cluster nuclear blocks into TYPES ─────────────────────────────────
def _cluster_types(block_sigs, *, sim_threshold, method="union"):
    """Return a list of int type-ids (0-based, first-occurrence order)."""
    m = len(block_sigs)
    if method == "greedy":
        # block8's convention: match the first earlier rep above threshold.
        reps = []  # (type_id, sig)
        ids = []
        for bs in block_sigs:
            assigned = None
            for tid, rsig in reps:
                if _block_sim(bs, rsig) >= sim_threshold:
                    assigned = tid
                    break
            if assigned is None:
                assigned = len(reps)
                reps.append((assigned, bs))
            ids.append(assigned)
        return ids
    # method == "union": ALL-PAIRS similarity graph + connected components.
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(m):
        for j in range(i + 1, m):
            if _block_sim(block_sigs[i], block_sigs[j]) >= sim_threshold:
                union(i, j)
    # relabel components in first-occurrence order
    remap = {}
    ids = []
    for i in range(m):
        r = find(i)
        if r not in remap:
            remap[r] = len(remap)
        ids.append(remap[r])
    return ids


# ── Step 3: RePair grammar induction ──────────────────────────────────────────
class Node:
    __slots__ = ("sym", "b0", "b1", "kids")

    def __init__(self, sym, b0, b1, kids=None):
        self.sym = sym          # symbol id (terminal type-id or composite id)
        self.b0, self.b1 = b0, b1
        self.kids = kids        # None (terminal) or (left_node, right_node)


def repair(type_ids, spans, *, merge_threshold=2):
    """RePair on the type-id sequence. Returns (levels, final_nodes, rules).

    levels: list of top-level sym-sequences, level 0 = type_ids, then after each
            merge (coarser as index grows).
    final_nodes: list[Node] = the fully-merged top-level parse forest (with spans).
    rules: dict composite_id -> (left_sym, right_sym).
    """
    # working sequence of Nodes (positional, keeps exact bar spans)
    seq = [Node(t, s, e) for t, (s, e) in zip(type_ids, spans)]
    levels = [[nd.sym for nd in seq]]
    rules = {}
    pair_to_new = {}                 # (a,b) -> composite id (consistent reuse)
    next_id = (max(type_ids) + 1) if type_ids else 0

    while True:
        # count adjacent symbol pairs (non-overlapping-safe count of positions)
        counts = collections.Counter()
        for i in range(len(seq) - 1):
            counts[(seq[i].sym, seq[i + 1].sym)] += 1
        if not counts:
            break
        (best_pair, best_ct) = max(counts.items(), key=lambda kv: kv[1])
        if best_ct < merge_threshold:
            break
        a, b = best_pair
        if best_pair in pair_to_new:
            new_sym = pair_to_new[best_pair]
        else:
            new_sym = next_id
            next_id += 1
            pair_to_new[best_pair] = new_sym
            rules[new_sym] = best_pair
        # left-to-right non-overlapping replacement
        out = []
        i = 0
        while i < len(seq):
            if (i < len(seq) - 1 and seq[i].sym == a and seq[i + 1].sym == b):
                l, r = seq[i], seq[i + 1]
                out.append(Node(new_sym, l.b0, r.b1, (l, r)))
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
        levels.append([nd.sym for nd in seq])
        if len(seq) < 2:
            break
    return levels, seq, rules


# ── per-bar labelling helpers ─────────────────────────────────────────────────
def _nodes_to_bar_labels(nodes, n):
    """Each top-level node -> its sym as label, filling its bar span."""
    lab = ["A"] * n
    for nd in nodes:
        for k in range(nd.b0, nd.b1):
            lab[k] = "S%d" % nd.sym
    return lab


def _seq_level_to_bar_labels(sym_seq, spans_after_level, n):
    # not used directly; we rebuild from nodes per level below
    raise NotImplementedError


def build(feat, *, nuclear=4, sim_threshold=0.75, merge_threshold=2,
          cluster="union"):
    """Full pipeline for one song. Returns dict with types, levels, nodes, rules
    and the list of per-level top-level Node lists (for level/frontier oracles)."""
    n = len(feat)
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    spans = _nuclear_spans(n, nuclear)
    block_sigs = [sigs[s:e] for s, e in spans]
    type_ids = _cluster_types(block_sigs, sim_threshold=sim_threshold,
                              method=cluster)
    levels, final_nodes, rules = repair(type_ids, spans,
                                        merge_threshold=merge_threshold)

    # Reconstruct the top-level Node list at EACH merge level (for level oracle).
    # Re-run the merge, snapshotting node lists.
    seq = [Node(t, s, e) for t, (s, e) in zip(type_ids, spans)]
    node_levels = [list(seq)]
    pair_to_new = {}
    next_id = (max(type_ids) + 1) if type_ids else 0
    while True:
        counts = collections.Counter()
        for i in range(len(seq) - 1):
            counts[(seq[i].sym, seq[i + 1].sym)] += 1
        if not counts:
            break
        best_pair, best_ct = max(counts.items(), key=lambda kv: kv[1])
        if best_ct < merge_threshold:
            break
        a, b = best_pair
        new_sym = pair_to_new.get(best_pair)
        if new_sym is None:
            new_sym = next_id
            next_id += 1
            pair_to_new[best_pair] = new_sym
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i].sym == a and seq[i + 1].sym == b:
                l, r = seq[i], seq[i + 1]
                out.append(Node(new_sym, l.b0, r.b1, (l, r)))
                i += 2
            else:
                out.append(seq[i]); i += 1
        seq = out
        node_levels.append(list(seq))
        if len(seq) < 2:
            break
    return {"n": n, "types": type_ids, "node_levels": node_levels,
            "final_nodes": final_nodes, "rules": rules}


# ── Automatic INTRO detection ─────────────────────────────────────────────────
def detect_intro(feat, *, nuclear=2, sim_threshold=0.75, cluster="union"):
    """Intro = leading run of SINGLETON nuclear blocks (types occurring only once
    in the whole song) up to the first block whose type also occurs elsewhere
    (i.e. belongs to ANY repeat group). Returns (intro_bars, n_intro_blocks,
    type_ids, spans). intro_bars=0 means no intro (song opens on a repeat)."""
    n = len(feat)
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    spans = _nuclear_spans(n, nuclear)
    block_sigs = [sigs[s:e] for s, e in spans]
    type_ids = _cluster_types(block_sigs, sim_threshold=sim_threshold,
                              method=cluster)
    cnt = collections.Counter(type_ids)
    first_repeat = None
    for i, t in enumerate(type_ids):
        if cnt[t] >= 2:
            first_repeat = i
            break
    if not first_repeat:            # None or 0 -> opens directly on a repeat
        return 0, 0, type_ids, spans
    intro_bars = spans[first_repeat][0]   # bars before first repeated block
    return intro_bars, first_repeat, type_ids, spans


# ── Step 4: default cut (NO GT) ───────────────────────────────────────────────
def default_cut_labels(info, *, lo=2, hi=8):
    """Coarsest merge-level whose per-bar labelling has lo..hi distinct symbols."""
    n = info["n"]
    node_levels = info["node_levels"]
    chosen = None
    # scan coarsest (last) -> finest, take first level in [lo,hi]
    for nodes in reversed(node_levels):
        lab = _nodes_to_bar_labels(nodes, n)
        d = len(set(lab))
        if lo <= d <= hi:
            chosen = lab
            break
    if chosen is None:
        # fall back: finest level with >=2 distinct, else level 0
        for nodes in node_levels:
            lab = _nodes_to_bar_labels(nodes, n)
            if len(set(lab)) >= 2:
                chosen = lab
                break
        if chosen is None:
            chosen = _nodes_to_bar_labels(node_levels[0], n)
    return chosen


# ── Step 5a: LEVEL oracle (best global merge-level per song) ───────────────────
def level_oracle_vf(info, gt):
    n = info["n"]
    best = 0.0
    for nodes in info["node_levels"]:
        lab = _nodes_to_bar_labels(nodes, n)
        best = max(best, vmeasure(gt, lab)[0])
    return best


# ── Step 5b: FRONTIER oracle (best MIXED-depth cut through the parse forest) ───
def _node_frontiers(nd, cap):
    """List of frontiers for a subtree. Each frontier = list of (b0,b1,sym).
    Capped: if the product would exceed `cap`, keep only {whole-node, full-expand}
    to stay tractable while still spanning shallow..deep for this node."""
    self_seg = [[(nd.b0, nd.b1, nd.sym)]]
    if nd.kids is None:
        return self_seg
    lf = _node_frontiers(nd.kids[0], cap)
    rf = _node_frontiers(nd.kids[1], cap)
    if len(lf) * len(rf) > cap:
        # reduce to shallow (this node) + deepest-only (fully expanded children)
        deep = lf[-1] + rf[-1]
        return self_seg + [deep]
    combos = [a + b for a in lf for b in rf]
    return self_seg + combos


def frontier_oracle_vf(info, gt, *, cap=4000, total_cap=60000):
    """Best V_F over mixed-depth cuts. Enumerate the cartesian product across
    top-level nodes; if it would exceed total_cap, fall back to level oracle."""
    n = info["n"]
    nodes = info["final_nodes"]
    per = [_node_frontiers(nd, cap) for nd in nodes]
    total = 1
    for p in per:
        total *= len(p)
        if total > total_cap:
            return level_oracle_vf(info, gt), True  # fell back
    best = 0.0
    # iterate cartesian product
    import itertools
    for combo in itertools.product(*per):
        lab = ["A"] * n
        for seg_list in combo:
            for (b0, b1, sym) in seg_list:
                for k in range(b0, b1):
                    lab[k] = "S%d" % sym
        best = max(best, vmeasure(gt, lab)[0])
    return best, False


# ── ALL-PAIRS per-scale flat block match (union-find, non-adjacency-aware) ────
def predict_blockmatch_union(feat, *, base_bars=8, sim_threshold=0.75):
    """Like symstruct.predict_blockmatch but clusters blocks with ALL-PAIRS
    union-find (connected components) instead of greedy left-to-right — so a
    non-adjacent repeat (e.g. the final A of an AABA form) links to the opening
    block regardless of position."""
    n = len(feat)
    if n < base_bars:
        return ["A"] * n
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    spans = _nuclear_spans(n, base_bars)
    block_sigs = [sigs[s:e] for s, e in spans]
    ids = _cluster_types(block_sigs, sim_threshold=sim_threshold, method="union")
    lab = ["A"] * n
    for (s, e), t in zip(spans, ids):
        for k in range(s, e):
            lab[k] = "S%d" % t
    return lab


# ── Clean-GT sanity: does all-pairs block matching recover GT sections? ────────
def _block_gt_label(spans, gt):
    """Majority GT section label per block."""
    out = []
    for s, e in spans:
        c = collections.Counter(gt[s:e])
        out.append(c.most_common(1)[0][0])
    return out


def sanity_allpairs(corpus, *, sizes=(4, 8), sim_threshold=0.75):
    """For each block size: all-pairs union-find over nuclear blocks on CLEAN
    iReal grids, then measure at the BLOCK level whether co-clustered blocks
    really share a GT section label. Pooled pairwise precision/recall + per-song
    block-level V-measure. Isolates matching logic from hierarchy logic."""
    print("=== CLEAN-GT all-pairs matching sanity (block level) ===")
    print("Q: when all-pairs groups two blocks, are they really the same GT section?")
    for size in sizes:
        tp = fp = fn = 0          # pooled over co-cluster / same-label block pairs
        vfs = []
        for c in corpus:
            feat, gt = c["feat"], c["labels"]
            n = len(feat)
            if n < size:
                continue
            sigs = [_bar_sig(feat[i]) for i in range(n)]
            spans = _nuclear_spans(n, size)
            bsigs = [sigs[s:e] for s, e in spans]
            cid = _cluster_types(bsigs, sim_threshold=sim_threshold, method="union")
            glab = _block_gt_label(spans, gt)
            m = len(spans)
            # per-song block-level V-measure (cluster-id vs GT block label)
            if m >= 2:
                vfs.append(vmeasure([str(x) for x in glab],
                                    [str(x) for x in cid])[0])
            for i in range(m):
                for j in range(i + 1, m):
                    same_cluster = cid[i] == cid[j]
                    same_gt = glab[i] == glab[j]
                    if same_cluster and same_gt:
                        tp += 1
                    elif same_cluster and not same_gt:
                        fp += 1
                    elif (not same_cluster) and same_gt:
                        fn += 1
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        print("  size=%2d bars: co-cluster PRECISION=%.3f (grouped blocks share GT sec) "
              "RECALL=%.3f (same-GT blocks get grouped)  block-lvl V_F=%.3f (n=%d)"
              % (size, prec, rec, np.mean(vfs), len(vfs)))
    print()


def aaba_spotcheck(corpus, *, size=8, sim_threshold=0.75, n_show=6):
    """Find tunes whose GT form is AABA-like (labels A A B A over 4 equal sections)
    and show whether all-pairs block matching links the final A to the opening A."""
    print("=== AABA spot-check: does all-pairs link the NON-ADJACENT final A? ===")
    shown = 0
    for c in corpus:
        feat, gt = c["feat"], c["labels"]
        n = len(feat)
        # collapse GT to section run-labels
        runs = []
        for l in gt:
            if not runs or runs[-1] != l:
                runs.append(l)
        # classic 4-section AABA only: run-labels == [A, A, B, A]
        if not (len(runs) == 4 and runs[0] == runs[1] == runs[3]
                and runs[2] != runs[0]):
            continue
        pred = predict_blockmatch_union(feat, base_bars=size,
                                        sim_threshold=sim_threshold)
        greedy = predict_blockmatch(feat, base_bars=size)
        # per-8bar-block cluster ids for compact display
        spans = _nuclear_spans(n, size)
        cid_u = [pred[s] for s, e in spans]
        cid_g = [greedy[s] for s, e in spans]
        glab = _block_gt_label(spans, gt)
        vf_u = vmeasure(gt, pred)[0]
        vf_g = vmeasure(gt, greedy)[0]
        print("  %-32s GT-blocks=%s" % (c["title"][:32], glab))
        print("    all-pairs=%s V_F=%.2f | greedy=%s V_F=%.2f"
              % (cid_u, vf_u, cid_g, vf_g))
        shown += 1
        if shown >= n_show:
            break
    if shown == 0:
        print("  (no clean 4-section AABA tunes found by the run-label heuristic)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="union", choices=["union", "greedy"])
    ap.add_argument("--merge-threshold", type=int, default=2)
    ap.add_argument("--sim-threshold", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0, help="subsample N tunes (0=all)")
    args = ap.parse_args()

    print("loading corpus...", file=sys.stderr)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    if args.limit:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(corpus))[:args.limit]
        corpus = [corpus[i] for i in idx]
    print("corpus: %d multi-section tunes" % len(corpus), file=sys.stderr)

    print("\ncluster=%s merge_threshold=%d sim_threshold=%.2f" %
          (args.cluster, args.merge_threshold, args.sim_threshold))
    print("baselines: flat block8(greedy)=0.681  fixed-scale-hier oracle=0.732\n")

    # ── (1) FIRST: clean-GT all-pairs matching sanity (block level) ──────────
    sanity_allpairs(corpus, sizes=(4, 8, 16), sim_threshold=args.sim_threshold)

    # ── (2) AABA spot-check ─────────────────────────────────────────────────
    aaba_spotcheck(corpus, size=8, sim_threshold=args.sim_threshold)

    # ── (3) per-scale ALL-PAIRS (union-find) flat block match vs greedy ─────
    print("=== per-scale ALL-PAIRS (union-find) block match vs greedy block8 ===")
    for size in (4, 8, 16, 32):
        vu = np.array([vmeasure(c["labels"],
                       predict_blockmatch_union(c["feat"], base_bars=size,
                       sim_threshold=args.sim_threshold))[0] for c in corpus])
        print("  union block%-2d  V_F mean=%.3f median=%.3f" %
              (size, vu.mean(), np.median(vu)))
    vg = np.array([vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0]
                   for c in corpus])
    print("  greedy block8  V_F mean=%.3f median=%.3f  (prior baseline)" %
          (vg.mean(), np.median(vg)))
    print()

    if args.limit == 0 and len(corpus) > 500:
        # frontier enumeration is O(product of cuts); bound it to a subsample
        rng = np.random.default_rng(7)
        gidx = rng.permutation(len(corpus))[:500]
        gcorpus = [corpus[i] for i in gidx]
        print("(grammar/frontier numbers on a 500-tune subsample for tractability)\n")
    else:
        gcorpus = corpus

    for nuclear in (4, 2):
        b8, defcut, lvl_or, fr_or = [], [], [], []
        n_fallback = 0
        for c in gcorpus:
            gt = c["labels"]
            b8.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=8))[0])
            info = build(c["feat"], nuclear=nuclear,
                         sim_threshold=args.sim_threshold,
                         merge_threshold=args.merge_threshold,
                         cluster=args.cluster)
            defcut.append(vmeasure(gt, default_cut_labels(info))[0])
            lvl_or.append(level_oracle_vf(info, gt))
            fv, fell = frontier_oracle_vf(info, gt)
            fr_or.append(fv); n_fallback += int(fell)
        b8 = np.array(b8); defcut = np.array(defcut)
        lvl_or = np.array(lvl_or); fr_or = np.array(fr_or)
        print("=== RePair grammar, nuclear = %d-bar ===" % nuclear)
        print("  flat block8 (ref)          V_F mean=%.3f median=%.3f" % (b8.mean(), np.median(b8)))
        print("  DEFAULT CUT (no GT)        V_F mean=%.3f median=%.3f   <- deployable" % (defcut.mean(), np.median(defcut)))
        print("  LEVEL oracle (global lvl)  V_F mean=%.3f median=%.3f" % (lvl_or.mean(), np.median(lvl_or)))
        print("  FRONTIER oracle (mixed cut)V_F mean=%.3f median=%.3f   <- info-in-tree ceiling (fallbacks=%d)" % (fr_or.mean(), np.median(fr_or), n_fallback))
        print()
