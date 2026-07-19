"""symstruct_adaptive.py — ADAPTIVE agglomerative hierarchy driven by the LEARNED
key-normalized similarity (replaces fixed 4->8->16->32 doubling + hard match).

Motivation (docs/known_issues.md 2026-07-17): the fixed-scale hierarchy forces
one power-of-2 block size per song; the learned key-norm encoder can embed a span
of ANY bar-length, so we let a section grow bottom-up as far as real recurrence
supports it — no doubling assumption. A locally-8-bar section stays 8; an
irregular 11/12-bar section is not snapped to 8 or 16.

Algorithm (per song, key-normalized):
  1. Nuclear = fine `nuclear`-bar spans; encode each with the trained BiLSTM.
  2. ADAPTIVE MERGE (agglomerative, adjacency-restricted): repeatedly try merging
     each adjacent segment pair. Merge score = RECURRENCE of the merged span =
     max cosine(emb(merged), emb(other equal-bar-length window elsewhere)). Merge
     the pair with the highest recurrence >= tau_merge. A section grows only while
     the larger unit still recurs; when extending it kills recurrence, it stops.
  3. LABEL final variable-length segments: union-find on cosine(emb) >= tau_label
     -> per-bar repeat labels. Evaluate with the same V-measure.

Loads a trained encoder saved by symstruct_learned.py (--save). No audio/commits.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import (BlockEncoder, nuclear_spans, bar_tokens, key_pc,
                               NC_ROOT, NC_QUAL, MAXSPAN)
from symstruct import load_corpus, vmeasure, predict_blockmatch


def load_encoder(path):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    model = BlockEncoder(hidden=a["hidden"], emb=a["emb"], arch=a["arch"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def _span_tokens(feat, s, e, shift, width):
    toks = [bar_tokens(feat[i]) for i in range(s, e)][:width]
    if shift:
        toks = [((r + shift) % 12 if r < NC_ROOT else r, q) for r, q in toks]
    roots = np.full(width, NC_ROOT, np.int64)
    quals = np.full(width, NC_QUAL, np.int64)
    for k, (r, q) in enumerate(toks):
        roots[k] = r; quals[k] = q
    return roots, quals, min(len(toks), width)


class SongEmbedder:
    """Embeds arbitrary bar-spans of one (key-normalized) song, with a cache."""
    def __init__(self, feat, model, shift, maxwidth):
        self.feat = feat; self.model = model; self.shift = shift
        self.maxw = maxwidth; self.cache = {}

    def emb(self, spans):
        """spans: list[(s,e)] -> (N,D) normalized tensor (cached per span)."""
        need = [sp for sp in spans if sp not in self.cache]
        if need:
            R, Q, L = [], [], []
            for (s, e) in need:
                r, q, l = _span_tokens(self.feat, s, e, self.shift, self.maxw)
                R.append(r); Q.append(q); L.append(l)
            with torch.no_grad():
                z = self.model(torch.tensor(np.stack(R)),
                               torch.tensor(np.stack(Q)),
                               torch.tensor(np.array(L)))
            for sp, v in zip(need, z):
                self.cache[sp] = v
        return torch.stack([self.cache[sp] for sp in spans])


def _recurrence(se, seg_span, other_windows):
    """Max cosine of merged span vs equal-length windows elsewhere."""
    if not other_windows:
        return -1.0
    e_main = se.emb([seg_span])[0]
    e_others = se.emb(other_windows)
    return float((e_others @ e_main).max())


def adaptive_segment(feat, model, shift, nuclear, tau_merge, stride=None,
                     floor=0, se=None):
    """Return list of final (s,e) segments via adaptive agglomerative merging.

    Phase 1 (recurrence): merge the adjacent pair whose merged span most strongly
    RECURS elsewhere, while that recurrence >= tau_merge. Repeated sections grow to
    their true (variable) length; a section stops growing when extending it kills
    recurrence.
    Phase 2 (floor cleanup, optional): sections with NO repeat signal never grow in
    phase 1. If `floor`>0, iteratively merge any sub-`floor`-bar segment into its
    most embedding-SIMILAR adjacent neighbor until all segments reach ~section
    scale. This restores a section-size prior ONLY where the data gives no
    recurrence evidence, keeping adaptivity where it does."""
    n = len(feat)
    spans = nuclear_spans(n, nuclear)
    if len(spans) <= 1:
        return spans
    stride = stride or nuclear
    maxw = MAXSPAN  # encoder was trained up to MAXSPAN bars; longer spans truncated
    if se is None:
        se = SongEmbedder(feat, model, shift, maxw)
    segs = list(spans)
    # ── phase 1: recurrence-driven merge ─────────────────────────────────────
    while len(segs) > 1:
        best_i, best_rec = -1, -1.0
        for i in range(len(segs) - 1):
            ms, me = segs[i][0], segs[i + 1][1]
            L = me - ms
            wins = []
            t = 0
            while t + L <= n:
                if not (t < me and t + L > ms):   # non-overlapping with merged span
                    wins.append((t, t + L))
                t += stride
            rec = _recurrence(se, (ms, me), wins)
            if rec > best_rec:
                best_rec, best_i = rec, i
        if best_rec < tau_merge:
            break
        i = best_i
        segs[i:i + 2] = [(segs[i][0], segs[i + 1][1])]
    # ── phase 2: section-floor cleanup ───────────────────────────────────────
    if floor > 0:
        while len(segs) > 1:
            # find the shortest sub-floor segment
            short = [(e - s, k) for k, (s, e) in enumerate(segs) if (e - s) < floor]
            if not short:
                break
            _, k = min(short)
            # pick the adjacent neighbor with higher embedding similarity
            cand = []
            if k > 0:
                cand.append(k - 1)
            if k < len(segs) - 1:
                cand.append(k + 1)
            E = se.emb([segs[k]] + [segs[c] for c in cand])
            sims = (E[1:] @ E[0]).tolist()
            nb = cand[int(np.argmax(sims))]
            lo, hi = min(k, nb), max(k, nb)
            segs[lo:hi + 1] = [(segs[lo][0], segs[hi][1])]
    return segs


def label_segments(feat, model, shift, segs, tau_label, maxw, se=None):
    """Union-find over final-segment embeddings -> per-bar labels."""
    n = len(feat)
    if se is None:
        se = SongEmbedder(feat, model, shift, maxw)
    E = se.emb(segs)
    S = (E @ E.t()).numpy()
    m = len(segs)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau_label:
                union(i, j)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(segs):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def predict_adaptive(feat, model, keystr, nuclear, tau_merge, tau_label,
                     keynorm=True, floor=0):
    n = len(feat)
    shift = (-key_pc(keystr) % 12) if keynorm else 0
    segs = adaptive_segment(feat, model, shift, nuclear, tau_merge, floor=floor)
    return label_segments(feat, model, shift, segs, tau_label, maxw=MAXSPAN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="scratchpad/keynorm_enc_s2.pt")
    ap.add_argument("--nuclear", type=int, default=2)
    args = ap.parse_args()

    model, ck = load_encoder(args.enc)
    keynorm = ck["args"].get("keynorm", True)
    corpus_all = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    test = [corpus_all[i] for i in ck["test_ids"]]
    val = [corpus_all[i] for i in ck["val_ids"]]
    print("loaded %s (keynorm=%s); val=%d test=%d songs" %
          (args.enc, keynorm, len(val), len(test)))

    merges = [0.55, 0.65, 0.75, 0.85]
    labels = [0.65, 0.75, 0.85]
    floors = [0, 4, 6, 8]

    def sweep(split):
        """Per-song single-embedder sweep over all (tm,tl,fl). Returns
        {(tm,tl,fl): mean_VF} and (for the caller) reuses embeddings."""
        acc = {(tm, tl, fl): [] for tm in merges for tl in labels for fl in floors}
        for c in split:
            feat, gt = c["feat"], c["labels"]
            n = len(feat)
            shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
            se = SongEmbedder(feat, model, shift, MAXSPAN)
            for tm in merges:
                for fl in floors:
                    segs = adaptive_segment(feat, model, shift, args.nuclear,
                                            tm, floor=fl, se=se)
                    for tl in labels:
                        lab = label_segments(feat, model, shift, segs, tl, n, se=se)
                        acc[(tm, tl, fl)].append(vmeasure(gt, lab)[0])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    val_sub = val[:150]
    res = sweep(val_sub)
    (tm, tl, fl), best_v = max(res.items(), key=lambda kv: kv[1])
    print("val(%d) best (tau_merge=%.2f tau_label=%.2f floor=%d) V_F=%.3f"
          % (len(val_sub), tm, tl, fl, best_v))

    va = [vmeasure(c["labels"], predict_adaptive(
        c["feat"], model, c.get("key"), args.nuclear, tm, tl, keynorm, fl))[0]
        for c in test]
    vb8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0]
           for c in test]
    seglens = []
    for c in test:
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        segs = adaptive_segment(c["feat"], model, shift, args.nuclear, tm, floor=fl)
        seglens += [e - s for s, e in segs]
    seglens = np.array(seglens)
    print("\n=== ADAPTIVE hierarchy (learned key-norm sim), TEST (nuclear=%d) ===" % args.nuclear)
    print("  adaptive V_F      = %.3f" % np.mean(va))
    print("  flat block8 (ref) = %.3f" % np.mean(vb8))
    print("  [full-corpus refs: block8=0.681, fixed-scale oracle=0.732]")
    print("  segment lengths (bars): mean=%.1f median=%d min=%d max=%d  distinct=%s"
          % (seglens.mean(), int(np.median(seglens)), seglens.min(),
             seglens.max(), sorted(set(seglens.tolist()))[:12]))


if __name__ == "__main__":
    main()
