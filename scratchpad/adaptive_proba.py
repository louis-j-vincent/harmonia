"""adaptive_proba.py — Task 2: the ADAPTIVE agglomerative hierarchy
(symstruct_adaptive.py's algorithm), re-run with the PROBABILISTIC-root
VARIABLE-SPAN encoder (keynorm_proba_varspan.pt, trained by
symstruct_proba_varspan.py) as the merge-similarity source, per the brief's
explicit instruction. Also evaluates the SAME encoder at fixed scales (4-32)
as a control, so a like-for-like comparison against the token encoder
(keynorm_varspan.pt) is possible at every level, not just the adaptive one.

Hypothesis for why this might rescue the adaptive merge (which scored 0.46
with the discrete-token encoder, well below flat block8's 0.695): the
diagnosed failure mode there was "free-form recurrence merge over-merges via
spurious long-span matches" -- a smoother, soft-probability input (already
shown in symstruct_proba.py to be more ROBUST to noise than hard tokens) may
produce a less spiky, more conservative similarity landscape and reduce
spurious long-range merges. This is a genuine test of that hypothesis, not
an assumption -- report the number either way.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from symstruct_learned import BlockEncoder, nuclear_spans, bar_tokens, key_pc, NC_ROOT, MAXSPAN
from symstruct_proba import onehot13
from symstruct import load_corpus, vmeasure, predict_blockmatch
from phase_fix import nuclear_spans_phase


def load_proba_encoder(path):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    model = BlockEncoder(hidden=a["hidden"], emb=a["emb"], arch=a["arch"],
                         root_mode=a["root_mode"], qual_mode=a["qual_mode"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def _span_proba(feat, s, e, shift, width):
    roots = []
    for i in range(s, e):
        r, _ = bar_tokens(feat[i])
        if shift and r < NC_ROOT:
            r = (r + shift) % 12
        roots.append(r)
    vecs = np.stack([onehot13(r) for r in roots]) if roots else np.zeros((0, 13), np.float32)
    padded = np.zeros((width, 13), np.float32)
    L = min(len(vecs), width)
    if L:
        padded[:L] = vecs[:width]
    return padded, L


class ProbaSongEmbedder:
    """Embeds arbitrary bar-spans of one key-normalized song using the
    probabilistic-root variable-span encoder, with a cache (mirrors
    symstruct_adaptive.SongEmbedder but for 13-d proba input instead of
    discrete tokens)."""
    def __init__(self, feat, model, shift, maxwidth=MAXSPAN):
        self.feat = feat; self.model = model; self.shift = shift
        self.maxw = maxwidth; self.cache = {}

    def emb(self, spans):
        need = [sp for sp in spans if sp not in self.cache]
        if need:
            R, L = [], []
            for (s, e) in need:
                r, l = _span_proba(self.feat, s, e, self.shift, self.maxw)
                R.append(r); L.append(l)
            with torch.no_grad():
                z = self.model(torch.tensor(np.stack(R)), None,
                               torch.tensor(np.array(L)))
            for sp, v in zip(need, z):
                self.cache[sp] = v
        return torch.stack([self.cache[sp] for sp in spans])


def _recurrence(se, seg_span, other_windows):
    if not other_windows:
        return -1.0
    e_main = se.emb([seg_span])[0]
    e_others = se.emb(other_windows)
    return float((e_others @ e_main).max())


def adaptive_segment(feat, model, shift, nuclear, tau_merge, stride=None,
                     floor=0, se=None, phase=0):
    n = len(feat)
    spans = nuclear_spans_phase(n, nuclear, phase) if phase else nuclear_spans(n, nuclear)
    if len(spans) <= 1:
        return spans
    stride = stride or nuclear
    if se is None:
        se = ProbaSongEmbedder(feat, model, shift)
    segs = list(spans)
    while len(segs) > 1:
        best_i, best_rec = -1, -1.0
        for i in range(len(segs) - 1):
            ms, me = segs[i][0], segs[i + 1][1]
            L = me - ms
            wins = []
            t = 0
            while t + L <= n:
                if not (t < me and t + L > ms):
                    wins.append((t, t + L))
                t += stride
            rec = _recurrence(se, (ms, me), wins)
            if rec > best_rec:
                best_rec, best_i = rec, i
        if best_rec < tau_merge:
            break
        i = best_i
        segs[i:i + 2] = [(segs[i][0], segs[i + 1][1])]
    if floor > 0:
        while len(segs) > 1:
            short = [(e - s, k) for k, (s, e) in enumerate(segs) if (e - s) < floor]
            if not short:
                break
            _, k = min(short)
            cand = []
            if k > 0: cand.append(k - 1)
            if k < len(segs) - 1: cand.append(k + 1)
            E = se.emb([segs[k]] + [segs[c] for c in cand])
            sims = (E[1:] @ E[0]).tolist()
            nb = cand[int(np.argmax(sims))]
            lo, hi = min(k, nb), max(k, nb)
            segs[lo:hi + 1] = [(segs[lo][0], segs[hi][1])]
    return segs


def label_segments(feat, model, shift, segs, tau_label, se=None):
    n = feat.shape[0] if hasattr(feat, "shape") else len(feat)
    n = max(e for s, e in segs) if segs else 0
    if se is None:
        se = ProbaSongEmbedder(feat, model, shift)
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


def fixed_union_labels(feat, model, shift, size, tau, se=None, phase=0):
    n = len(feat)
    if n < size:
        return ["A"] * n
    spans = nuclear_spans_phase(n, size, phase) if phase else nuclear_spans(n, size)
    if se is None:
        se = ProbaSongEmbedder(feat, model, shift)
    return label_segments(feat, model, shift, spans, tau, se=se)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="scratchpad/keynorm_proba_varspan.pt")
    ap.add_argument("--nuclear", type=int, default=2)
    args = ap.parse_args()

    model, ck = load_proba_encoder(args.enc)
    keynorm = ck["args"].get("keynorm", True)
    corpus_all = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    val = [corpus_all[i] for i in ck["val_ids"]]
    test = [corpus_all[i] for i in ck["test_ids"]]
    print("loaded %s; val=%d test=%d" % (args.enc, len(val), len(test)))

    # ---- fixed-scale control (like-for-like vs token encoder) ----
    scales = (4, 8, 16, 32)
    taus_fixed = np.round(np.arange(0.50, 0.96, 0.05), 2)

    def eval_fixed(split, size, tau):
        vs = []
        for c in split:
            shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
            lab = fixed_union_labels(c["feat"], model, shift, size, tau)
            vs.append(vmeasure(c["labels"], lab)[0])
        return float(np.mean(vs))

    print("\n=== FIXED-SCALE control (proba varspan encoder) ===")
    val_sub = val[:150]
    for size in scales:
        best_tau, best_v = None, -1
        for tau in taus_fixed:
            v = eval_fixed(val_sub, size, tau)
            if v > best_v:
                best_v, best_tau = v, tau
        vt = eval_fixed(test, size, best_tau)
        print("  scale=%-2d  val_tau*=%.2f  TEST V_F=%.3f" % (size, best_tau, vt))

    # ---- adaptive agglomerative merge, proba similarity ----
    print("\n=== ADAPTIVE (proba varspan sim), nuclear=%d ===" % args.nuclear)
    merges = [0.55, 0.65, 0.75, 0.85]
    labels = [0.55, 0.65, 0.75, 0.85]
    floors = [0, 4, 8]

    def sweep(split):
        acc = {(tm, tl, fl): [] for tm in merges for tl in labels for fl in floors}
        for c in split:
            feat, gt = c["feat"], c["labels"]
            shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
            se = ProbaSongEmbedder(feat, model, shift)
            for tm in merges:
                for fl in floors:
                    segs = adaptive_segment(feat, model, shift, args.nuclear, tm, floor=fl, se=se)
                    for tl in labels:
                        lab = label_segments(feat, model, shift, segs, tl, se=se)
                        acc[(tm, tl, fl)].append(vmeasure(gt, lab)[0])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    res = sweep(val_sub)
    (tm, tl, fl), best_v = max(res.items(), key=lambda kv: kv[1])
    print("val(%d) best (tau_merge=%.2f tau_label=%.2f floor=%d) V_F=%.3f"
          % (len(val_sub), tm, tl, fl, best_v))

    va = []
    for c in test:
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        segs = adaptive_segment(c["feat"], model, shift, args.nuclear, tm, floor=fl)
        lab = label_segments(c["feat"], model, shift, segs, tl)
        va.append(vmeasure(c["labels"], lab)[0])
    vb8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0] for c in test]
    print("  adaptive V_F (proba sim) = %.3f" % np.mean(va))
    print("  flat block8 (ref)       = %.3f" % np.mean(vb8))


if __name__ == "__main__":
    main()
