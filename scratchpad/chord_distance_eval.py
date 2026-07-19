"""chord_distance_eval.py — compare V1(binary)/V2(weighted)/V3(TIV) chord-tone
similarity as the clustering criterion for structure detection, on the same
1992-tune clean-iReal corpus / V-measure setup used all night.

No neural net, no training — pure hand-crafted similarity, union-find at a
swept threshold tau (val-chosen, test-reported), same protocol as
symstruct_learned.py's downstream eval for a fair comparison.
"""
from __future__ import annotations
import sys, io, collections, random
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import (chord_vector_binary, chord_vector_weighted,
                             chord_vector_tiv, cosine)
from symstruct import vmeasure, predict_blockmatch, qbucket

FILES = ["jazz1460", "pop400", "blues50", "brazilian220",
         "country", "dixieland1", "latin_salsa50"]

SCHEMES = {
    "V1_binary": chord_vector_binary,
    "V2_weighted": chord_vector_weighted,
    "V3_tiv": chord_vector_tiv,
}


def key_pc(keystr):
    if not keystr:
        return 0
    pc = chord_root_pc(keystr.rstrip("-"))
    return pc if pc is not None else 0


def load_corpus_vectors(keynorm=True):
    """Per tune: per-bar chord vector (for EACH scheme) + labels."""
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(Path("data/ireal/%s.txt" % f))
        except Exception:
            continue
        for t in tunes:
            try:
                mc = tune_to_mma(t)
            except Exception:
                continue
            shift = (-key_pc(mc.key) % 12) if keynorm else 0
            bar_vecs = {name: [] for name in SCHEMES}
            labels = []
            for bar_no, section, slots in mc.timeline:
                accum = {name: None for name in SCHEMES}
                for (_, _, mma) in slots:
                    pc = chord_root_pc(mma)
                    if pc is None:
                        continue
                    rpc = (pc + shift) % 12
                    q = qbucket(mma)
                    for name, fn in SCHEMES.items():
                        v = fn(rpc, q)
                        accum[name] = v if accum[name] is None else accum[name] + v
                for name in SCHEMES:
                    empty = np.zeros(6, dtype=complex) if name == "V3_tiv" else np.zeros(12)
                    bar_vecs[name].append(accum[name] if accum[name] is not None else empty)
                labels.append(section)
            if len(labels) < 8 or len(set(labels)) < 2:
                continue
            out.append({"title": mc.title, "bar_vecs": bar_vecs, "labels": labels})
    return out


def nuclear_spans(n, size):
    sp = [(s, min(s + size, n)) for s in range(0, n, size)]
    if len(sp) >= 2 and (sp[-1][1] - sp[-1][0]) < size / 2:
        s, e = sp.pop()
        sp[-1] = (sp[-1][0], e)
    return sp


def block_sim(bars_a, bars_b):
    """POSITION-ALIGNED similarity: concatenate each block's per-bar vectors
    in order and cosine the concatenation. Expands to sum_k bars_a[k].bars_b[k]
    in the numerator (per the user's worked example), NOT a pool-then-dot
    (which would wrongly include cross-position terms)."""
    L = min(len(bars_a), len(bars_b))
    num = sum(np.vdot(bars_a[k], bars_b[k]).real for k in range(L))
    na = np.sqrt(sum(np.vdot(v, v).real for v in bars_a[:L]))
    nb = np.sqrt(sum(np.vdot(v, v).real for v in bars_b[:L]))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def predict_union(bar_vecs, n, size, tau):
    spans = nuclear_spans(n, size)
    block_bars = [bar_vecs[s:e] for (s, e) in spans]
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            if block_sim(block_bars[i], block_bars[j]) >= tau:
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
    random.seed(0)
    print("loading corpus (chord-vector, 3 schemes)...", file=sys.stderr)
    corpus = load_corpus_vectors(keynorm=True)
    print("corpus: %d multi-section tunes" % len(corpus), file=sys.stderr)

    ids = list(range(len(corpus)))
    random.shuffle(ids)
    nval = len(ids) // 5
    val_ids, test_ids = ids[:nval], ids[nval:]
    val = [corpus[i] for i in val_ids]
    test = [corpus[i] for i in test_ids]

    for size in (2, 8):
        print("\n=== nuclear size=%d ===" % size)
        for name in SCHEMES:
            taus = np.round(np.arange(0.3, 0.99, 0.03), 2)
            best_tau, best_v = None, -1
            for tau in taus:
                vs = [vmeasure(c["labels"], predict_union(
                    c["bar_vecs"][name], len(c["labels"]), size, tau))[0]
                    for c in val]
                mv = np.mean(vs)
                if mv > best_v:
                    best_v, best_tau = mv, tau
            vs_test = [vmeasure(c["labels"], predict_union(
                c["bar_vecs"][name], len(c["labels"]), size, best_tau))[0]
                for c in test]
            print("  %-12s val_tau*=%.2f  TEST V_F=%.3f (n=%d)" %
                  (name, best_tau, np.mean(vs_test), len(vs_test)))


if __name__ == "__main__":
    main()
