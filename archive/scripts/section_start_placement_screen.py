"""iid noise is the WRONG noise model: a real recogniser makes the SAME mistake
on every occurrence of the same chord, which PRESERVES repeat structure.
Model that (systematic per-chord confusion) to bracket the honest number.
Also: fusion of the symbolic repeat score with the chroma novelty."""
import os, sys, numpy as np, mirdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from billboard_bar_gt import parse, bar_grid
rng = np.random.default_rng(0)
d = mirdata.initialize("billboard"); ts = d.load_tracks()

def prep(t):
    meta, lines = parse(t.salami_path); ch = t.chroma; cd = t.chords_majmin
    dur = float(ch[-1,0]); g = bar_grid(lines, dur)
    if len(g) < 4: return None
    edges = []; ss = []
    for (t0, bd, n, L) in g:
        if L: ss.append(len(edges))
        for j in range(n): edges.append(t0 + j*bd)
    edges.append(edges[-1]+g[-1][1]); nb = len(edges)-1
    if nb < 32 or len(ss) < 3: return None
    iv, lab = cd.intervals, cd.labels
    seq = []
    for i in range(nb):
        m = 0.5*(edges[i]+edges[i+1])
        j = np.searchsorted(iv[:,0], m, 'right')-1
        seq.append(lab[j] if 0 <= j < len(lab) else "N")
    return np.array(seq), ss, nb

def score(seq, sb, nb):
    sc = []
    for c in range(sb-4, sb+5):
        if c < 0 or c+8 > nb: sc.append(-9.0); continue
        A = seq[c:c+8]; best = 0.0
        for o in range(0, nb-8):
            if abs(o-c) < 4: continue
            best = max(best, float((A == seq[o:o+8]).mean()))
        sc.append(best)
    return np.array(sc)

for p in (0.15, 0.30, 0.45):
    out = []
    for k in list(ts)[:80]:
        try: r = prep(ts[k])
        except Exception: r = None
        if r is None: continue
        seq, ss, nb = r
        vocab = np.unique(seq)
        if len(vocab) > 1:                       # SYSTEMATIC confusion map
            cmap = {v: (rng.choice(vocab) if rng.random() < p else v) for v in vocab}
            seq = np.array([cmap[x] for x in seq])
        for sb in ss:
            if sb < 8 or sb > nb-9: continue
            sc = score(seq, sb, nb)
            out.append(int(np.argmax(sc - 1e-6*np.abs(np.arange(-4,5)))) - 4)
    out = np.array(out)
    print(f"SYSTEMATIC per-chord error {p:.0%}: exact-bar {np.mean(out==0)*100:.1f}%  |err|<=1 {np.mean(np.abs(out)<=1)*100:.1f}%  (n={len(out)})")
