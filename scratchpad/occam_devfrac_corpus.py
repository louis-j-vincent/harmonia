"""Part A premise-check: is the GT deviation-fraction distribution informative?

For every iReal chart, compute the deviation fraction vs its OWN dominant
reciprocal-bigram vamp — the exact Occam vocabulary selection (top unordered
root pair by c[x->y]+c[y->x]) — and the minimal vocabulary size (pattern length).
If pop dev-fracs are flat/uniform, the corpus prior is uninformative → stop.
"""
import sys
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from pathlib import Path
from collections import Counter
import numpy as np
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords

_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def root_pc(tok: str):
    tok = tok.strip()
    if not tok or tok[0] not in _LETTER:
        return None
    pc = _LETTER[tok[0]]
    i = 1
    while i < len(tok) and tok[i] in "#b":
        pc += 1 if tok[i] == "#" else -1
        i += 1
    return pc % 12


def tune_roots(tune):
    roots = []
    for _label, measure in sectionized_measures(tune):
        for tok in split_chords(measure):
            t = tok.strip()
            if not t or t[0] in "npW":       # N.C./repeat/bass-only
                continue
            r = root_pc(t)
            if r is not None:
                roots.append(r)
    return roots


def dev_frac(roots):
    """Occam vocabulary: dominant reciprocal bigram pair; dev-frac = fraction of
    roots not in the pair. Returns (dev_frac, has_dominant_alternation, min_vocab90)."""
    m = len(roots)
    if m < 4:
        return None
    big = Counter()
    for k in range(m - 1):
        a, b = roots[k], roots[k + 1]
        if a != b:
            big[frozenset((a, b))] += 1
    if not big:
        return (0.0, False, 1)
    ranked = big.most_common()
    top_pair, top_n = ranked[0]
    second = next((c for p, c in ranked[1:] if not (p & top_pair)), 0)
    n_changes = sum(big.values())
    dominant = top_n >= max(2, 0.30 * n_changes) and top_n >= 1.5 * max(second, 1e-9)
    df = sum(1 for r in roots if r not in top_pair) / m
    # minimal vocabulary covering >=90% of bar-roots (pattern length)
    counts = Counter(roots).most_common()
    cum, kv = 0, 0
    for _r, c in counts:
        cum += c; kv += 1
        if cum / m >= 0.90:
            break
    return (df, dominant, kv)


def analyze(path, name):
    tunes = load_playlist(Path(path))
    dfs, doms, vocs = [], [], []
    for t in tunes:
        r = tune_roots(t)
        res = dev_frac(r)
        if res is None:
            continue
        df, dom, kv = res
        dfs.append(df); doms.append(dom); vocs.append(kv)
    dfs = np.array(dfs); vocs = np.array(vocs)
    print(f"\n=== {name} (n={len(dfs)}) ===")
    print(f"  dev-frac: mean={dfs.mean():.3f} median={np.median(dfs):.3f} "
          f"p10={np.percentile(dfs,10):.3f} p90={np.percentile(dfs,90):.3f} std={dfs.std():.3f}")
    print(f"  frac dominant-2-chord-alternation: {np.mean(doms):.2f}")
    print(f"  min-vocab(>=90%): median={int(np.median(vocs))} "
          f"2-chord={np.mean(vocs<=2):.2f} 3-4chord={np.mean((vocs>=3)&(vocs<=4)):.2f} 5+={np.mean(vocs>=5):.2f}")
    print(f"  where 0.335 sits: percentile={100*np.mean(dfs<=0.335):.0f}% "
          f"(frac of tunes with dev-frac <= 0.335)")
    # crude histogram
    hist, edges = np.histogram(dfs, bins=10, range=(0, 1))
    print("  hist[0..1]:", list(hist))
    return dfs, vocs


pop_df, pop_voc = analyze("data/ireal/pop400.txt", "pop400")
jazz_df, jazz_voc = analyze("data/ireal/jazz1460.txt", "jazz1460")
np.savez("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad/devfrac.npz",
         pop_df=pop_df, jazz_df=jazz_df, pop_voc=pop_voc, jazz_voc=jazz_voc)
