"""Anti-crush gate #1 (symbolic, zero-cost): run the FULL razor + Bayes prior on
the GT symbolic chord sequences of pop400. On clean, certain data the razor must
be a near-no-op: target >= 99% of GT bars unchanged. Report the exact figure and
the worst-offending charts.
"""
import sys
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from pathlib import Path
from collections import Counter
import numpy as np
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords
from harmonia.models.chord_pipeline_v1 import occam_compress_bars

_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def root_pc(tok):
    tok = tok.strip()
    if not tok or tok[0] not in _LETTER:
        return None
    pc = _LETTER[tok[0]]; i = 1
    while i < len(tok) and tok[i] in "#b":
        pc += 1 if tok[i] == "#" else -1; i += 1
    return pc % 12


def bar_roots(tune):
    out = []
    for _l, measure in sectionized_measures(tune):
        r = None
        for tok in split_chords(measure):
            t = tok.strip()
            if t and t[0] not in "npW":
                r = root_pc(t)
                if r is not None:
                    break
        out.append(r)
    return out


tunes = load_playlist(Path("data/ireal/pop400.txt"))
tot_bars = 0
changed_root = 0
changed_any = 0
worst = []
n_applied = 0
for t in tunes:
    rb = bar_roots(t)
    roots = [r for r in rb if r is not None]
    m = len(roots)
    if m < 8:
        continue
    # one-hot posteriors (GT certain), conf = 1 everywhere
    post = np.full((m, 12), 1e-4)
    for i, r in enumerate(roots):
        post[i, r] = 1.0
    conf = np.ones(m)
    quals = [None] * m
    fam = [0] * m
    nr, nq, dec = occam_compress_bars(roots, quals, post, fam, bar_conf=conf)
    applied = any(d.get("applied") for d in dec)
    if applied:
        n_applied += 1
    ch_root = sum(1 for a, b in zip(roots, nr) if a != b)
    tot_bars += m
    changed_root += ch_root
    if ch_root:
        worst.append((t.title, ch_root, m))

print(f"pop400 symbolic self-test: {len(tunes)} tunes, {tot_bars} GT bars")
print(f"  vamp families APPLIED (razor acted): {n_applied} tunes")
print(f"  GT root-bars UNCHANGED: {100*(tot_bars-changed_root)/tot_bars:.2f}%  "
      f"(changed {changed_root}/{tot_bars})")
print(f"  charts with ANY root change: {len(worst)}")
for title, c, m in sorted(worst, key=lambda x: -x[1])[:8]:
    print(f"    {title}: {c}/{m} bars changed")
