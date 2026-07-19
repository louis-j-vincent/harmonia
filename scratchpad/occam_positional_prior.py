"""Part B: positional exception prior — P(deviation | position in 8-bar phrase).

For each pop tune, take one root per BAR (first chord), find its dominant
reciprocal-bigram vamp pair, mark bars whose root is off-vamp as deviations, and
tally deviation rate by bar-position mod 8 (and mod 4). Expected: concentrated at
phrase ends (turnarounds/cadences).
"""
import sys
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from pathlib import Path
from collections import Counter
import numpy as np
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords

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
    roots = []
    for _l, measure in sectionized_measures(tune):
        chosen = None
        for tok in split_chords(measure):
            t = tok.strip()
            if not t or t[0] in "npW":
                continue
            r = root_pc(t)
            if r is not None:
                chosen = r; break
        roots.append(chosen)          # None = empty/NC bar
    return roots


def vamp_pair(roots):
    rs = [r for r in roots if r is not None]
    big = Counter()
    for k in range(len(rs) - 1):
        if rs[k] != rs[k + 1]:
            big[frozenset((rs[k], rs[k + 1]))] += 1
    if not big:
        return None
    return set(big.most_common(1)[0][0])


for path, name in [("data/ireal/pop400.txt", "pop400")]:
    tunes = load_playlist(Path(path))
    dev8 = np.zeros(8); tot8 = np.zeros(8)
    dev4 = np.zeros(4); tot4 = np.zeros(4)
    for t in tunes:
        rb = bar_roots(t)
        pair = vamp_pair(rb)
        if pair is None:
            continue
        for i, r in enumerate(rb):
            if r is None:
                continue
            p8 = i % 8; p4 = i % 4
            tot8[p8] += 1; tot4[p4] += 1
            if r not in pair:
                dev8[p8] += 1; dev4[p4] += 1
    rate8 = dev8 / np.maximum(tot8, 1)
    rate4 = dev4 / np.maximum(tot4, 1)
    print(f"=== {name} ===")
    print("P(dev | pos mod 8):", [f"{x:.2f}" for x in rate8])
    print("P(dev | pos mod 4):", [f"{x:.2f}" for x in rate4])
    print("overall dev rate:", f"{dev8.sum()/tot8.sum():.3f}")
    np.savez("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad/positional.npz",
             rate8=rate8, rate4=rate4, overall=dev8.sum()/tot8.sum())
