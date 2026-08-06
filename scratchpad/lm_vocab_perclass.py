"""Per-class counts for the extended chord-LM vocabulary decision.

The handoff's coverage screen answered "does the key space stay dense enough"
(yes: 92.9% -> 83.7% of occurrences in keys with >=20 observations, Q5 -> Q16).
It did NOT answer which classes actually earn a slot. A class with a few
hundred occurrences corpus-wide cannot be estimated in a trigram table no
matter how healthy the aggregate looks, and it drags the whole arm's tail down.

This prints the per-class share so the fold decisions are made on counts.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
import harmonia_min.chord_context_prior as cp                      # noqa: E402
from scratchpad.lm_vocab_coverage import Q8, Q16, make_parser, _orig_parse  # noqa: E402


def counts_for(table):
    parser, names = make_parser(table)
    cp.parse_harte_lite = parser
    try:
        songs, _ = cp.load_all_corpus_sequences()
    finally:
        cp.parse_harte_lite = _orig_parse
    c = Counter()
    for seqs in songs.values():
        for seq in seqs:
            for _r, q in seq:
                c[names[q]] += 1
    return c


for label, table in (("Q8", Q8), ("Q16", Q16)):
    c = counts_for(table)
    tot = sum(c.values())
    print(f"\n=== {label}: {len(c)} classes, {tot} chord tokens ===")
    print(f"  {'class':<10} {'count':>9} {'share':>8}")
    for name, n in c.most_common():
        print(f"  {name:<10} {n:>9} {100*n/tot:>7.2f}%")
    thin = [n for n, k in c.items() if k / tot < 0.005]
    print(f"  under 0.5% of tokens: {sorted(thin)}")
