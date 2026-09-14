"""Sanity-check the chord-LM tokenizer and print corpus statistics.

Run:  .venv/bin/python scripts/chord_lm_corpus_report.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, grid, vocab


def main() -> None:
    print("=" * 72)
    print("1. UNIT CHECKS on the within-bar placement rule")
    print("=" * 72)
    cases = [
        (1, 4, [0]), (2, 4, [0, 2]), (3, 4, [0, 2, 3]), (4, 4, [0, 1, 2, 3]),
        (2, 3, [0, 2]), (3, 3, [0, 1, 2]), (5, 4, [0, 1, 2, 3]),
    ]
    for n, b, want in cases:
        got = grid.place_in_bar(n, b)
        flag = "ok " if got == want else "FAIL"
        print(f"  {flag} {n} chords in {b}/4 -> {got}  (want {want})")

    print("\n  half-bar slots for a 4/4 bar:")
    C = vocab.chord_id(0, "maj")
    F = vocab.chord_id(5, "maj")
    G = vocab.chord_id(7, "dom")
    A = vocab.chord_id(9, "min")
    # 3 chords = 2+1+1, so the 2nd half-bar starts on the SECOND chord (beat 2),
    # and the third chord (beat 3) is the one a half-bar grid cannot hold.
    for toks, want in [([C], [C, C]), ([C, F], [C, F]), ([C, F, G], [C, F]),
                       ([C, F, G, A], [C, G])]:
        got = grid.bar_to_slots(toks, 2, 4)
        flag = "ok " if got == want else "FAIL"
        names = " ".join(vocab.token_name(t) for t in toks)
        print(f"  {flag} [{names}] -> {[vocab.token_name(t) for t in got]}")

    print("\n  REP round-trip:")
    absolute = [C, C, F, F, F, vocab.NC, F, G]
    toks, n_rep = grid.apply_repeats(absolute)
    back = grid.expand_repeats(toks)
    print("    absolute :", [vocab.token_name(t) for t in absolute])
    print("    tokens   :", [vocab.token_name(t) for t in toks])
    print("    expanded :", [vocab.token_name(t) for t in back])
    print("    round-trip", "ok" if back == absolute else "FAIL")

    print("\n" + "=" * 72)
    print("2. CORPUS")
    print("=" * 72)
    splits = corpus.load_splits(slots_per_bar=2)
    print("  " + splits.summary())
    allc = splits.train + splits.val + splits.test
    sym = sum(c.n_symbols for c in allc)
    unm = sum(c.n_unmapped for c in allc)
    print(f"  chord symbols {sym}, unmapped {unm} ({100 * unm / sym:.3f}%)")
    bars = sum(c.n_bars for c in allc)
    print(f"  bars {bars}, mean bars/song {bars / len(allc):.1f}")

    print("\n  token distribution:")
    cnt = collections.Counter(t for c in allc for t in c.tokens)
    tot = sum(cnt.values())
    print(f"    REP   {cnt[vocab.REP]:7d}  {100 * cnt[vocab.REP] / tot:5.2f}%"
          "   <- 'no change this half-bar'")
    print(f"    N.C.  {cnt[vocab.NC]:7d}  {100 * cnt[vocab.NC] / tot:5.2f}%")
    chords = tot - cnt[vocab.REP] - cnt[vocab.NC]
    print(f"    chord {chords:7d}  {100 * chords / tot:5.2f}%   <- a change happens")

    fam = collections.Counter()
    root = collections.Counter()
    for t, c in cnt.items():
        if vocab.is_chord(t):
            r, f = vocab.split_chord(t)
            fam[f] += c
            root[r] += c
    print("\n  family shares (of written-out chords):")
    for f, c in fam.most_common():
        print(f"    {f:5} {c:7d}  {100 * c / chords:5.2f}%")

    print(f"\n  distinct chord tokens used: {sum(1 for t in cnt if vocab.is_chord(t))}"
          f" / {vocab.N_CHORD}")
    print("  top 15 tokens:")
    for t, c in cnt.most_common(15):
        print(f"    {vocab.token_name(t):<9} {c:7d}  {100 * c / tot:5.2f}%")

    print("\n" + "=" * 72)
    print("3. EYEBALL — first 16 bars of three tunes")
    print("=" * 72)
    wanted = ["Autumn Leaves", "Blue Bossa", "All The Things You Are"]
    by_title = {c.title: c for c in allc}
    for w in wanted:
        c = by_title.get(w) or next((x for x in allc if x.title.startswith(w)), None)
        if c is None:
            continue
        print(f"\n  {c.title}  ({c.key}, {c.style}, {c.n_bars} bars, "
              f"split={corpus.split_of(c.title)})")
        head = grid.GriddedChart(title=c.title, tokens=c.tokens[:32],
                                 slots_per_bar=2, beats_per_bar=4,
                                 sections=c.sections[:16])
        for line in head.render(bars_per_line=4).splitlines():
            print("    " + line)


if __name__ == "__main__":
    main()
