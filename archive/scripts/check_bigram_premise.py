"""Premise check for issue #21 (CLAUDE.md rule #2).

Question: what fraction of consecutive chord pairs over the iReal corpus fall
into the top-50 transpose-invariant bigrams?  If >=70% a fixed bigram prior
covers enough of the corpus mass to be worth wiring; if <50% we should skip to
trigrams / a full sequence model.

Transpose-invariant representation of a pair (chord_i -> chord_j):

    (root_interval = (root_j - root_i) mod 12, quality_i, quality_j)

so "ii-V-I in Bb" and "ii-V-I in G" collapse to the same bigrams.

Chord data comes from data/accomp_db/db.jsonl (the same iReal-derived corpus
used by train_online / eval_irealb_e2e), via song_chord_spans().
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
TOP_K = 50
CORPORA = ("jazz1460", "pop400", "blues50")


def pairs_for_corpus(recs: list[dict]) -> list[tuple[int, str, str]]:
    """All consecutive-chord transpose-invariant bigrams across the given recs.

    A pair is only formed between two *distinct* chord spans within one song;
    identical consecutive spans are already merged by song_chord_spans, so a
    (interval=0, q, q) self-repeat pair means a genuine quality change on the
    same root, not a held chord.
    """
    out: list[tuple[int, str, str]] = []
    for rec in recs:
        spans = song_chord_spans(rec)  # (t0, t1, root_pc, quality)
        seq = [(r % 12, q) for _, _, r, q in spans]
        for (ri, qi), (rj, qj) in zip(seq, seq[1:]):
            out.append(((rj - ri) % 12, qi, qj))
    return out


def main() -> None:
    recs = [json.loads(l) for l in open(DB)]
    by_corpus = {c: [r for r in recs if r.get("corpus") == c] for c in CORPORA}

    print(f"DB: {len(recs)} songs total\n")
    print(f"{'corpus':<10} {'songs':>6} {'pairs':>8} {'uniq':>6} "
          f"{'top50%':>8} {'top20%':>8}")
    print("-" * 52)

    for corpus in CORPORA + ("ALL",):
        if corpus == "ALL":
            these = [r for c in CORPORA for r in by_corpus[c]]
        else:
            these = by_corpus[corpus]
        pairs = pairs_for_corpus(these)
        if not pairs:
            continue
        counts = Counter(pairs)
        total = len(pairs)
        ranked = counts.most_common()
        top50 = sum(c for _, c in ranked[:TOP_K]) / total
        top20 = sum(c for _, c in ranked[:20]) / total
        print(f"{corpus:<10} {len(these):>6} {total:>8} {len(counts):>6} "
              f"{top50:>7.1%} {top20:>7.1%}")

    # Detailed verdict on the eval target (jazz1460).
    jz_pairs = pairs_for_corpus(by_corpus["jazz1460"])
    jz_counts = Counter(jz_pairs)
    jz_total = len(jz_pairs)
    jz_top50 = sum(c for _, c in jz_counts.most_common(TOP_K)) / jz_total

    NOTE = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    print(f"\nTop-15 jazz1460 bigrams (interval, q_prev -> q_next):")
    for (iv, qi, qj), c in jz_counts.most_common(15):
        print(f"  +{iv:>2} {qi:>7} -> {qj:<7}  {c:>6}  ({c/jz_total:.1%})")

    # Secondary diagnostic: does knowing the previous chord actually reduce
    # uncertainty about the next?  Top-50 coverage is a proxy tuned for a small
    # hand-crafted rule set; a *learned smoothed* matrix benefits from the whole
    # distribution, so we also report the information gain directly.
    #   H(next)         : marginal entropy of the next (interval, quality)
    #   H(next | prev)  : entropy after conditioning on the previous chord
    # A large drop => a bigram prior is informative even if top-50 mass is < 70%.
    import math
    from collections import defaultdict

    # next symbol = (interval, q_next); condition = q_prev (transpose-invariant,
    # so the previous root is factored out and only its quality conditions).
    next_marg: Counter = Counter()
    cond: dict[str, Counter] = defaultdict(Counter)
    for iv, qi, qj in jz_pairs:
        next_marg[(iv, qj)] += 1
        cond[qi][(iv, qj)] += 1

    def entropy(counter: Counter) -> float:
        tot = sum(counter.values())
        return -sum((c / tot) * math.log2(c / tot) for c in counter.values())

    h_marg = entropy(next_marg)
    h_cond = sum(sum(c.values()) for c in cond.values())
    h_cond = sum(
        (sum(c.values()) / jz_total) * entropy(c) for c in cond.values()
    )
    print(f"\nInformation diagnostic (jazz1460, transpose-invariant):")
    print(f"  H(next)            = {h_marg:.2f} bits")
    print(f"  H(next | q_prev)   = {h_cond:.2f} bits")
    print(f"  info gain          = {h_marg - h_cond:.2f} bits "
          f"({(h_marg - h_cond) / h_marg:.0%} of marginal uncertainty removed)")

    print(f"\nVERDICT (jazz1460): top-{TOP_K} coverage = {jz_top50:.1%}")
    if jz_top50 >= 0.70:
        print("  >= 70% -> PREMISE PASSES: fixed bigram prior worth building.")
    elif jz_top50 >= 0.50:
        print("  50-70% -> MARGINAL: bigram prior partial; consider smoothing / trigrams.")
    else:
        print("  < 50% -> PREMISE FAILS: skip to trigrams / full sequence model.")


if __name__ == "__main__":
    main()
