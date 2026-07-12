"""Premise check for issue #21 — TRIGRAM version (CLAUDE.md rule #2).

Follow-up to scripts/check_bigram_premise.py. Agent C found transpose-invariant
*bigrams* cover only 63.8% of jazz1460 pairs (< 70% gate). Key hypothesis: jazz
is organised in *trigrams* (ii-V-I), and a bigram splits the cadence into ii->V
and V->I, so it cannot enforce the whole thing. This script asks the parallel
question one order up:

    what fraction of consecutive chord *triples* over jazz1460 fall into the
    top-50 transpose-invariant trigrams?

Transpose-invariant representation of a triple (chord_0 -> chord_1 -> chord_2):

    (interval_1 = (root_1 - root_0) mod 12, q0,
     interval_2 = (root_2 - root_1) mod 12, q1, q2)

so "ii-V-I in Bb" and "ii-V-I in G" collapse to the same trigram.

Gate (pre-registered, same spirit as the bigram check):
    top-50 coverage >= 70%  -> build the trigram Markov matrix AND the encoder
    top-50 coverage <  70%  -> skip the Markov prior, go straight to the
                               attention encoder (captures long patterns without
                               an explicit n-gram table).

Chord data comes from data/accomp_db/db.jsonl via song_chord_spans().
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
TOP_K = 50
CORPORA = ("jazz1460", "pop400", "blues50")

Trigram = tuple[int, str, int, str, str]


def triples_for_corpus(recs: list[dict]) -> list[Trigram]:
    """All consecutive-chord transpose-invariant trigrams across the recs.

    A triple is only formed within one song. song_chord_spans already merges
    identical consecutive spans, so an interval-0 step is a genuine quality
    change on the same root, not a held chord.
    """
    out: list[Trigram] = []
    for rec in recs:
        spans = song_chord_spans(rec)  # (t0, t1, root_pc, quality)
        seq = [(r % 12, q) for _, _, r, q in spans]
        for (r0, q0), (r1, q1), (r2, q2) in zip(seq, seq[1:], seq[2:]):
            out.append(((r1 - r0) % 12, q0, (r2 - r1) % 12, q1, q2))
    return out


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values())


def main() -> None:
    recs = [json.loads(l) for l in open(DB)]
    by_corpus = {c: [r for r in recs if r.get("corpus") == c] for c in CORPORA}

    print(f"DB: {len(recs)} songs total\n")
    print(f"{'corpus':<10} {'songs':>6} {'triples':>8} {'uniq':>7} "
          f"{'top50%':>8} {'top20%':>8}")
    print("-" * 54)

    for corpus in CORPORA + ("ALL",):
        if corpus == "ALL":
            these = [r for c in CORPORA for r in by_corpus[c]]
        else:
            these = by_corpus[corpus]
        triples = triples_for_corpus(these)
        if not triples:
            continue
        counts = Counter(triples)
        total = len(triples)
        ranked = counts.most_common()
        top50 = sum(c for _, c in ranked[:TOP_K]) / total
        top20 = sum(c for _, c in ranked[:20]) / total
        print(f"{corpus:<10} {len(these):>6} {total:>8} {len(counts):>7} "
              f"{top50:>7.1%} {top20:>7.1%}")

    # ── detailed verdict on the eval target (jazz1460) ─────────────────────────
    jz = triples_for_corpus(by_corpus["jazz1460"])
    jz_counts = Counter(jz)
    jz_total = len(jz)
    jz_top50 = sum(c for _, c in jz_counts.most_common(TOP_K)) / jz_total

    print(f"\nTop-15 jazz1460 trigrams (+iv1 q0 -> +iv2 q1 -> q2):")
    for (iv1, q0, iv2, q1, q2), c in jz_counts.most_common(15):
        print(f"  +{iv1:>2} {q0:>7} -> +{iv2:>2} {q1:>7} -> {q2:<7}  "
              f"{c:>5}  ({c/jz_total:.2%})")

    # information gain: does knowing the *pair* (q0,iv2,q1) reduce uncertainty
    # about the next (interval, quality) relative to the bigram condition?
    next_marg: Counter = Counter()
    cond_bi: dict[tuple, Counter] = defaultdict(Counter)   # condition on q1 only
    cond_tri: dict[tuple, Counter] = defaultdict(Counter)  # condition on (q0,iv2,q1)
    for iv1, q0, iv2, q1, q2 in jz:
        nxt = (iv2, q2)
        next_marg[nxt] += 1
        cond_bi[(q1,)][nxt] += 1
        cond_tri[(q0, iv2, q1)][nxt] += 1

    h_marg = entropy(next_marg)
    h_bi = sum((sum(c.values()) / jz_total) * entropy(c) for c in cond_bi.values())
    h_tri = sum((sum(c.values()) / jz_total) * entropy(c) for c in cond_tri.values())

    print(f"\nInformation diagnostic (jazz1460, transpose-invariant):")
    print(f"  H(next)                 = {h_marg:.2f} bits")
    print(f"  H(next | q_prev)        = {h_bi:.2f} bits  "
          f"(bigram, gain {h_marg - h_bi:.2f})")
    print(f"  H(next | q0,iv2,q_prev) = {h_tri:.2f} bits  "
          f"(trigram, gain {h_marg - h_tri:.2f})")
    print(f"  extra gain from trigram context = {h_bi - h_tri:.2f} bits")

    print()
    gate = 0.70
    verdict = "PASS" if jz_top50 >= gate else "FAIL"
    print(f"GATE: jazz1460 top-50 trigram coverage = {jz_top50:.1%} "
          f"vs {gate:.0%} -> {verdict}")
    if verdict == "FAIL":
        print("  -> skip the explicit trigram Markov matrix; go straight to the")
        print("     attention encoder (issue #21 section 3).")
    else:
        print("  -> build both the trigram Markov matrix and the encoder.")


if __name__ == "__main__":
    main()
