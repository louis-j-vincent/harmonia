"""section_pairs.py — 2026-07-18, section-level suggestion tool, task 2 infra.

Builds (block_sim, label) pairs at a given grain (8-bar standard, 4-bar
secondary) for the FULL iReal corpus, reusing `tau_auto_search.py`'s
corpus loader VERBATIM (bass/treble proxy vectors + per-bar `sections`
list, already parsed from the same MMA chart data) — no new corpus
parsing path.

Feature: per-tune 1-bar raw Gram matrices (bass, treble), derived to the
target grain via `hierarchy_shortcut.py`'s exact diagonal-prefix-sum
shortcut (position-aligned block_sim, NOT pool-then-cosine — matches the
project's established convention for block similarity, see
chord_distance_eval.block_sim's docstring). sim_combined = 50/50 average
of bass-grain-sim and treble-grain-sim, matching tau_auto_search's
sim_combined convention at bar level.

GT label (section-level, NOT bar-level chord-identity — deliberately
different from tau_auto_search's corrected label): label=1 iff the two
blocks' MAJORITY section letter (over each block's bars) match AND are
non-adjacent (>=min_gap blocks apart). This is the SAME "same GT section"
label the bar-level thread found broken at bar grain (~50% error even at
sim==1.0) -- but per that finding's own caveat, grain=8 blocks carry
"enough context to make section identity approximately decodable" (this is
also the flat-block8 V-measure structure-detection convention already
validated corpus-wide at V_F 0.68-0.70, docs/known_issues.md). Blocks with
an ambiguous majority (tie, or >50% of bars un-sectioned) are excluded.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tau_auto_search import load_corpus_bar_chords
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans

# CORRECTED 2026-07-18 (caught while cross-checking the premise-check
# example against the generated candidates -- Autumn Leaves' two A
# occurrences, bars 1-8 vs 9-16, are ADJACENT 8-bar blocks (block index 0
# vs 1). An earlier version set MIN_GAP_BLOCKS=1 by naively porting the
# bar-level MIN_GAP=4 "exclude trivial adjacent-sustain" convention to
# block grain -- but at block grain, adjacent blocks are NOT trivial; they
# are exactly the primary use case (back-to-back verse/verse, A/A section
# repeats). MIN_GAP_BLOCKS=1 silently excluded the user's own worked
# example from the entire pair pool used for both ROC calibration and
# candidate generation. Set to 0 (only exclude i==j, which the i<j loop
# already does).
MIN_GAP_BLOCKS = 0


def majority_section(sections_slice):
    sections_slice = [s for s in sections_slice if s]
    if not sections_slice:
        return None
    c = Counter(sections_slice)
    label, count = c.most_common(1)[0]
    if count / len(sections_slice) < 0.5:
        return None  # no clear majority
    return label


def block_gram_sim(prefix, sq, n, i0, j0, L):
    """`prefix` must be the PRECOMPUTED diagonal_prefix_sums(G) table for
    this tune's Gram matrix — computing it here per-pair (as an earlier
    version of this function did) is O(n^2) per call and made the corpus
    sweep untractable (~4min and still running on a single tune-count smoke
    test); callers must build it ONCE per tune, outside the pair loop."""
    d = j0 - i0
    num = diag_sum(prefix, n, d, i0, L)
    na = np.sqrt(float(np.sum(sq[i0:i0 + L])))
    nb = np.sqrt(float(np.sum(sq[j0:j0 + L])))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def build_section_pairs(corpus, grain, min_gap_blocks=MIN_GAP_BLOCKS):
    """Returns per_tune: list of lists of (sim_combined, label) tuples."""
    per_tune = []
    for c in corpus:
        bass = np.array(c["bass"])
        treb = np.array(c["treble"])
        sections = c["sections"]
        n = len(sections)
        if n < grain * 2:
            continue
        spans = nuclear_spans(n, grain)
        m = len(spans)
        if m < 2:
            continue
        maj = [majority_section(sections[s:e]) for (s, e) in spans]

        Gb = bass @ bass.T
        sqb = np.diag(Gb).copy()
        Gt = treb @ treb.T
        sqt = np.diag(Gt).copy()
        prefix_b = diagonal_prefix_sums(Gb)
        prefix_t = diagonal_prefix_sums(Gt)

        rows = []
        for i in range(m):
            if maj[i] is None:
                continue
            si, ei = spans[i]
            for j in range(i + 1 + min_gap_blocks, m):
                if maj[j] is None:
                    continue
                sj, ej = spans[j]
                L = min(ei - si, ej - sj)
                sb = block_gram_sim(prefix_b, sqb, n, si, sj, L)
                st = block_gram_sim(prefix_t, sqt, n, si, sj, L)
                sim = 0.5 * (sb + st)
                label = 1 if maj[i] == maj[j] else 0
                rows.append((sim, label))
        if rows:
            per_tune.append(rows)
    return per_tune


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Loading corpus...")
    corpus = load_corpus_bar_chords(max_tunes=None)
    print("  %d tunes, %.1fs" % (len(corpus), time.time() - t0))
    for grain in (4, 8):
        per_tune = build_section_pairs(corpus, grain)
        total = sum(len(r) for r in per_tune)
        pos = sum(sum(l for _, l in r) for r in per_tune)
        print("grain=%d: %d tunes usable, %d pairs, %d positive (%.1f%%)" %
              (grain, len(per_tune), total, pos, 100.0 * pos / total if total else 0))
