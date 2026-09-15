"""Control: do n-grams close the gap on the transformer if you just give them
more context?

The obvious objection to "attention beats a trigram" is that the transformer
simply sees more history, and an n-gram with the same window would do as well.
This pushes the interpolated n-gram to order 6 (a 5-slot = 2.5-bar context) and
reports where it saturates. Counted in all 12 keys, exactly like the models in
scripts/chord_lm_train.py, so the comparison is like-for-like.

    .venv/bin/python scripts/chord_lm_ngram_ceiling.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, data, evaluate


def main() -> None:
    splits = corpus.load_splits(slots_per_bar=2)
    tr = [c for ch in splits.train for c in data.chunk(ch, 256)]
    va = [c for ch in splits.val for c in data.chunk(ch, 256)]
    te = [c for ch in splits.test for c in data.chunk(ch, 256)]
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} sequences\n")
    print(evaluate.HEADER)
    print("-" * 100)
    for order in (2, 3, 4, 5, 6):
        t0 = time.time()
        ng = evaluate.NGramLM(order=order).fit(tr)
        ng.tune_lambdas(va[:120], n_iter=12)
        acc = evaluate.MetricAccumulator()
        for s in te:
            if len(s) >= 2:
                acc.add(ng.logprobs(s)[1:], np.array(s[1:], dtype=np.int64))
        m = acc.result()
        n_ctx = len(ng.counts[order - 1])
        print(m.table_row(f"{order}-gram"),
              f"   ctx={n_ctx:>8}  lam_top={ng.lambdas[-1]:.2f}  ({time.time()-t0:.0f}s)")
        del ng
        gc.collect()


if __name__ == "__main__":
    main()
