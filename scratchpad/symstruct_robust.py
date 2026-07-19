"""symstruct_robust.py — how well does the symbolic block-match structure model
survive NOISY, bar-drifted chord input (the real-predicted-chords condition)?

Clean iReal chords give block8 V_F 0.68. Real music-x-lab chords on real audio
are ~83% root-correct (tonight's boundary eval) AND the bar grid is not clean
(tempo/downbeat error → inserted/deleted bars). We corrupt the clean iReal bars
to mimic that and re-measure, to quantify transfer risk WITHOUT needing audio.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratchpad.symstruct import (load_corpus, vmeasure, predict_blockmatch,
                                   predict_fixed8, NQ)


def corrupt(feat, labels, *, p_root, p_qual, p_drift, rng):
    """Return corrupted (feat, labels) with the SAME bar->label correspondence
    maintained across insert/delete so GT stays valid."""
    new_rows = []
    new_labels = []
    for i in range(len(feat)):
        row = feat[i].copy()
        # root substitution
        if row[:12].sum() > 0 and rng.random() < p_root:
            r0 = int(np.argmax(row[:12]))
            row[:12] = 0.0
            row[rng.integers(12)] = 1.0
        # quality substitution
        if row[12:].sum() > 0 and rng.random() < p_qual:
            row[12:] = 0.0
            row[12 + rng.integers(NQ)] = 1.0
        # bar drift: delete this bar, or duplicate it
        u = rng.random()
        if u < p_drift / 2:
            continue  # delete bar (its label vanishes too)
        new_rows.append(row)
        new_labels.append(labels[i])
        if u > 1 - p_drift / 2:
            new_rows.append(row.copy())  # duplicate bar (phase drift)
            new_labels.append(labels[i])
    if not new_rows:
        return feat, labels
    return np.stack(new_rows), new_labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    rng2 = np.random.default_rng(1)
    idx = rng2.permutation(len(corpus))[:400]
    sample = [corpus[i] for i in idx]
    print("robustness sample: %d multi-section tunes" % len(sample))
    print("%-40s %8s %8s" % ("condition (p_root/p_qual/p_drift)", "block8", "fixed8"))
    conditions = [
        (0.00, 0.00, 0.00),   # clean
        (0.10, 0.15, 0.00),   # label noise only
        (0.17, 0.25, 0.00),   # music-x-lab-level label noise
        (0.00, 0.00, 0.10),   # bar-drift only
        (0.00, 0.00, 0.20),
        (0.17, 0.25, 0.10),   # realistic combined
        (0.17, 0.25, 0.20),   # pessimistic combined
    ]
    for p_root, p_qual, p_drift in conditions:
        b8, fx = [], []
        for c in sample:
            f2, l2 = corrupt(c["feat"], c["labels"],
                             p_root=p_root, p_qual=p_qual, p_drift=p_drift, rng=rng)
            b8.append(vmeasure(l2, predict_blockmatch(f2, base_bars=8))[0])
            fx.append(vmeasure(l2, predict_fixed8(f2))[0])
        print("%-40s %8.3f %8.3f" % (
            "root=%.2f qual=%.2f drift=%.2f" % (p_root, p_qual, p_drift),
            np.mean(b8), np.mean(fx)))
