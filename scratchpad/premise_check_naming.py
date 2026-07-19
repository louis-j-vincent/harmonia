"""Premise check: is V-measure sensitive to LABEL NAMING (does it matter whether
a cluster is called 'A' or 'B'), or does it only care about the grouping
(partition) structure? Demonstrate directly on 'Isn't She Lovely?'."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from symstruct import vmeasure

songs = json.loads(Path("scratchpad/premise_check_examples.json").read_text())
s = next(x for x in songs if x["title"] == "Isn't She Lovely?")
gt, pred = s["gt"], s["pred"]

print("gt:  ", " ".join(gt))
print("pred:", " ".join(pred))

vf_asis = vmeasure(gt, pred)[0]
print("\nV_F as-is:                          %.3f" % vf_asis)

# relabel pred with a DIFFERENT but internally-consistent naming (A->Z, B->Y, C->X)
remap = {"A": "Z", "B": "Y", "C": "X"}
pred_renamed = [remap[x] for x in pred]
vf_renamed = vmeasure(gt, pred_renamed)[0]
print("V_F after renaming pred's clusters:  %.3f  (identical => naming is irrelevant)" % vf_renamed)

# what's the best possible score if we optimally re-pair GT letters <-> pred
# letters (Hungarian assignment on the contingency table), i.e. best-case
# per-bar ACCURACY under the most charitable possible naming?
from collections import Counter
import itertools
gt_labels = sorted(set(gt))
pred_labels = sorted(set(pred))
try:
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((len(gt_labels), len(pred_labels)))
    for i, gl in enumerate(gt_labels):
        for j, pl in enumerate(pred_labels):
            overlap = sum(1 for a, b in zip(gt, pred) if a == gl and b == pl)
            cost[i, j] = -overlap
    row, col = linear_sum_assignment(cost)
    best_map = {pred_labels[j]: gt_labels[i] for i, j in zip(row, col)}
    matched = sum(1 for x in pred_labels if x not in best_map)
    correct = sum(1 for a, b in zip(gt, pred) if best_map.get(b) == a)
    print("\nBest possible per-bar ACCURACY under optimal 1:1 label matching: %d/%d = %.1f%%"
          % (correct, len(gt), 100 * correct / len(gt)))
    print("optimal mapping:", best_map)
    print("\nbar-by-bar (mismatches marked *):")
    for i, (a, b) in enumerate(zip(gt, pred)):
        mark = "" if best_map.get(b) == a else "  *"
        print("  bar %2d: gt=%-2s pred=%-2s -> mapped=%-2s%s" % (i + 1, a, b, best_map.get(b), mark))
except ImportError:
    print("scipy not available for Hungarian check")
