"""chord_distance_multiseed.py — Task 2 of the 2026-07-18 Call 3 brief.

chord_distance_eval.py's single run (random.seed(0)) reported V1=0.682,
V2=0.683, V3=0.675 at nuclear size=8 and claimed they're "inside the flat-
block8 range" but explicitly flagged NOT multi-seed validated. These schemes
have NO trained parameters, so "seed" here means the random VAL/TEST SPLIT
(chord_distance_eval.main()'s `random.shuffle(ids)` over the corpus), not a
training seed. Re-run the full val-tau-sweep -> test-report protocol across
>=5 different split seeds and report mean+-spread, per the Honesty-bar rule
(mandatory before citing any ranking among V1/V2/V3).
"""
from __future__ import annotations
import sys, io, random
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from chord_distance_eval import load_corpus_vectors, predict_union, SCHEMES
from symstruct import vmeasure

SEEDS = [0, 1, 2, 3, 4, 5, 6]
SIZE = 8


def run_one_seed(corpus, seed):
    ids = list(range(len(corpus)))
    random.Random(seed).shuffle(ids)
    nval = len(ids) // 5
    val_ids, test_ids = ids[:nval], ids[nval:]
    val = [corpus[i] for i in val_ids]
    test = [corpus[i] for i in test_ids]

    out = {}
    for name in SCHEMES:
        taus = np.round(np.arange(0.3, 0.99, 0.03), 2)
        best_tau, best_v = None, -1
        for tau in taus:
            vs = [vmeasure(c["labels"], predict_union(
                c["bar_vecs"][name], len(c["labels"]), SIZE, tau))[0]
                for c in val]
            mv = np.mean(vs)
            if mv > best_v:
                best_v, best_tau = mv, tau
        vs_test = [vmeasure(c["labels"], predict_union(
            c["bar_vecs"][name], len(c["labels"]), SIZE, best_tau))[0]
            for c in test]
        out[name] = {"tau_star": float(best_tau), "test_vf": float(np.mean(vs_test))}
    return out


def main():
    print("loading corpus once (shared across all seeds — same vectors, "
          "only the split changes)...", file=sys.stderr)
    buf = io.StringIO()
    with redirect_stdout(buf):
        corpus = load_corpus_vectors(keynorm=True)
    print("corpus: %d multi-section tunes" % len(corpus), file=sys.stderr)

    per_seed = {}
    for seed in SEEDS:
        res = run_one_seed(corpus, seed)
        per_seed[seed] = res
        print("seed=%d  " % seed + "  ".join(
            "%s: tau*=%.2f V_F=%.3f" % (n, r["tau_star"], r["test_vf"])
            for n, r in res.items()))

    print("\n=== summary over %d seeds (size=%d) ===" % (len(SEEDS), SIZE))
    summary = {}
    for name in SCHEMES:
        vals = np.array([per_seed[s][name]["test_vf"] for s in SEEDS])
        summary[name] = {"mean": float(vals.mean()), "std": float(vals.std()),
                          "min": float(vals.min()), "max": float(vals.max()),
                          "values": vals.tolist()}
        print("  %-12s mean=%.4f  std=%.4f  range=[%.3f, %.3f]" %
              (name, vals.mean(), vals.std(), vals.min(), vals.max()))

    # pairwise comparison: is any ranking real or within noise?
    print("\n=== pairwise mean differences (seed-paired, same splits) ===")
    names = list(SCHEMES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diffs = np.array([per_seed[s][a]["test_vf"] - per_seed[s][b]["test_vf"]
                              for s in SEEDS])
            print("  %s - %s: mean=%+.4f  std=%.4f  sign-consistent=%s (%d/%d positive)" %
                  (a, b, diffs.mean(), diffs.std(),
                   abs(diffs.mean()) > diffs.std(),
                   int((diffs > 0).sum()), len(diffs)))

    import json
    OUT = Path(__file__).resolve().parent / "chord_distance_multiseed_results.json"
    OUT.write_text(json.dumps({"per_seed": {str(k): v for k, v in per_seed.items()},
                                "summary": summary, "seeds": SEEDS, "size": SIZE},
                               indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
