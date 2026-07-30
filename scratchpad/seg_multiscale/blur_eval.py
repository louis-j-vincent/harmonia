"""(b) How good is the blurred-SSM boundary prior, corpus-wide?

Reference = the SHARP detector's own letter-change bars (`vocab_sections`). This
measures AGREEMENT, not truth — stated up front, because the blur's job is to be
a prior on the sharp pass, and a prior is only useful if it lands where the sharp
pass would have looked anyway.

Tolerance ±2 bars, matching the tolerance Louis used by hand on Don't Know Why.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from blur import change_zones, novelty, gaussian_blur, peaks   # noqa: E402
from harness import all_charts                                 # noqa: E402
from harmonia.models.section_vocab import (                    # noqa: E402
    build_slots, chord_ssm, vocab_sections)

TOL = 2


def prf(pred: list[int], ref: list[int], tol: int = TOL):
    """Greedy one-to-one match within ±tol."""
    used = set()
    hit = 0
    for p in pred:
        cand = [r for r in ref if r not in used and abs(r - p) <= tol]
        if cand:
            used.add(min(cand, key=lambda r: abs(r - p)))
            hit += 1
    prec = hit / len(pred) if pred else 0.0
    rec = hit / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, hit


def _major(v, sigma):
    """Letter-change bars a COARSE pass could possibly see: both the section that
    ends and the section that begins are at least `sigma` bars long. Scoring
    recall against every letter change punishes the blur for skipping exactly the
    short fragments it is designed to skip."""
    out = []
    for i in range(1, len(v)):
        a, b = v[i - 1], v[i]
        if (a["bar1"] - a["bar0"]) >= sigma and (b["bar1"] - b["bar0"]) >= sigma:
            out.append(b["bar0"])
    return out


def main():
    charts = all_charts()
    sigmas = [2, 3, 4, 6, 8]
    rows = []
    for c in charts:
        v = vocab_sections(c["bars"], c["n_bars"], tonic_pc=c["tonic_pc"],
                           bpb=c["bpb"])
        if not v or len(v) < 2:
            continue
        ref = [s["bar0"] for s in v[1:]]                   # letter changes
        tok, roots, known = build_slots(c["bars"], c["n_bars"],
                                        tonic_pc=c["tonic_pc"], bpb=c["bpb"])
        S = chord_ssm(tok)
        row = {"slug": c["slug"], "n_bars": c["n_bars"], "ref": ref}
        for s in sigmas:
            pred = change_zones(S, s)
            row[f"s{s}"] = {"pred": pred, "prf": prf(pred, ref),
                            "prf_major": prf(pred, _major(v, s))}
        # control: the SHARP matrix with no blur at all, same kernel
        pred = peaks(novelty(S, 4.0), 4.0)
        row["sharp"] = {"pred": pred, "prf": prf(pred, ref),
                        "prf_major": prf(pred, _major(v, 4))}
        rows.append(row)

    (HERE / "blur_eval.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} charts where the sharp detector fires "
          f"(reference = its letter-change bars, ±{TOL} bars)\n")
    hdr = f"{'variant':10s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'#pred/chart':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for key in [f"s{s}" for s in sigmas] + ["sharp"]:
        P = np.mean([r[key]["prf"][0] for r in rows])
        R = np.mean([r[key]["prf"][1] for r in rows])
        F = np.mean([r[key]["prf"][2] for r in rows])
        N = np.mean([len(r[key]["pred"]) for r in rows])
        name = "no blur" if key == "sharp" else f"sigma={key[1:]} bars"
        print(f"{name:10s} {P:6.3f} {R:6.3f} {F:6.3f} {N:12.1f}")
    print("\n--- fair recall: reference restricted to letter changes between two "
          "sections BOTH >= sigma bars long ---")
    print(hdr)
    print("-" * len(hdr))
    for key in [f"s{s}" for s in sigmas] + ["sharp"]:
        P = np.mean([r[key]["prf_major"][0] for r in rows])
        R = np.mean([r[key]["prf_major"][1] for r in rows])
        F = np.mean([r[key]["prf_major"][2] for r in rows])
        N = np.mean([len(r[key]["pred"]) for r in rows])
        name = "no blur" if key == "sharp" else f"sigma={key[1:]} bars"
        print(f"{name:10s} {P:6.3f} {R:6.3f} {F:6.3f} {N:12.1f}")
    print(f"\nmean #letter-changes/chart (the reference) = "
          f"{np.mean([len(r['ref']) for r in rows]):.1f}")


if __name__ == "__main__":
    main()
