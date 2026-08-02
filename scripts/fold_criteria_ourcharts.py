"""What the recommended rule would do on OUR OWN six charts.

Rule under test: two same-letter occurrences merge only if
    chroma_align >= 0.85   AND   len_agree >= 0.95
(chroma_align = time-normalised 64-slot cosine of the NNLS chord-tone chroma,
the pairwise analogue of the shipped STACK_COHERENCE.)

Prints, per chart: each letter's occurrence lengths, the pairwise values, and
the groups the rule produces vs. the single group the pipeline writes today.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
from fold_criteria_billboard import slot_cos  # noqa: E402

CHARTS = os.path.join(HERE, "harmonia_min/state/charts")
AUDIO = os.path.join(HERE, "docs/audio")
TAU_ALIGN, TAU_LEN = 0.85, 0.95


def chroma_span(C, dt, t0, t1, n=64):
    a, b = int(t0 / dt), max(int(t0 / dt) + 1, int(t1 / dt))
    X = C[a:min(b, len(C))]
    if len(X) < 4:
        return None
    lo = np.clip(np.linspace(0, len(X), n + 1)[:-1].astype(int), 0, len(X) - 1)
    hi = np.clip(np.linspace(0, len(X), n + 1)[1:].astype(int), 1, len(X))
    return np.array([X[i:j].mean(0) for i, j in zip(lo, hi)])


def main():
    from harmonia_min.nnls_features import extract_bothchroma
    out = {}
    for f in sorted(os.listdir(CHARTS)):
        if not f.startswith("min_") or not f.endswith(".json"):
            continue
        ch = json.load(open(os.path.join(CHARTS, f)))
        stem = f[4:-5]
        ap = Path(AUDIO) / f"{stem}.m4a"
        if not ap.exists():
            print(f"{stem}: no audio, skipped")
            continue
        arr, times = extract_bothchroma(ap)
        arr = np.asarray(arr)
        X = arr[:, 12:24] + arr[:, :12] if arr.shape[1] >= 24 else arr[:, :12]
        X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
        dt = float(np.median(np.diff(times)))
        grid = np.array(ch["barGrid"], float)
        print(f"\n=== {ch.get('title') or stem}  ({ch['nBars']} bars) ===")
        rec = {}
        for sec in ch["sections"]:
            rng = [tuple(r) for r in sec["barRanges"]]
            if len(rng) < 2:
                print(f"  {sec['label']}: 1 occurrence, nothing to decide")
                continue
            lens = [b - a + 1 for a, b in rng]
            cs = [chroma_span(X, dt, grid[a], grid[min(b + 1, len(grid) - 1)])
                  for a, b in rng]
            print(f"  {sec['label']}: {len(rng)} occurrences, lengths {lens}, "
                  f"written as {len(sec['bars'])} bars")
            groups, gi = [], {}
            for i in range(len(rng)):
                for g in groups:
                    j = g[0]
                    if cs[i] is None or cs[j] is None:
                        continue
                    al = slot_cos(cs[j], cs[i])
                    la = min(lens[i], lens[j]) / max(lens[i], lens[j])
                    if al >= TAU_ALIGN and la >= TAU_LEN:
                        g.append(i)
                        break
                else:
                    groups.append([i])
            for i in range(len(rng)):
                for j in range(i + 1, len(rng)):
                    if cs[i] is None or cs[j] is None:
                        continue
                    al = slot_cos(cs[i], cs[j])
                    la = min(lens[i], lens[j]) / max(lens[i], lens[j])
                    v = "MERGE" if (al >= TAU_ALIGN and la >= TAU_LEN) else "keep apart"
                    print(f"      occ{i+1}({lens[i]}b) vs occ{j+1}({lens[j]}b): "
                          f"align {al:.3f}  len {la:.3f}  -> {v}")
            written = sum(lens[g[0]] for g in groups)
            print(f"      RULE -> {len(groups)} block(s) "
                  f"{[[i+1 for i in g] for g in groups]}, {written} bars written "
                  f"(today: {len(sec['bars'])} bars for all {sum(lens)})")
            rec[sec["label"]] = {"lengths": lens, "groups": groups,
                                 "bars_rule": written,
                                 "bars_today": len(sec["bars"])}
        out[stem] = rec
    json.dump(out, open(os.path.join(HERE, "scratchpad/fold_criteria/ourcharts.json"),
                        "w"), indent=1)


if __name__ == "__main__":
    main()
