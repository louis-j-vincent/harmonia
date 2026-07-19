"""Extract a handful of representative songs (size=8 clustering) for the
metric-granularity artifact chart: per-bar ref/est labels + both V_F scores."""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from symstruct import load_corpus, vmeasure, _bar_sig
from symstruct_grammar import _nuclear_spans, _cluster_types

corpus = load_corpus()
multi = [c for c in corpus if len(set(c["labels"])) >= 2]

size = 8
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rows = []
for c in multi:
    feat, gt = c["feat"], c["labels"]
    n = len(feat)
    if n < size or n > 64:
        continue
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    spans = _nuclear_spans(n, size)
    bsigs = [sigs[s:e] for s, e in spans]
    cid = _cluster_types(bsigs, sim_threshold=0.75, method="union")
    m = len(spans)
    if m < 2:
        continue

    glab_block = []
    for (s, e) in spans:
        cnt = collections.Counter(gt[s:e])
        glab_block.append(cnt.most_common(1)[0][0])
    block_vf = vmeasure([str(x) for x in glab_block], [str(x) for x in cid])[0]

    per_bar_pred = []
    for (s, e), cidx in zip(spans, cid):
        lab = letters[cidx % 26] if cidx is not None else "?"
        per_bar_pred += [lab] * (e - s)
    per_bar_pred = per_bar_pred[:n]
    if len(per_bar_pred) < n:
        per_bar_pred += [per_bar_pred[-1] if per_bar_pred else "A"] * (n - len(per_bar_pred))
    perbar_vf = vmeasure(gt, per_bar_pred)[0]

    # what the BLOCK-level scorer actually compares against: the majority-vote
    # GT label repeated across each block's bars (this is the information loss)
    gt_block_view = []
    for (s, e), lab in zip(spans, glab_block):
        gt_block_view += [lab] * (e - s)
    gt_block_view = gt_block_view[:n]
    if len(gt_block_view) < n:
        gt_block_view += [gt_block_view[-1] if gt_block_view else gt[-1]] * (n - len(gt_block_view))

    block_bounds = [s for (s, e) in spans if s > 0]

    rows.append({
        "title": c["title"], "file": c["file"], "n_bars": n,
        "gt": gt, "gt_block_view": gt_block_view, "pred": per_bar_pred,
        "block_bounds": block_bounds,
        "block_vf": round(float(block_vf), 3),
        "perbar_vf": round(float(perbar_vf), 3),
        "gap": round(float(block_vf - perbar_vf), 3),
    })

rows.sort(key=lambda r: -r["gap"])
print("n candidates:", len(rows))

# pick a spread: biggest gap, ~75th pct, ~median, near-zero gap (all with n_bars<=40 for readability)
readable = [r for r in rows if r["n_bars"] <= 40]
readable.sort(key=lambda r: -r["gap"])
picks = []
picks.append(readable[0])                                   # biggest gap
picks.append(readable[len(readable)//8])                    # high gap
mean_gap = float(np.mean([r["gap"] for r in readable]))
nontrivial2 = [r for r in readable if r["block_vf"] < 0.95]
typical = min(nontrivial2, key=lambda r: abs(r["gap"] - mean_gap))
picks.append(typical)                                        # ~typical (mean) gap, nontrivial
nontrivial = [r for r in readable if r["block_vf"] < 0.95]
near_zero = min(nontrivial, key=lambda r: abs(r["gap"]))
picks.append(near_zero)                                      # near-zero gap, nontrivial

seen = set()
uniq = []
for p in picks:
    if p["title"] not in seen:
        uniq.append(p)
        seen.add(p["title"])

for p in uniq:
    print("%-40s n=%2d block_vf=%.3f perbar_vf=%.3f gap=%.3f" % (
        p["title"][:40], p["n_bars"], p["block_vf"], p["perbar_vf"], p["gap"]))

Path("scratchpad/premise_check_examples.json").write_text(json.dumps(uniq, indent=2))
print("wrote scratchpad/premise_check_examples.json (%d songs)" % len(uniq))
