"""Diagnostic plot: temporal vs pooled bass/inversion, per-seed + mean.

Reads the cv_result_*.json produced by bass_temporal_cv.py and renders a
grouped comparison of the key metrics (inv precision, inv recall, bass-pc acc,
net root acc after gate) for POOLED vs TEMPORAL, with per-seed jitter and the
prior full-corpus pooled reference lines.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
D = REPO / "scratchpad/bass_temporal"

configs = sys.argv[1:] or ["cv_result_bassnote_ctx0.4.json", "cv_result_bassnote_ctx1.0.json"]

fig, axes = plt.subplots(1, len(configs), figsize=(6*len(configs), 5), squeeze=False)
metrics = [
    ("inv precision", "pool_inv_pre", "tmp_inv_pre"),
    ("inv recall", "pool_inv_rec", "tmp_inv_rec"),
    ("bass-pc acc", "pool_bass_acc", "tmp_bass_acc"),
    ("net root acc\n(after gate)", "pool_gate_all", "tmp_gate_all"),
]
for ci, cfg in enumerate(configs):
    p = D / cfg
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    ax = axes[0][ci]
    x = np.arange(len(metrics))
    for i, (lbl, pk, tk) in enumerate(metrics):
        pv = np.array(r[pk]); tv = np.array(r[tk])
        ax.bar(i-0.18, pv.mean(), 0.34, color="#6b7280", label="pooled" if i==0 else None)
        ax.bar(i+0.18, tv.mean(), 0.34, color="#2563eb", label="temporal" if i==0 else None)
        ax.errorbar(i-0.18, pv.mean(), pv.std(), color="k", capsize=3)
        ax.errorbar(i+0.18, tv.mean(), tv.std(), color="k", capsize=3)
        ax.scatter(np.full_like(pv, i-0.18)+np.random.uniform(-.05,.05,len(pv)), pv, s=12, color="k", zorder=5)
        ax.scatter(np.full_like(tv, i+0.18)+np.random.uniform(-.05,.05,len(tv)), tv, s=12, color="k", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_title(cfg.replace("cv_result_","").replace(".json",""))
    ax.axhline(0.664, ls="--", color="#16a34a", lw=1, label="prior pooled bass .664")
    ax.axhline(0.204, ls=":", color="#dc2626", lw=1, label="prior pooled inv-P .204")
    ax.legend(fontsize=7); ax.set_ylim(0, 0.85); ax.grid(axis="y", alpha=.3)

fig.suptitle("RWC bass/inversion: TEMPORAL (bi-GRU over frames) vs POOLED snapshot", fontsize=12)
fig.tight_layout()
out = REPO / "docs/plots/bass_temporal_rwc.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
